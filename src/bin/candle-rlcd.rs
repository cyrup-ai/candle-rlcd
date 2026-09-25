//! `candle-rlcd` command line: run a model, train one, evaluate it, and prepare data.
//!
//! ```text
//! candle-rlcd run --model <dir> --request request.json
//! candle-rlcd serve --model <dir> --port 8080
//! candle-rlcd bench --model <dir> --data requests.jsonl --concurrency 8
//! candle-rlcd train --train train.jsonl --eval eval.jsonl --out runs/x --init-encoder <modernbert>
//! candle-rlcd eval --model runs/x/final --data test.jsonl
//! candle-rlcd data ag-news --input train.csv --output train.jsonl
//! candle-rlcd tokenizer --data train.jsonl --out tok --vocab-size 8192
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_rlcd::bench;
use clap::{Parser, Subcommand};
use serde_json::Value;

#[derive(Parser)]
#[command(about = "Train and run ModernBERT + RLCD typed-decision models on Candle")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Answer a Jev-style request (`{"state": .., "questions": {..}}`) as JSON.
    Run {
        /// Model directory (rl_agent_config.json, model.safetensors, encoder/, tokenizer/).
        #[arg(long)]
        model: PathBuf,
        /// Request JSON file; reads stdin when omitted.
        #[arg(long)]
        request: Option<PathBuf>,
        /// f32, f16 or bf16.
        #[arg(long, default_value = "f32")]
        dtype: String,
        /// Run on CPU even when a GPU backend is compiled in.
        #[arg(long)]
        cpu: bool,
        /// Print load and inference timings to stderr.
        #[arg(long)]
        timings: bool,
    },
    /// Serve the model over HTTP with TypeSafe's Jev API (`POST /v1/systemone`,
    /// `GET /v1/models`).
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        /// Port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Address to bind (use 0.0.0.0 to accept remote connections).
        #[arg(long, default_value = "127.0.0.1")]
        host: std::net::IpAddr,
        /// Name reported in responses and by /v1/models (default: the model directory's name).
        #[arg(long)]
        model_name: Option<String>,
        /// Require `Authorization: Bearer <key>` (also read from CANDLE_RLCD_API_KEY).
        #[arg(long, env = "CANDLE_RLCD_API_KEY", hide_env_values = true)]
        api_key: Option<String>,
        /// Add laya's extra answer fields (answer_confidence, act_probability) and how each
        /// question was fitted (state_chunks, option_batches, truncated).
        #[arg(long)]
        extended: bool,
        /// A question whose instructions and options don't fit the model's input: `strict`
        /// answers 422, `report` cuts it and counts it in the x-truncated-questions header.
        #[arg(long, default_value = "strict", value_parser = ["strict", "report"])]
        truncation: String,
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// Load-test the engine in process, or a running server with --url.
    Bench {
        /// Model directory (in-process benchmark).
        #[arg(long, required_unless_present = "url")]
        model: Option<PathBuf>,
        #[arg(long, default_value = "f32")]
        dtype: String,
        #[arg(long)]
        cpu: bool,
        /// Benchmark a running server instead, e.g. http://127.0.0.1:8080.
        #[arg(long)]
        url: Option<String>,
        /// Requests to replay (JSONL of `{"state", "questions"}` records; extra keys ignored).
        #[arg(long)]
        data: PathBuf,
        /// Concurrent clients.
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        /// Total requests (cycles through the data).
        #[arg(long, default_value_t = 200)]
        requests: usize,
        /// Requests per client sent before timing starts.
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// Train with the direct proper-scoring loss.
    Train {
        #[command(flatten)]
        cfg: Box<candle_rlcd::train::TrainConfig>,
        #[arg(long)]
        cpu: bool,
    },
    /// Accuracy, NLL, Brier, ECE (raw and with the model's fitted temperatures).
    Eval {
        #[arg(long)]
        model: PathBuf,
        /// Labelled records (JSONL).
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        cpu: bool,
    },
    /// Convert public datasets to training records, or generate synthetic ones.
    Data {
        #[command(subcommand)]
        kind: DataCmd,
    },
    /// Train a byte-level BPE tokenizer on the text of training records (for models trained
    /// from scratch; fine-tuning keeps the pretrained tokenizer).
    Tokenizer {
        /// Records (JSONL); repeatable.
        #[arg(long, required = true)]
        data: Vec<PathBuf>,
        /// Output directory for tokenizer.json and tokenizer_config.json.
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 8192)]
        vocab_size: usize,
    },
}

#[derive(clap::Args)]
struct ModelArgs {
    /// Model directory (rl_agent_config.json, model.safetensors, encoder/, tokenizer/).
    #[arg(long)]
    model: PathBuf,
    /// f32, f16 or bf16.
    #[arg(long, default_value = "f32")]
    dtype: String,
    /// Run on CPU even when a GPU backend is compiled in.
    #[arg(long)]
    cpu: bool,
}

/// How inference runs; see `candle_rlcd::serve` for the design. Defaults come from
/// `bench` on a 4-core CPU: one worker per core whose thread count adapts to the load (a lone
/// request gets every core) and no cross-request batching; on a GPU, one worker that batches.
#[derive(clap::Args, Clone)]
struct EngineArgs {
    /// Inference worker threads sharing one copy of the weights (default: cores on CPU, 1 on GPU).
    #[arg(long)]
    workers: Option<usize>,
    /// CPU threads per worker for matmuls (default: cores / workers). 0 lets all workers share
    /// one pool of every core.
    #[arg(long)]
    threads_per_worker: Option<usize>,
    /// Most question rows batched into one forward pass across concurrent requests
    /// (default: 1, no batching, on CPU; 64 on GPU).
    #[arg(long)]
    max_batch_rows: Option<usize>,
    /// Milliseconds a worker waits to fill a batch (0: take only what is already queued).
    #[arg(long, default_value_t = 0.0)]
    batch_wait_ms: f64,
    /// Queued requests beyond this are refused with 529 Overloaded.
    #[arg(long, default_value_t = 1024)]
    max_queue: usize,
    /// Keep every worker at --threads-per-worker instead of widening it (up to every core)
    /// when fewer requests are in flight than there are cores.
    #[arg(long)]
    no_adaptive: bool,
}

impl EngineArgs {
    fn config(&self, dev: &Device) -> candle_rlcd::serve::EngineConfig {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let cpu = dev.is_cpu();
        let workers = self.workers.unwrap_or(if cpu { cores } else { 1 }).max(1);
        let threads = self.threads_per_worker.unwrap_or((cores / workers).max(1));
        candle_rlcd::serve::EngineConfig {
            workers,
            threads_per_worker: threads,
            max_batch_rows: self
                .max_batch_rows
                .unwrap_or(if cpu { 1 } else { 64 })
                .max(1),
            batch_wait: std::time::Duration::from_secs_f64(self.batch_wait_ms.max(0.0) / 1e3),
            max_queue: self.max_queue.max(1),
            adaptive: !self.no_adaptive,
        }
    }
}

fn parse_dtype(d: &str) -> Result<DType> {
    Ok(match d {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        d => anyhow::bail!("unknown dtype {d}"),
    })
}

/// Candle sizes its matmul split from RAYON_NUM_THREADS; match it to the worker pools so a
/// one-thread worker runs matmuls inline instead of splitting them across a pool of one.
fn set_matmul_threads(threads: usize) {
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }
}

fn load_engine(
    model: &std::path::Path,
    dtype: &str,
    cpu: bool,
    engine: &EngineArgs,
) -> Result<candle_rlcd::serve::Engine> {
    let dev = device(cpu)?;
    let cfg = engine.config(&dev);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    // Matmuls split into as many tasks as the widest pool that runs them.
    set_matmul_threads(match cfg.threads_per_worker {
        0 => cores,
        n if cfg.adaptive => cores.max(n),
        n => n,
    });
    let t0 = Instant::now();
    let laya = candle_rlcd::Laya::load(model, &dev, parse_dtype(dtype)?)?;
    eprintln!(
        "loaded {} ({:?} layout) in {:.1?} on {dev:?}; {} worker(s) x {} thread(s), batches up to {} rows{}",
        model.display(),
        laya.cfg.layout,
        t0.elapsed(),
        cfg.workers,
        cfg.threads_per_worker,
        cfg.max_batch_rows,
        if cfg.adaptive { ", adaptive width" } else { "" }
    );
    candle_rlcd::serve::Engine::new(std::sync::Arc::new(laya), cfg)
}

#[derive(Subcommand)]
enum DataCmd {
    /// AG News CSV (`class,title,description`) to `choice` records.
    AgNews {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// BoolQ JSONL to `noul` records.
    Boolq {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Synthetic support tickets with `choice` / `noul` / `score` questions.
    Synthetic {
        #[arg(long)]
        n: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        output: PathBuf,
    },
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    if candle_core::utils::cuda_is_available() {
        return Ok(Device::new_cuda(0)?);
    }
    if candle_core::utils::metal_is_available() {
        return Ok(Device::new_metal(0)?);
    }
    Ok(Device::Cpu)
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Run {
            model,
            request,
            dtype,
            cpu,
            timings,
        } => {
            let dtype = parse_dtype(&dtype)?;
            let dev = device(cpu)?;
            let t0 = Instant::now();
            let model = candle_rlcd::Laya::load(&model, &dev, dtype)?;
            let t_load = t0.elapsed();
            let req: Value = match &request {
                Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
                None => serde_json::from_reader(std::io::stdin())?,
            };
            let state = req.get("state").cloned().unwrap_or(Value::Null);
            let questions = req
                .get("questions")
                .and_then(Value::as_object)
                .context("request needs a \"questions\" object")?;
            let t1 = Instant::now();
            let answers = model.system_one(&state, questions)?;
            let t_run = t1.elapsed();
            println!("{}", serde_json::to_string_pretty(&answers)?);
            if timings {
                eprintln!("load {t_load:?}, inference {t_run:?} on {dev:?} {dtype:?}");
            }
        }
        Cmd::Serve {
            model,
            port,
            host,
            model_name,
            api_key,
            extended,
            truncation,
            engine,
        } => {
            let engine = load_engine(&model.model, &model.dtype, model.cpu, &engine)?;
            let dir = std::fs::canonicalize(&model.model)?;
            let name = model_name.unwrap_or_else(|| {
                dir.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "candle-rlcd".into())
            });
            let cfg = candle_rlcd::serve::ServeConfig {
                addr: std::net::SocketAddr::new(host, port),
                model_name: name,
                description: format!(
                    "ModernBERT + RLCD System One model served by candle-rlcd ({:?} layout)",
                    engine.laya.cfg.layout
                )
                .to_lowercase(),
                release_date: release_date(&dir),
                api_key,
                extended,
                truncation: match truncation.as_str() {
                    "report" => candle_rlcd::serve::TruncationPolicy::Report,
                    _ => candle_rlcd::serve::TruncationPolicy::Strict,
                },
            };
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()?
                .block_on(candle_rlcd::serve::serve(engine, cfg))?;
        }
        Cmd::Bench {
            model,
            dtype,
            cpu,
            url,
            data,
            concurrency,
            requests,
            warmup,
            engine,
        } => {
            let reqs = bench::load_requests(&data)?;
            let target = match (&url, &model) {
                (Some(u), _) => bench::Target::Http(u.clone()),
                (None, Some(m)) => bench::Target::Engine(std::sync::Arc::new(load_engine(
                    m, &dtype, cpu, &engine,
                )?)),
                (None, None) => unreachable!("clap requires --model or --url"),
            };
            let report = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()?
                .block_on(bench::run(target, reqs, concurrency, requests, warmup))?;
            println!("{}", serde_json::to_string(&report)?);
        }
        Cmd::Train { cfg, cpu } => {
            let dev = device(cpu)?;
            let mut trainer = candle_rlcd::train::Trainer::new(*cfg, &dev)?;
            trainer.run()?;
        }
        Cmd::Eval { model, data, cpu } => {
            let m = candle_rlcd::train::evaluate_dir(&model, &data, &device(cpu)?)?;
            println!("{}", serde_json::to_string_pretty(&m)?);
        }
        Cmd::Data { kind } => {
            let (output, n) = match kind {
                DataCmd::AgNews { input, output } => {
                    let mut w = std::io::BufWriter::new(std::fs::File::create(&output)?);
                    let n = candle_rlcd::data::convert_ag_news(&input, &mut w)?;
                    (output, n)
                }
                DataCmd::Boolq { input, output } => {
                    let mut w = std::io::BufWriter::new(std::fs::File::create(&output)?);
                    let n = candle_rlcd::data::convert_boolq(&input, &mut w)?;
                    (output, n)
                }
                DataCmd::Synthetic { n, seed, output } => {
                    let mut w = std::io::BufWriter::new(std::fs::File::create(&output)?);
                    candle_rlcd::data::synthetic(n, seed, &mut w)?;
                    (output, n)
                }
            };
            eprintln!("wrote {n} records to {}", output.display());
        }
        Cmd::Tokenizer {
            data,
            out,
            vocab_size,
        } => {
            candle_rlcd::data::train_tokenizer(&data, &out, vocab_size)?;
            eprintln!("wrote {}", out.join("tokenizer.json").display());
        }
    }
    Ok(())
}

/// `YYYY-MM-DD` of the checkpoint's weights file (or today's date when unavailable).
fn release_date(dir: &std::path::Path) -> String {
    let t = std::fs::metadata(dir.join("model.safetensors"))
        .and_then(|m| m.modified())
        .unwrap_or_else(|_| std::time::SystemTime::now());
    let days = t
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86_400) as i64;
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}
