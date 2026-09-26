//! `candle-rlcd` command line: run a model, train one, evaluate it, and prepare data.
//!
//! ```text
//! candle-rlcd run --model <dir> --request request.json
//! candle-rlcd serve --port 8080          # convaiinnovations/laya, downloaded on first run
//! candle-rlcd serve --model laya=convaiinnovations/laya --model runs/x/final
//! candle-rlcd bench --model <dir> --data requests.jsonl --concurrency 8
//! candle-rlcd train --train train.jsonl --eval eval.jsonl --out runs/x --init-encoder <modernbert>
//! candle-rlcd eval --model runs/x/final --data test.jsonl
//! candle-rlcd calibrate --model convaiinnovations/laya --data labelled.jsonl --out cal.json
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
        /// Model directory, or a Hub id such as convaiinnovations/laya (downloaded and cached).
        #[arg(long, default_value = DEFAULT_MODEL)]
        model: String,
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
        /// Temperatures from `candle-rlcd calibrate --out`, in place of the model's.
        #[arg(long)]
        calibration: Option<PathBuf>,
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
        /// Add laya's extra answer fields (answer_confidence, act_probability).
        #[arg(long)]
        extended: bool,
        /// Refuse requests with more questions than this (422). Bounds how long one request can
        /// hold an inference worker.
        #[arg(long)]
        max_questions: Option<usize>,
        /// Refuse requests that would read more tokens than this (422); Jev's cap is 65536.
        /// 0 turns it off.
        #[arg(long, default_value_t = candle_rlcd::serve::JEV_MAX_REQUEST_TOKENS)]
        max_request_tokens: usize,
        /// Answer 504 to requests not done within this many seconds (0: no timeout).
        #[arg(long, default_value_t = 120.0)]
        timeout_secs: f64,
        /// Temperatures from `candle-rlcd calibrate --out`: `file` for the default model, or
        /// `name=file`. Repeatable.
        #[arg(long)]
        calibration: Vec<String>,
        #[command(flatten)]
        engine: EngineArgs,
    },
    /// Load-test the engine in process, or a running server with --url.
    Bench {
        /// Model directory or Hub id (in-process benchmark).
        #[arg(long, required_unless_present = "url")]
        model: Option<String>,
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
        /// Model directory or Hub id.
        #[arg(long)]
        model: String,
        /// Labelled records (JSONL).
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        cpu: bool,
        /// Temperatures from `candle-rlcd calibrate --out`, in place of the model's.
        #[arg(long)]
        calibration: Option<PathBuf>,
    },
    /// Refit the model's confidence temperatures on your own labelled requests (Jev-shaped
    /// requests with `targets`, or logged requests with the `answers` you accepted), so
    /// `confidence` thresholds mean what they say on your traffic. Weights are unchanged.
    Calibrate {
        /// Model directory or Hub id.
        #[arg(long, default_value = DEFAULT_MODEL)]
        model: String,
        /// Labelled records (JSONL): `{"state", "questions", "targets"}` or `"answers"`.
        #[arg(long)]
        data: PathBuf,
        /// Write the temperatures to this file, for `serve`/`run`/`eval --calibration`.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Write them into the model directory's rl_agent_config.json (the original is kept
        /// as rl_agent_config.orig.json). Local directories only.
        #[arg(long)]
        write: bool,
        /// Fewest questions a (type, option-count) bucket needs for its own temperature.
        #[arg(long, default_value_t = 30)]
        min_examples: usize,
        /// Questions per forward pass.
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
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

/// Served when no --model is given.
const DEFAULT_MODEL: &str = "convaiinnovations/laya";

#[derive(clap::Args)]
struct ModelArgs {
    /// A model to serve: a directory (rl_agent_config.json, model.safetensors, encoder/,
    /// tokenizer/) or a Hub id (`org/repo[/sub-folder][@revision]`), downloaded to the Hugging
    /// Face cache on first use. Prefix `name=` to choose the name requests use. Repeat to serve
    /// several; the first is the default, which answers `jev-latest` and unknown names when it
    /// is the only one.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: Vec<String>,
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

/// Splits `name=spec` (a name has no `/`); plain `spec` gives no name.
fn split_model_arg(arg: &str) -> (Option<&str>, &str) {
    match arg.split_once('=') {
        Some((n, spec)) if !n.is_empty() && !n.contains('/') => (Some(n), spec),
        _ => (None, arg),
    }
}

fn read_calibration(path: &std::path::Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading calibration {}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

fn load_engine(
    model: &std::path::Path,
    dtype: &str,
    cpu: bool,
    engine: &EngineArgs,
) -> Result<candle_rlcd::serve::Engine> {
    let busy = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    load_engine_shared(model, dtype, cpu, engine, busy, None)
}

fn load_engine_shared(
    model: &std::path::Path,
    dtype: &str,
    cpu: bool,
    engine: &EngineArgs,
    busy: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    calibration: Option<&Value>,
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
    let mut laya = candle_rlcd::Laya::load(model, &dev, parse_dtype(dtype)?)?;
    if let Some(c) = calibration {
        laya.apply_calibration(c)?;
    }
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
    candle_rlcd::serve::Engine::with_shared_busy(std::sync::Arc::new(laya), cfg, busy)
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
        // Candle's Metal backend needs macOS 15 (MTLResidencySet) and panics on older
        // systems; fall back to the CPU there instead of crashing.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let metal = std::panic::catch_unwind(|| Device::new_metal(0));
        std::panic::set_hook(hook);
        match metal {
            Ok(dev) => return Ok(dev?),
            Err(_) => eprintln!("Metal is unavailable (it needs macOS 15 or later); using the CPU"),
        }
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
            calibration,
        } => {
            let dtype = parse_dtype(&dtype)?;
            let dev = device(cpu)?;
            let dir = candle_rlcd::hub::resolve(&model)?.dir;
            let t0 = Instant::now();
            let mut model = candle_rlcd::Laya::load(&dir, &dev, dtype)?;
            if let Some(c) = &calibration {
                model.apply_calibration(&read_calibration(c)?)?;
            }
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
            max_questions,
            max_request_tokens,
            timeout_secs,
            calibration,
            engine,
        } => {
            // Resolve (and download) everything before loading anything.
            let mut specs = vec![];
            for (i, arg) in model.model.iter().enumerate() {
                let (name, spec) = split_model_arg(arg);
                let r = candle_rlcd::hub::resolve(spec)?;
                let name = name
                    .map(str::to_string)
                    .or_else(|| model_name.clone().filter(|_| i == 0))
                    .unwrap_or_else(|| r.name.clone());
                specs.push((name, r));
            }
            let mut names = std::collections::HashSet::new();
            for (n, _) in &specs {
                anyhow::ensure!(
                    names.insert(n.clone()),
                    "two models are named {n:?}; name them with --model <name>=<model>"
                );
            }
            let mut cals: std::collections::HashMap<String, Value> = Default::default();
            for c in &calibration {
                let (name, file) = split_model_arg(c);
                let name = name.map_or_else(|| specs[0].0.clone(), str::to_string);
                anyhow::ensure!(
                    names.contains(&name),
                    "--calibration names model {name:?}, which isn't served"
                );
                cals.insert(name, read_calibration(std::path::Path::new(file))?);
            }
            let busy = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut served = vec![];
            for (name, r) in specs {
                let engine = load_engine_shared(
                    &r.dir,
                    &model.dtype,
                    model.cpu,
                    &engine,
                    busy.clone(),
                    cals.get(&name),
                )?;
                let mut aliases = vec![];
                let mut description = format!(
                    "ModernBERT + RLCD System One model served by candle-rlcd ({:?} layout)",
                    engine.laya.cfg.layout
                )
                .to_lowercase();
                if let Some((hub, commit)) = &r.hub {
                    // A pinned name, so clients can hold on to one exact version.
                    aliases.push(format!("{name}@{}", &commit[..7.min(commit.len())]));
                    description.push_str(&format!("; {} at {commit}", hub.repo));
                }
                served.push(candle_rlcd::serve::ServedModel {
                    name,
                    aliases,
                    description,
                    release_date: release_date(&r.dir),
                    engine,
                });
            }
            let cfg = candle_rlcd::serve::ServeConfig {
                addr: std::net::SocketAddr::new(host, port),
                model_name: served[0].name.clone(),
                description: served[0].description.clone(),
                release_date: served[0].release_date.clone(),
                api_key,
                extended,
                max_questions,
                max_request_tokens: (max_request_tokens > 0).then_some(max_request_tokens),
                timeout: (timeout_secs > 0.0)
                    .then(|| std::time::Duration::from_secs_f64(timeout_secs)),
            };
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()?
                .block_on(candle_rlcd::serve::serve_models(served, cfg))?;
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
                    &candle_rlcd::hub::resolve(m)?.dir,
                    &dtype,
                    cpu,
                    &engine,
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
        Cmd::Train { mut cfg, cpu } => {
            // `--init` also takes a Hub id, e.g. convaiinnovations/laya.
            if let Some(init) = cfg.init.as_ref().filter(|p| !p.is_dir()) {
                cfg.init = Some(candle_rlcd::hub::resolve(&init.to_string_lossy())?.dir);
            }
            let dev = device(cpu)?;
            let mut trainer = candle_rlcd::train::Trainer::new(*cfg, &dev)?;
            trainer.run()?;
        }
        Cmd::Eval {
            model,
            data,
            cpu,
            calibration,
        } => {
            let dir = candle_rlcd::hub::resolve(&model)?.dir;
            let cal = calibration.as_deref().map(read_calibration).transpose()?;
            let m =
                candle_rlcd::train::evaluate_dir_with(&dir, &data, &device(cpu)?, cal.as_ref())?;
            println!("{}", serde_json::to_string_pretty(&m)?);
        }
        Cmd::Calibrate {
            model,
            data,
            out,
            write,
            min_examples,
            batch,
            seed,
            cpu,
        } => {
            let r = candle_rlcd::hub::resolve(&model)?;
            anyhow::ensure!(
                !(write && r.hub.is_some()),
                "--write changes a local model directory; for a Hub model use --out <file> and \
                 pass it to serve/run/eval with --calibration"
            );
            let laya = candle_rlcd::Laya::load(&r.dir, &device(cpu)?, DType::F32)?;
            let cfg = candle_rlcd::calibrate::CalibrateConfig {
                min_examples,
                batch,
                seed,
            };
            let (report, file) = candle_rlcd::calibrate::calibrate(&laya, &data, &cfg)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if let Some(out) = &out {
                std::fs::write(out, serde_json::to_string_pretty(&file)?)?;
                eprintln!("wrote {}", out.display());
            }
            if write {
                candle_rlcd::calibrate::write_into(&r.dir, &file)?;
                eprintln!("updated {}", r.dir.join("rl_agent_config.json").display());
            }
            if out.is_none() && !write {
                eprintln!(
                    "nothing written: pass --out <file> or --write to keep these temperatures"
                );
            }
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
