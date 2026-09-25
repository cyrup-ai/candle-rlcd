//! `candle-rlcd` command line: run a model, train one, evaluate it, and prepare data.
//!
//! ```text
//! candle-rlcd run --model <dir> --request request.json
//! candle-rlcd train --train train.jsonl --eval eval.jsonl --out runs/x --init-encoder <modernbert>
//! candle-rlcd eval --model runs/x/final --data test.jsonl
//! candle-rlcd data ag-news --input train.csv --output train.jsonl
//! candle-rlcd tokenizer --data train.jsonl --out tok --vocab-size 8192
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
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
            let dtype = match dtype.as_str() {
                "f32" => DType::F32,
                "f16" => DType::F16,
                "bf16" => DType::BF16,
                d => anyhow::bail!("unknown dtype {d}"),
            };
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
