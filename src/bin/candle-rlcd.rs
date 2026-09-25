//! `candle-rlcd --model <dir> --request request.json` prints Jev-style answers as JSON.
//!
//! The request is `{"state": ..., "questions": {id: {"t": ..., "ins": ..., "crit": ...}}}`.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::Parser;
use serde_json::Value;

#[derive(Parser)]
#[command(about = "Run a laya-compatible RLCD decision model on Candle")]
struct Args {
    /// Checkpoint directory (rl_agent_config.json, model.safetensors, encoder/, tokenizer/).
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
    let args = Args::parse();
    let dtype = match args.dtype.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        d => anyhow::bail!("unknown dtype {d}"),
    };
    let dev = device(args.cpu)?;
    let t0 = Instant::now();
    let model = candle_rlcd::Laya::load(&args.model, &dev, dtype)?;
    let t_load = t0.elapsed();
    let req: Value = match &args.request {
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
    if args.timings {
        eprintln!(
            "load {:?}, inference {:?} on {:?} {:?}",
            t_load, t_run, dev, dtype
        );
    }
    Ok(())
}
