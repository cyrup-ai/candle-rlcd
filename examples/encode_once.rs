//! Latency of laya's layout (the state re-encoded per question) vs the prefix layout (the state
//! encoded once per request), on a ModernBERT-base-shaped model with random weights.
//!
//! ```sh
//! cargo run --release --example encode_once              # CPU
//! cargo run --release --features cuda --example encode_once
//! ```

use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use candle_rlcd::agent::{build_model, load_tokenizer};
use candle_rlcd::config::{AgentConfig, EncoderConfig};
use candle_rlcd::model::Layout;
use candle_rlcd::sequence::Question;
use candle_rlcd::Laya;
use serde_json::json;

fn main() -> Result<()> {
    let dev = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    // ModernBERT-base shape (the vocabulary only has to cover the fixture tokenizer).
    let enc = EncoderConfig::from_json(
        &json!({"vocab_size": 1024, "hidden_size": 768, "num_hidden_layers": 22,
                "num_attention_heads": 12, "intermediate_size": 1152, "max_position_embeddings": 8192,
                "global_attn_every_n_layers": 3, "local_attention": 128,
                "global_rope_theta": 160000.0, "local_rope_theta": 10000.0, "hidden_activation": "gelu"})
        .to_string(),
    )?;
    let (tok, sp) = load_tokenizer(&fixture)?;
    let state = json!("Customer says the package arrived damaged and wants a refund. ".repeat(40));
    let all_qs: Vec<Question> = (0..8)
        .map(|i| {
            Question::from_json(&json!({"t": "choice", "ins": format!("Question {i}: which team handles this?"),
                "crit": {"billing": "payments and refunds", "shipping": "delivery and damage", "account": "logins"}}))
        })
        .collect::<Result<_>>()?;
    println!("device {dev:?}, ModernBERT-base shape, max_len 512");
    println!(
        "{:>9} {:>14} {:>14} {:>8}",
        "questions", "laya (ms)", "prefix (ms)", "speedup"
    );
    // Random init once, then plain (non-trainable) tensors so inference takes the fused paths.
    let init_cfg: AgentConfig = serde_json::from_value(json!({"act_costs": {"act": 0.0}}))?;
    let vm = VarMap::new();
    build_model(
        VarBuilder::from_varmap(&vm, DType::F32, &dev),
        &init_cfg,
        &enc,
    )?;
    let weights: std::collections::HashMap<String, candle_core::Tensor> = vm
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
        .collect();
    let mut models = Vec::new();
    for layout in [Layout::Laya, Layout::Prefix] {
        let cfg: AgentConfig =
            serde_json::from_value(json!({"layout": layout, "act_costs": {"act": 0.0}}))?;
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let model = build_model(vb, &cfg, &enc)?;
        models.push(Laya::from_parts(
            model,
            cfg,
            enc.clone(),
            tok.clone(),
            sp.clone(),
            &dev,
        ));
    }
    for n in [1usize, 2, 4, 8] {
        let qs = &all_qs[..n];
        let mut ms = Vec::new();
        for m in &models {
            let rows = m.encode(&state, qs)?;
            m.forward(&rows)?; // warm up
            let reps = 3;
            let t = Instant::now();
            for _ in 0..reps {
                m.forward(&rows)?;
            }
            ms.push(t.elapsed().as_secs_f64() * 1000.0 / reps as f64);
        }
        println!(
            "{n:>9} {:>14.1} {:>14.1} {:>7.1}x",
            ms[0],
            ms[1],
            ms[0] / ms[1]
        );
    }
    Ok(())
}
