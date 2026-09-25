//! Parity against PyTorch outputs from laya's own `DecisionModel` / `build_sequence`
//! (`scripts/make_tiny_fixture.py`), on a tiny random-weight checkpoint.
//!
//! Set `LAYA_DIR` (a downloaded `convaiinnovations/laya` snapshot) and `LAYA_GOLDEN`
//! (`scripts/golden_from_laya.py` output) to run the same checks on the real weights.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_rlcd::sequence::Question;
use candle_rlcd::Laya;
use serde_json::Value;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}

fn load_golden(path: &Path) -> Vec<Value> {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - y).abs())
        .fold(0.0, f64::max)
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// Returns the worst logit error over all rows.
fn check(model: &Laya, golden: &[Value], logit_tol: f64, prob_tol: f64) -> f64 {
    let mut worst: f64 = 0.0;
    for req in golden {
        let qs: Vec<Question> = req["questions"]
            .as_object()
            .unwrap()
            .values()
            .map(|q| Question::from_json(q).unwrap())
            .collect();
        let rows = model.encode(&req["state"], &qs).unwrap();
        let expected = req["rows"].as_array().unwrap();
        for (r, g) in rows.iter().zip(expected) {
            let ids: Vec<u32> = g["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(r.ids, ids, "token ids differ for {}", g["id"]);
            let markers: Vec<usize> = g["markers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(r.markers, markers, "markers differ for {}", g["id"]);
        }
        let outs = model.forward(&rows).unwrap();
        for ((out, g), q) in outs.iter().zip(expected).zip(&qs) {
            let d = max_abs_diff(&out.logits, &floats(&g["logits"]));
            worst = worst.max(d);
            assert!(d < logit_tol, "{}: logits off by {d}", g["id"]);
            let d = max_abs_diff(&out.act_probs, &floats(&g["act_probs"]));
            assert!(d < prob_tol, "{}: act probs off by {d}", g["id"]);
            let t = model.temperature(q.t, out.logits.len());
            assert!((t - g["temperature"].as_f64().unwrap()).abs() < 1e-9);
            let ans = model.decode(q, out);
            let probs = floats(&g["probs"]);
            if let Some(p) = ans.get("probabilities").and_then(Value::as_object) {
                for (v, e) in p.values().zip(&probs) {
                    assert!((v.as_f64().unwrap() - e).abs() < prob_tol + 1e-4);
                }
            } else {
                let v = ans["noul"].as_f64().unwrap();
                assert!((v - probs[1]).abs() < prob_tol + 1e-4);
            }
        }
    }
    worst
}

#[test]
fn tiny_f32_matches_pytorch() {
    let model = Laya::load(fixture(), &Device::Cpu, DType::F32).unwrap();
    let worst = check(
        &model,
        &load_golden(&fixture().join("golden.json")),
        1e-4,
        1e-5,
    );
    eprintln!("f32 worst logit diff {worst:e}");
}

#[test]
fn bf16_on_cpu_is_rejected_clearly() {
    let err = Laya::load(fixture(), &Device::Cpu, DType::BF16)
        .err()
        .unwrap();
    assert!(err
        .to_string()
        .contains("BF16 needs a CUDA or Metal device"));
}

/// BF16 parity on a GPU backend (`--features cuda` or `metal`); skipped on CPU-only builds.
#[test]
fn tiny_bf16_close_to_pytorch_on_gpu() {
    let dev = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap()
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap()
    } else {
        eprintln!("skipped: no GPU backend compiled in");
        return;
    };
    let model = Laya::load(fixture(), &dev, DType::BF16).unwrap();
    let worst = check(
        &model,
        &load_golden(&fixture().join("golden.json")),
        0.1,
        0.05,
    );
    eprintln!("bf16 worst logit diff {worst:e}");
}

#[test]
fn tiny_f16_close_to_pytorch() {
    let model = Laya::load(fixture(), &Device::Cpu, DType::F16).unwrap();
    let worst = check(
        &model,
        &load_golden(&fixture().join("golden.json")),
        0.03,
        0.02,
    );
    eprintln!("f16 worst logit diff {worst:e}");
}

/// Rows must not depend on what else is in the batch (padding is fully masked).
#[test]
fn batching_is_padding_invariant() {
    let model = Laya::load(fixture(), &Device::Cpu, DType::F32).unwrap();
    let mut all = vec![];
    for req in load_golden(&fixture().join("golden.json")) {
        let qs: Vec<Question> = req["questions"]
            .as_object()
            .unwrap()
            .values()
            .map(|q| Question::from_json(q).unwrap())
            .collect();
        all.extend(model.encode(&req["state"], &qs).unwrap());
    }
    let batched = model.forward(&all).unwrap();
    for (row, b) in all.iter().zip(&batched) {
        let single = &model.forward(std::slice::from_ref(row)).unwrap()[0];
        let d = single
            .logits
            .iter()
            .zip(&b.logits)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(d < 1e-4, "batched vs single differ by {d}");
    }
}

#[test]
fn real_laya_matches_pytorch() {
    let (Ok(dir), Ok(golden)) = (std::env::var("LAYA_DIR"), std::env::var("LAYA_GOLDEN")) else {
        eprintln!("skipped: set LAYA_DIR and LAYA_GOLDEN to run against the published weights");
        return;
    };
    let model = Laya::load(&dir, &Device::Cpu, DType::F32).unwrap();
    let worst = check(&model, &load_golden(Path::new(&golden)), 1e-3, 1e-4);
    eprintln!("laya f32 worst logit diff {worst:e}");
}
