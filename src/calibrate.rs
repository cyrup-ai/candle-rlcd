//! Refit a model's temperatures on the user's own labelled requests (`candle-rlcd calibrate`).
//!
//! Confidence-gated routing (act when `confidence` is high, escalate otherwise) only works when
//! confidence means what it says on the user's traffic. laya ships temperatures fit on its own
//! data that are overconfident elsewhere, so this refits them from labelled requests in Jev's
//! shape (see `data.rs`), without touching the weights:
//!
//! 1. Run the model once over every labelled question and keep the raw logits.
//! 2. Fit on a random half and report accuracy, NLL, Brier and ECE on the other half, with the
//!    current and the new temperatures, so the gain is measured on data the fit didn't see.
//! 3. Refit on everything for the temperatures written out.
//!
//! A `(type, option-count bucket)` gets its own temperature only with at least `min_examples`
//! questions; a type with enough questions gets a new per-type temperature, and its old
//! bucket temperatures are dropped so they fall back to it. Types without enough data keep
//! the model's temperatures.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{ensure, Result};
use serde_json::{json, Value};

use crate::agent::{effective_temperature, temperature_key};
use crate::data::{load_jsonl, Example, Rng};
use crate::sequence::QType;
use crate::train::{fit_temperature, metrics, Temperatures};
use crate::Laya;

#[derive(Debug, Clone)]
pub struct CalibrateConfig {
    /// Fewest questions a bucket or type needs for its own temperature.
    pub min_examples: usize,
    /// Rows per forward pass.
    pub batch: usize,
    pub seed: u64,
}

impl Default for CalibrateConfig {
    fn default() -> Self {
        Self {
            min_examples: 30,
            batch: 8,
            seed: 0,
        }
    }
}

const TYPES: [QType; 3] = [QType::Choice, QType::Score, QType::Noul];

/// One option count per temperature bucket a type can have (a Noul has 2 options and a Score
/// at most 10 levels).
fn bucket_sizes(t: QType) -> &'static [usize] {
    match t {
        QType::Choice => &[2, 3, 6, 11],
        QType::Score => &[2, 3, 6],
        QType::Noul => &[2],
    }
}

/// The temperatures a model answers with now, written out bucket by bucket (so the 11+ floor
/// in [`effective_temperature`] is included).
pub fn current_temperatures(laya: &Laya) -> Temperatures {
    let mut by_options = BTreeMap::new();
    for t in TYPES {
        for &k in bucket_sizes(t) {
            by_options.insert(
                temperature_key(t, k),
                effective_temperature(&laya.cfg, t, k),
            );
        }
    }
    Temperatures {
        by_options,
        by_type: TYPES
            .iter()
            .map(|&t| effective_temperature(&laya.cfg, t, 3))
            .collect(),
    }
}

/// Raw logits for each example, `batch` questions per forward pass.
pub fn predict(laya: &Laya, exs: &[Example], batch: usize) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(exs.len());
    for (i, chunk) in exs.chunks(batch.max(1)).enumerate() {
        let rows = chunk
            .iter()
            .map(|e| laya.encode(&e.state, std::slice::from_ref(&e.question)))
            .collect::<Result<Vec<_>>>()?;
        let groups: Vec<&[_]> = rows.iter().map(Vec::as_slice).collect();
        for g in laya.forward_many(&groups)? {
            out.extend(g.into_iter().map(|r| r.logits));
        }
        let done = ((i + 1) * batch).min(exs.len());
        if exs.len() > 50 && (i % 10 == 9 || done == exs.len()) {
            eprintln!("scored {done}/{} questions", exs.len());
        }
    }
    Ok(out)
}

/// Fits temperatures on `idx`, starting from `base` (see the module docs).
fn fit(
    base: &Temperatures,
    exs: &[Example],
    logits: &[Vec<f32>],
    idx: &[usize],
    min: usize,
) -> (Temperatures, Value) {
    let mut by_key: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut by_type: [Vec<usize>; 3] = Default::default();
    for &i in idx {
        let q = exs[i].question.t;
        by_key
            .entry(temperature_key(q, logits[i].len()))
            .or_default()
            .push(i);
        by_type[q as usize].push(i);
    }
    let run =
        |ix: &[usize]| fit_temperature(ix.iter().map(|&i| (&logits[i][..], &exs[i].target[..])));
    let mut out = base.clone();
    let mut counts = serde_json::Map::new();
    for t in TYPES {
        let ix = &by_type[t as usize];
        counts.insert(t.name().into(), json!(ix.len()));
        if ix.len() < min {
            continue;
        }
        out.by_type[t as usize] = run(ix);
        let prefix = format!("{}:", t.name());
        out.by_options.retain(|k, _| !k.starts_with(&prefix));
    }
    for (key, ix) in &by_key {
        counts.insert(key.clone(), json!(ix.len()));
        if ix.len() >= min {
            out.by_options.insert(key.clone(), run(ix));
        }
    }
    (out, Value::Object(counts))
}

/// Refits `laya`'s temperatures on labelled records. Returns the report and the calibration
/// file (`temperature`, `temperature_by_options`, `calibration`), which
/// [`Laya::apply_calibration`] and `rl_agent_config.json` both accept.
pub fn calibrate(laya: &Laya, data: &Path, cfg: &CalibrateConfig) -> Result<(Value, Value)> {
    let exs = load_jsonl(data)?;
    ensure!(
        !exs.is_empty(),
        "{} has no labelled questions (records need \"targets\" or \"answers\")",
        data.display()
    );
    let logits = predict(laya, &exs, cfg.batch)?;
    let current = current_temperatures(laya);

    let mut order: Vec<usize> = (0..exs.len()).collect();
    Rng(cfg.seed ^ 0xCA1B).shuffle(&mut order);
    let (fit_half, test_half) = order.split_at(order.len() / 2);
    let mut report = json!({"questions": exs.len()});
    if !fit_half.is_empty() && !test_half.is_empty() {
        let (held, _) = fit(&current, &exs, &logits, fit_half, cfg.min_examples / 2);
        let pick = |ix: &[usize]| -> (Vec<Example>, Vec<Vec<f32>>) {
            (
                ix.iter().map(|&i| exs[i].clone()).collect(),
                ix.iter().map(|&i| logits[i].clone()).collect(),
            )
        };
        let (te, tl) = pick(test_half);
        report["holdout"] = json!({
            "note": "temperatures fit on one half, measured on the other",
            "fit_on": fit_half.len(),
            "measured_on": test_half.len(),
            "raw": metrics(&te, &tl, None),
            "current": metrics(&te, &tl, Some(&current)),
            "calibrated": metrics(&te, &tl, Some(&held)),
        });
    }
    let (fitted, counts) = fit(&current, &exs, &logits, &order, cfg.min_examples);
    report["examples_by_bucket"] = counts;
    report["current"] = json!(current);
    report["fitted"] = json!(fitted);
    let file = json!({
        "temperature": fitted.by_type,
        "temperature_by_options": fitted.by_options,
        "calibration": {
            "by": "candle-rlcd calibrate",
            "data": data.file_name().map(|n| n.to_string_lossy()),
            "questions": exs.len(),
            "min_examples": cfg.min_examples,
        },
    });
    Ok((report, file))
}

/// Writes a calibration into a checkpoint's `rl_agent_config.json`, keeping the original as
/// `rl_agent_config.orig.json` the first time.
pub fn write_into(dir: &Path, file: &Value) -> Result<()> {
    let path = dir.join("rl_agent_config.json");
    let backup = dir.join("rl_agent_config.orig.json");
    let text = std::fs::read_to_string(&path)?;
    if !backup.exists() {
        std::fs::write(&backup, &text)?;
    }
    let mut cfg: Value = serde_json::from_str(&text)?;
    for k in ["temperature", "temperature_by_options", "calibration"] {
        cfg[k] = file[k].clone();
    }
    std::fs::write(&path, serde_json::to_string_pretty(&cfg)?)?;
    Ok(())
}
