//! Load a laya checkpoint directory and answer typed questions about a state.
//!
//! Directory layout (as published on the Hub):
//! `rl_agent_config.json`, `model.safetensors`, `encoder/config.json`, `tokenizer/tokenizer.json`
//! (+ `tokenizer/tokenizer_config.json`).

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde_json::{json, Map, Value};
use tokenizers::Tokenizer;

use crate::config::{clamp_temperature, AgentConfig, EncoderConfig};
use crate::head::DecisionHead;
use crate::model::{DecisionModel, Layout};
use crate::modernbert::ModernBert;
use crate::pipeline::{Budget, Plan};
use crate::sequence::{
    build_prefix_sequence, build_sequence, build_state_prefix, encode_state, Criteria, Encoded,
    QType, Question, Specials,
};

pub struct Laya {
    pub model: DecisionModel,
    pub cfg: AgentConfig,
    pub encoder_cfg: EncoderConfig,
    pub tokenizer: Tokenizer,
    pub specials: Specials,
    device: Device,
}

/// Raw per-row outputs of one batched forward.
#[derive(Debug, Clone)]
pub struct RowOutput {
    /// Uncalibrated logits for this row's `k` options.
    pub logits: Vec<f32>,
    /// Softmax of the act head (`[act, escalate]`).
    pub act_probs: Vec<f32>,
}

impl Laya {
    pub fn load(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = dir.as_ref();
        anyhow::ensure!(
            !(device.is_cpu() && dtype == DType::BF16),
            "BF16 needs a CUDA or Metal device: Candle's CPU backend has no BF16 matmul (use f16 or f32 on CPU)"
        );
        let read =
            |p: &str| std::fs::read_to_string(dir.join(p)).with_context(|| format!("reading {p}"));
        let cfg: AgentConfig = serde_json::from_str(&read("rl_agent_config.json")?)?;
        let encoder_cfg = EncoderConfig::from_json(&read("encoder/config.json")?)?;
        let (tokenizer, specials) = load_tokenizer(dir)?;
        let weights = dir.join("model.safetensors");
        // SAFETY: the file is memory-mapped read-only and not modified while the model lives.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let model = build_model(vb, &cfg, &encoder_cfg)?;
        Ok(Self {
            model,
            cfg,
            encoder_cfg,
            tokenizer,
            specials,
            device: device.clone(),
        })
    }

    /// Assemble from already-built parts (used by training).
    pub fn from_parts(
        model: DecisionModel,
        cfg: AgentConfig,
        encoder_cfg: EncoderConfig,
        tokenizer: Tokenizer,
        specials: Specials,
        device: &Device,
    ) -> Self {
        Self {
            model,
            cfg,
            encoder_cfg,
            tokenizer,
            specials,
            device: device.clone(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Tokenize a state against each question. The state is tokenized once and shared.
    pub fn encode(&self, state: &Value, questions: &[Question]) -> Result<Vec<Encoded>> {
        let state_ids = encode_state(&self.tokenizer, &self.specials, state)?;
        // Conversations (lists) keep the newest turn: truncate from the left.
        let truncate_left = state.is_array();
        if self.cfg.layout == Layout::Prefix {
            let prefix = build_state_prefix(
                &self.specials,
                &state_ids,
                self.cfg.max_len,
                self.cfg.head_max_len,
                truncate_left,
            );
            return questions
                .iter()
                .map(|q| {
                    build_prefix_sequence(
                        &self.tokenizer,
                        &self.specials,
                        q,
                        &prefix,
                        self.cfg.head_max_len,
                    )
                })
                .collect();
        }
        questions
            .iter()
            .map(|q| {
                build_sequence(
                    &self.tokenizer,
                    &self.specials,
                    q,
                    &state_ids,
                    self.cfg.max_len,
                    self.cfg.head_max_len,
                    truncate_left,
                )
            })
            .collect()
    }

    /// One forward over all rows of a request. In the prefix layout each run of rows sharing
    /// a state prefix encodes it once; otherwise every row goes through one padded batch.
    pub fn forward(&self, rows: &[Encoded]) -> Result<Vec<RowOutput>> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        Ok(self.forward_many(&[rows])?.remove(0))
    }

    /// Several requests in one forward: each group is one request's rows (from
    /// [`Self::encode`]). In the prefix layout the states are encoded together, left-padded
    /// ([`DecisionModel::forward_prefix_groups`]); in laya's layout all rows form one padded
    /// batch. Padding is masked, so each request gets what it would alone.
    pub fn forward_many(&self, groups: &[&[Encoded]]) -> Result<Vec<Vec<RowOutput>>> {
        if groups.iter().all(|g| g.is_empty()) {
            return Ok(groups.iter().map(|_| vec![]).collect());
        }
        let pad = self.specials.pad;
        // Prefix layout: a request whose state runs in chunks has one prefix per chunk, so
        // split its rows into runs that share one.
        let nonempty: Vec<&[Encoded]> = groups
            .iter()
            .flat_map(|g| g.chunk_by(|a, b| a.ids[..a.prefix_len] == b.ids[..b.prefix_len]))
            .collect();
        let out = match self.model.layout {
            Layout::Prefix => self
                .model
                .forward_prefix_groups(&nonempty, pad, &self.device)?,
            Layout::Laya => {
                let rows: Vec<Encoded> = nonempty.iter().flat_map(|g| g.iter().cloned()).collect();
                let batch = self.model.batch(&rows, pad, &self.device)?;
                self.model.forward(&batch, false)?
            }
        };
        let logits = out.logits.to_vec2::<f32>()?;
        let act = candle_nn::ops::softmax_last_dim(&out.act_logits)?.to_vec2::<f32>()?;
        let mut flat = logits.into_iter().zip(act);
        Ok(groups
            .iter()
            .map(|g| {
                g.iter()
                    .map(|r| {
                        let (l, a) = flat.next().expect("one output row per input row");
                        RowOutput {
                            logits: l[..r.markers.len()].to_vec(),
                            act_probs: a,
                        }
                    })
                    .collect()
            })
            .collect())
    }

    /// Fitted temperature for a question type and option count; see [`effective_temperature`].
    pub fn temperature(&self, qtype: QType, k: usize) -> f64 {
        effective_temperature(&self.cfg, qtype, k)
    }

    /// Replaces the temperatures with a calibration file's (`candle-rlcd calibrate --out`):
    /// `temperature`, `temperature_by_options` and `calibration`, as in `rl_agent_config.json`.
    pub fn apply_calibration(&mut self, cal: &Value) -> Result<()> {
        let c: CalibrationFile = serde_json::from_value(cal.clone())
            .context("a calibration file needs \"temperature\" and \"temperature_by_options\"")?;
        anyhow::ensure!(
            c.temperature.len() == 3,
            "\"temperature\" needs 3 values (choice, score, noul)"
        );
        self.cfg.temperature = c.temperature;
        self.cfg.temperature_by_options = c.temperature_by_options;
        self.cfg.calibration = Some(c.calibration.unwrap_or_else(|| json!({})));
        Ok(())
    }

    /// Jev `/v1/systemone`-shaped call: `questions` is an ordered `{id: question}` object.
    pub fn system_one(&self, state: &Value, questions: &Map<String, Value>) -> Result<Value> {
        let parsed = questions
            .iter()
            .map(|(id, q)| Question::from_json(q).with_context(|| format!("question {id:?}")))
            .collect::<Result<Vec<_>>>()?;
        let plan = Plan::new(self, &Budget::for_model(self), state, parsed)?;
        let (outs, _) = plan.run(|rows| self.forward(&rows))?;
        let mut answers = Map::new();
        for (id, (q, out)) in questions.keys().zip(outs) {
            answers.insert(id.clone(), self.decode(&q, &out));
        }
        Ok(Value::Object(answers))
    }

    /// Calibrated probabilities for one row: softmax at the fitted temperature.
    pub fn probabilities(&self, q: &Question, out: &RowOutput) -> Vec<f32> {
        let temp = self.temperature(q.t, out.logits.len()) as f32;
        softmax(&out.logits, temp)
    }

    /// Turn one row's logits into a typed answer, with calibrated probabilities.
    pub fn decode(&self, q: &Question, out: &RowOutput) -> Value {
        let k = out.logits.len();
        let temp = self.temperature(q.t, k) as f32;
        let p = softmax(&out.logits, temp);
        let argmax = p
            .iter()
            .enumerate()
            .fold(0, |best, (i, &v)| if v > p[best] { i } else { best });
        let answer_conf = round4(p[argmax] as f64);
        let entropy_conf = entropy_confidence(&p);
        let action = json!({"act_probability": round4(out.act_probs[0] as f64)});
        let keys = q.option_keys();
        let probs: Map<String, Value> = keys
            .iter()
            .zip(&p)
            .map(|(k, &v)| (k.clone(), json!(round4(v as f64))))
            .collect();
        match q.t {
            QType::Choice => json!({
                "type": "choice", "choice": keys[argmax], "probabilities": probs,
                "confidence": entropy_conf, "answer_confidence": answer_conf, "action": action,
            }),
            QType::Score => {
                let exp: f64 = p
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| i as f64 * v as f64)
                    .sum();
                let legend: Map<String, Value> = match &q.crit {
                    Criteria::Score(c) => c
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i.to_string(), v.clone()))
                        .collect(),
                    _ => Map::new(),
                };
                json!({
                    "type": "score", "score": round4(exp), "legend": legend, "probabilities": probs,
                    "confidence": entropy_conf, "answer_confidence": answer_conf, "action": action,
                })
            }
            QType::Noul => {
                let pt = p[1] as f64;
                json!({
                    "type": "noul", "noul": round4(pt), "confidence": round4(pt.max(1.0 - pt)),
                    "answer_confidence": answer_conf, "action": action,
                })
            }
        }
    }
}

#[derive(serde::Deserialize)]
struct CalibrationFile {
    temperature: Vec<f64>,
    temperature_by_options: std::collections::BTreeMap<String, f64>,
    calibration: Option<Value>,
}

/// Option counts from which a shipped temperature may not sharpen.
pub const MANY_OPTIONS: usize = 11;

/// The temperature answers use for a question type and option count: the `(type, bucket)`
/// fit, else the per-type one, clamped to [0.5, 5].
///
/// laya's shipped `choice:11+` is 0.1, fit on few examples; even clamped to 0.5 it sharpens,
/// so an unanswerable question's confidence jumped from 0.24 at 10 options to 0.56 at 11 and
/// broke confidence-gated routing. Unless the temperatures were refit by `candle-rlcd
/// calibrate` or `train` (`calibration` in the config), 11+ options never go below 1.0.
pub fn effective_temperature(cfg: &AgentConfig, qtype: QType, k: usize) -> f64 {
    let t = cfg
        .temperature_by_options
        .get(&temperature_key(qtype, k))
        .copied()
        .or_else(|| cfg.temperature.get(qtype as usize).copied())
        .unwrap_or(1.0);
    let t = clamp_temperature(t);
    if k >= MANY_OPTIONS && cfg.calibration.is_none() {
        t.max(1.0)
    } else {
        t
    }
}

/// laya's temperature bucket, e.g. `choice:3-5`.
pub fn temperature_key(qtype: QType, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", qtype.name())
}

/// Tokenizer and special tokens from a checkpoint's `tokenizer/` directory.
pub fn load_tokenizer(dir: &Path) -> Result<(Tokenizer, Specials)> {
    load_tokenizer_from(&dir.join("tokenizer"))
}

/// Tokenizer and special tokens from a directory holding `tokenizer.json` (and optionally
/// `tokenizer_config.json`).
pub fn load_tokenizer_from(dir: &Path) -> Result<(Tokenizer, Specials)> {
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer from {}: {e}", dir.display()))?;
    let tok_cfg: Option<Value> = std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    let specials = Specials::resolve(&tokenizer, tok_cfg.as_ref())?;
    Ok((tokenizer, specials))
}

/// Builds the model from a `VarBuilder` rooted at the checkpoint (laya's key names).
pub fn build_model(
    vb: VarBuilder,
    cfg: &AgentConfig,
    encoder_cfg: &EncoderConfig,
) -> Result<DecisionModel> {
    let encoder = ModernBert::load(vb.pp("encoder"), encoder_cfg).context("loading encoder")?;
    let head = DecisionHead::load(
        vb,
        encoder_cfg.hidden_size,
        cfg.head_layers,
        cfg.n_act(),
        encoder_cfg.initializer_range,
    )
    .context("loading decision head")?;
    Ok(DecisionModel {
        encoder,
        head,
        layout: cfg.layout,
    })
}

pub(crate) fn softmax(logits: &[f32], temp: f32) -> Vec<f32> {
    let z: Vec<f32> = logits.iter().map(|l| l / temp).collect();
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// `1 - H(p) / log k`: how concentrated the distribution is. Not calibrated.
fn entropy_confidence(p: &[f32]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let h: f64 = p
        .iter()
        .map(|&v| {
            let v = (v as f64).clamp(1e-12, 1.0);
            -v * v.ln()
        })
        .sum();
    round4((1.0 - h / (k as f64).ln()).clamp(0.0, 1.0))
}

pub(crate) fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
