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

    /// One forward over all rows of a request. In the prefix layout the shared state is
    /// encoded once; otherwise every row goes through one padded batch.
    pub fn forward(&self, rows: &[Encoded]) -> Result<Vec<RowOutput>> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        let pad = self.specials.pad;
        let out = match self.model.layout {
            Layout::Prefix => self.model.forward_prefix(rows, pad, &self.device)?,
            Layout::Laya => {
                let batch = self.model.batch(rows, pad, &self.device)?;
                self.model.forward(&batch, false)?
            }
        };
        let logits = out.logits.to_vec2::<f32>()?;
        let act = candle_nn::ops::softmax_last_dim(&out.act_logits)?.to_vec2::<f32>()?;
        Ok(rows
            .iter()
            .zip(logits)
            .zip(act)
            .map(|((r, l), a)| RowOutput {
                logits: l[..r.markers.len()].to_vec(),
                act_probs: a,
            })
            .collect())
    }

    /// Fitted temperature for a question type and option count, clamped to [0.5, 5].
    pub fn temperature(&self, qtype: QType, k: usize) -> f64 {
        let key = temperature_key(qtype, k);
        let t = self
            .cfg
            .temperature_by_options
            .get(&key)
            .copied()
            .or_else(|| self.cfg.temperature.get(qtype as usize).copied())
            .unwrap_or(1.0);
        clamp_temperature(t)
    }

    /// Jev `/v1/systemone`-shaped call: `questions` is an ordered `{id: question}` object.
    pub fn system_one(&self, state: &Value, questions: &Map<String, Value>) -> Result<Value> {
        let parsed = questions
            .iter()
            .map(|(id, q)| Question::from_json(q).with_context(|| format!("question {id:?}")))
            .collect::<Result<Vec<_>>>()?;
        let rows = self.encode(state, &parsed)?;
        let outs = self.forward(&rows)?;
        let mut answers = Map::new();
        for ((id, q), out) in questions.keys().zip(&parsed).zip(outs) {
            answers.insert(id.clone(), self.decode(q, &out));
        }
        Ok(Value::Object(answers))
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

fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
