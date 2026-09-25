//! Load a laya checkpoint directory and answer typed questions about a state.
//!
//! Directory layout (as published on the Hub):
//! `rl_agent_config.json`, `model.safetensors`, `encoder/config.json`, `tokenizer/tokenizer.json`
//! (+ `tokenizer/tokenizer_config.json`).

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::{json, Map, Value};
use tokenizers::Tokenizer;

use crate::config::{clamp_temperature, AgentConfig, EncoderConfig};
use crate::head::DecisionHead;
use crate::modernbert::ModernBert;
use crate::sequence::{build_sequence, encode_state, Criteria, Encoded, QType, Question, Specials};

pub struct Laya {
    pub encoder: ModernBert,
    pub head: DecisionHead,
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
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        let tok_cfg: Option<Value> = read("tokenizer/tokenizer_config.json")
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        let specials = Specials::resolve(&tokenizer, tok_cfg.as_ref())?;

        let weights = dir.join("model.safetensors");
        // SAFETY: the file is memory-mapped read-only and not modified while the model lives.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device)? };
        let encoder =
            ModernBert::load(vb.pp("encoder"), &encoder_cfg).context("loading encoder")?;
        let head = DecisionHead::load(
            vb.clone(),
            encoder_cfg.hidden_size,
            cfg.head_layers,
            cfg.n_act(),
        )
        .context("loading decision head")?;
        Ok(Self {
            encoder,
            head,
            cfg,
            encoder_cfg,
            tokenizer,
            specials,
            device: device.clone(),
        })
    }

    /// Tokenize a state against each question. The state is tokenized once and shared.
    pub fn encode(&self, state: &Value, questions: &[Question]) -> Result<Vec<Encoded>> {
        let state_ids = encode_state(&self.tokenizer, &self.specials, state)?;
        // Conversations (lists) keep the newest turn: truncate from the left.
        let truncate_left = state.is_array();
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

    /// One padded forward over all rows (every question of a request goes in one batch).
    pub fn forward(&self, rows: &[Encoded]) -> Result<Vec<RowOutput>> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        let n = rows.len();
        let t = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let kmax = rows.iter().map(|r| r.markers.len()).max().unwrap_or(0);
        let pad = self.specials.pad;
        let mut ids = vec![pad; n * t];
        let mut att = vec![0u32; n * t];
        let mut mpos = vec![0u32; n * kmax];
        let mut mmask = vec![0f32; n * kmax];
        let mut qt = vec![0u32; n];
        for (i, r) in rows.iter().enumerate() {
            ids[i * t..i * t + r.ids.len()].copy_from_slice(&r.ids);
            att[i * t..i * t + r.ids.len()].fill(1);
            for (j, &m) in r.markers.iter().enumerate() {
                mpos[i * kmax + j] = m as u32;
                mmask[i * kmax + j] = 1.0;
            }
            qt[i] = r.qtype as u32;
        }
        let dev = &self.device;
        let ids = Tensor::from_vec(ids, (n, t), dev)?;
        let att = Tensor::from_vec(att, (n, t), dev)?;
        let mpos = Tensor::from_vec(mpos, (n, kmax), dev)?;
        let mmask = Tensor::from_vec(mmask, (n, kmax), dev)?;
        let qt = Tensor::from_vec(qt, n, dev)?;

        let h = self.encoder.forward(&ids, &att)?;
        let out = self.head.forward(&h, &att, &mpos, &mmask, &qt)?;
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
        let size = match k {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        let key = format!("{}:{size}", qtype.name());
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

fn softmax(logits: &[f32], temp: f32) -> Vec<f32> {
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
