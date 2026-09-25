//! Config parsing for the encoder (`encoder/config.json`) and the laya agent (`rl_agent_config.json`).

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

/// ModernBERT config, normalised across transformers 4.x and 5.x layouts.
///
/// transformers 4.x writes `global_rope_theta` / `local_rope_theta`; 5.x writes
/// `rope_parameters = {"full_attention": {"rope_theta": ..}, "sliding_attention": {..}}`.
/// mmBERT uses 160000 for both, and transformers 4.x silently fell back to 10000 for the local
/// theta when only `rope_parameters` was present, so we read both forms and prefer the 5.x one.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub norm_bias: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
    /// Total local window; each token sees keys with `|i - j| <= local_attention / 2`.
    pub local_attention: usize,
    /// Per-layer sliding-window flag (transformers 5 `layer_types`, else every layer not a
    /// multiple of `global_attn_every_n_layers`).
    pub local_layers: Vec<bool>,
}

#[derive(Deserialize)]
struct RawEncoderConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    norm_eps: Option<f64>,
    layer_norm_eps: Option<f64>,
    #[serde(default)]
    norm_bias: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    pad_token_id: Option<u32>,
    #[serde(default = "default_global_every")]
    global_attn_every_n_layers: usize,
    global_rope_theta: Option<f64>,
    local_rope_theta: Option<f64>,
    #[serde(default = "default_local_attention")]
    local_attention: usize,
    rope_parameters: Option<Value>,
    layer_types: Option<Vec<String>>,
    hidden_activation: Option<String>,
}

fn default_max_pos() -> usize {
    8192
}
fn default_global_every() -> usize {
    3
}
fn default_local_attention() -> usize {
    128
}

impl EncoderConfig {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        let raw: RawEncoderConfig = serde_json::from_str(s)?;
        if let Some(act) = &raw.hidden_activation {
            anyhow::ensure!(
                act == "gelu",
                "unsupported hidden_activation {act:?}: only exact (erf) gelu is implemented"
            );
        }
        let mut global = raw.global_rope_theta.unwrap_or(160_000.0);
        let mut local = raw.local_rope_theta.unwrap_or(10_000.0);
        if let Some(Value::Object(rope)) = &raw.rope_parameters {
            let flat = rope.get("rope_theta").and_then(Value::as_f64);
            let pick = |key: &str| {
                rope.get(key)
                    .and_then(|p| p.get("rope_theta"))
                    .and_then(Value::as_f64)
                    .or(flat)
            };
            if let Some(t) = pick("full_attention") {
                global = t;
            }
            if let Some(t) = pick("sliding_attention") {
                local = t;
            }
        }
        let local_layers = match &raw.layer_types {
            Some(types) => {
                anyhow::ensure!(
                    types.len() == raw.num_hidden_layers,
                    "layer_types has {} entries for {} layers",
                    types.len(),
                    raw.num_hidden_layers
                );
                types
                    .iter()
                    .map(|t| match t.as_str() {
                        "sliding_attention" => Ok(true),
                        "full_attention" => Ok(false),
                        t => anyhow::bail!("unknown layer type {t:?}"),
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
            }
            None => (0..raw.num_hidden_layers)
                .map(|i| i % raw.global_attn_every_n_layers != 0)
                .collect(),
        };
        Ok(Self {
            local_layers,
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            intermediate_size: raw.intermediate_size,
            max_position_embeddings: raw.max_position_embeddings,
            norm_eps: raw.norm_eps.or(raw.layer_norm_eps).unwrap_or(1e-5),
            norm_bias: raw.norm_bias,
            attention_bias: raw.attention_bias,
            mlp_bias: raw.mlp_bias,
            pad_token_id: raw.pad_token_id.unwrap_or(0),
            global_attn_every_n_layers: raw.global_attn_every_n_layers,
            global_rope_theta: global,
            local_rope_theta: local,
            local_attention: raw.local_attention,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// The subset of `rl_agent_config.json` inference needs (training writes the same file).
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct AgentConfig {
    /// Sequence layout; absent in laya checkpoints, which use the joint layout.
    #[serde(default)]
    pub layout: crate::model::Layout,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default)]
    pub act_costs: BTreeMap<String, Value>,
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    #[serde(default = "default_head_max_len")]
    pub head_max_len: usize,
    /// Per-qtype temperature, indexed by [choice, score, noul].
    #[serde(default = "default_temperature")]
    pub temperature: Vec<f64>,
    /// Per `(qtype, k-bucket)` temperature, e.g. `"choice:3-5"`.
    #[serde(default)]
    pub temperature_by_options: BTreeMap<String, f64>,
}

fn default_head_layers() -> usize {
    2
}
fn default_max_len() -> usize {
    512
}
fn default_head_max_len() -> usize {
    192
}
fn default_temperature() -> Vec<f64> {
    vec![1.0, 1.0, 1.0]
}

impl AgentConfig {
    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }
}

/// A fitted temperature below 0.5 sharpens rather than calibrates (the shipped laya root
/// checkpoint has `choice:11+ = 0.1006`); laya clamps to [0.5, 5.0] at load and so do we.
pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

pub fn clamp_temperature(t: f64) -> f64 {
    if !t.is_finite() {
        return 1.0;
    }
    t.clamp(TEMP_MIN, TEMP_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_parameters_override_flat_thetas() {
        let cfg = EncoderConfig::from_json(
            r#"{"vocab_size": 10, "hidden_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 16,
                "global_rope_theta": 160000, "local_rope_theta": 10000,
                "rope_parameters": {"full_attention": {"rope_theta": 160000.0},
                                    "sliding_attention": {"rope_theta": 160000.0}}}"#,
        )
        .unwrap();
        assert_eq!(cfg.local_rope_theta, 160000.0);
        assert_eq!(cfg.global_rope_theta, 160000.0);
    }
}
