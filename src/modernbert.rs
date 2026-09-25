//! ModernBERT encoder, vendored from `candle-transformers` (huggingface/candle 66a8cf1) and fixed.
//!
//! Changes against upstream:
//! - Bias-free norms are built with `LayerNorm::new_no_bias`-equivalent code instead of
//!   `layer_norm_no_bias`, which demands a `bias` tensor and so failed to load `mlp_norm`
//!   (`attn_norm` only got past it through a blanket `.ok()` that also hid real errors). The
//!   layer-0 `attn_norm` is now optional only when the tensor is genuinely absent.
//! - F16/BF16 work: masks are built in F32 and the softmax and norms run in F32, so padded keys
//!   get a true `-inf`-like score instead of an overflowed or mismatched-dtype add. RoPE tables
//!   are computed in F32 from integer positions and cast once, so BF16 keeps exact positions
//!   past 256.
//! - Loads HF `ModernBertModel` weights at any prefix (laya stores them under `encoder.`), and
//!   honours optional attention/MLP/norm biases from the config.
//! - The sliding-window mask is built once per sequence length and shared by the local layers.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::config::EncoderConfig;

/// Norm computed in F32 regardless of the model dtype.
#[derive(Clone, Debug)]
pub struct Norm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl Norm {
    pub fn load(vb: VarBuilder, size: usize, eps: f64, bias: bool) -> Result<Self> {
        let weight = vb.get(size, "weight")?.to_dtype(DType::F32)?;
        let bias = if bias {
            vb.get(size, "bias")?.to_dtype(DType::F32)?
        } else {
            Tensor::zeros(size, DType::F32, vb.device())?
        };
        Ok(Self {
            weight,
            bias,
            eps: eps as f32,
        })
    }
}

impl Module for Norm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let x = xs.to_dtype(DType::F32)?.contiguous()?;
        candle_nn::ops::layer_norm(&x, &self.weight, &self.bias, self.eps)?.to_dtype(dtype)
    }
}

fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Clone, Debug)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &EncoderConfig, theta: f64, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let max_len = cfg.max_position_embeddings;
        let t = Tensor::arange(0u32, max_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, t)?.contiguous()?;
        let sin = self.sin.narrow(0, 0, t)?.contiguous()?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

/// Scaled dot-product attention with an additive F32 mask, softmax in F32.
///
/// `q, k, v`: `[b, h, t, hd]`; `mask`: broadcastable to `[b, h, t, t]`, F32.
pub(crate) fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let dtype = q.dtype();
    let scale = (q.dim(D::Minus1)? as f64).powf(-0.5);
    let q = (q * scale)?;
    let att = q.matmul(&k.t()?.contiguous()?)?;
    let att = att.to_dtype(DType::F32)?.broadcast_add(mask)?;
    let att = candle_nn::ops::softmax_last_dim(&att)?.to_dtype(dtype)?;
    att.matmul(&v.contiguous()?)
}

#[derive(Clone, Debug)]
struct Attention {
    wqkv: Linear,
    wo: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            wqkv: linear(vb.pp("Wqkv"), d, 3 * d, cfg.attention_bias)?,
            wo: linear(vb.pp("Wo"), d, d, cfg.attention_bias)?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.wqkv)?
            .reshape((b, t, 3, self.heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let (q, k) = rope.apply(&qkv.get(0)?, &qkv.get(1)?)?;
        let v = qkv.get(2)?;
        let out = attention(&q, &k, &v, mask)?;
        out.transpose(1, 2)?.reshape((b, t, d))?.apply(&self.wo)
    }
}

#[derive(Clone, Debug)]
struct Mlp {
    wi: Linear,
    wo: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let (d, f) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            wi: linear(vb.pp("Wi"), d, 2 * f, cfg.mlp_bias)?,
            wo: linear(vb.pp("Wo"), f, d, cfg.mlp_bias)?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.wi)?;
        let halves = xs.chunk(2, D::Minus1)?;
        (halves[0].gelu_erf()? * &halves[1])?.apply(&self.wo) // GeGLU
    }
}

#[derive(Clone, Debug)]
struct Layer {
    attn_norm: Option<Norm>,
    attn: Attention,
    mlp_norm: Norm,
    mlp: Mlp,
    local: bool,
}

impl Layer {
    fn load(vb: VarBuilder, cfg: &EncoderConfig, layer_id: usize) -> Result<Self> {
        let d = cfg.hidden_size;
        // HF uses Identity for layer 0's attn_norm; every other layer must have one.
        let attn_norm = if layer_id == 0 && !vb.contains_tensor("attn_norm.weight") {
            None
        } else {
            Some(Norm::load(
                vb.pp("attn_norm"),
                d,
                cfg.norm_eps,
                cfg.norm_bias,
            )?)
        };
        Ok(Self {
            attn_norm,
            attn: Attention::load(vb.pp("attn"), cfg)?,
            mlp_norm: Norm::load(vb.pp("mlp_norm"), d, cfg.norm_eps, cfg.norm_bias)?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
            local: cfg.local_layers[layer_id],
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let h = match &self.attn_norm {
            Some(n) => xs.apply(n)?,
            None => xs.clone(),
        };
        let xs = (xs + self.attn.forward(&h, mask, rope)?)?;
        let m = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        xs + m
    }
}

/// Additive F32 key-padding mask `[b, 1, 1, t]`: 0 for real tokens, a large negative for padding.
pub fn padding_mask(attention_mask: &Tensor) -> Result<Tensor> {
    let m = attention_mask.to_dtype(DType::F32)?;
    let m = ((1.0 - m)? * f32::MIN as f64)?;
    m.unsqueeze(1)?.unsqueeze(1)
}

/// Additive F32 sliding-window mask `[t, t]`: keeps keys with `|i - j| <= half_window`.
fn window_mask(t: usize, half_window: usize, dev: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..t)
        .flat_map(|i| {
            (0..t).map(move |j| {
                if i.abs_diff(j) > half_window {
                    f32::MIN
                } else {
                    0.0
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (t, t), dev)
}

/// HF `ModernBertModel`: returns the last hidden state `[b, t, d]` after `final_norm`.
#[derive(Clone, Debug)]
pub struct ModernBert {
    embeddings: Embedding,
    emb_norm: Norm,
    layers: Vec<Layer>,
    final_norm: Norm,
    global_rope: RotaryEmbedding,
    local_rope: RotaryEmbedding,
    half_window: usize,
    dtype: DType,
}

impl ModernBert {
    /// `vb` points at the `ModernBertModel` root (e.g. `vb.pp("encoder")` for laya checkpoints).
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let embeddings =
            candle_nn::embedding(cfg.vocab_size, d, vb.pp("embeddings.tok_embeddings"))?;
        let emb_norm = Norm::load(vb.pp("embeddings.norm"), d, cfg.norm_eps, cfg.norm_bias)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(vb.pp(format!("layers.{i}")), cfg, i))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = Norm::load(vb.pp("final_norm"), d, cfg.norm_eps, cfg.norm_bias)?;
        let dev = vb.device();
        Ok(Self {
            embeddings,
            emb_norm,
            layers,
            final_norm,
            global_rope: RotaryEmbedding::new(vb.dtype(), cfg, cfg.global_rope_theta, dev)?,
            local_rope: RotaryEmbedding::new(vb.dtype(), cfg, cfg.local_rope_theta, dev)?,
            half_window: cfg.local_attention / 2,
            dtype: vb.dtype(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// `input_ids`: `[b, t]` u32; `attention_mask`: `[b, t]` (1 = token, 0 = pad).
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let t = input_ids.dim(1)?;
        let dev = input_ids.device();
        let global_mask = padding_mask(attention_mask)?;
        // Clamp rather than let MIN + MIN overflow to -inf: HF uses masked_fill here, and a padded
        // query row whose whole window is padding must stay finite, or its NaNs leak into the
        // real rows through the next layer's keys and values.
        let local_mask = global_mask
            .broadcast_add(&window_mask(t, self.half_window, dev)?)?
            .clamp(f32::MIN, 0f32)?;
        let mut xs = input_ids.apply(&self.embeddings)?.apply(&self.emb_norm)?;
        for layer in &self.layers {
            let (mask, rope) = if layer.local {
                (&local_mask, &self.local_rope)
            } else {
                (&global_mask, &self.global_rope)
            };
            xs = layer.forward(&xs, mask, rope)?;
        }
        xs.apply(&self.final_norm)
    }
}
