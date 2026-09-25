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
//! - Trains: RoPE and softmax use the fused kernels with the backward passes in
//!   [`crate::autograd`] (upstream RoPE has none, so Q/K never trained), norms switch to the
//!   differentiable composite when gradients are tracked, and fresh weights get ModernBERT's
//!   init (norms at 1/0, embeddings and linears at N(0, 0.02)).
//! - Prefix caching: [`ModernBert::encode_prefix`] runs a state once and keeps each layer's
//!   keys/values; [`ModernBert::forward_suffix`] runs question rows against that cache. The
//!   result equals a joint forward where state tokens cannot see question tokens.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, Init, Linear, VarBuilder};

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
        let weight = vb
            .get_with_hints(size, "weight", Init::Const(1.0))?
            .to_dtype(DType::F32)?;
        let bias = if bias {
            vb.get_with_hints(size, "bias", Init::Const(0.0))?
                .to_dtype(DType::F32)?
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
        // The fused kernel has no backward; the composite is exact and differentiable.
        let y = if x.track_op() || self.weight.track_op() || self.bias.track_op() {
            candle_nn::ops::layer_norm_slow(&x, &self.weight, &self.bias, self.eps)?
        } else {
            candle_nn::ops::layer_norm(&x, &self.weight, &self.bias, self.eps)?
        };
        y.to_dtype(dtype)
    }
}

/// N(0, std²) init (the config's `initializer_range`); only used when a weight is created
/// rather than loaded.
pub(crate) fn init_normal(std: f64) -> Init {
    Init::Randn {
        mean: 0.0,
        stdev: std,
    }
}

pub(crate) fn linear(
    vb: VarBuilder,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    std: f64,
) -> Result<Linear> {
    let w = vb.get_with_hints((out_dim, in_dim), "weight", init_normal(std))?;
    let b = if bias {
        Some(vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?)
    } else {
        None
    };
    Ok(Linear::new(w, b))
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

    /// Rotates `q, k` (`[b, h, t, hd]`) for positions `offset..offset + t`.
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let t = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, t)?.contiguous()?;
        let sin = self.sin.narrow(0, offset, t)?.contiguous()?;
        let q = crate::autograd::rope(q, &cos, &sin)?;
        let k = crate::autograd::rope(k, &cos, &sin)?;
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
    let att = crate::autograd::softmax_last_dim(&att)?.to_dtype(dtype)?;
    att.matmul(&v.contiguous()?)
}

/// Keys and values of a cached prefix for one layer, `[g, h, t, hd]` (one row per state).
#[derive(Clone, Debug)]
pub struct Kv {
    pub k: Tensor,
    pub v: Tensor,
}

impl Kv {
    /// Keys/values for a batch of `b` query rows: a cache with one row is broadcast to all of
    /// them, and a cache with `b` rows (one per query row, see [`Kv::select`]) is used as is.
    pub(crate) fn expand(&self, b: usize) -> Result<(Tensor, Tensor)> {
        if self.k.dim(0)? == b {
            return Ok((self.k.clone(), self.v.clone()));
        }
        let grow = |x: &Tensor| -> Result<Tensor> {
            let (_, h, t, hd) = x.dims4()?;
            x.broadcast_as((b, h, t, hd))?.contiguous()
        };
        Ok((grow(&self.k)?, grow(&self.v)?))
    }

    /// Gathers cache rows so row `i` holds state `rows[i]`.
    pub fn select(&self, rows: &Tensor) -> Result<Self> {
        Ok(Self {
            k: self.k.index_select(rows, 0)?,
            v: self.v.index_select(rows, 0)?,
        })
    }
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
            wqkv: linear(
                vb.pp("Wqkv"),
                d,
                3 * d,
                cfg.attention_bias,
                cfg.initializer_range,
            )?,
            wo: linear(vb.pp("Wo"), d, d, cfg.attention_bias, cfg.initializer_range)?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Returns the output and this call's rotated keys and values (`[b, h, t, hd]`). With
    /// `past`, queries also attend to cached prefix keys/values placed before `xs`, and `xs`
    /// sits at positions `offset..`.
    fn forward(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
        past: Option<&Kv>,
    ) -> Result<(Tensor, Kv)> {
        let (b, t, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.wqkv)?
            .reshape((b, t, 3, self.heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let (q, k) = rope.apply(&qkv.get(0)?, &qkv.get(1)?, offset)?;
        let v = qkv.get(2)?.contiguous()?;
        let out = match past {
            None => attention(&q, &k, &v, mask)?,
            Some(p) => {
                let (pk, pv) = p.expand(b)?;
                attention(
                    &q,
                    &Tensor::cat(&[&pk, &k], 2)?,
                    &Tensor::cat(&[&pv, &v], 2)?,
                    mask,
                )?
            }
        };
        let out = out.transpose(1, 2)?.reshape((b, t, d))?.apply(&self.wo)?;
        Ok((out, Kv { k, v }))
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
            wi: linear(vb.pp("Wi"), d, 2 * f, cfg.mlp_bias, cfg.initializer_range)?,
            wo: linear(vb.pp("Wo"), f, d, cfg.mlp_bias, cfg.initializer_range)?,
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

    fn forward(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        offset: usize,
        past: Option<&Kv>,
    ) -> Result<(Tensor, Kv)> {
        let h = match &self.attn_norm {
            Some(n) => xs.apply(n)?,
            None => xs.clone(),
        };
        let (a, kv) = self.attn.forward(&h, mask, rope, offset, past)?;
        let xs = (xs + a)?;
        let m = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        Ok(((xs + m)?, kv))
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

/// Attention masks for a batch, one per attention type, broadcastable to `[b, h, q, k]`.
#[derive(Clone, Debug)]
pub struct Masks {
    pub global: Tensor,
    pub local: Tensor,
}

impl Masks {
    /// Plain key-padding masks (laya's joint layout).
    pub fn padding(attention_mask: &Tensor, half_window: usize) -> Result<Self> {
        let t = attention_mask.dim(1)?;
        let global = padding_mask(attention_mask)?;
        // Clamp rather than let MIN + MIN overflow to -inf: HF uses masked_fill here, and a padded
        // query row whose whole window is padding must stay finite, or its NaNs leak into the
        // real rows through the next layer's keys and values.
        let local = global
            .broadcast_add(&window_mask(t, half_window, attention_mask.device())?)?
            .clamp(f32::MIN, 0f32)?;
        Ok(Self { global, local })
    }

    /// Prefix layout: row `i` has `lens[i]` real tokens of which the first `prefix[i]` are the
    /// state. State tokens attend only to state tokens; the rest attend to every real token.
    /// Queries sit at absolute positions `q_offset..` and keys at `0..` (for cached prefixes).
    pub fn prefix(
        lens: &[usize],
        prefix: &[usize],
        t_q: usize,
        t_k: usize,
        q_offset: usize,
        half_window: usize,
        dev: &Device,
    ) -> Result<Self> {
        let b = lens.len();
        let mut global = vec![0f32; b * t_q * t_k];
        let mut local = vec![0f32; b * t_q * t_k];
        for r in 0..b {
            for qi in 0..t_q {
                let q = qi + q_offset;
                for k in 0..t_k {
                    let visible = k < lens[r] && (q >= prefix[r] || k < prefix[r]);
                    let idx = (r * t_q + qi) * t_k + k;
                    if !visible {
                        global[idx] = f32::MIN;
                    }
                    if !visible || q.abs_diff(k) > half_window {
                        local[idx] = f32::MIN;
                    }
                }
            }
        }
        Ok(Self {
            global: Tensor::from_vec(global, (b, 1, t_q, t_k), dev)?,
            local: Tensor::from_vec(local, (b, 1, t_q, t_k), dev)?,
        })
    }

    /// Masks for rows whose state is left-padded to `p_max` tokens, so every state ends at the
    /// same position and question tokens start at `p_max` in every row. Row `r`'s state is keys
    /// `pads[r]..p_max` and its question keys `p_max..p_max + sufs[r]`; queries sit at positions
    /// `q_offset..q_offset + t_q`. State queries see only their state, question queries see
    /// their state and question. RoPE and the sliding window depend only on relative position,
    /// so a left-padded row computes what it would alone. Padded state queries see only
    /// themselves and padded question queries see the row's real keys, so no row is empty.
    #[allow(clippy::too_many_arguments)]
    pub fn left_padded(
        pads: &[usize],
        p_max: usize,
        sufs: &[usize],
        t_q: usize,
        t_k: usize,
        q_offset: usize,
        half_window: usize,
        dev: &Device,
    ) -> Result<Self> {
        let b = pads.len();
        let mut global = vec![f32::MIN; b * t_q * t_k];
        let mut local = vec![f32::MIN; b * t_q * t_k];
        for r in 0..b {
            for qi in 0..t_q {
                let q = qi + q_offset;
                for k in 0..t_k {
                    let state_key = k >= pads[r] && k < p_max;
                    let visible = if q < pads[r] {
                        k == q
                    } else if q < p_max {
                        state_key
                    } else {
                        state_key || (k >= p_max && k < p_max + sufs[r])
                    };
                    if visible {
                        let idx = (r * t_q + qi) * t_k + k;
                        global[idx] = 0.0;
                        if q.abs_diff(k) <= half_window {
                            local[idx] = 0.0;
                        }
                    }
                }
            }
        }
        Ok(Self {
            global: Tensor::from_vec(global, (b, 1, t_q, t_k), dev)?,
            local: Tensor::from_vec(local, (b, 1, t_q, t_k), dev)?,
        })
    }
}

/// States encoded once: per-layer keys/values and the final hidden state `[g, t, d]`.
#[derive(Clone, Debug)]
pub struct PrefixCache {
    pub layers: Vec<Kv>,
    pub hidden: Tensor,
}

impl PrefixCache {
    pub fn len(&self) -> usize {
        self.hidden.dim(1).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
        let embeddings = Embedding::new(
            vb.pp("embeddings.tok_embeddings").get_with_hints(
                (cfg.vocab_size, d),
                "weight",
                init_normal(cfg.initializer_range),
            )?,
            d,
        );
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

    pub fn half_window(&self) -> usize {
        self.half_window
    }

    /// `input_ids`: `[b, t]` u32; `attention_mask`: `[b, t]` (1 = token, 0 = pad).
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let masks = Masks::padding(attention_mask, self.half_window)?;
        self.forward_masked(input_ids, &masks)
    }

    /// Forward with explicit attention masks (see [`Masks`]).
    pub fn forward_masked(&self, input_ids: &Tensor, masks: &Masks) -> Result<Tensor> {
        Ok(self.run(input_ids, masks, 0, None)?.0)
    }

    /// Encodes a state prefix (`[1, t]`, all real tokens) and keeps every layer's keys/values.
    pub fn encode_prefix(&self, input_ids: &Tensor) -> Result<PrefixCache> {
        let t = input_ids.dim(1)?;
        let masks = Masks::prefix(&[t], &[t], t, t, 0, self.half_window, input_ids.device())?;
        self.encode_prefix_masked(input_ids, &masks)
    }

    /// Encodes a batch of states (`[g, t]`) under `masks` (see [`Masks::left_padded`]) and
    /// keeps every layer's keys/values.
    pub fn encode_prefix_masked(&self, input_ids: &Tensor, masks: &Masks) -> Result<PrefixCache> {
        let (hidden, layers) = self.run(input_ids, masks, 0, None)?;
        Ok(PrefixCache { layers, hidden })
    }

    /// Runs `input_ids` (`[b, t]`) positioned after a cached prefix, attending to it.
    /// `masks` covers keys `[prefix; suffix]` (see [`Masks::prefix`] with `q_offset = prefix`).
    pub fn forward_suffix(
        &self,
        input_ids: &Tensor,
        masks: &Masks,
        cache: &PrefixCache,
    ) -> Result<Tensor> {
        Ok(self
            .run(input_ids, masks, cache.len(), Some(&cache.layers))?
            .0)
    }

    fn run(
        &self,
        input_ids: &Tensor,
        masks: &Masks,
        offset: usize,
        past: Option<&[Kv]>,
    ) -> Result<(Tensor, Vec<Kv>)> {
        let mut xs = input_ids.apply(&self.embeddings)?.apply(&self.emb_norm)?;
        let mut kvs = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let (mask, rope) = if layer.local {
                (&masks.local, &self.local_rope)
            } else {
                (&masks.global, &self.global_rope)
            };
            let (x, kv) = layer.forward(&xs, mask, rope, offset, past.map(|p| &p[i]))?;
            xs = x;
            kvs.push(kv);
        }
        Ok((xs.apply(&self.final_norm)?, kvs))
    }
}
