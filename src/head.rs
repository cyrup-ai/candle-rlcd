//! laya's typed decision head on top of the encoder.
//!
//! ```text
//! h  = encoder(ids, mask) + type_emb[qtype]           (added at every position)
//! h  = 2 x pre-norm TransformerEncoderLayer(h)        (PyTorch defaults: ReLU FFN, biased, full attn)
//! lg = scorer(h[marker_pos])                          (LayerNorm -> Linear -> GELU -> Linear(d, 1))
//! act = act_head([h[:, 0], top1, top1 - top2, H(p)/log k, k/255])
//! ```

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::modernbert::{attention, padding_mask, Norm};

/// PyTorch `nn.TransformerEncoderLayer(norm_first=True, activation=relu)` in eval mode.
#[derive(Clone, Debug)]
struct HeadLayer {
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: Norm,
    norm2: Norm,
    heads: usize,
}

impl HeadLayer {
    fn load(vb: VarBuilder, d: usize) -> Result<Self> {
        let sa = vb.pp("self_attn");
        let in_proj = Linear::new(
            sa.get((3 * d, d), "in_proj_weight")?,
            Some(sa.get(3 * d, "in_proj_bias")?),
        );
        Ok(Self {
            in_proj,
            out_proj: candle_nn::linear(d, d, sa.pp("out_proj"))?,
            linear1: candle_nn::linear(d, 4 * d, vb.pp("linear1"))?,
            linear2: candle_nn::linear(4 * d, d, vb.pp("linear2"))?,
            // PyTorch LayerNorm default eps.
            norm1: Norm::load(vb.pp("norm1"), d, 1e-5, true)?,
            norm2: Norm::load(vb.pp("norm2"), d, 1e-5, true)?,
            heads: (d / 64).max(1),
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let hd = d / self.heads;
        let qkv = xs
            .apply(&self.norm1)?
            .apply(&self.in_proj)?
            .reshape((b, t, 3, self.heads, hd))?
            .permute((2, 0, 3, 1, 4))?;
        let a = attention(&qkv.get(0)?, &qkv.get(1)?, &qkv.get(2)?, mask)?;
        let a = a
            .transpose(1, 2)?
            .reshape((b, t, d))?
            .apply(&self.out_proj)?;
        let xs = (xs + a)?;
        let f = xs
            .apply(&self.norm2)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        xs + f
    }
}

#[derive(Clone, Debug)]
pub struct DecisionHead {
    type_emb: Embedding,
    layers: Vec<HeadLayer>,
    scorer_norm: Norm,
    scorer_fc: Linear,
    scorer_out: Linear,
    act_fc: Linear,
    act_out: Linear,
    /// Temperature buffer stored in the checkpoint (per qtype). The agent config's fitted
    /// temperatures are what decoding uses; this is kept for inspection.
    pub temperature: Option<Tensor>,
}

/// Raw head outputs for a batch of question rows.
#[derive(Debug)]
pub struct HeadOutput {
    /// `[n, kmax]` F32 option logits; invalid options are `-1e4`.
    pub logits: Tensor,
    /// `[n, n_act]` F32 act-head logits.
    pub act_logits: Tensor,
}

impl DecisionHead {
    pub fn load(vb: VarBuilder, d: usize, head_layers: usize, n_act: usize) -> Result<Self> {
        let layers = (0..head_layers)
            .map(|i| HeadLayer::load(vb.pp(format!("head.layers.{i}")), d))
            .collect::<Result<Vec<_>>>()?;
        let temperature = vb.get(3, "temperature").ok();
        Ok(Self {
            type_emb: candle_nn::embedding(3, d, vb.pp("type_emb"))?,
            layers,
            scorer_norm: Norm::load(vb.pp("scorer.0"), d, 1e-5, true)?,
            scorer_fc: candle_nn::linear(d, d, vb.pp("scorer.1"))?,
            scorer_out: candle_nn::linear(d, 1, vb.pp("scorer.3"))?,
            act_fc: candle_nn::linear(d + 4, 256, vb.pp("act_head.0"))?,
            act_out: candle_nn::linear(256, n_act, vb.pp("act_head.2"))?,
            temperature,
        })
    }

    /// `h`: encoder output `[n, t, d]`; `attention_mask`: `[n, t]`; `marker_pos`: `[n, kmax]` u32
    /// (0 for padding); `marker_mask`: `[n, kmax]` F32 (1 valid, 0 pad); `qtype`: `[n]` u32.
    pub fn forward(
        &self,
        h: &Tensor,
        attention_mask: &Tensor,
        marker_pos: &Tensor,
        marker_mask: &Tensor,
        qtype: &Tensor,
    ) -> Result<HeadOutput> {
        let (n, _t, d) = h.dims3()?;
        let kmax = marker_pos.dim(1)?;
        let dtype = h.dtype();
        let mut h = h.broadcast_add(&qtype.apply(&self.type_emb)?.unsqueeze(1)?)?;
        if !self.layers.is_empty() {
            let mask = padding_mask(attention_mask)?;
            for layer in &self.layers {
                h = layer.forward(&h, &mask)?;
            }
        }
        let idx = marker_pos
            .unsqueeze(2)?
            .broadcast_as((n, kmax, d))?
            .contiguous()?;
        let m = h.contiguous()?.gather(&idx, 1)?;
        let logits = m
            .apply(&self.scorer_norm)?
            .apply(&self.scorer_fc)?
            .gelu_erf()?
            .apply(&self.scorer_out)?
            .squeeze(2)?
            .to_dtype(DType::F32)?;
        // masked_fill(~marker_mask, -1e4)
        let logits = ((logits * marker_mask)? + ((1.0 - marker_mask)? * -1e4)?)?;

        // Confidence features from the (uncalibrated) softmax.
        let p = candle_nn::ops::softmax_last_dim(&logits)?;
        let k = marker_mask.sum(1)?.clamp(2f32, f32::MAX)?;
        let ent = (p.clamp(1e-9f32, 1f32)?.log()? * &p)?.sum(1)?.neg()?;
        let ent = (ent / k.log()?)?;
        let (top1, top2) = top_two(&p)?;
        let feats = Tensor::stack(&[&top1, &(&top1 - &top2)?, &ent, &(k / 255.0)?], 1)?;
        let pooled = h.narrow(1, 0, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
        let act_in = Tensor::cat(&[&pooled, &feats], 1)?.to_dtype(dtype)?;
        let act_logits = act_in
            .apply(&self.act_fc)?
            .gelu_erf()?
            .apply(&self.act_out)?
            .to_dtype(DType::F32)?;
        Ok(HeadOutput { logits, act_logits })
    }
}

/// Largest and second-largest entries of each row of `p` (`[n, k]`); second is 0 when `k == 1`.
fn top_two(p: &Tensor) -> Result<(Tensor, Tensor)> {
    let sorted = p.contiguous()?.sort_last_dim(false)?.0;
    let top1 = sorted.narrow(D::Minus1, 0, 1)?.squeeze(1)?;
    let top2 = if p.dim(1)? >= 2 {
        sorted.narrow(D::Minus1, 1, 1)?.squeeze(1)?
    } else {
        top1.zeros_like()?
    };
    Ok((top1, top2))
}
