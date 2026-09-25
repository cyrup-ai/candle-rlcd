//! laya's typed decision head on top of the encoder.
//!
//! ```text
//! h  = encoder(ids, mask) + type_emb[qtype]           (added at every question position)
//! h  = 2 x pre-norm TransformerEncoderLayer(h)        (PyTorch defaults: ReLU FFN, biased, full attn)
//! lg = scorer(h[marker_pos])                          (LayerNorm -> Linear -> GELU -> Linear(d, 1))
//! act = act_head([h[pooled], top1, top1 - top2, H(p)/log k, k/255])
//! ```
//!
//! In laya's joint layout the type embedding goes on every position and `pooled` is `[CLS]`. In
//! the prefix layout it goes only on the question part (so the state's hidden states stay
//! question-independent and cacheable) and `pooled` is the question part's own `[CLS]`.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Embedding, Init, Linear, VarBuilder};

use crate::modernbert::{attention, init_normal, linear, Kv, Norm};

/// Dropout used by PyTorch's `TransformerEncoderLayer` default, applied only when training.
pub const HEAD_DROPOUT: f32 = 0.1;

fn dropout(x: &Tensor, train: bool) -> Result<Tensor> {
    if train {
        candle_nn::ops::dropout(x, HEAD_DROPOUT)
    } else {
        Ok(x.clone())
    }
}

/// PyTorch `nn.TransformerEncoderLayer(norm_first=True, activation=relu)`.
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
    fn load(vb: VarBuilder, d: usize, std: f64) -> Result<Self> {
        let sa = vb.pp("self_attn");
        let in_proj = Linear::new(
            sa.get_with_hints((3 * d, d), "in_proj_weight", init_normal(std))?,
            Some(sa.get_with_hints(3 * d, "in_proj_bias", Init::Const(0.0))?),
        );
        Ok(Self {
            in_proj,
            out_proj: linear(sa.pp("out_proj"), d, d, true, std)?,
            linear1: linear(vb.pp("linear1"), d, 4 * d, true, std)?,
            linear2: linear(vb.pp("linear2"), 4 * d, d, true, std)?,
            // PyTorch LayerNorm default eps.
            norm1: Norm::load(vb.pp("norm1"), d, 1e-5, true)?,
            norm2: Norm::load(vb.pp("norm2"), d, 1e-5, true)?,
            heads: (d / 64).max(1),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        past: Option<&Kv>,
        train: bool,
    ) -> Result<(Tensor, Kv)> {
        let (b, t, d) = xs.dims3()?;
        let hd = d / self.heads;
        let qkv = xs
            .apply(&self.norm1)?
            .apply(&self.in_proj)?
            .reshape((b, t, 3, self.heads, hd))?
            .permute((2, 0, 3, 1, 4))?;
        let (q, k, v) = (
            qkv.get(0)?.contiguous()?,
            qkv.get(1)?.contiguous()?,
            qkv.get(2)?.contiguous()?,
        );
        let a = match past {
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
        let a = a
            .transpose(1, 2)?
            .reshape((b, t, d))?
            .apply(&self.out_proj)?;
        let xs = (xs + dropout(&a, train)?)?;
        let f = xs.apply(&self.norm2)?.apply(&self.linear1)?.relu()?;
        let f = dropout(&f, train)?.apply(&self.linear2)?;
        Ok(((xs + dropout(&f, train)?)?, Kv { k, v }))
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
    /// temperatures are what decoding uses; this is kept for inspection and round-tripping.
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

/// Per-row inputs the head needs besides the hidden states.
#[derive(Debug, Clone)]
pub struct HeadInputs {
    /// `[n, kmax]` u32 marker positions (0 for padding).
    pub marker_pos: Tensor,
    /// `[n, kmax]` F32, 1 for a real option.
    pub marker_mask: Tensor,
    /// `[n]` u32 question type.
    pub qtype: Tensor,
    /// `[n, t, 1]` F32: where the type embedding is added (all ones in laya's layout).
    pub type_mask: Option<Tensor>,
    /// `[n]` u32 position pooled into the act head (0 = `[CLS]` in laya's layout).
    pub pooled_pos: Tensor,
}

impl DecisionHead {
    /// `std`: init scale for weights created rather than loaded.
    pub fn load(
        vb: VarBuilder,
        d: usize,
        head_layers: usize,
        n_act: usize,
        std: f64,
    ) -> Result<Self> {
        let layers = (0..head_layers)
            .map(|i| HeadLayer::load(vb.pp(format!("head.layers.{i}")), d, std))
            .collect::<Result<Vec<_>>>()?;
        let temperature = vb.get_with_hints(3, "temperature", Init::Const(1.0)).ok();
        Ok(Self {
            type_emb: Embedding::new(
                vb.pp("type_emb")
                    .get_with_hints((3, d), "weight", init_normal(std))?,
                d,
            ),
            layers,
            scorer_norm: Norm::load(vb.pp("scorer.0"), d, 1e-5, true)?,
            scorer_fc: linear(vb.pp("scorer.1"), d, d, true, std)?,
            scorer_out: linear(vb.pp("scorer.3"), d, 1, true, std)?,
            act_fc: linear(vb.pp("act_head.0"), d + 4, 256, true, std)?,
            act_out: linear(vb.pp("act_head.2"), 256, n_act, true, std)?,
            temperature,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// `h`: encoder output `[n, t, d]`; `mask`: additive F32 attention mask broadcastable to
    /// `[n, heads, t, t]`.
    pub fn forward(
        &self,
        h: &Tensor,
        mask: &Tensor,
        inp: &HeadInputs,
        train: bool,
    ) -> Result<HeadOutput> {
        let mut h = self.add_type(h, inp)?;
        for layer in &self.layers {
            h = layer.forward(&h, mask, None, train)?.0;
        }
        self.score(&h, inp)
    }

    /// Runs the head layers over a cached state's hidden states (`[1, t, d]`, no type
    /// embedding) and returns each layer's keys/values for [`Self::forward_suffix`].
    pub fn encode_prefix(&self, h: &Tensor, mask: &Tensor) -> Result<Vec<Kv>> {
        let mut h = h.clone();
        let mut kvs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (x, kv) = layer.forward(&h, mask, None, false)?;
            h = x;
            kvs.push(kv);
        }
        Ok(kvs)
    }

    /// Question rows (`[n, t, d]`, positions after the prefix) against cached head-layer
    /// keys/values. `inp` positions are relative to the question rows.
    pub fn forward_suffix(
        &self,
        h: &Tensor,
        mask: &Tensor,
        past: &[Kv],
        inp: &HeadInputs,
    ) -> Result<HeadOutput> {
        let mut h = self.add_type(h, inp)?;
        for (layer, kv) in self.layers.iter().zip(past) {
            h = layer.forward(&h, mask, Some(kv), false)?.0;
        }
        self.score(&h, inp)
    }

    fn add_type(&self, h: &Tensor, inp: &HeadInputs) -> Result<Tensor> {
        let te = inp.qtype.apply(&self.type_emb)?.unsqueeze(1)?;
        match &inp.type_mask {
            None => h.broadcast_add(&te),
            Some(m) => h + te.broadcast_mul(&m.to_dtype(h.dtype())?)?,
        }
    }

    fn score(&self, h: &Tensor, inp: &HeadInputs) -> Result<HeadOutput> {
        let (n, _t, d) = h.dims3()?;
        let (marker_pos, marker_mask) = (&inp.marker_pos, &inp.marker_mask);
        let kmax = marker_pos.dim(1)?;
        let dtype = h.dtype();
        let h = h.contiguous()?;
        let idx = marker_pos
            .unsqueeze(2)?
            .broadcast_as((n, kmax, d))?
            .contiguous()?;
        let m = h.gather(&idx, 1)?;
        let logits = m
            .apply(&self.scorer_norm)?
            .apply(&self.scorer_fc)?
            .gelu_erf()?
            .apply(&self.scorer_out)?
            .squeeze(2)?
            .to_dtype(DType::F32)?;
        // masked_fill(~marker_mask, -1e4)
        let logits = ((logits * marker_mask)? + ((1.0 - marker_mask)? * -1e4)?)?;

        // Confidence features from the (uncalibrated, detached) softmax, as in laya.
        let p = candle_nn::ops::softmax_last_dim(&logits.detach())?;
        let k = marker_mask.sum(1)?.clamp(2f32, f32::MAX)?;
        let ent = (p.clamp(1e-9f32, 1f32)?.log()? * &p)?.sum(1)?.neg()?;
        let ent = (ent / k.log()?)?;
        let (top1, top2) = top_two(&p)?;
        let feats = Tensor::stack(&[&top1, &(&top1 - &top2)?, &ent, &(k / 255.0)?], 1)?;
        let pidx = inp
            .pooled_pos
            .reshape((n, 1, 1))?
            .broadcast_as((n, 1, d))?
            .contiguous()?;
        let pooled = h.gather(&pidx, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
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
