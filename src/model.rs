//! Encoder + decision head as one model, in either sequence layout.
//!
//! - [`Layout::Laya`]: laya's joint `[CLS] question [SEP] options [SEP] state [SEP]` with full
//!   attention. Loads laya checkpoints as-is; every question re-encodes the state.
//! - [`Layout::Prefix`]: `[CLS] state [SEP] [CLS] question [SEP] options [SEP]`, where state
//!   tokens attend only to the state. Same parameters as laya (a laya checkpoint is a valid
//!   starting point for fine-tuning), but the state's hidden states no longer depend on the
//!   question, so inference encodes the state once per request and runs each question as a
//!   short suffix against its cached keys/values ([`DecisionModel::forward_prefix`]).
//!   Training runs the joint masked forward, which is exactly equal to the cached path.

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::head::{DecisionHead, HeadInputs, HeadOutput};
use crate::modernbert::{Masks, ModernBert};
use crate::sequence::Encoded;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    #[default]
    Laya,
    Prefix,
}

#[derive(Clone, Debug)]
pub struct DecisionModel {
    pub encoder: ModernBert,
    pub head: DecisionHead,
    pub layout: Layout,
}

/// A padded batch of question rows as tensors.
#[derive(Debug, Clone)]
pub struct Batch {
    pub ids: Tensor,
    pub masks: Masks,
    /// Mask for the head's own (global) attention layers.
    pub head_mask: Tensor,
    pub head: HeadInputs,
    /// Real option count per row.
    pub ks: Vec<usize>,
}

impl Batch {
    pub fn new(
        rows: &[Encoded],
        pad: u32,
        layout: Layout,
        half_window: usize,
        dev: &Device,
    ) -> Result<Self> {
        let n = rows.len();
        let t = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let mut ids = vec![pad; n * t];
        let mut att = vec![0u32; n * t];
        for (i, r) in rows.iter().enumerate() {
            ids[i * t..i * t + r.ids.len()].copy_from_slice(&r.ids);
            att[i * t..i * t + r.ids.len()].fill(1);
        }
        let ids = Tensor::from_vec(ids, (n, t), dev)?;
        let (masks, head_mask, type_mask, pooled) = match layout {
            Layout::Laya => {
                let att = Tensor::from_vec(att, (n, t), dev)?;
                let masks = Masks::padding(&att, half_window)?;
                let head_mask = masks.global.clone();
                (masks, head_mask, None, vec![0u32; n])
            }
            Layout::Prefix => {
                let lens: Vec<usize> = rows.iter().map(|r| r.ids.len()).collect();
                let prefix: Vec<usize> = rows.iter().map(|r| r.prefix_len).collect();
                let masks = Masks::prefix(&lens, &prefix, t, t, 0, half_window, dev)?;
                let head_mask = masks.global.clone();
                let tm: Vec<f32> = rows
                    .iter()
                    .flat_map(|r| (0..t).map(move |j| if j >= r.prefix_len { 1.0 } else { 0.0 }))
                    .collect();
                let tm = Tensor::from_vec(tm, (n, t, 1), dev)?;
                let pooled = rows.iter().map(|r| r.prefix_len as u32).collect();
                (masks, head_mask, Some(tm), pooled)
            }
        };
        let head = head_inputs(rows, 0, type_mask, pooled, dev)?;
        Ok(Self {
            ids,
            masks,
            head_mask,
            head,
            ks: rows.iter().map(|r| r.markers.len()).collect(),
        })
    }
}

/// Marker/type tensors; marker positions are shifted down by `shift`.
fn head_inputs(
    rows: &[Encoded],
    shift: usize,
    type_mask: Option<Tensor>,
    pooled: Vec<u32>,
    dev: &Device,
) -> Result<HeadInputs> {
    let n = rows.len();
    let kmax = rows.iter().map(|r| r.markers.len()).max().unwrap_or(0);
    let mut mpos = vec![0u32; n * kmax];
    let mut mmask = vec![0f32; n * kmax];
    for (i, r) in rows.iter().enumerate() {
        for (j, &m) in r.markers.iter().enumerate() {
            mpos[i * kmax + j] = (m - shift) as u32;
            mmask[i * kmax + j] = 1.0;
        }
    }
    let qt: Vec<u32> = rows.iter().map(|r| r.qtype as u32).collect();
    Ok(HeadInputs {
        marker_pos: Tensor::from_vec(mpos, (n, kmax), dev)?,
        marker_mask: Tensor::from_vec(mmask, (n, kmax), dev)?,
        qtype: Tensor::from_vec(qt, n, dev)?,
        type_mask,
        pooled_pos: Tensor::from_vec(pooled, n, dev)?,
    })
}

impl DecisionModel {
    pub fn batch(&self, rows: &[Encoded], pad: u32, dev: &Device) -> Result<Batch> {
        Batch::new(rows, pad, self.layout, self.encoder.half_window(), dev)
    }

    /// Joint forward over a padded batch (training, and inference in laya's layout).
    pub fn forward(&self, b: &Batch, train: bool) -> Result<HeadOutput> {
        let h = self.encoder.forward_masked(&b.ids, &b.masks)?;
        self.head.forward(&h, &b.head_mask, &b.head, train)
    }

    /// Prefix layout inference: the state prefix is encoded once and every question row
    /// (whose `ids` start with that same prefix) runs as a suffix against its cache.
    pub fn forward_prefix(&self, rows: &[Encoded], pad: u32, dev: &Device) -> Result<HeadOutput> {
        if self.layout != Layout::Prefix {
            candle_core::bail!("forward_prefix needs a prefix-layout model");
        }
        let Some(first) = rows.first() else {
            candle_core::bail!("no rows");
        };
        let p = first.prefix_len;
        let state = &first.ids[..p];
        if rows
            .iter()
            .any(|r| r.prefix_len != p || &r.ids[..p] != state)
        {
            candle_core::bail!("rows must share one state prefix");
        }
        let hw = self.encoder.half_window();
        let cache = self
            .encoder
            .encode_prefix(&Tensor::from_vec(state.to_vec(), (1, p), dev)?)?;
        let state_mask = Tensor::zeros((1, 1, p, p), DType::F32, dev)?;
        let head_kv = self.head.encode_prefix(&cache.hidden, &state_mask)?;

        let n = rows.len();
        let t = rows.iter().map(|r| r.ids.len() - p).max().unwrap_or(0);
        let mut ids = vec![pad; n * t];
        for (i, r) in rows.iter().enumerate() {
            ids[i * t..i * t + r.ids.len() - p].copy_from_slice(&r.ids[p..]);
        }
        let ids = Tensor::from_vec(ids, (n, t), dev)?;
        let lens: Vec<usize> = rows.iter().map(|r| r.ids.len()).collect();
        let masks = Masks::prefix(&lens, &vec![p; n], t, p + t, p, hw, dev)?;
        let h = self.encoder.forward_suffix(&ids, &masks, &cache)?;
        let inp = head_inputs(rows, p, None, vec![0; n], dev)?;
        self.head.forward_suffix(&h, &masks.global, &head_kv, &inp)
    }
}
