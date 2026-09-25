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

use candle_core::{Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::head::{DecisionHead, HeadInputs, HeadOutput};
use crate::modernbert::{Masks, ModernBert, PrefixCache};
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
        self.forward_prefix_groups(&[rows], pad, dev)
    }

    /// Prefix layout inference over several requests at once: each group is one state's
    /// question rows (sharing that state's prefix). The states are encoded together,
    /// left-padded to a common length (see [`Masks::left_padded`]), then every question row of
    /// every group runs as one batch against its own state's cache. Output rows follow the
    /// groups in order.
    pub fn forward_prefix_groups(
        &self,
        groups: &[&[Encoded]],
        pad: u32,
        dev: &Device,
    ) -> Result<HeadOutput> {
        if self.layout != Layout::Prefix {
            candle_core::bail!("forward_prefix needs a prefix-layout model");
        }
        if groups.is_empty() || groups.iter().any(|g| g.is_empty()) {
            candle_core::bail!("no rows");
        }
        let mut states = Vec::with_capacity(groups.len());
        for rows in groups {
            let p = rows[0].prefix_len;
            let state = &rows[0].ids[..p];
            if rows
                .iter()
                .any(|r| r.prefix_len != p || &r.ids[..p] != state)
            {
                candle_core::bail!("rows must share one state prefix");
            }
            states.push(state);
        }
        let hw = self.encoder.half_window();
        let g = states.len();
        let p_max = states.iter().map(|s| s.len()).max().unwrap_or(0);
        let pads: Vec<usize> = states.iter().map(|s| p_max - s.len()).collect();
        let mut state_ids = vec![pad; g * p_max];
        for (i, s) in states.iter().enumerate() {
            state_ids[i * p_max + pads[i]..(i + 1) * p_max].copy_from_slice(s);
        }
        let state_ids = Tensor::from_vec(state_ids, (g, p_max), dev)?;
        let state_masks = Masks::left_padded(&pads, p_max, &vec![0; g], p_max, p_max, 0, hw, dev)?;
        let cache = self
            .encoder
            .encode_prefix_masked(&state_ids, &state_masks)?;
        let head_kv = self
            .head
            .encode_prefix(&cache.hidden, &state_masks.global)?;

        let rows: Vec<&Encoded> = groups.iter().flat_map(|g| g.iter()).collect();
        let n = rows.len();
        let t = rows
            .iter()
            .map(|r| r.ids.len() - r.prefix_len)
            .max()
            .unwrap_or(0);
        let mut ids = vec![pad; n * t];
        let mut row_pads = Vec::with_capacity(n);
        let mut sufs = Vec::with_capacity(n);
        let mut owner = Vec::with_capacity(n);
        for (gi, rows) in groups.iter().enumerate() {
            for r in rows.iter() {
                let i = owner.len();
                let suffix = &r.ids[r.prefix_len..];
                ids[i * t..i * t + suffix.len()].copy_from_slice(suffix);
                row_pads.push(pads[gi]);
                sufs.push(suffix.len());
                owner.push(gi as u32);
            }
        }
        let ids = Tensor::from_vec(ids, (n, t), dev)?;
        let masks = Masks::left_padded(&row_pads, p_max, &sufs, t, p_max + t, p_max, hw, dev)?;
        // One cache row per question row (a single state is broadcast instead).
        let (enc_cache, head_kv) = if g == 1 {
            (cache, head_kv)
        } else {
            let owner = Tensor::from_vec(owner, n, dev)?;
            let layers = cache
                .layers
                .iter()
                .map(|kv| kv.select(&owner))
                .collect::<Result<Vec<_>>>()?;
            let head_kv = head_kv
                .iter()
                .map(|kv| kv.select(&owner))
                .collect::<Result<Vec<_>>>()?;
            (
                PrefixCache {
                    layers,
                    hidden: cache.hidden,
                },
                head_kv,
            )
        };
        let h = self.encoder.forward_suffix(&ids, &masks, &enc_cache)?;
        let inp = head_inputs_rows(&rows, dev)?;
        self.head.forward_suffix(&h, &masks.global, &head_kv, &inp)
    }
}

/// Marker/type tensors for question rows run as suffixes: marker positions are relative to
/// each row's own question part.
fn head_inputs_rows(rows: &[&Encoded], dev: &Device) -> Result<HeadInputs> {
    let n = rows.len();
    let kmax = rows.iter().map(|r| r.markers.len()).max().unwrap_or(0);
    let mut mpos = vec![0u32; n * kmax];
    let mut mmask = vec![0f32; n * kmax];
    for (i, r) in rows.iter().enumerate() {
        for (j, &m) in r.markers.iter().enumerate() {
            mpos[i * kmax + j] = (m - r.prefix_len) as u32;
            mmask[i * kmax + j] = 1.0;
        }
    }
    let qt: Vec<u32> = rows.iter().map(|r| r.qtype as u32).collect();
    Ok(HeadInputs {
        marker_pos: Tensor::from_vec(mpos, (n, kmax), dev)?,
        marker_mask: Tensor::from_vec(mmask, (n, kmax), dev)?,
        qtype: Tensor::from_vec(qt, n, dev)?,
        type_mask: None,
        pooled_pos: Tensor::from_vec(vec![0u32; n], n, dev)?,
    })
}
