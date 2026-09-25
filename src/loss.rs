//! The training objective: a direct proper-scoring loss on the option distribution.
//!
//! laya's RLCD treats `q = softmax(ℓ + ε)` as an action and estimates `∇ E_ε R(q, t)` with
//! REINFORCE. `R` is a known differentiable function of `q`, so we backpropagate through it
//! instead (the reparameterised gradient of the same objective, ~40x lower variance; see
//! the spec's §3.1):
//!
//! ```text
//! R(q, t) = w_log · Σ t·max(log q, floor) + w_sph · (t·q)/‖q‖ − w_rps · RPS(q, t) [score only]
//! loss    = −mean over rows and G noise samples of R(softmax(ℓ + ε_g), t)
//! ```
//!
//! All three terms are strictly proper. `ε` is zero-mean Gaussian noise over the valid options
//! (σ = 0 gives the deterministic loss); the RPS term is laya's ordinal penalty for `score`.

use candle_core::{Device, Result, Tensor, D};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, clap::Args)]
pub struct LossConfig {
    /// Weight of the (floored) log score. With the others at 0 this is soft cross-entropy.
    #[arg(long, default_value_t = 1.0)]
    pub w_log: f64,
    /// Weight of the spherical score (laya: 0.75).
    #[arg(long, default_value_t = 0.75)]
    pub w_sph: f64,
    /// Weight of the ranked probability score on `score` questions (laya: 1.0).
    #[arg(long, default_value_t = 1.0)]
    pub w_rps: f64,
    /// Floor on log q (laya: ln 1e-4 = -9.21).
    #[arg(long, default_value_t = -9.21)]
    pub log_floor: f64,
    /// Logit noise σ at the start of training (0 = deterministic loss; laya anneals 0.4 -> 0.1).
    #[arg(long, default_value_t = 0.0)]
    pub sigma_start: f64,
    /// Logit noise σ at the end of training.
    #[arg(long, default_value_t = 0.0)]
    pub sigma_end: f64,
    /// Noise samples per row when σ > 0.
    #[arg(long, default_value_t = 4)]
    pub noise_samples: usize,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            w_log: 1.0,
            w_sph: 0.75,
            w_rps: 1.0,
            log_floor: -9.21,
            sigma_start: 0.0,
            sigma_end: 0.0,
            noise_samples: 4,
        }
    }
}

impl LossConfig {
    /// Linear σ schedule over training progress `frac ∈ [0, 1]`.
    pub fn sigma(&self, frac: f64) -> f64 {
        self.sigma_start + (self.sigma_end - self.sigma_start) * frac.clamp(0.0, 1.0)
    }
}

/// Targets for a batch, padded to the logits' `kmax`.
pub struct Targets {
    /// `[n, kmax]` target distribution (0 on padded options).
    pub dist: Tensor,
    /// `[n, kmax]` 1 for real options.
    pub mask: Tensor,
    /// `[n]` 1 for `score` rows (RPS applies).
    pub is_score: Tensor,
    /// `[n]` 1 / (k - 1), for RPS normalisation.
    pub inv_km1: Tensor,
}

impl Targets {
    pub fn new(rows: &[(Vec<f32>, bool)], kmax: usize, dev: &Device) -> Result<Self> {
        let n = rows.len();
        let mut dist = vec![0f32; n * kmax];
        let mut mask = vec![0f32; n * kmax];
        let mut is_score = vec![0f32; n];
        let mut inv = vec![0f32; n];
        for (i, (t, score)) in rows.iter().enumerate() {
            dist[i * kmax..i * kmax + t.len()].copy_from_slice(t);
            mask[i * kmax..i * kmax + t.len()].fill(1.0);
            is_score[i] = if *score { 1.0 } else { 0.0 };
            inv[i] = 1.0 / (t.len().max(2) - 1) as f32;
        }
        Ok(Self {
            dist: Tensor::from_vec(dist, (n, kmax), dev)?,
            mask: Tensor::from_vec(mask, (n, kmax), dev)?,
            is_score: Tensor::from_vec(is_score, n, dev)?,
            inv_km1: Tensor::from_vec(inv, n, dev)?,
        })
    }
}

/// Mean negative reward over rows (and noise samples). `logits`: `[n, kmax]` F32 with padded
/// options already at a large negative value.
pub fn proper_scoring_loss(
    logits: &Tensor,
    t: &Targets,
    cfg: &LossConfig,
    sigma: f64,
) -> Result<Tensor> {
    let samples = if sigma > 0.0 {
        cfg.noise_samples.max(1)
    } else {
        1
    };
    let mut total: Option<Tensor> = None;
    for _ in 0..samples {
        let z = if sigma > 0.0 {
            // Zero-mean over the valid options, zero on padding.
            let e = (logits.randn_like(0.0, sigma)? * &t.mask)?;
            let mean = (e.sum_keepdim(D::Minus1)? / t.mask.sum_keepdim(D::Minus1)?)?;
            let e = (e.broadcast_sub(&mean)? * &t.mask)?;
            (logits + e)?
        } else {
            logits.clone()
        };
        let r = reward(&z, t, cfg)?;
        total = Some(match total {
            None => r,
            Some(acc) => (acc + r)?,
        });
    }
    let r = (total.expect("at least one sample") / samples as f64)?;
    r.mean_all()?.neg()
}

/// Per-row reward `[n]`.
fn reward(z: &Tensor, t: &Targets, cfg: &LossConfig) -> Result<Tensor> {
    let mut r = z.zeros_like()?.sum(D::Minus1)?;
    if cfg.w_log != 0.0 {
        let logq = candle_nn::ops::log_softmax(z, D::Minus1)?;
        let floor = logq.ones_like()?.affine(cfg.log_floor, 0.0)?;
        let logq = logq.maximum(&floor)?;
        r = (r + ((logq * &t.dist)?.sum(D::Minus1)? * cfg.w_log)?)?;
    }
    if cfg.w_sph == 0.0 && cfg.w_rps == 0.0 {
        return Ok(r);
    }
    let q = crate::autograd::softmax_last_dim(z)?;
    if cfg.w_sph != 0.0 {
        let tq = (&q * &t.dist)?.sum(D::Minus1)?;
        let norm = q.sqr()?.sum(D::Minus1)?.sqrt()?;
        r = (r + ((tq / norm)? * cfg.w_sph)?)?;
    }
    if cfg.w_rps != 0.0 {
        // Padded options have q ≈ 0 and t = 0, so the CDFs are flat there and add nothing.
        let diff = (&q - &t.dist)?.cumsum(D::Minus1)?;
        let rps = (diff.sqr()?.sum(D::Minus1)? * &t.inv_km1)?;
        r = (r - ((rps * &t.is_score)? * cfg.w_rps)?)?;
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    #[test]
    fn log_only_is_soft_cross_entropy() {
        let dev = Device::Cpu;
        let l = Tensor::new(&[[1.0f32, -0.5, 0.2, -1e4]], &dev).unwrap();
        let t = Targets::new(&[(vec![0.6, 0.3, 0.1], false)], 4, &dev).unwrap();
        let cfg = LossConfig {
            w_sph: 0.0,
            w_rps: 0.0,
            ..Default::default()
        };
        let got = proper_scoring_loss(&l, &t, &cfg, 0.0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let v = [1.0f64, -0.5, 0.2];
        let lse = v.iter().map(|x| x.exp()).sum::<f64>().ln();
        let want = -(0.6 * (v[0] - lse) + 0.3 * (v[1] - lse) + 0.1 * (v[2] - lse));
        assert!((got as f64 - want).abs() < 1e-5, "{got} vs {want}");
    }

    #[test]
    fn minimised_at_the_target() {
        // Proper scoring: gradient at logits = log t is ~0 for every term.
        let dev = Device::Cpu;
        let t = vec![0.5f32, 0.3, 0.2];
        let l = Var::from_vec(t.iter().map(|x| x.ln()).collect::<Vec<_>>(), (1, 3), &dev).unwrap();
        let tg = Targets::new(&[(t, true)], 3, &dev).unwrap();
        let loss = proper_scoring_loss(&l, &tg, &LossConfig::default(), 0.0).unwrap();
        let g = loss.backward().unwrap();
        let g = g
            .get(&l)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(g.iter().all(|x| x.abs() < 1e-5), "{g:?}");
    }

    #[test]
    fn noise_is_zero_mean_over_valid_options() {
        let dev = Device::Cpu;
        let l = Var::from_vec(vec![0.3f32, 0.1, -1e4], (1, 3), &dev).unwrap();
        let tg = Targets::new(&[(vec![1.0, 0.0], false)], 3, &dev).unwrap();
        let loss = proper_scoring_loss(&l, &tg, &LossConfig::default(), 0.4).unwrap();
        assert!(loss.to_scalar::<f32>().unwrap().is_finite());
        let g = loss.backward().unwrap();
        let g = g
            .get(&l)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(g[2].abs() < 1e-12 && g[0] < 0.0);
    }
}
