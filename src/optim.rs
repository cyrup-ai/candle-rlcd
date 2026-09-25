//! Training pieces Candle lacks: AdamW with saveable F32 state and per-group learning rates,
//! clip-by-global-norm, and a warmup + cosine schedule.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Result, Tensor, Var};

/// One trainable tensor with its checkpoint name and parameter group.
pub struct Param {
    pub name: String,
    pub var: Var,
    pub group: usize,
    /// Decoupled weight decay applies to matrices only, not to norms, biases or 1-D buffers.
    pub decay: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Group {
    /// Peak learning rate.
    pub lr: f64,
    pub weight_decay: f64,
}

pub struct AdamW {
    pub params: Vec<Param>,
    pub groups: Vec<Group>,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Completed optimizer steps.
    pub step: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(params: Vec<Param>, groups: Vec<Group>) -> Result<Self> {
        let zeros = |p: &Param| Tensor::zeros(p.var.shape(), DType::F32, p.var.device());
        let m = params.iter().map(zeros).collect::<Result<Vec<_>>>()?;
        let v = params.iter().map(zeros).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            params,
            groups,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step: 0,
            m,
            v,
        })
    }

    /// One update. `grads[i]` belongs to `params[i]` (None = no gradient, skipped);
    /// `lr_scale` multiplies every group's peak learning rate (the schedule).
    pub fn step(&mut self, grads: &[Option<Tensor>], lr_scale: f64) -> Result<()> {
        self.step += 1;
        let t = self.step as i32;
        let bc1 = 1.0 - self.beta1.powi(t);
        let bc2 = 1.0 - self.beta2.powi(t);
        for (i, p) in self.params.iter().enumerate() {
            let Some(g) = &grads[i] else { continue };
            let g = g.to_dtype(DType::F32)?;
            let group = self.groups[p.group];
            let lr = group.lr * lr_scale;
            let m = ((&self.m[i] * self.beta1)? + (&g * (1.0 - self.beta1))?)?;
            let v = ((&self.v[i] * self.beta2)? + (g.sqr()? * (1.0 - self.beta2))?)?;
            let update = ((&m / bc1)? / ((&v / bc2)?.sqrt()? + self.eps)?)?;
            let theta = p.var.as_tensor().to_dtype(DType::F32)?;
            let theta = if p.decay && group.weight_decay > 0.0 {
                (theta * (1.0 - lr * group.weight_decay))?
            } else {
                theta
            };
            let theta = (theta - (update * lr)?)?;
            p.var.set(&theta.to_dtype(p.var.dtype())?)?;
            self.m[i] = m;
            self.v[i] = v;
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for (i, p) in self.params.iter().enumerate() {
            map.insert(format!("m.{}", p.name), self.m[i].clone());
            map.insert(format!("v.{}", p.name), self.v[i].clone());
        }
        map.insert(
            "step".into(),
            Tensor::new(&[self.step as u32], &candle_core::Device::Cpu)?,
        );
        candle_core::safetensors::save(&map, path)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let dev = self.params.first().map(|p| p.var.device().clone());
        let dev = dev.unwrap_or(candle_core::Device::Cpu);
        let map = candle_core::safetensors::load(path, &dev)?;
        for (i, p) in self.params.iter().enumerate() {
            let get = |k: String| {
                map.get(&k)
                    .cloned()
                    .ok_or_else(|| candle_core::Error::Msg(format!("optimizer state missing {k}")))
            };
            self.m[i] = get(format!("m.{}", p.name))?;
            self.v[i] = get(format!("v.{}", p.name))?;
        }
        if let Some(s) = map.get("step") {
            self.step = s.to_vec1::<u32>()?[0] as usize;
        }
        Ok(())
    }
}

/// Scales gradients in place so their global L2 norm is at most `max_norm`; returns the norm
/// before clipping.
pub fn clip_grad_norm(grads: &mut [Option<Tensor>], max_norm: f64) -> Result<f64> {
    let mut sq = 0f64;
    for g in grads.iter().flatten() {
        sq += g
            .to_dtype(DType::F32)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()? as f64;
    }
    let norm = sq.sqrt();
    if max_norm > 0.0 && norm > max_norm {
        let s = max_norm / (norm + 1e-6);
        for g in grads.iter_mut().flatten() {
            *g = (&*g * s)?;
        }
    }
    Ok(norm)
}

/// Linear warmup to 1, then cosine decay to `min_ratio` at `total` steps.
#[derive(Debug, Clone, Copy)]
pub struct Schedule {
    pub warmup: usize,
    pub total: usize,
    pub min_ratio: f64,
}

impl Schedule {
    /// Multiplier for the learning rate of optimizer step `step` (0-based).
    pub fn scale(&self, step: usize) -> f64 {
        if step < self.warmup {
            return (step + 1) as f64 / self.warmup as f64;
        }
        let span = self.total.saturating_sub(self.warmup).max(1);
        let p = ((step - self.warmup) as f64 / span as f64).min(1.0);
        self.min_ratio + (1.0 - self.min_ratio) * 0.5 * (1.0 + (std::f64::consts::PI * p).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn schedule_shape() {
        let s = Schedule {
            warmup: 10,
            total: 110,
            min_ratio: 0.1,
        };
        assert!((s.scale(0) - 0.1).abs() < 1e-9);
        assert!((s.scale(9) - 1.0).abs() < 1e-9);
        assert!((s.scale(10) - 1.0).abs() < 1e-9);
        assert!((s.scale(60) - 0.55).abs() < 1e-9);
        assert!((s.scale(110) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn clipping_caps_the_global_norm() {
        let dev = Device::Cpu;
        let mut g = vec![
            Some(Tensor::new(&[3f32], &dev).unwrap()),
            None,
            Some(Tensor::new(&[4f32], &dev).unwrap()),
        ];
        let n = clip_grad_norm(&mut g, 1.0).unwrap();
        assert!((n - 5.0).abs() < 1e-6);
        let a = g[0].as_ref().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((a - 0.6).abs() < 1e-5);
    }

    #[test]
    fn adamw_minimises_and_round_trips() {
        let dev = Device::Cpu;
        let x = Var::new(&[2f32, -3.0], &dev).unwrap();
        let params = vec![Param {
            name: "x".into(),
            var: x.clone(),
            group: 0,
            decay: false,
        }];
        let mut opt = AdamW::new(
            params,
            vec![Group {
                lr: 0.1,
                weight_decay: 0.0,
            }],
        )
        .unwrap();
        for _ in 0..200 {
            let g = x
                .as_tensor()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .backward()
                .unwrap();
            opt.step(&[g.get(&x).cloned()], 1.0).unwrap();
        }
        let v = x.as_tensor().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|a| a.abs() < 0.1), "{v:?}");
        let dir = std::env::temp_dir().join(format!("adamw-{}.safetensors", std::process::id()));
        opt.save(&dir).unwrap();
        let before = opt.m[0].to_vec1::<f32>().unwrap();
        opt.m[0] = opt.m[0].zeros_like().unwrap();
        opt.step = 0;
        opt.load(&dir).unwrap();
        assert_eq!(opt.step, 200);
        assert_eq!(opt.m[0].to_vec1::<f32>().unwrap(), before);
        std::fs::remove_file(dir).ok();
    }
}
