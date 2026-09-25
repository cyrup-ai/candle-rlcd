//! Backward passes for the fused Candle ops the model uses.
//!
//! Candle's fused `rope`, `softmax_last_dim` and `layer_norm` are forward-only
//! (`apply_op*_no_bwd`), so gradients silently stop at them: with upstream `rope`, the Q and K
//! slices of `Wqkv` never train. Here the fused kernel still runs the forward, and
//! [`attach_grad`] splices a hand-written backward onto its output. When nothing upstream tracks
//! gradients (inference), the fused op runs alone at no extra cost.

use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CudaStorage, CustomOp2, Layout, MetalStorage, Result, Shape, Tensor, D,
};

type BwdFn = dyn Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor> + Send + Sync;

/// Identity on `y` in the forward pass; in the backward pass sends `bwd(x, y, grad_y)` to `x`.
struct WithGrad {
    name: &'static str,
    bwd: Box<BwdFn>,
}

fn check_layout(l: &Layout) -> Result<()> {
    if !l.is_contiguous() || l.start_offset() != 0 {
        candle_core::bail!("attach_grad: output must be contiguous and start at offset 0")
    }
    Ok(())
}

impl CustomOp2 for WithGrad {
    fn name(&self) -> &'static str {
        self.name
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        y: &CpuStorage,
        ly: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        check_layout(ly)?;
        Ok((y.try_clone(ly)?, ly.shape().clone()))
    }

    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        y: &CudaStorage,
        ly: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        check_layout(ly)?;
        Ok((y.try_clone(ly)?, ly.shape().clone()))
    }

    fn metal_fwd(
        &self,
        _: &MetalStorage,
        _: &Layout,
        y: &MetalStorage,
        ly: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        check_layout(ly)?;
        Ok((y.try_clone(ly)?, ly.shape().clone()))
    }

    fn bwd(
        &self,
        x: &Tensor,
        _y: &Tensor,
        res: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((Some((self.bwd)(x, res, grad)?), None))
    }
}

/// Returns `y` (computed from `x` by a forward-only op) with `bwd(x, y, dL/dy) = dL/dx` attached.
pub fn attach_grad(
    name: &'static str,
    x: &Tensor,
    y: Tensor,
    bwd: impl Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor> + Send + Sync + 'static,
) -> Result<Tensor> {
    if !x.track_op() {
        return Ok(y);
    }
    let y = y.contiguous()?;
    x.apply_op2_arc(
        &y,
        Arc::new(Box::new(WithGrad {
            name,
            bwd: Box::new(bwd),
        })),
    )
}

/// Non-interleaved RoPE (`candle_nn::rotary_emb::rope`) with a backward pass.
///
/// RoPE is a rotation by `+θ` of each `(x_i, x_{i+d/2})` pair, so its transpose is the rotation
/// by `-θ`: the gradient is `rope(grad, cos, -sin)`.
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let x = x.contiguous()?;
    let y = candle_nn::rotary_emb::rope(&x, cos, sin)?;
    let (cos, sin) = (cos.clone(), sin.clone());
    attach_grad("rope-bwd", &x, y, move |_, _, g| {
        candle_nn::rotary_emb::rope(&g.contiguous()?, &cos, &sin.neg()?)
    })
}

/// Fused softmax over the last dim with a backward pass: `dx = y * (g - Σ g·y)`.
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let y = candle_nn::ops::softmax_last_dim(x)?;
    attach_grad("softmax-bwd", x, y, |_, y, g| {
        let dot = (g * y)?.sum_keepdim(D::Minus1)?;
        y * g.broadcast_sub(&dot)?
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Var};

    fn tables(t: usize, d: usize, dev: &Device) -> (Tensor, Tensor) {
        let f: Vec<f32> = (0..t)
            .flat_map(|p| (0..d / 2).map(move |i| p as f32 * 0.3 / (1.0 + i as f32)))
            .collect();
        let f = Tensor::from_vec(f, (t, d / 2), dev).unwrap();
        (f.cos().unwrap(), f.sin().unwrap())
    }

    fn close(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn rope_grad_matches_composite() {
        let dev = Device::Cpu;
        let (cos, sin) = tables(5, 8, &dev);
        let x = Var::randn(0f32, 1., (2, 3, 5, 8), &dev).unwrap();
        let w = Tensor::randn(0f32, 1., (2, 3, 5, 8), &dev).unwrap();
        let fused = rope(&x, &cos, &sin).unwrap();
        let slow = candle_nn::rotary_emb::rope_slow(&x, &cos, &sin).unwrap();
        assert!(close(&fused, &slow) < 1e-5);
        let g1 = (fused * &w).unwrap().sum_all().unwrap().backward().unwrap();
        let g2 = (slow * &w).unwrap().sum_all().unwrap().backward().unwrap();
        let (g1, g2) = (g1.get(&x).unwrap(), g2.get(&x).unwrap());
        assert!(close(g1, g2) < 1e-5, "rope grad mismatch");
    }

    #[test]
    fn softmax_grad_matches_composite() {
        let dev = Device::Cpu;
        let x = Var::randn(0f32, 2., (3, 7), &dev).unwrap();
        let w = Tensor::randn(0f32, 1., (3, 7), &dev).unwrap();
        let g1 = (softmax_last_dim(&x).unwrap() * &w)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let g2 = (candle_nn::ops::softmax(&x, D::Minus1).unwrap() * &w)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert!(close(g1.get(&x).unwrap(), g2.get(&x).unwrap()) < 1e-6);
    }

    #[test]
    fn inference_skips_the_extra_op() {
        let x = Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap();
        assert!(!softmax_last_dim(&x).unwrap().track_op());
    }
}
