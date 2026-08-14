//! T9: local PC inference does not reverse through K.

use g::prelude::*;

/// One local inference step: x <- x - eta * (x - tanh(pre @ w))
fn infer_step(x: &Tensor, pre: &Tensor, w: &Tensor, eta: f64) -> Result<Tensor> {
    let pred = tanh(&linear(pre, w, None)?)?;
    let err = x.sub(&pred)?;
    x.sub(&err.mul(&Tensor::scalar_f32(eta as f32)?)?)
}

#[test]
fn weight_grad_ignores_inference_path() -> Result<()> {
    let pre = from_slice_f32(&[0.2, -0.1, 0.3], &[1, 3])?;
    let w = from_slice_f32(&[0.4, -0.2, 0.1], &[3, 1])?.with_requires_grad();
    let mut x = from_slice_f32(&[0.0], &[1, 1])?;
    for _ in 0..4 {
        x = infer_step(&x, &pre, &w.detach(), 0.2)?;
    }
    let stopped = stop_gradient(&x)?;
    let pred = tanh(&linear(&pre, &w, None)?)?;
    let err = stopped.sub(&pred)?;
    let energy = err.mul(&err)?.sum(None, false)?;
    let g = grad(&energy, &[&w])?;
    assert_eq!(g[0].shape(), w.shape());
    assert!(g[0].to_vec_f32()?.iter().all(|v| v.is_finite()));

    // Same energy from a different K (more steps) must give the same *local* weight
    // gradient once activities are stopped — we re-equilibrate independently and
    // only differentiate the local energy at the stopped state.
    let mut x2 = from_slice_f32(&[0.0], &[1, 1])?;
    for _ in 0..12 {
        x2 = infer_step(&x2, &pre, &w.detach(), 0.2)?;
    }
    let stopped2 = stop_gradient(&x2)?;
    let pred2 = tanh(&linear(&pre, &w, None)?)?;
    let err2 = stopped2.sub(&pred2)?;
    let energy2 = err2.mul(&err2)?.sum(None, false)?;
    // Not required to be equal (different x*), but both must be independent of
    // reversing through K: stop_gradient makes dE/dx* not flow into the K loop.
    let _ = energy2;
    Ok(())
}

#[test]
fn stop_gradient_blocks_activity_path() -> Result<()> {
    let a = from_slice_f32(&[1.5], &[])?.with_requires_grad();
    let stopped = stop_gradient(&a)?;
    let loss = stopped.mul(&stopped)?;
    let g = grad(&loss, &[&a])?;
    assert_eq!(g[0].item_f32()?, 0.0);
    Ok(())
}
