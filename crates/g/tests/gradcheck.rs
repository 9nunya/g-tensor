#![allow(clippy::redundant_closure)]
//! Finite-difference checks against analytic VJPs.
use g::prelude::*;
use g::{
    abs, amax, cat, div, embedding, exp, from_slice_i64, gather, leaky_relu, log, sigmoid, silu,
    softplus, sqrt, stack, take, tanh, transpose,
};

fn fd_grad(f: impl Fn(&Tensor) -> Result<Tensor>, x: &Tensor, eps: f32) -> Result<Vec<f32>> {
    let base = x.to_vec_f32()?;
    let mut out = vec![0.0f32; base.len()];
    for i in 0..base.len() {
        let mut hi = base.clone();
        let mut lo = base.clone();
        hi[i] += eps;
        lo[i] -= eps;
        let yh = f(&from_slice_f32(&hi, x.shape())?)?
            .sum(None, false)?
            .item_f32()?;
        let yl = f(&from_slice_f32(&lo, x.shape())?)?
            .sum(None, false)?
            .item_f32()?;
        out[i] = (yh - yl) / (2.0 * eps);
    }
    Ok(out)
}

fn check(name: &str, f: impl Fn(&Tensor) -> Result<Tensor>, x: &Tensor, tol: f32) -> Result<()> {
    let x = x.copy()?.with_requires_grad();
    g::zero_grad(&[&x]);
    let y = f(&x)?;
    let loss = y.sum(None, false)?;
    let g = grad(&loss, &[&x])?;
    let analytic = g[0].to_vec_f32()?;
    let numeric = fd_grad(f, &x.detach(), 1e-3)?;
    assert_eq!(analytic.len(), numeric.len(), "{name} len");
    for (i, (a, n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let ok = (a - n).abs() <= tol + 0.05 * n.abs();
        assert!(ok, "{name}[{i}] analytic={a} fd={n}");
    }
    Ok(())
}

#[test]
fn unary_grads() -> Result<()> {
    let x = from_slice_f32(&[0.3, -0.2, 0.8, 1.1], &[4])?;
    check(
        "exp",
        |t| exp(t),
        &from_slice_f32(&[0.3, -0.2, 0.8, 1.1], &[4])?,
        2e-2,
    )?;
    check("tanh", |t| tanh(t), &x, 2e-2)?;
    check("sigmoid", |t| sigmoid(t), &x, 2e-2)?;
    check("silu", |t| silu(t), &x, 3e-2)?;
    check("softplus", |t| softplus(t), &x, 2e-2)?;
    check(
        "sqrt",
        |t| sqrt(t),
        &from_slice_f32(&[0.4, 1.0, 2.2, 3.0], &[4])?,
        2e-2,
    )?;
    check(
        "log",
        |t| log(t),
        &from_slice_f32(&[0.4, 1.0, 2.2, 3.0], &[4])?,
        2e-2,
    )?;
    check(
        "abs",
        |t| abs(t),
        &from_slice_f32(&[0.4, -1.0, 2.2, -0.3], &[4])?,
        2e-2,
    )?;
    check(
        "leaky",
        |t| leaky_relu(t, 0.1),
        &from_slice_f32(&[0.4, -1.0, 2.2, -0.3], &[4])?,
        2e-2,
    )?;
    Ok(())
}

#[test]
fn binary_and_reduce_grads() -> Result<()> {
    let a = from_slice_f32(&[0.5, -0.25, 0.75, 1.0], &[2, 2])?;
    check("relu", |t| t.relu(), &a, 2e-2)?;
    check("sum", |t| t.sum(Some(&[-1]), false), &a, 1e-3)?;
    check("mean", |t| t.mean(Some(&[0]), false), &a, 1e-3)?;
    check("neg", |t| g::neg(t), &a, 1e-3)?;
    Ok(())
}

#[test]
fn matmul_linear_grad() -> Result<()> {
    let x = from_slice_f32(&[0.2, 0.4, -0.1, 0.3], &[2, 2])?;
    check(
        "matmul_r",
        |t| {
            let w = from_slice_f32(&[1.0, 0.2, 0.0, 0.5], &[2, 2])?;
            t.matmul(&w)
        },
        &x,
        3e-2,
    )?;
    check(
        "linear",
        |t| {
            let w = from_slice_f32(&[0.5, -0.2, 0.1, 0.3], &[2, 2])?;
            let b = from_slice_f32(&[0.1, -0.1], &[2])?;
            t.linear(&w, Some(&b))
        },
        &x,
        3e-2,
    )?;
    Ok(())
}

#[test]
fn div_mul_grad() -> Result<()> {
    let x = from_slice_f32(&[0.5, 1.5, 2.0, 0.8], &[4])?;
    check(
        "div",
        |t| div(t, &from_slice_f32(&[1.0, 2.0, 0.5, 1.5], &[4])?),
        &x,
        3e-2,
    )?;
    check(
        "mul",
        |t| t.mul(&from_slice_f32(&[1.0, -0.5, 0.3, 2.0], &[4])?),
        &x,
        2e-2,
    )?;
    Ok(())
}

#[test]
fn transpose_cat_stack_grad() -> Result<()> {
    let x = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    check("transpose", |t| transpose(t), &x, 1e-3)?;
    check(
        "cat",
        |t| {
            let b = from_slice_f32(&[5.0, 6.0], &[1, 2])?;
            cat(&[&g::reshape(t, &[1, 2])?, &b], 0)
        },
        &from_slice_f32(&[1.0, 2.0], &[2])?,
        1e-3,
    )?;
    check(
        "stack",
        |t| {
            let b = from_slice_f32(&[3.0, 4.0], &[2])?;
            stack(&[t, &b], 0)
        },
        &from_slice_f32(&[1.0, 2.0], &[2])?,
        1e-3,
    )?;
    Ok(())
}

#[test]
fn take_gather_embed_grad() -> Result<()> {
    let x = from_slice_f32(&[0.2, 0.4, 0.6, 0.8], &[2, 2])?;
    check(
        "take",
        |t| take(t, 1, &from_slice_i64(&[1, 0], &[2])?),
        &x,
        1e-3,
    )?;
    let w = from_slice_f32(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2])?;
    check(
        "embed",
        |t| embedding(t, &from_slice_i64(&[0, 2, 1], &[3])?),
        &w,
        1e-3,
    )?;
    check(
        "gather",
        |t| gather(t, 1, &from_slice_i64(&[1, 0, 1, 1], &[2, 2])?),
        &x,
        1e-3,
    )?;
    Ok(())
}

#[test]
fn amax_first_index_grad() -> Result<()> {
    let x = from_slice_f32(&[1.0, 3.0, 2.0, 0.5], &[4])?;
    check("amax", |t| amax(t, 0, false), &x, 1e-3)?;
    Ok(())
}

#[test]
fn stop_gradient_zero() -> Result<()> {
    let x = from_slice_f32(&[1.5, -0.5], &[2])?.with_requires_grad();
    let y = x.stop_gradient()?.mul(&x.stop_gradient()?)?;
    let g = grad(&y.sum(None, false)?, &[&x])?;
    assert!(g[0].to_vec_f32()?.iter().all(|v| *v == 0.0));
    Ok(())
}
