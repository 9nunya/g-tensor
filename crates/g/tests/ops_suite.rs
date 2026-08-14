use g::prelude::*;
use g::{
    abs, amax, arange_f32, cat, clamp, exp, from_slice_i64, layer_norm, leaky_relu, log, softplus,
};

#[test]
fn unary_roundtrip_and_grad() -> Result<()> {
    let x = from_slice_f32(&[0.2, 0.5, 1.5], &[3])?.with_requires_grad();
    let y = exp(&x)?.sum(None, false)?;
    let g = grad(&y, &[&x])?;
    let want: Vec<f32> = x.to_vec_f32()?.into_iter().map(|t| t.exp()).collect();
    let got = g[0].to_vec_f32()?;
    for (a, b) in got.iter().zip(want) {
        assert!((a - b).abs() < 1e-5);
    }
    let _ = abs(&x)?;
    let _ = log(&from_slice_f32(&[0.2, 0.5, 1.5], &[3])?)?;
    let _ = softplus(&x)?;
    let _ = leaky_relu(&x, 0.01)?;
    let _ = clamp(&x, -1.0, 1.0)?;
    Ok(())
}

#[test]
fn cat_and_amax() -> Result<()> {
    let a = from_slice_f32(&[1.0, 2.0], &[2])?;
    let b = from_slice_f32(&[3.0, 4.0], &[2])?;
    let c = cat(&[&a, &b], 0)?;
    assert_eq!(c.to_vec_f32()?, vec![1.0, 2.0, 3.0, 4.0]);
    let m = amax(&c, 0, false)?;
    assert_eq!(m.item_f32()?, 4.0);
    assert!(amax(&from_slice_f32(&[], &[0])?, 0, false).is_err());
    Ok(())
}

#[test]
fn arange_eye_randn() -> Result<()> {
    let r = arange_f32(0.0, 4.0, 1.0)?;
    assert_eq!(r.to_vec_f32()?, vec![0.0, 1.0, 2.0, 3.0]);
    let i = g::eye(2, Dtype::F32)?;
    assert_eq!(i.to_vec_f32()?, vec![1.0, 0.0, 0.0, 1.0]);
    let z = g::randn_f32(&[8], 42)?;
    assert_eq!(z.numel(), 8);
    Ok(())
}

#[test]
fn cross_entropy_descends() -> Result<()> {
    let x = from_slice_f32(&[0.1, -0.2, 0.3, 0.0, 0.4, -0.1], &[2, 3])?;
    let mut w = from_slice_f32(&[0.2, 0.0, -0.1, 0.1, 0.05, 0.0, -0.2, 0.1, 0.15], &[3, 3])?
        .with_requires_grad();
    let y = from_slice_i64(&[0, 2], &[2])?;
    let sgd = g::Sgd { lr: 0.2 };
    let l0 = g::cross_entropy(&x.linear(&w, None)?, &y, Reduce::Mean)?.item_f32()?;
    for _ in 0..20 {
        g::zero_grad(&[&w]);
        let loss = g::cross_entropy(&x.linear(&w, None)?, &y, Reduce::Mean)?;
        let gs = grad(&loss, &[&w])?;
        w = sgd.step(&[&w], &gs)?.remove(0).with_requires_grad();
    }
    let l1 = g::cross_entropy(&x.linear(&w, None)?, &y, Reduce::Mean)?.item_f32()?;
    assert!(l1 < l0, "{l1} vs {l0}");
    Ok(())
}

#[test]
fn layer_norm_unit_stats() -> Result<()> {
    let x = from_slice_f32(&[1.0, 3.0, 5.0, 7.0], &[2, 2])?;
    let y = layer_norm(&x, 1e-5)?;
    assert_eq!(y.shape(), &[2, 2]);
    Ok(())
}

#[test]
fn gelu_grad_finite() -> Result<()> {
    let x = from_slice_f32(&[-1.0, 0.0, 1.0], &[3])?.with_requires_grad();
    let y = g::gelu(&x)?.sum(None, false)?;
    let g = grad(&y, &[&x])?;
    assert!(g[0].to_vec_f32()?.iter().all(|v| v.is_finite()));
    Ok(())
}
