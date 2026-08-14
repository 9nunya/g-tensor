//! Local PC inference (no reverse through K) + Hebbian-like weight step.
use g::prelude::*;

fn main() -> Result<()> {
    let x0 = from_slice_f32(&[0.2, -0.1, 0.3], &[1, 3])?;
    let mut x1 = g::zeros(&[1, 2], Dtype::F32)?;
    let w = g::randn_f32(&[3, 2], 7)?.with_requires_grad();
    for _ in 0..16 {
        let pred = x0.linear(&w.detach(), None)?.tanh()?;
        let err = x1.sub(&pred)?;
        x1 = x1.sub(&err.mul(&from_slice_f32(&[0.2], &[])?)?)?;
    }
    let stopped = x1.stop_gradient()?;
    let pred = x0.linear(&w, None)?.tanh()?;
    let energy = stopped
        .sub(&pred)?
        .mul(&stopped.sub(&pred)?)?
        .sum(None, false)?;
    let g = grad(&energy, &[&w])?;
    println!(
        "energy {} |grad| {}",
        energy.item_f32()?,
        g[0].to_vec_f32()?.iter().map(|v| v.abs()).sum::<f32>()
    );
    Ok(())
}
