//! Three-class blobs with cross-entropy + SGD.
use g::prelude::*;

fn main() -> Result<()> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for c in 0..3 {
        let cx = (c as f32) - 1.0;
        for i in 0..20 {
            let n = ((i * 17 + c * 9) % 11) as f32 / 11.0 - 0.5;
            xs.extend_from_slice(&[cx + n * 0.3, n]);
            ys.push(c as i64);
        }
    }
    let x = from_slice_f32(&xs, &[60, 2])?;
    let y = g::from_slice_i64(&ys, &[60])?;
    let mut w1 = g::randn_f32(&[2, 16], 1)?
        .mul(&from_slice_f32(&[0.4], &[])?)?
        .with_requires_grad();
    let mut b1 = g::zeros(&[16], Dtype::F32)?.with_requires_grad();
    let mut w2 = g::randn_f32(&[16, 3], 2)?
        .mul(&from_slice_f32(&[0.4], &[])?)?
        .with_requires_grad();
    let mut b2 = g::zeros(&[3], Dtype::F32)?.with_requires_grad();
    let opt = g::Sgd { lr: 0.15 };
    for step in 0..80 {
        g::zero_grad(&[&w1, &b1, &w2, &b2]);
        let logits = x.linear(&w1, Some(&b1))?.relu()?.linear(&w2, Some(&b2))?;
        let loss = g::cross_entropy(&logits, &y, Reduce::Mean)?;
        let gs = grad(&loss, &[&w1, &b1, &w2, &b2])?;
        let u = opt.step(&[&w1, &b1, &w2, &b2], &gs)?;
        w1 = u[0].clone().with_requires_grad();
        b1 = u[1].clone().with_requires_grad();
        w2 = u[2].clone().with_requires_grad();
        b2 = u[3].clone().with_requires_grad();
        if step % 20 == 0 {
            let p = softmax(&logits, -1)?.to_vec_f32()?;
            let mut acc = 0;
            for i in 0..60 {
                let (mut best, mut bi) = (p[i * 3], 0);
                for a in 1..3 {
                    if p[i * 3 + a] > best {
                        best = p[i * 3 + a];
                        bi = a;
                    }
                }
                if bi as i64 == ys[i] {
                    acc += 1;
                }
            }
            println!("step {step} loss {:.4} acc {}/60", loss.item_f32()?, acc);
        }
    }
    Ok(())
}
