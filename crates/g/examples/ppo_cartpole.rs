//! PPO (clipped) on CartPole with separate actor/critic.
use g::prelude::*;

struct Pole {
    x: f32,
    xd: f32,
    th: f32,
    thd: f32,
}
impl Pole {
    fn new() -> Self {
        Self {
            x: 0.0,
            xd: 0.0,
            th: 0.015,
            thd: 0.0,
        }
    }
    fn obs(&self) -> [f32; 4] {
        [self.x, self.xd, self.th, self.thd]
    }
    fn step(&mut self, left: bool) -> (f32, bool) {
        let force = if left { -10.0 } else { 10.0 };
        let (g, mc, mp, l, tau) = (9.8, 1.0, 0.1, 0.5, 0.02);
        let total = mc + mp;
        let c = self.th.cos();
        let s = self.th.sin();
        let temp = (force + mp * l * self.thd * self.thd * s) / total;
        let thacc = (g * s - c * temp) / (l * (4.0 / 3.0 - mp * c * c / total));
        let xacc = temp - mp * l * thacc * c / total;
        self.x += tau * self.xd;
        self.xd += tau * xacc;
        self.th += tau * self.thd;
        self.thd += tau * thacc;
        (1.0, self.x.abs() > 2.4 || self.th.abs() > 0.21)
    }
}

fn sample(p: &[f32], u: f32) -> usize {
    let mut c = 0.0;
    for (i, pi) in p.iter().enumerate() {
        c += *pi;
        if u < c {
            return i;
        }
    }
    p.len() - 1
}

fn main() -> Result<()> {
    let mut aw1 = g::randn_f32(&[4, 24], 1)?
        .mul(&from_slice_f32(&[0.25], &[])?)?
        .with_requires_grad();
    let mut ab1 = g::zeros(&[24], Dtype::F32)?.with_requires_grad();
    let mut aw2 = g::randn_f32(&[24, 2], 2)?
        .mul(&from_slice_f32(&[0.25], &[])?)?
        .with_requires_grad();
    let mut ab2 = g::zeros(&[2], Dtype::F32)?.with_requires_grad();
    let mut cw1 = g::randn_f32(&[4, 24], 3)?
        .mul(&from_slice_f32(&[0.25], &[])?)?
        .with_requires_grad();
    let mut cb1 = g::zeros(&[24], Dtype::F32)?.with_requires_grad();
    let mut cw2 = g::randn_f32(&[24, 1], 4)?
        .mul(&from_slice_f32(&[0.25], &[])?)?
        .with_requires_grad();
    let mut cb2 = g::zeros(&[1], Dtype::F32)?.with_requires_grad();
    let opt = g::Sgd { lr: 0.008 };
    let mut rng = 0xDEAD_BEEFu64;
    let mut nxt = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f32 / ((1u64 << 53) as f32)
    };

    for update in 0..40 {
        let mut obs = Vec::new();
        let mut acts = Vec::new();
        let mut rews = Vec::new();
        let mut vals = Vec::new();
        let mut oldlp = Vec::new();
        let mut dones = Vec::new();
        let mut env = Pole::new();
        let mut ep_len = 0;
        for _ in 0..256 {
            let o = env.obs();
            let xt = from_slice_f32(&o, &[1, 4])?;
            let logits = xt
                .linear(&aw1, Some(&ab1))?
                .tanh()?
                .linear(&aw2, Some(&ab2))?;
            let v = xt
                .linear(&cw1, Some(&cb1))?
                .tanh()?
                .linear(&cw2, Some(&cb2))?;
            let p = softmax(&logits, -1)?.to_vec_f32()?;
            let a = sample(&p, nxt());
            let lp = g::categorical_log_prob(&logits, &g::from_slice_i64(&[a as i64], &[1])?)?;
            obs.extend_from_slice(&o);
            acts.push(a as i64);
            vals.push(v.item_f32()?);
            oldlp.push(lp.item_f32()?);
            let (r, done) = env.step(a == 0);
            rews.push(r);
            dones.push(done);
            ep_len += 1;
            if done {
                env = Pole::new();
            }
        }
        // GAE
        let mut adv = vec![0.0f32; rews.len()];
        let mut last = 0.0;
        for i in (0..rews.len()).rev() {
            let nextv = if dones[i] || i + 1 == rews.len() {
                0.0
            } else {
                vals[i + 1]
            };
            let delta = rews[i] + 0.99 * nextv - vals[i];
            last = if dones[i] {
                delta
            } else {
                delta + 0.99 * 0.95 * last
            };
            adv[i] = last;
        }
        let ret: Vec<f32> = adv.iter().zip(vals.iter()).map(|(a, v)| a + v).collect();
        let xt = from_slice_f32(&obs, &[rews.len(), 4])?;
        let at = g::from_slice_i64(&acts, &[rews.len()])?;
        let old = from_slice_f32(&oldlp, &[rews.len()])?;
        let advt = g::whiten(&from_slice_f32(&adv, &[rews.len()])?)?;
        let rett = from_slice_f32(&ret, &[rews.len(), 1])?;

        for _epoch in 0..3 {
            g::zero_grad(&[&aw1, &ab1, &aw2, &ab2, &cw1, &cb1, &cw2, &cb2]);
            let logits = xt
                .linear(&aw1, Some(&ab1))?
                .tanh()?
                .linear(&aw2, Some(&ab2))?;
            let v = xt
                .linear(&cw1, Some(&cb1))?
                .tanh()?
                .linear(&cw2, Some(&cb2))?;
            let lp = g::categorical_log_prob(&logits, &at)?;
            let ratio = g::exp(&lp.sub(&old)?)?;
            let surr1 = ratio.mul(&advt)?;
            let clipped = g::clamp(&ratio, 0.8, 1.2)?.mul(&advt)?;
            let pg = g::neg(&g::minimum(&surr1, &clipped)?.mean(None, false)?)?;
            let vf = v.mse_loss(&rett, Reduce::Mean)?;
            let ent = g::neg(&g::categorical_entropy(&logits)?.mean(None, false)?)?;
            let loss = pg
                .add(&vf.mul(&from_slice_f32(&[0.5], &[])?)?)?
                .add(&ent.mul(&from_slice_f32(&[0.01], &[])?)?)?;
            let params = [&aw1, &ab1, &aw2, &ab2, &cw1, &cb1, &cw2, &cb2];
            let gs = grad(&loss, &params)?;
            let u = opt.step(&params, &gs)?;
            aw1 = u[0].clone().with_requires_grad();
            ab1 = u[1].clone().with_requires_grad();
            aw2 = u[2].clone().with_requires_grad();
            ab2 = u[3].clone().with_requires_grad();
            cw1 = u[4].clone().with_requires_grad();
            cb1 = u[5].clone().with_requires_grad();
            cw2 = u[6].clone().with_requires_grad();
            cb2 = u[7].clone().with_requires_grad();
        }
        if update % 5 == 0 {
            println!(
                "update {update} last_ep_chunk {ep_len} mean_rew {:.2}",
                rews.iter().sum::<f32>()
            );
        }
    }
    Ok(())
}
