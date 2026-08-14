//! REINFORCE on a built-in CartPole.
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
            th: 0.02,
            thd: 0.0,
        }
    }
    fn obs(&self) -> [f32; 4] {
        [self.x, self.xd, self.th, self.thd]
    }
    fn step(&mut self, left: bool) -> (f32, bool) {
        let force = if left { -10.0 } else { 10.0 };
        let g = 9.8;
        let mc = 1.0;
        let mp = 0.1;
        let l = 0.5;
        let tau = 0.02;
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
        let done = self.x.abs() > 2.4 || self.th.abs() > 0.21;
        (1.0, done)
    }
}

fn sample(probs: &[f32], u: f32) -> usize {
    let mut c = 0.0;
    for (i, p) in probs.iter().enumerate() {
        c += *p;
        if u < c {
            return i;
        }
    }
    probs.len() - 1
}

fn main() -> Result<()> {
    let mut w1 = g::randn_f32(&[4, 16], 11)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let mut b1 = g::zeros(&[16], Dtype::F32)?.with_requires_grad();
    let mut w2 = g::randn_f32(&[16, 2], 12)?
        .mul(&from_slice_f32(&[0.3], &[])?)?
        .with_requires_grad();
    let mut b2 = g::zeros(&[2], Dtype::F32)?.with_requires_grad();
    let opt = g::Sgd { lr: 0.01 };
    let mut rng = 0xC0FFEE_u64;
    let mut nxt = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f32) / (u64::MAX as f32)
    };
    for ep in 0..80 {
        let mut env = Pole::new();
        let mut obs = Vec::new();
        let mut acts = Vec::new();
        let mut rews = Vec::new();
        for _t in 0..200 {
            let o = env.obs();
            obs.extend_from_slice(&o);
            let xt = from_slice_f32(&o, &[1, 4])?;
            let logits = xt.linear(&w1, Some(&b1))?.tanh()?.linear(&w2, Some(&b2))?;
            let p = softmax(&logits, -1)?.to_vec_f32()?;
            let a = sample(&p, nxt());
            acts.push(a as i64);
            let (r, done) = env.step(a == 0);
            rews.push(r);
            if done {
                break;
            }
        }
        let tlen = rews.len();
        let mut ret = vec![0.0f32; tlen];
        let mut gae = 0.0;
        for i in (0..tlen).rev() {
            gae = rews[i] + 0.99 * gae;
            ret[i] = gae;
        }
        let mean = ret.iter().sum::<f32>() / tlen as f32;
        for r in &mut ret {
            *r -= mean;
        }
        g::zero_grad(&[&w1, &b1, &w2, &b2]);
        let xt = from_slice_f32(&obs, &[tlen, 4])?;
        let logits = xt.linear(&w1, Some(&b1))?.tanh()?.linear(&w2, Some(&b2))?;
        let lp = g::categorical_log_prob(&logits, &g::from_slice_i64(&acts, &[tlen])?)?;
        let adv = from_slice_f32(&ret, &[tlen])?;
        let loss = g::neg(&lp.mul(&adv)?.mean(None, false)?)?;
        let gs = grad(&loss, &[&w1, &b1, &w2, &b2])?;
        let u = opt.step(&[&w1, &b1, &w2, &b2], &gs)?;
        w1 = u[0].clone().with_requires_grad();
        b1 = u[1].clone().with_requires_grad();
        w2 = u[2].clone().with_requires_grad();
        b2 = u[3].clone().with_requires_grad();
        if ep % 10 == 0 {
            println!("ep {ep} steps {tlen} loss {:.4}", loss.item_f32()?);
        }
    }
    Ok(())
}
