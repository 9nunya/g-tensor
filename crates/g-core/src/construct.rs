use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Half-open range `[start, end)` with step `step`.
pub fn arange_f32(start: f32, end: f32, step: f32) -> Result<Tensor> {
    if step == 0.0 {
        return Err(Error::shape("arange", "step must not be 0"));
    }
    let mut v = Vec::new();
    let mut x = start;
    if step > 0.0 {
        while x < end {
            v.push(x);
            x += step;
        }
    } else {
        while x > end {
            v.push(x);
            x += step;
        }
    }
    let n = v.len();
    Tensor::from_slice_f32(&v, &[n])
}

/// `num` equally spaced points from `start` to `end` inclusive.
pub fn linspace_f32(start: f32, end: f32, num: usize) -> Result<Tensor> {
    if num == 0 {
        return Tensor::from_slice_f32(&[], &[0]);
    }
    if num == 1 {
        return Tensor::from_slice_f32(&[start], &[1]);
    }
    let mut v = Vec::with_capacity(num);
    let den = (num - 1) as f32;
    for i in 0..num {
        v.push(start + (end - start) * (i as f32) / den);
    }
    Tensor::from_slice_f32(&v, &[num])
}

/// `n × n` identity.
pub fn eye(n: usize, dtype: Dtype) -> Result<Tensor> {
    match dtype {
        Dtype::F32 => {
            let mut v = vec![0.0f32; n * n];
            for i in 0..n {
                v[i * n + i] = 1.0;
            }
            Tensor::from_slice_f32(&v, &[n, n])
        }
        Dtype::F64 => {
            let mut v = vec![0.0f64; n * n];
            for i in 0..n {
                v[i * n + i] = 1.0;
            }
            Tensor::from_slice_f64(&v, &[n, n])
        }
        Dtype::I64 => {
            let mut v = vec![0i64; n * n];
            for i in 0..n {
                v[i * n + i] = 1;
            }
            Tensor::from_slice_i64(&v, &[n, n])
        }
    }
}

/// Seeded standard-normal `f32` tensor (Box–Muller, no global RNG).
pub fn randn_f32(shape: &[usize], seed: u64) -> Result<Tensor> {
    let n = crate::shape::numel(shape)?;
    let mut state = seed | 1;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (state >> 11) as f64 / ((1u64 << 53) as f64)
    };
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let u1 = next().max(1e-12);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        v.push((r * theta.cos()) as f32);
        if v.len() < n {
            v.push((r * theta.sin()) as f32);
        }
    }
    Tensor::from_slice_f32(&v, shape)
}

/// Every element equal to `value`.
pub fn full_f32(shape: &[usize], value: f32) -> Result<Tensor> {
    Tensor::full_f32(shape, value)
}
