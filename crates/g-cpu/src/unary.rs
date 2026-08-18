use crate::fast;
use g_core::{Dtype, Error, Result, Tensor};

/// Scalar fallback used for `f64` (and as the reference implementation).
fn map_float(
    op: &'static str,
    a: &Tensor,
    f32e: impl Fn(f32) -> f32,
    f64e: impl Fn(f64) -> f64,
) -> Result<Tensor> {
    match a.dtype() {
        Dtype::F32 => {
            let mut v: Vec<f32> = a.to_vec_f32()?;
            for x in v.iter_mut() {
                *x = f32e(*x);
            }
            Tensor::from_vec_f32(v, a.shape())
        }
        Dtype::F64 => {
            let mut v: Vec<f64> = a.to_vec_f64()?;
            for x in v.iter_mut() {
                *x = f64e(*x);
            }
            Tensor::from_vec_f64(v, a.shape())
        }
        Dtype::I64 => Err(Error::dtype(op, "expected float")),
    }
}

/// Elementwise absolute value.
pub fn abs(a: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        return fast::map_f32(a, |x| x.abs());
    }
    map_float("abs", a, |x| x.abs(), |x| x.abs())
}

/// Elementwise sign: `-1`, `0`, or `1`.
pub fn sign(a: &Tensor) -> Result<Tensor> {
    #[inline]
    fn s32(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
    if a.dtype() == Dtype::F32 {
        return fast::map_f32(a, s32);
    }
    map_float("sign", a, s32, |x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

/// Elementwise natural exponential.
pub fn exp(a: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        return fast::unary_f32(a, fast::k_exp);
    }
    map_float("exp", a, |x| x.exp(), |x| x.exp())
}

/// Elementwise natural logarithm.
pub fn log(a: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        return fast::unary_f32(a, fast::k_log);
    }
    map_float("log", a, |x| x.ln(), |x| x.ln())
}

/// Elementwise square root.
pub fn sqrt(a: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        return fast::unary_f32(a, fast::k_sqrt);
    }
    map_float("sqrt", a, |x| x.sqrt(), |x| x.sqrt())
}

/// Elementwise logistic sigmoid `1 / (1 + e^-x)`.
pub fn sigmoid(a: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        return fast::unary_f32(a, fast::k_sigmoid);
    }
    map_float(
        "sigmoid",
        a,
        |x| 1.0 / (1.0 + (-x).exp()),
        |x| 1.0 / (1.0 + (-x).exp()),
    )
}

/// Elementwise SiLU `x * sigmoid(x)`.
pub fn silu(a: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        return fast::unary_f32(a, fast::k_silu);
    }
    map_float(
        "silu",
        a,
        |x| x / (1.0 + (-x).exp()),
        |x| x / (1.0 + (-x).exp()),
    )
}

/// Elementwise GELU (tanh approximation).
pub fn gelu(a: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        return fast::unary_f32(a, fast::k_gelu);
    }
    map_float(
        "gelu",
        a,
        |x| {
            let z = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + z.tanh())
        },
        |x| {
            let z = (2.0f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + z.tanh())
        },
    )
}

/// Elementwise softplus `ln(1 + e^x)`, numerically stabilized.
pub fn softplus(a: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        return fast::map_f32(a, |x| if x > 20.0 { x } else { (1.0 + x.exp()).ln() });
    }
    map_float(
        "softplus",
        a,
        |x| if x > 20.0 { x } else { (1.0 + x.exp()).ln() },
        |x| if x > 20.0 { x } else { (1.0 + x.exp()).ln() },
    )
}

/// Elementwise leaky ReLU with the given negative slope.
pub fn leaky_relu(a: &Tensor, slope: f64) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        let s = slope as f32;
        return fast::map_f32(a, move |x| if x >= 0.0 { x } else { x * s });
    }
    map_float(
        "leaky_relu",
        a,
        move |x| if x >= 0.0 { x } else { x * slope as f32 },
        move |x| if x >= 0.0 { x } else { x * slope },
    )
}

/// Elementwise clamp to `[min, max]`.
pub fn clamp(a: &Tensor, min: f64, max: f64) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        let (lo, hi) = (min as f32, max as f32);
        return fast::map_f32(a, move |x| x.clamp(lo, hi));
    }
    map_float(
        "clamp",
        a,
        move |x| x.clamp(min as f32, max as f32),
        move |x| x.clamp(min, max),
    )
}

/// Elementwise `a^p` for scalar `p`.
pub fn pow_scalar(a: &Tensor, p: f64) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 {
        let pf = p as f32;
        // Integer powers are far cheaper than a general powf.
        if pf == 2.0 {
            return fast::map_f32(a, |x| x * x);
        }
        if pf == 0.5 {
            return fast::unary_f32(a, fast::k_sqrt);
        }
        return fast::map_f32(a, move |x| x.powf(pf));
    }
    map_float("pow", a, move |x| x.powf(p as f32), move |x| x.powf(p))
}

/// Elementwise `a / b` with broadcasting.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() == Dtype::F32 && b.dtype() == Dtype::F32 {
        return fast::binary_f32("div", a, b, |x, y| x / y);
    }
    super::binary_float("div", a, b, |x, y| x / y, |x, y| x / y)
}
