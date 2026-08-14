use g_core::{Dtype, Error, Result, Tensor};

fn map_float(
    op: &'static str,
    a: &Tensor,
    f32e: impl Fn(f32) -> f32,
    f64e: impl Fn(f64) -> f64,
) -> Result<Tensor> {
    match a.dtype() {
        Dtype::F32 => {
            let v: Vec<f32> = a.to_vec_f32()?.into_iter().map(f32e).collect();
            Tensor::from_slice_f32(&v, a.shape())
        }
        Dtype::F64 => {
            let v: Vec<f64> = a.to_vec_f64()?.into_iter().map(f64e).collect();
            Tensor::from_slice_f64(&v, a.shape())
        }
        Dtype::I64 => Err(Error::dtype(op, "expected float")),
    }
}

pub fn abs(a: &Tensor) -> Result<Tensor> {
    map_float("abs", a, |x| x.abs(), |x| x.abs())
}

pub fn sign(a: &Tensor) -> Result<Tensor> {
    map_float(
        "sign",
        a,
        |x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
        |x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
    )
}

pub fn exp(a: &Tensor) -> Result<Tensor> {
    map_float("exp", a, |x| x.exp(), |x| x.exp())
}

pub fn log(a: &Tensor) -> Result<Tensor> {
    map_float("log", a, |x| x.ln(), |x| x.ln())
}

pub fn sqrt(a: &Tensor) -> Result<Tensor> {
    map_float("sqrt", a, |x| x.sqrt(), |x| x.sqrt())
}

pub fn sigmoid(a: &Tensor) -> Result<Tensor> {
    map_float(
        "sigmoid",
        a,
        |x| 1.0 / (1.0 + (-x).exp()),
        |x| 1.0 / (1.0 + (-x).exp()),
    )
}

pub fn silu(a: &Tensor) -> Result<Tensor> {
    map_float(
        "silu",
        a,
        |x| x / (1.0 + (-x).exp()),
        |x| x / (1.0 + (-x).exp()),
    )
}

pub fn gelu(a: &Tensor) -> Result<Tensor> {
    // tanh approximation
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

pub fn softplus(a: &Tensor) -> Result<Tensor> {
    map_float(
        "softplus",
        a,
        |x| {
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        },
        |x| {
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        },
    )
}

pub fn leaky_relu(a: &Tensor, slope: f64) -> Result<Tensor> {
    map_float(
        "leaky_relu",
        a,
        move |x| if x >= 0.0 { x } else { x * slope as f32 },
        move |x| if x >= 0.0 { x } else { x * slope },
    )
}

pub fn clamp(a: &Tensor, min: f64, max: f64) -> Result<Tensor> {
    map_float(
        "clamp",
        a,
        move |x| x.clamp(min as f32, max as f32),
        move |x| x.clamp(min, max),
    )
}

pub fn pow_scalar(a: &Tensor, p: f64) -> Result<Tensor> {
    map_float("pow", a, move |x| x.powf(p as f32), move |x| x.powf(p))
}

pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    super::binary_float("div", a, b, |x, y| x / y, |x, y| x / y)
}
