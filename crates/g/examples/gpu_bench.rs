use std::time::Instant;

use g::Result;

fn bench(name: &str, f: impl Fn() -> Result<()>, iters: usize) {
    let _ = f();
    let t = Instant::now();
    for _ in 0..iters {
        f().unwrap();
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    println!("{name:24} {:>9.3} ms", dt * 1e3);
}

#[cfg(feature = "gpu")]
fn run() -> Result<()> {
    use g::{from_slice_f32, set_include_integrated};

    let m = 1024usize;
    let n = 1024usize;
    let k = 1024usize;
    let b = 16usize;

    let mut a = vec![0.0f32; b * m * k];
    for bi in 0..b {
        for i in 0..m {
            for j in 0..k {
                a[(bi * m + i) * k + j] = ((i * 3 + j * 5 + bi) % 991) as f32;
            }
        }
    }
    let mut rhs = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            rhs[i * n + j] = ((i * 7 + j * 11) % 983) as f32;
        }
    }
    let at = from_slice_f32(&a, &[b, m, k])?;
    let bt = from_slice_f32(&rhs, &[k, n])?;

    let iters = 10usize;

    set_include_integrated(false);
    bench(
        "dGPU only",
        || {
            let _ = g::g_apple::matmul(&at, &bt)?;
            Ok(())
        },
        iters,
    );

    bench(
        "dGPU + iGPU",
        || {
            set_include_integrated(true);
            let r = g::g_apple::matmul_multi_device(&at, &bt).map(|_| ());
            set_include_integrated(false);
            r
        },
        iters,
    );

    bench(
        "CPU + dGPU",
        || {
            set_include_integrated(false);
            g::matmul(&at, &bt).map(|_| ())
        },
        iters,
    );

    bench(
        "CPU + dGPU + iGPU",
        || {
            set_include_integrated(true);
            let r = g::matmul(&at, &bt).map(|_| ());
            set_include_integrated(false);
            r
        },
        iters,
    );
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run() -> Result<()> {
    println!("build with `--features gpu` to run this benchmark");
    Ok(())
}

fn main() -> Result<()> {
    run()
}
