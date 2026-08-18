#![cfg(feature = "gpu")]

use g::{from_slice_f32, gpu_device_count, gpu_device_name, matmul, Result};

fn close(a: f32, b: f32) -> bool {
    // CPU and GPU accumulate the same dot products in different orders, so a
    // tiny relative error is expected at f32 magnitudes around 1e8.
    (a - b).abs() <= 1e-3 + 1e-5 * a.abs().max(b.abs())
}

#[test]
fn gpu_reports_metal_device() {
    assert!(gpu_device_count() >= 1);
    if let Some(name) = gpu_device_name() {
        assert!(!name.is_empty());
        println!("gpu device: {name}");
    }
}

#[test]
fn batched_rank3_times_rank2_matches_cpu_reference() -> Result<()> {
    // [2, 512, 512] @ [512, 512]. Sits exactly on the offload crossover so the
    // GPU path is exercised, and the identity LHS makes the expected output
    // trivial to verify without materializing a second matmul.
    let m = 512usize;
    let n = 512usize;
    let k = 512usize;
    let b = 2usize;

    let mut a = vec![0.0f32; b * m * k];
    for bi in 0..b {
        for i in 0..m {
            a[(bi * m + i) * k + i] = 1.0;
        }
    }
    let mut rhs = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            rhs[i * n + j] = ((i * 7 + j * 3) % 997) as f32;
        }
    }

    let at = from_slice_f32(&a, &[b, m, k])?;
    let bt = from_slice_f32(&rhs, &[k, n])?;
    let y = matmul(&at, &bt)?;
    assert_eq!(y.shape(), &[b, m, n]);

    let got = y.to_vec_f32()?;
    for bi in 0..b {
        for (i, j) in [(0usize, 0usize), (1, 511), (511, 1), (511, 511)] {
            let expected = rhs[i * n + j];
            let actual = got[(bi * m + i) * n + j];
            assert!(
                close(actual, expected),
                "{bi} {i} {j}: {actual} != {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn multi_device_rank3_times_rank2_matches_single() -> Result<()> {
    let m = 512usize;
    let n = 512usize;
    let k = 512usize;
    let b = 2usize;

    let mut a = vec![0.0f32; b * m * k];
    for bi in 0..b {
        for i in 0..m {
            a[(bi * m + i) * k + i] = 1.0;
        }
    }
    let mut rhs = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            rhs[i * n + j] = ((i * 7 + j * 3) % 997) as f32;
        }
    }

    let at = from_slice_f32(&a, &[b, m, k])?;
    let bt = from_slice_f32(&rhs, &[k, n])?;
    let single = matmul(&at, &bt)?;
    let multi = g::g_apple::matmul_multi_device(&at, &bt)?;

    let sv = single.to_vec_f32()?;
    let mv = multi.to_vec_f32()?;
    assert_eq!(sv.len(), mv.len());
    for (i, (s, m)) in sv.iter().zip(&mv).enumerate() {
        assert!(close(*s, *m), "{i}: {s} != {m}");
    }
    Ok(())
}

#[test]
fn cpu_gpu_batched_matches_single() -> Result<()> {
    let m = 512usize;
    let n = 512usize;
    let k = 512usize;
    let b = 6usize;

    let mut a = vec![0.0f32; b * m * k];
    for bi in 0..b {
        for i in 0..m {
            a[(bi * m + i) * k + i] = 1.0;
        }
    }
    let mut rhs = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            rhs[i * n + j] = ((i * 7 + j * 3) % 997) as f32;
        }
    }

    let at = from_slice_f32(&a, &[b, m, k])?;
    let bt = from_slice_f32(&rhs, &[k, n])?;
    let cpu_gpu = matmul(&at, &bt)?;
    let single = g::g_apple::matmul(&at, &bt)?;

    let a = cpu_gpu.to_vec_f32()?;
    let c = single.to_vec_f32()?;
    assert_eq!(a.len(), c.len());
    for (i, (x, y)) in a.iter().zip(&c).enumerate() {
        assert!(close(*x, *y), "{i}: {x} != {y}");
    }
    Ok(())
}

#[test]
fn cpu_gpu_rank3_times_rank3_matches_single() -> Result<()> {
    let m = 512usize;
    let n = 512usize;
    let k = 512usize;
    let b = 6usize;

    let mut a = vec![0.0f32; b * m * k];
    let mut bmat = vec![0.0f32; b * k * n];
    for bi in 0..b {
        for i in 0..m {
            for j in 0..k {
                a[(bi * m + i) * k + j] = ((i * 3 + j * 5 + bi) % 991) as f32;
            }
        }
        for i in 0..k {
            for j in 0..n {
                bmat[(bi * k + i) * n + j] = ((i * 7 + j * 11 + bi) % 983) as f32;
            }
        }
    }

    let at = from_slice_f32(&a, &[b, m, k])?;
    let bt = from_slice_f32(&bmat, &[b, k, n])?;
    let cpu_gpu = matmul(&at, &bt)?;
    let single = g::g_apple::matmul(&at, &bt)?;

    let a = cpu_gpu.to_vec_f32()?;
    let c = single.to_vec_f32()?;
    assert_eq!(a.len(), c.len());
    for (i, (x, y)) in a.iter().zip(&c).enumerate() {
        assert!(close(*x, *y), "{i}: {x} != {y}");
    }
    Ok(())
}

#[test]
fn cpu_gpu_rank2_matches_single() -> Result<()> {
    let m = 1536usize;
    let n = 512usize;
    let k = 512usize;

    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for i in 0..m {
        for j in 0..k {
            a[i * k + j] = ((i * 3 + j * 5) % 991) as f32;
        }
    }
    for i in 0..k {
        for j in 0..n {
            b[i * n + j] = ((i * 7 + j * 11) % 983) as f32;
        }
    }

    let at = from_slice_f32(&a, &[m, k])?;
    let bt = from_slice_f32(&b, &[k, n])?;
    let cpu_gpu = matmul(&at, &bt)?;
    let single = g::g_apple::matmul(&at, &bt)?;

    let a = cpu_gpu.to_vec_f32()?;
    let c = single.to_vec_f32()?;
    assert_eq!(a.len(), c.len());
    for (i, (x, y)) in a.iter().zip(&c).enumerate() {
        assert!(close(*x, *y), "{i}: {x} != {y}");
    }
    Ok(())
}

#[test]
fn multi_device_single_batch_rank3_matches_single() -> Result<()> {
    let m = 768usize;
    let n = 768usize;
    let k = 768usize;

    let mut a = vec![0.0f32; m * k];
    for i in 0..m {
        a[i * k + i] = 1.0;
    }
    let mut rhs = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            rhs[i * n + j] = ((i * 7 + j * 3) % 997) as f32;
        }
    }
    let at = from_slice_f32(&a, &[1, m, k])?;
    let bt = from_slice_f32(&rhs, &[k, n])?;

    let single = g::g_apple::matmul(&at, &bt)?;
    let multi = g::g_apple::matmul_multi_device(&at, &bt)?;
    let sv = single.to_vec_f32()?;
    let mv = multi.to_vec_f32()?;
    assert_eq!(sv.len(), mv.len());
    for (i, (s, m)) in sv.iter().zip(&mv).enumerate() {
        assert!(close(*s, *m), "{i}: {s} != {m}");
    }
    Ok(())
}

#[test]
fn multi_device_rank2_matches_single() -> Result<()> {
    let m = 512usize;
    let n = 512usize;
    let k = 1024usize;

    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for i in 0..m {
        for j in 0..k {
            a[i * k + j] = ((i * 3 + j * 5) % 991) as f32;
        }
    }
    for i in 0..k {
        for j in 0..n {
            b[i * n + j] = ((i * 7 + j * 11) % 983) as f32;
        }
    }

    let at = from_slice_f32(&a, &[m, k])?;
    let bt = from_slice_f32(&b, &[k, n])?;
    let single = matmul(&at, &bt)?;
    let multi = g::g_apple::matmul_multi_device(&at, &bt)?;

    let sv = single.to_vec_f32()?;
    let mv = multi.to_vec_f32()?;
    assert_eq!(sv.len(), mv.len());
    for (i, (s, m)) in sv.iter().zip(&mv).enumerate() {
        assert!(close(*s, *m), "{i}: {s} != {m}");
    }
    Ok(())
}

#[test]
fn batched_rank3_times_rank3_keeps_batch_pairing() -> Result<()> {
    // [2, 512, 512] @ [2, 512, 512]. A diagonal LHS scaled per batch means
    // batch bi of the output must only depend on batch bi of the RHS.
    let m = 512usize;
    let n = 512usize;
    let k = 512usize;
    let b = 2usize;

    let mut a = vec![0.0f32; b * m * k];
    let mut rhs = vec![0.0f32; b * k * n];
    for bi in 0..b {
        let scale = (bi + 1) as f32;
        for i in 0..m {
            a[(bi * m + i) * k + i] = scale;
        }
        for i in 0..k {
            for j in 0..n {
                rhs[(bi * k + i) * n + j] = ((i * 5 + j * 11 + bi * 3) % 991) as f32;
            }
        }
    }

    let at = from_slice_f32(&a, &[b, m, k])?;
    let bt = from_slice_f32(&rhs, &[b, k, n])?;
    let y = matmul(&at, &bt)?;
    assert_eq!(y.shape(), &[b, m, n]);

    let got = y.to_vec_f32()?;
    for bi in 0..b {
        let scale = (bi + 1) as f32;
        for (i, j) in [(0usize, 0usize), (7, 500), (500, 7), (511, 511)] {
            let expected = scale * rhs[(bi * k + i) * n + j];
            let actual = got[(bi * m + i) * n + j];
            assert!(
                close(actual, expected),
                "{bi} {i} {j}: {actual} != {expected}"
            );
        }
    }
    Ok(())
}
