#![cfg(feature = "gpu")]

use g::{from_slice_f32, matmul, Result};

#[test]
fn large_gemm_matches_cpu_small_patch() -> Result<()> {
    // 1024 meets offload rule (max dim >= 1024 and volume).
    let n = 1024usize;
    let mut a = vec![0.0f32; n * n];
    let mut b = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
        b[i * n + (i % n)] = (i % 7) as f32 * 0.1;
    }
    let at = from_slice_f32(&a, &[n, n])?;
    let bt = from_slice_f32(&b, &[n, n])?;
    let y = matmul(&at, &bt)?;
    // I @ B = B
    let got = y.to_vec_f32()?;
    for i in 0..64 {
        assert!((got[i] - b[i]).abs() < 1e-4, "{i} {} {}", got[i], b[i]);
    }
    Ok(())
}
