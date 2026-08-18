#![cfg(feature = "gpu")]

//! Probe: does MPSGraph on the Intel iGPU stay stable when the backend runs
//! both the AMD dGPU and Intel iGPU in parallel? Run this single test alone.

use g::{
    from_slice_f32, gpu_all_device_names, gpu_device_names, matmul, set_include_integrated, Result,
};

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3 + 1e-5 * a.abs().max(b.abs())
}

#[test]
fn igpu_include_loop_matches_single() -> Result<()> {
    println!("all: {:?}", gpu_all_device_names());
    println!("eligible default: {:?}", gpu_device_names());
    set_include_integrated(true);
    println!("eligible with igpu: {:?}", gpu_device_names());

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

    set_include_integrated(false);
    let single = matmul(&at, &bt)?;
    let ref_v = single.to_vec_f32()?;
    set_include_integrated(true);

    for iter in 0..6 {
        let multi = matmul(&at, &bt)?;
        let mv = multi.to_vec_f32()?;
        for (i, (x, y)) in ref_v.iter().zip(&mv).enumerate() {
            assert!(close(*x, *y), "iter {iter} idx {i}: {x} != {y}");
        }
        println!("iter {iter} ok");
    }

    // A single-batch rank-3 GEMM must still route to the dGPU even when the
    // iGPU is eligible (there is no batch axis worth fanning out).
    let m1 = 768usize;
    let mut a1 = vec![0.0f32; m1 * m1];
    for i in 0..m1 {
        a1[i * m1 + i] = 1.0;
    }
    let mut rhs1 = vec![0.0f32; m1 * m1];
    for i in 0..m1 {
        for j in 0..m1 {
            rhs1[i * m1 + j] = ((i * 5 + j * 7) % 997) as f32;
        }
    }
    let a1t = from_slice_f32(&a1, &[1, m1, m1])?;
    let b1t = from_slice_f32(&rhs1, &[m1, m1])?;
    let got = matmul(&a1t, &b1t)?.to_vec_f32()?;
    set_include_integrated(false);
    let want = matmul(&a1t, &b1t)?.to_vec_f32()?;
    set_include_integrated(true);
    for (i, (x, y)) in got.iter().zip(&want).enumerate() {
        assert!(close(*x, *y), "single-batch idx {i}: {x} != {y}");
    }

    set_include_integrated(false);
    Ok(())
}
