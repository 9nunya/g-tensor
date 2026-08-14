use crate::error::{Error, Result};

pub fn numel(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| Error::shape("numel", "shape volume overflow"))
}

pub fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc = 1isize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i] as isize);
    }
    strides
}

pub fn is_contiguous(shape: &[usize], strides: &[isize], offset: usize) -> bool {
    if offset != 0 {
        return false;
    }
    strides == row_major_strides(shape)
}

/// Right-aligned broadcast. Size 0 with 1 → 0.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        if da == db {
            out[i] = da;
        } else if da == 1 {
            out[i] = db;
        } else if db == 1 {
            out[i] = da;
        } else {
            return Err(Error::shape(
                "broadcast",
                format!("incompatible {a:?} vs {b:?}"),
            ));
        }
    }
    Ok(out)
}

pub fn broadcast_strides(
    from_shape: &[usize],
    from_strides: &[isize],
    to_shape: &[usize],
) -> Result<Vec<isize>> {
    let _ = broadcast_shapes(from_shape, to_shape)?;
    let rank = to_shape.len();
    let mut strides = vec![0isize; rank];
    let pad = rank - from_shape.len();
    for i in 0..rank {
        if i < pad {
            strides[i] = 0;
        } else {
            let d = from_shape[i - pad];
            strides[i] = if d == 1 { 0 } else { from_strides[i - pad] };
        }
    }
    Ok(strides)
}

pub fn normalize_axis(axis: isize, rank: usize, op: &'static str) -> Result<usize> {
    let r = rank as isize;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        Err(Error::shape(op, format!("axis {axis} out of rank {rank}")))
    } else {
        Ok(a as usize)
    }
}

pub fn resolve_reshape(old_shape: &[usize], new: &[isize], op: &'static str) -> Result<Vec<usize>> {
    let old_numel = numel(old_shape)?;
    let mut infer = None;
    let mut known = 1usize;
    let mut out = Vec::with_capacity(new.len());
    for (i, &d) in new.iter().enumerate() {
        if d == -1 {
            if infer.is_some() {
                return Err(Error::shape(op, "only one -1 is allowed"));
            }
            infer = Some(i);
            out.push(0);
        } else if d < 0 {
            return Err(Error::shape(op, "negative dimension other than -1"));
        } else {
            let u = d as usize;
            known = known
                .checked_mul(u)
                .ok_or_else(|| Error::shape(op, "reshape overflow"))?;
            out.push(u);
        }
    }
    if let Some(i) = infer {
        if known == 0 {
            if old_numel != 0 {
                return Err(Error::shape(op, "cannot infer -1 with a zero dimension"));
            }
            let mut specified_pos = 1usize;
            for (j, &d) in new.iter().enumerate() {
                if j != i && d > 0 {
                    specified_pos = specified_pos.saturating_mul(d as usize);
                }
            }
            let old_pos = old_shape
                .iter()
                .copied()
                .filter(|&d| d > 0)
                .fold(1usize, |a, d| a.saturating_mul(d));
            out[i] = old_pos.checked_div(specified_pos).unwrap_or(0);
        } else if old_numel % known != 0 {
            return Err(Error::shape(
                op,
                format!("cannot infer -1: numel {old_numel} not divisible by {known}"),
            ));
        } else {
            out[i] = old_numel / known;
        }
    } else if known != old_numel {
        return Err(Error::shape(
            op,
            format!("reshape numel mismatch: {old_numel} vs {known}"),
        ));
    }
    Ok(out)
}

pub fn offset_of(index: &[usize], offset: usize, strides: &[isize]) -> usize {
    let mut o = offset as isize;
    for (&i, &s) in index.iter().zip(strides) {
        o += i as isize * s;
    }
    o as usize
}

pub fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    if shape.is_empty() {
        f(&[]);
        return;
    }
    let mut idx = vec![0usize; shape.len()];
    loop {
        f(&idx);
        let mut k = shape.len() - 1;
        loop {
            idx[k] += 1;
            if idx[k] < shape[k] {
                break;
            }
            idx[k] = 0;
            if k == 0 {
                return;
            }
            k -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_zero_with_one() {
        assert_eq!(broadcast_shapes(&[0, 3], &[1, 3]).unwrap(), vec![0, 3]);
    }

    #[test]
    fn broadcast_incompatible() {
        assert!(broadcast_shapes(&[2, 3], &[3, 2]).is_err());
    }

    #[test]
    fn reshape_infer_empty() {
        assert_eq!(
            resolve_reshape(&[0, 3], &[0, -1], "reshape").unwrap(),
            vec![0, 3]
        );
    }
}


/// Iterate row-major over `shape`, yielding the *storage offset* of each element.
///
/// Unlike [`for_each_index`] this keeps a running offset (odometer) instead of
/// recomputing `offset_of` per element, and never allocates inside the loop.
pub fn for_each_offset(base: usize, shape: &[usize], strides: &[isize], mut f: impl FnMut(usize)) {
    if shape.contains(&0) {
        return;
    }
    if shape.is_empty() {
        f(base);
        return;
    }
    let rank = shape.len();
    let mut idx = vec![0usize; rank];
    let mut off = base as isize;
    loop {
        f(off as usize);
        let mut k = rank - 1;
        loop {
            idx[k] += 1;
            off += strides[k];
            if idx[k] < shape[k] {
                break;
            }
            off -= strides[k] * shape[k] as isize;
            idx[k] = 0;
            if k == 0 {
                return;
            }
            k -= 1;
        }
    }
}

/// Split a logical shape/stride pair into contiguous inner "runs".
///
/// Calls `f(start_offset, run_len)` for each maximal run of elements that are
/// unit-stride in storage, so callers can use slice-level (auto-vectorized)
/// inner loops instead of per-element stride math.
pub fn for_each_run(
    base: usize,
    shape: &[usize],
    strides: &[isize],
    mut f: impl FnMut(usize, usize),
) {
    if shape.contains(&0) {
        return;
    }
    if shape.is_empty() {
        f(base, 1);
        return;
    }
    let rank = shape.len();
    // Trailing dims with unit stride collapse into one contiguous run.
    let mut run = 1usize;
    let mut split = rank;
    let mut expect: isize = 1;
    for k in (0..rank).rev() {
        if strides[k] == expect && shape[k] > 0 {
            run *= shape[k];
            expect *= shape[k] as isize;
            split = k;
        } else {
            break;
        }
    }
    if split == 0 {
        f(base, run);
        return;
    }
    let outer_shape = &shape[..split];
    let outer_strides = &strides[..split];
    for_each_offset(base, outer_shape, outer_strides, |off| f(off, run));
}
