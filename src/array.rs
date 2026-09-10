//! The Rust-side array: an owned, C-contiguous float64 buffer plus inline
//! dims. Python never sees this type directly; `python.rs` wraps it.

use crate::dims::{Dims, DimsError};
use std::fmt;

#[derive(Clone)]
pub struct Array {
    pub(crate) data: Vec<f64>,
    pub(crate) dims: Dims,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrayError {
    ShapeMismatch(Vec<usize>, Vec<usize>),
    SizeMismatch { expected: usize, got: usize },
    Dims(DimsError),
    IndexOutOfBounds { index: isize, axis: usize, len: usize },
    TooManyIndices { given: usize, ndim: usize },
    Message(String),
}

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayError::ShapeMismatch(a, b) => {
                write!(f, "operands could not be broadcast together with shapes {} {}", shape_str(a), shape_str(b))
            }
            ArrayError::SizeMismatch { expected, got } => {
                write!(f, "cannot reshape array of size {got} into shape with {expected} elements")
            }
            ArrayError::Dims(e) => write!(f, "{e}"),
            ArrayError::IndexOutOfBounds { index, axis, len } => {
                write!(f, "index {index} is out of bounds for axis {axis} with size {len}")
            }
            ArrayError::TooManyIndices { given, ndim } => write!(
                f,
                "too many indices for array: array is {ndim}-dimensional, but {given} were indexed"
            ),
            ArrayError::Message(m) => write!(f, "{m}"),
        }
    }
}

/// NumPy-style shape text: `(2, 3)`, `(3,)`, `()`.
fn shape_str(shape: &[usize]) -> String {
    match shape {
        [] => "()".into(),
        [n] => format!("({n},)"),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

/// Shape two shapes broadcast to under NumPy's rules.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut out = vec![0usize; ndim];
    for k in 0..ndim {
        let da = if k + a.len() >= ndim { a[k + a.len() - ndim] } else { 1 };
        let db = if k + b.len() >= ndim { b[k + b.len() - ndim] } else { 1 };
        out[k] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(ArrayError::ShapeMismatch(a.to_vec(), b.to_vec())),
        };
    }
    Ok(out)
}

/// Element strides of `shape` viewed in `out` (right-aligned), zero where
/// the axis is broadcast.
fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; out.len()];
    let offset = out.len() - shape.len();
    let mut stride = 1;
    for k in (0..shape.len()).rev() {
        strides[k + offset] = if shape[k] == 1 { 0 } else { stride };
        stride *= shape[k];
    }
    strides
}

impl From<DimsError> for ArrayError {
    fn from(e: DimsError) -> Self {
        ArrayError::Dims(e)
    }
}

pub type Result<T> = std::result::Result<T, ArrayError>;

impl Array {
    pub fn new(data: Vec<f64>, shape: &[usize]) -> Result<Array> {
        let dims = Dims::from_shape(shape)?;
        if dims.size() != data.len() {
            return Err(ArrayError::SizeMismatch { expected: dims.size(), got: data.len() });
        }
        Ok(Array { data, dims })
    }

    pub fn scalar(value: f64) -> Array {
        Array { data: vec![value], dims: Dims::scalar() }
    }

    pub fn filled(shape: &[usize], value: f64) -> Result<Array> {
        let dims = Dims::from_shape(shape)?;
        Ok(Array { data: vec![value; dims.size()], dims })
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.dims.shape()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.ndim()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// `a[i, j, ...] = value` for a full integer index.
    pub fn set(&mut self, indices: &[isize], value: f64) -> Result<()> {
        let (offset, dims) = self.locate(indices)?;
        if dims.ndim() != 0 {
            return Err(ArrayError::Message(format!(
                "setting an array element with a sequence: {} indices given for a {}-dimensional array",
                indices.len(),
                self.ndim()
            )));
        }
        self.data[offset] = value;
        Ok(())
    }

    /// `self = f(self, other)` element-wise in place, with `other` broadcast
    /// to `self`'s shape (the result may not change shape, like NumPy's
    /// in-place operators).
    pub fn zip_map_inplace<F: Fn(f64, f64) -> f64>(&mut self, other: &Array, f: F) -> Result<()> {
        if self.dims == other.dims {
            for (a, &b) in self.data.iter_mut().zip(&other.data) {
                *a = f(*a, b);
            }
            return Ok(());
        }
        let out_shape = broadcast_shape(self.shape(), other.shape())?;
        if out_shape.as_slice() != self.shape() {
            return Err(ArrayError::Message(format!(
                "non-broadcastable output operand with shape {} doesn't match the broadcast shape {}",
                shape_str(self.shape()),
                shape_str(&out_shape)
            )));
        }
        let result = self.zip_map(other, f)?;
        self.data.copy_from_slice(&result.data);
        Ok(())
    }

    /// `self = f(self)` element-wise in place.
    pub fn map_inplace<F: Fn(f64) -> f64>(&mut self, f: F) {
        for a in self.data.iter_mut() {
            *a = f(*a);
        }
    }

    // ---- element-wise kernels ------------------------------------------

    /// `f(a[i], b[i])` with NumPy broadcasting. Equal shapes take the
    /// straight zipped loop; anything else goes through an odometer over the
    /// broadcast shape with zero strides on broadcast axes.
    #[inline]
    pub fn zip_map<F: Fn(f64, f64) -> f64>(&self, other: &Array, f: F) -> Result<Array> {
        if self.dims == other.dims {
            let data = self.data.iter().zip(&other.data).map(|(&a, &b)| f(a, b)).collect();
            return Ok(Array { data, dims: self.dims });
        }
        self.zip_map_broadcast(other, f)
    }

    #[inline(never)]
    fn zip_map_broadcast<F: Fn(f64, f64) -> f64>(&self, other: &Array, f: F) -> Result<Array> {
        let out_shape = broadcast_shape(self.shape(), other.shape())?;
        let size: usize = out_shape.iter().product();
        let mut out = Vec::with_capacity(size);
        if size > 0 {
            let sa = broadcast_strides(self.shape(), &out_shape);
            let sb = broadcast_strides(other.shape(), &out_shape);
            let n = out_shape.len();
            let mut idx = vec![0usize; n];
            let (mut ao, mut bo) = (0usize, 0usize);
            for _ in 0..size {
                out.push(f(self.data[ao], other.data[bo]));
                let mut k = n;
                while k > 0 {
                    k -= 1;
                    idx[k] += 1;
                    ao += sa[k];
                    bo += sb[k];
                    if idx[k] < out_shape[k] {
                        break;
                    }
                    ao -= sa[k] * idx[k];
                    bo -= sb[k] * idx[k];
                    idx[k] = 0;
                }
            }
        }
        Array::new(out, &out_shape)
    }

    /// `f(a[i])`.
    #[inline]
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Array {
        Array { data: self.data.iter().map(|&a| f(a)).collect(), dims: self.dims }
    }

    // ---- reductions ----------------------------------------------------

    /// Sum with 8 independent accumulators so the loop vectorizes; a plain
    /// sequential f64 sum is not reassociable and stays scalar.
    pub fn sum(&self) -> f64 {
        const LANES: usize = 8;
        let mut acc = [0.0f64; LANES];
        let chunks = self.data.chunks_exact(LANES);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for i in 0..LANES {
                acc[i] += chunk[i];
            }
        }
        let tail: f64 = remainder.iter().sum();
        ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
    }

    pub fn prod(&self) -> f64 {
        self.data.iter().product()
    }

    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            f64::NAN
        } else {
            self.sum() / self.data.len() as f64
        }
    }

    /// NaN-propagating maximum, like `np.max`. Returns None when empty.
    pub fn max(&self) -> Option<f64> {
        self.data.iter().copied().reduce(|a, b| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) })
    }

    pub fn min(&self) -> Option<f64> {
        self.data.iter().copied().reduce(|a, b| if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) })
    }

    /// Fold along one axis. `init` seeds the accumulator; when None the
    /// first element along the axis seeds it (for max/min), which requires a
    /// non-empty axis.
    pub fn fold_axis<F: Fn(f64, f64) -> f64>(&self, axis: isize, init: Option<f64>, f: F) -> Result<Array> {
        let ndim = self.ndim() as isize;
        if axis < -ndim || axis >= ndim {
            return Err(ArrayError::Message(format!("axis {axis} is out of bounds for array of dimension {ndim}")));
        }
        let axis = if axis < 0 { (axis + ndim) as usize } else { axis as usize };
        let shape = self.shape();
        let len = shape[axis];
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        if init.is_none() && len == 0 {
            return Err(ArrayError::Message("zero-size array to reduction operation which has no identity".into()));
        }
        let mut out = Vec::with_capacity(outer * inner);
        if inner == 1 {
            // Reducing the last axis: each output is a contiguous run.
            for o in 0..outer {
                let row = &self.data[o * len..(o + 1) * len];
                let (mut acc, rest) = match init {
                    Some(v) => (v, row),
                    None => (row[0], &row[1..]),
                };
                for &x in rest {
                    acc = f(acc, x);
                }
                out.push(acc);
            }
        } else {
            // Reducing an earlier axis: accumulate whole inner rows at a time so
            // the inner loop is contiguous and vectorizes.
            for o in 0..outer {
                let base = o * len * inner;
                let start = out.len();
                let start_j = match init {
                    Some(v) => {
                        out.resize(start + inner, v);
                        0
                    }
                    None => {
                        out.extend_from_slice(&self.data[base..base + inner]);
                        1
                    }
                };
                let acc = &mut out[start..start + inner];
                for j in start_j..len {
                    let row = &self.data[base + j * inner..base + (j + 1) * inner];
                    for (a, &x) in acc.iter_mut().zip(row) {
                        *a = f(*a, x);
                    }
                }
            }
        }
        let mut out_shape = shape[..axis].to_vec();
        out_shape.extend_from_slice(&shape[axis + 1..]);
        Array::new(out, &out_shape)
    }

    /// Population variance (ddof = 0), two-pass like NumPy.
    pub fn var(&self) -> f64 {
        if self.data.is_empty() {
            return f64::NAN;
        }
        let mean = self.mean();
        let n = self.data.len() as f64;
        self.data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n
    }

    /// Index of the first maximum; NaN wins, like NumPy.
    pub fn argmax(&self) -> Option<usize> {
        self.arg_extreme(|x, best| x > best)
    }

    pub fn argmin(&self) -> Option<usize> {
        self.arg_extreme(|x, best| x < best)
    }

    fn arg_extreme<F: Fn(f64, f64) -> bool>(&self, better: F) -> Option<usize> {
        let mut best_i = 0;
        let mut best = *self.data.first()?;
        if best.is_nan() {
            return Some(0);
        }
        for (i, &x) in self.data.iter().enumerate().skip(1) {
            if x.is_nan() {
                return Some(i);
            }
            if better(x, best) {
                best = x;
                best_i = i;
            }
        }
        Some(best_i)
    }

    pub fn any(&self) -> bool {
        self.data.iter().any(|&x| x != 0.0)
    }

    pub fn all(&self) -> bool {
        self.data.iter().all(|&x| x != 0.0)
    }

    /// Running fold over the flattened array (NumPy's `cumsum(axis=None)`).
    pub fn scan<F: Fn(f64, f64) -> f64>(&self, init: f64, f: F) -> Array {
        let mut acc = init;
        let data: Vec<f64> = self.data.iter().map(|&x| { acc = f(acc, x); acc }).collect();
        Array { data, dims: Dims::from_shape(&[self.size()]).expect("1-d") }
    }

    /// Inner product of two 1-D arrays of equal length.
    pub fn dot1d(&self, other: &Array) -> Result<f64> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(ArrayError::Message("dot1d expects 1-D arrays".into()));
        }
        if self.size() != other.size() {
            return Err(ArrayError::Message(format!(
                "shapes ({},) and ({},) not aligned",
                self.size(),
                other.size()
            )));
        }
        const LANES: usize = 8;
        let mut acc = [0.0f64; LANES];
        let a = self.data.chunks_exact(LANES);
        let b = other.data.chunks_exact(LANES);
        let tail: f64 = a.remainder().iter().zip(b.remainder()).map(|(x, y)| x * y).sum();
        for (ca, cb) in a.zip(b) {
            for i in 0..LANES {
                acc[i] += ca[i] * cb[i];
            }
        }
        Ok(((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail)
    }

    // ---- indexing ------------------------------------------------------

    /// Normalise a possibly negative index on `axis`.
    fn normalize_index(&self, index: isize, axis: usize) -> Result<usize> {
        let len = self.shape()[axis];
        let i = if index < 0 { index + len as isize } else { index };
        if i < 0 || i as usize >= len {
            return Err(ArrayError::IndexOutOfBounds { index, axis, len });
        }
        Ok(i as usize)
    }

    /// Index with one integer per leading axis. Returns the flat offset of
    /// the addressed sub-array and its dims.
    fn locate(&self, indices: &[isize]) -> Result<(usize, Dims)> {
        if indices.len() > self.ndim() {
            return Err(ArrayError::TooManyIndices { given: indices.len(), ndim: self.ndim() });
        }
        let mut offset = 0usize;
        for (axis, &idx) in indices.iter().enumerate() {
            let i = self.normalize_index(idx, axis)?;
            offset += i * (self.dims.strides()[axis] as usize / 8);
        }
        Ok((offset, self.dims.drop_leading(indices.len())))
    }

    /// `a[i, j, ...]` with integers only. A full index returns a scalar
    /// array (0-d); a partial one returns a copy of the sub-array.
    pub fn index(&self, indices: &[isize]) -> Result<Array> {
        let (offset, dims) = self.locate(indices)?;
        let n = dims.size();
        Ok(Array { data: self.data[offset..offset + n].to_vec(), dims })
    }

    /// Scalar element for a full integer index.
    pub fn get(&self, indices: &[isize]) -> Result<Option<f64>> {
        let (offset, dims) = self.locate(indices)?;
        Ok(if dims.ndim() == 0 { Some(self.data[offset]) } else { None })
    }

    /// General basic indexing: one selector per leading axis, remaining axes
    /// taken whole. Integer selectors drop their axis; slices keep it. The
    /// result is a copy (views arrive in Phase 4).
    pub fn select(&self, sels: &[Selector]) -> Result<Array> {
        if sels.len() > self.ndim() {
            return Err(ArrayError::TooManyIndices { given: sels.len(), ndim: self.ndim() });
        }
        let shape = self.shape();
        // Per axis: (start, step, count) in elements of that axis.
        let mut plan: Vec<(usize, isize, usize)> = Vec::with_capacity(self.ndim());
        let mut out_shape: Vec<usize> = Vec::with_capacity(self.ndim());
        for axis in 0..self.ndim() {
            match sels.get(axis) {
                Some(Selector::Int(i)) => {
                    let i = self.normalize_index(*i, axis)?;
                    plan.push((i, 1, 1));
                }
                Some(Selector::Slice { start, stop, step }) => {
                    let count = if *step > 0 {
                        if stop > start { ((stop - start - 1) / step + 1) as usize } else { 0 }
                    } else if start > stop {
                        ((start - stop - 1) / -step + 1) as usize
                    } else {
                        0
                    };
                    plan.push((*start as usize, *step, count));
                    out_shape.push(count);
                }
                None => {
                    plan.push((0, 1, shape[axis]));
                    out_shape.push(shape[axis]);
                }
            }
        }
        let out_size: usize = out_shape.iter().product();
        let mut out = Vec::with_capacity(out_size);
        if out_size > 0 {
            let elem_strides: Vec<usize> = self.dims.strides().iter().map(|&s| s as usize / 8).collect();
            gather(&self.data, &plan, &elem_strides, 0, 0, &mut out);
        }
        Array::new(out, &out_shape)
    }

    /// Join arrays of equal ndim along `axis`; all other dimensions must match.
    pub fn concatenate(parts: &[&Array], axis: isize) -> Result<Array> {
        let first = parts.first().ok_or_else(|| ArrayError::Message("need at least one array to concatenate".into()))?;
        let ndim = first.ndim() as isize;
        if ndim == 0 {
            return Err(ArrayError::Message("zero-dimensional arrays cannot be concatenated".into()));
        }
        if axis < -ndim || axis >= ndim {
            return Err(ArrayError::Message(format!("axis {axis} is out of bounds for array of dimension {ndim}")));
        }
        let axis = if axis < 0 { (axis + ndim) as usize } else { axis as usize };
        let mut out_shape = first.shape().to_vec();
        out_shape[axis] = 0;
        for p in parts {
            if p.ndim() != first.ndim() {
                return Err(ArrayError::Message("all the input arrays must have same number of dimensions".into()));
            }
            for (k, (&a, &b)) in p.shape().iter().zip(first.shape()).enumerate() {
                if k != axis && a != b {
                    return Err(ArrayError::Message(format!(
                        "all the input array dimensions except for the concatenation axis must match exactly, but along dimension {k}, sizes {b} and {a} differ"
                    )));
                }
            }
            out_shape[axis] += p.shape()[axis];
        }
        let outer: usize = first.shape()[..axis].iter().product();
        let inner: usize = first.shape()[axis + 1..].iter().product();
        let mut data = Vec::with_capacity(out_shape.iter().product());
        for o in 0..outer {
            for p in parts {
                let block = p.shape()[axis] * inner;
                data.extend_from_slice(&p.data[o * block..(o + 1) * block]);
            }
        }
        Array::new(data, &out_shape)
    }

    /// `where(cond, x, y)` with a boolean mask of the same shape.
    pub fn select_where(cond: &[bool], x: &Array, y: &Array) -> Result<Array> {
        if x.dims != y.dims || cond.len() != x.size() {
            return Err(ArrayError::ShapeMismatch(x.shape().to_vec(), y.shape().to_vec()));
        }
        let data = cond.iter().zip(&x.data).zip(&y.data).map(|((&c, &a), &b)| if c { a } else { b }).collect();
        Ok(Array { data, dims: x.dims })
    }

    /// Reverse the axes (NumPy's `.T`). 0-d and 1-d arrays are returned as
    /// copies; 2-d is transposed directly; higher dims go through a general
    /// permutation.
    pub fn transpose(&self) -> Array {
        match self.ndim() {
            0 | 1 => self.clone(),
            2 => {
                let (rows, cols) = (self.shape()[0], self.shape()[1]);
                let mut data = Vec::with_capacity(self.size());
                for c in 0..cols {
                    for r in 0..rows {
                        data.push(self.data[r * cols + c]);
                    }
                }
                Array { data, dims: Dims::from_shape(&[cols, rows]).expect("2-d") }
            }
            _ => {
                let shape = self.shape();
                let ndim = shape.len();
                let rev_shape: Vec<usize> = shape.iter().rev().copied().collect();
                let strides: Vec<usize> = self.dims.strides().iter().map(|&s| s as usize / 8).collect();
                let mut data = Vec::with_capacity(self.size());
                let mut idx = vec![0usize; ndim];
                for _ in 0..self.size() {
                    // idx is a multi-index into the transposed array; source offset uses reversed axes
                    let off: usize = (0..ndim).map(|k| idx[k] * strides[ndim - 1 - k]).sum();
                    data.push(self.data[off]);
                    for k in (0..ndim).rev() {
                        idx[k] += 1;
                        if idx[k] < rev_shape[k] {
                            break;
                        }
                        idx[k] = 0;
                    }
                }
                Array { data, dims: Dims::from_shape(&rev_shape).expect("same ndim") }
            }
        }
    }

    /// `a[mask]` with a boolean mask over the whole array (mask shape equals
    /// the array shape): the selected elements as a 1-D array.
    pub fn compress_flat(&self, mask: &[bool]) -> Result<Array> {
        if mask.len() != self.size() {
            return Err(ArrayError::Message(format!(
                "boolean index did not match indexed array; size {} but corresponding boolean size is {}",
                self.size(),
                mask.len()
            )));
        }
        let data: Vec<f64> = self.data.iter().zip(mask).filter(|(_, &m)| m).map(|(&x, _)| x).collect();
        let n = data.len();
        Array::new(data, &[n])
    }

    /// `a[[i, j, ...]]`: rows along the leading axis, negative indices allowed.
    pub fn take_leading(&self, indices: &[isize]) -> Result<Array> {
        if self.ndim() == 0 {
            return Err(ArrayError::TooManyIndices { given: 1, ndim: 0 });
        }
        let row = self.dims.drop_leading(1).size();
        let mut data = Vec::with_capacity(indices.len() * row);
        for &i in indices {
            let i = self.normalize_index(i, 0)?;
            data.extend_from_slice(&self.data[i * row..(i + 1) * row]);
        }
        let mut shape = vec![indices.len()];
        shape.extend_from_slice(&self.shape()[1..]);
        Array::new(data, &shape)
    }

    /// Sorted copy of a 1-D array, NaN last like NumPy.
    pub fn sorted_1d(&self) -> Result<Array> {
        if self.ndim() != 1 {
            return Err(ArrayError::Message("sorted_1d expects a 1-D array".into()));
        }
        let mut data = self.data.clone();
        data.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => a.partial_cmp(b).expect("non-NaN floats compare"),
        });
        Ok(Array { data, dims: self.dims })
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Array> {
        Array::new(self.data.clone(), shape)
    }
}

/// One axis of a basic index: an integer or a slice already resolved with
/// `slice.indices(len)` semantics.
#[derive(Debug, Clone, Copy)]
pub enum Selector {
    Int(isize),
    Slice { start: isize, stop: isize, step: isize },
}

fn gather(data: &[f64], plan: &[(usize, isize, usize)], strides: &[usize], axis: usize, offset: usize, out: &mut Vec<f64>) {
    let (start, step, count) = plan[axis];
    let last = axis + 1 == plan.len();
    let mut pos = start as isize;
    for _ in 0..count {
        let off = offset + pos as usize * strides[axis];
        if last {
            out.push(data[off]);
        } else {
            gather(data, plan, strides, axis + 1, off, out);
        }
        pos += step;
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array").field("shape", &self.shape()).field("data", &self.data).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_map_equal_and_incompatible() {
        let a = Array::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Array::new(vec![1.0, 2.0], &[2]).unwrap();
        let err = a.zip_map(&b, |x, y| x + y).unwrap_err().to_string();
        assert_eq!(err, "operands could not be broadcast together with shapes (3,) (2,)");
        let c = a.zip_map(&a, |x, y| x * y).unwrap();
        assert_eq!(c.data(), &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn zip_map_broadcasts() {
        let m = Array::new((0..6).map(|i| i as f64).collect(), &[2, 3]).unwrap();
        let row = Array::new(vec![10.0, 20.0, 30.0], &[3]).unwrap();
        let col = Array::new(vec![100.0, 200.0], &[2, 1]).unwrap();
        let r = m.zip_map(&row, |x, y| x + y).unwrap();
        assert_eq!((r.shape(), r.data()), (&[2, 3][..], &[10.0, 21.0, 32.0, 13.0, 24.0, 35.0][..]));
        let r = col.zip_map(&m, |x, y| x - y).unwrap();
        assert_eq!(r.data(), &[100.0, 99.0, 98.0, 197.0, 196.0, 195.0]);
        let r = col.zip_map(&row, |x, y| x + y).unwrap();
        assert_eq!((r.shape(), r.data()), (&[2, 3][..], &[110.0, 120.0, 130.0, 210.0, 220.0, 230.0][..]));
        let s = Array::scalar(2.0);
        assert_eq!(s.zip_map(&row, |x, y| x * y).unwrap().data(), &[20.0, 40.0, 60.0]);
        let empty = Array::new(vec![], &[0]).unwrap();
        assert_eq!(empty.zip_map(&Array::new(vec![1.0], &[1]).unwrap(), |x, y| x + y).unwrap().shape(), &[0]);
        assert_eq!(broadcast_shape(&[2, 1, 4], &[3, 1]).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn sum_matches_sequential_within_rounding() {
        let data: Vec<f64> = (0..1003).map(|i| i as f64 * 0.1).collect();
        let a = Array::new(data.clone(), &[1003]).unwrap();
        let seq: f64 = data.iter().sum();
        assert!((a.sum() - seq).abs() < 1e-9);
    }

    #[test]
    fn max_propagates_nan_and_handles_empty() {
        let a = Array::new(vec![1.0, f64::NAN, 3.0], &[3]).unwrap();
        assert!(a.max().unwrap().is_nan());
        assert_eq!(Array::new(vec![], &[0]).unwrap().max(), None);
    }

    #[test]
    fn integer_indexing() {
        let a = Array::new((0..6).map(|i| i as f64).collect(), &[2, 3]).unwrap();
        assert_eq!(a.get(&[1, 2]).unwrap(), Some(5.0));
        assert_eq!(a.get(&[-1, -1]).unwrap(), Some(5.0));
        assert_eq!(a.index(&[1]).unwrap().data(), &[3.0, 4.0, 5.0]);
        assert!(a.get(&[2, 0]).is_err());
        assert!(a.get(&[0, 0, 0]).is_err());
    }

    #[test]
    fn axis_reductions() {
        let a = Array::new((0..6).map(|i| i as f64).collect(), &[2, 3]).unwrap();
        let s0 = a.fold_axis(0, Some(0.0), |x, y| x + y).unwrap();
        assert_eq!((s0.shape(), s0.data()), (&[3][..], &[3.0, 5.0, 7.0][..]));
        let s1 = a.fold_axis(-1, Some(0.0), |x, y| x + y).unwrap();
        assert_eq!((s1.shape(), s1.data()), (&[2][..], &[3.0, 12.0][..]));
        let m = a.fold_axis(1, None, f64::max).unwrap();
        assert_eq!(m.data(), &[2.0, 5.0]);
        assert!(a.fold_axis(2, Some(0.0), |x, y| x + y).is_err());
        assert!(Array::new(vec![], &[0, 3]).unwrap().fold_axis(0, None, f64::max).is_err());
    }

    #[test]
    fn extra_kernels() {
        let a = Array::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], &[5]).unwrap();
        assert_eq!(a.argmax(), Some(4));
        assert_eq!(a.argmin(), Some(1));
        assert_eq!(Array::new(vec![1.0, f64::NAN, 9.0], &[3]).unwrap().argmax(), Some(1));
        assert!((a.var() - 2.56).abs() < 1e-12);
        assert_eq!(a.scan(0.0, |x, y| x + y).data(), &[3.0, 4.0, 8.0, 9.0, 14.0]);
        assert_eq!(a.dot1d(&a).unwrap(), 52.0);
        assert!(a.any() && a.all());
    }

    #[test]
    fn mask_and_take() {
        let a = Array::new((0..6).map(|i| i as f64).collect(), &[2, 3]).unwrap();
        let m = a.compress_flat(&[true, false, true, false, false, true]).unwrap();
        assert_eq!((m.shape(), m.data()), (&[3][..], &[0.0, 2.0, 5.0][..]));
        assert!(a.compress_flat(&[true]).is_err());
        let t = a.take_leading(&[1, -2, 1]).unwrap();
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(&t.data()[3..6], &[0.0, 1.0, 2.0]);
        assert!(a.take_leading(&[2]).is_err());
    }

    #[test]
    fn in_place_mutation() {
        let mut a = Array::new((0..6).map(|i| i as f64).collect(), &[2, 3]).unwrap();
        a.set(&[1, -1], 50.0).unwrap();
        assert_eq!(a.data()[5], 50.0);
        assert!(a.set(&[1], 0.0).is_err());
        assert!(a.set(&[2, 0], 0.0).is_err());
        let row = Array::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        a.zip_map_inplace(&row, |x, y| x + y).unwrap();
        assert_eq!(&a.data()[..3], &[1.0, 3.0, 5.0]);
        a.map_inplace(|x| -x);
        assert_eq!(a.data()[0], -1.0);
        let col = Array::new(vec![1.0, 2.0], &[2, 1]).unwrap();
        assert!(col.clone().zip_map_inplace(&a, |x, y| x + y).is_err()); // would change shape
    }

    #[test]
    fn sort_nan_last() {
        let a = Array::new(vec![3.0, f64::NAN, -1.0, 2.0], &[4]).unwrap();
        let s = a.sorted_1d().unwrap();
        assert_eq!(&s.data()[..3], &[-1.0, 2.0, 3.0]);
        assert!(s.data()[3].is_nan());
        assert!(Array::new(vec![1.0], &[1, 1]).unwrap().sorted_1d().is_err());
    }

    #[test]
    fn transpose_all_dims() {
        let a = Array::new((0..6).map(|i| i as f64).collect(), &[2, 3]).unwrap();
        let t = a.transpose();
        assert_eq!((t.shape(), t.data()), (&[3, 2][..], &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0][..]));
        let b = Array::new((0..24).map(|i| i as f64).collect(), &[2, 3, 4]).unwrap();
        let t = b.transpose();
        assert_eq!(t.shape(), &[4, 3, 2]);
        assert_eq!(t.get(&[3, 1, 1]).unwrap(), b.get(&[1, 1, 3]).unwrap());
        assert_eq!(t.transpose().data(), b.data());
    }

    #[test]
    fn concatenate_axes() {
        let a = Array::new((0..6).map(|i| i as f64).collect(), &[2, 3]).unwrap();
        let r = Array::concatenate(&[&a, &a], 0).unwrap();
        assert_eq!(r.shape(), &[4, 3]);
        let r = Array::concatenate(&[&a, &a], -1).unwrap();
        assert_eq!((r.shape(), &r.data()[..6]), (&[2, 6][..], &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0][..]));
        let b = Array::new(vec![9.0, 9.0], &[2, 1]).unwrap();
        assert_eq!(Array::concatenate(&[&a, &b], 1).unwrap().data()[3], 9.0);
        assert!(Array::concatenate(&[&a, &b], 0).is_err());
    }

    #[test]
    fn select_mixed() {
        let a = Array::new((0..24).map(|i| i as f64).collect(), &[2, 3, 4]).unwrap();
        let s = Selector::Slice { start: 0, stop: 3, step: 2 };
        let r = a.select(&[Selector::Int(1), s]).unwrap();
        assert_eq!(r.shape(), &[2, 4]);
        assert_eq!(r.data(), &[12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0]);
        let r = a.select(&[Selector::Slice { start: 1, stop: -1, step: -1 }, Selector::Int(-1), Selector::Int(0)]).unwrap();
        assert_eq!((r.shape(), r.data()), (&[2][..], &[20.0, 8.0][..]));
        let r = a.select(&[Selector::Slice { start: 0, stop: 0, step: 1 }]).unwrap();
        assert_eq!(r.shape(), &[0, 3, 4]);
        assert!(a.select(&[Selector::Int(0); 4]).is_err());
    }

}
