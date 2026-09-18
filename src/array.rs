//! The Rust-side array: a C-contiguous buffer (owned, or a view into another
//! array's buffer) plus inline dims. Python never sees this type directly; `python.rs` wraps it.

use crate::dims::{Dims, DimsError};
use std::fmt;

#[derive(Clone)]
pub struct Array<T = f64> {
    pub(crate) data: Storage<T>,
    pub(crate) dims: Dims,
}

/// The element buffer of an array: its own allocation, or a contiguous window
/// into the buffer of another (Python-level) array, which `base` keeps alive.
/// A base never reallocates its buffer, so the pointer stays valid. Cloning a
/// view gives an owned copy. Stored as raw parts (a boxed slice when owned) so
/// that taking the slice needs no branch and the array stays three cache lines.
pub struct Storage<T> {
    ptr: *mut T,
    len: usize,
    base: Option<pyo3::Py<pyo3::PyAny>>,
}

impl<T> Drop for Storage<T> {
    #[inline(always)]
    fn drop(&mut self) {
        if self.base.is_none() {
            // SAFETY: owned storage was created from a boxed slice of this length.
            unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.ptr, self.len))) }
        }
    }
}

impl<T> std::ops::Deref for Storage<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: owned allocation, or a window of `base`'s, which outlives us.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> std::ops::DerefMut for Storage<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as above; Python-level access is serialised by the GIL.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Clone> Clone for Storage<T> {
    fn clone(&self) -> Self {
        self.to_vec().into()
    }
}

impl<T> From<Vec<T>> for Storage<T> {
    #[inline(always)]
    fn from(v: Vec<T>) -> Self {
        // Exact-capacity vectors (everything the kernels build) convert for free.
        let len = v.len();
        let ptr = Box::into_raw(v.into_boxed_slice()) as *mut T;
        Storage { ptr, len, base: None }
    }
}

impl<T> FromIterator<T> for Storage<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<T>>().into()
    }
}

impl<'a, T> IntoIterator for &'a Storage<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> Storage<T> {
    /// The array whose buffer this storage points into, for views.
    pub fn base(&self) -> Option<&pyo3::Py<pyo3::PyAny>> {
        self.base.as_ref()
    }
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

/// Element types an `Array` can hold.
pub trait Element: Copy + PartialEq + Default + fmt::Debug + 'static {}
impl Element for f64 {}
impl Element for i64 {}
impl Element for bool {}

// ---- structure: construction, indexing, reshaping (any element type) ----
impl<T: Element> Array<T> {
    pub fn new(data: Vec<T>, shape: &[usize]) -> Result<Array<T>> {
        let dims = Dims::with_itemsize(shape, std::mem::size_of::<T>())?;
        if dims.size() != data.len() {
            return Err(ArrayError::SizeMismatch { expected: dims.size(), got: data.len() });
        }
        Ok(Array { data: data.into(), dims })
    }


    pub fn scalar(value: T) -> Array<T> {
        Array { data: vec![value].into(), dims: Dims::scalar_with(std::mem::size_of::<T>()) }
    }


    pub fn filled(shape: &[usize], value: T) -> Result<Array<T>> {
        let dims = Dims::with_itemsize(shape, std::mem::size_of::<T>())?;
        Ok(Array { data: vec![value; dims.size()].into(), dims })
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
    pub fn data(&self) -> &[T] {
        &self.data
    }


    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }


    /// `a[i, j, ...] = value` for a full integer index.
    pub fn set(&mut self, indices: &[isize], value: T) -> Result<()> {
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
            offset += i * (self.dims.strides()[axis] as usize / std::mem::size_of::<T>());
        }
        Ok((offset, self.dims.drop_leading(indices.len())))
    }


    /// `a[i, j, ...]` with integers only. A full index returns a scalar
    /// array (0-d); a partial one returns a copy of the sub-array.
    pub fn index(&self, indices: &[isize]) -> Result<Array<T>> {
        let (offset, dims) = self.locate(indices)?;
        let n = dims.size();
        Ok(Array { data: self.data[offset..offset + n].to_vec().into(), dims })
    }


    /// Scalar element for a full integer index.
    pub fn get(&self, indices: &[isize]) -> Result<Option<T>> {
        let (offset, dims) = self.locate(indices)?;
        Ok(if dims.ndim() == 0 { Some(self.data[offset]) } else { None })
    }


    /// Per axis `(start, step, count)` of a basic index, and the result shape.
    #[allow(clippy::type_complexity)]
    pub fn selection_plan(&self, sels: &[Selector]) -> Result<(Vec<(usize, isize, usize)>, Vec<usize>)> {
        if sels.len() > self.ndim() {
            return Err(ArrayError::TooManyIndices { given: sels.len(), ndim: self.ndim() });
        }
        let shape = self.shape();
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
        Ok((plan, out_shape))
    }


    fn elem_strides(&self) -> Vec<usize> {
        self.dims.strides().iter().map(|&s| s as usize / std::mem::size_of::<T>()).collect()
    }


    /// General basic indexing: one selector per leading axis, remaining axes
    /// taken whole. Integer selectors drop their axis; slices keep it. The
    /// result is a copy; `select_view` shares memory where it can.
    pub fn select(&self, sels: &[Selector]) -> Result<Array<T>> {
        let (plan, out_shape) = self.selection_plan(sels)?;
        self.gather_plan(&plan, &out_shape)
    }


    fn gather_plan(&self, plan: &[(usize, isize, usize)], out_shape: &[usize]) -> Result<Array<T>> {
        let out_size: usize = out_shape.iter().product();
        let mut out = Vec::with_capacity(out_size);
        if out_size > 0 {
            gather(&self.data, plan, &self.elem_strides(), 0, 0, &mut out);
        }
        Array::new(out, out_shape)
    }


    /// Like `select`, but sharing memory: a selection that is one contiguous
    /// block of the buffer (`a[i]`, `a[2:5]`, `a[1, 2:4]`, `a[1:3, :]`, ...)
    /// becomes a view kept alive by `base()`. None for strided selections,
    /// which the caller turns into a strided view.
    pub fn select_view(&self, sels: &[Selector], base: impl FnOnce() -> pyo3::Py<pyo3::PyAny>) -> Result<Option<Array<T>>> {
        let (plan, out_shape) = self.selection_plan(sels)?;
        self.plan_view(&plan, &out_shape, base)
    }


    /// `select_view` for a plan made by `selection_plan`.
    pub fn plan_view(&self, plan: &[(usize, isize, usize)], out_shape: &[usize], base: impl FnOnce() -> pyo3::Py<pyo3::PyAny>) -> Result<Option<Array<T>>> {
        let shape = self.shape();
        let strides = self.elem_strides();
        let mut offset = 0usize;
        let mut block_started = false;
        for (axis, &(start, step, count)) in plan.iter().enumerate() {
            if count == 0 {
                // empty result: nothing to share
                return Ok(Some(Array::new(Vec::new(), out_shape)?));
            }
            if block_started {
                if !(start == 0 && step == 1 && count == shape[axis]) {
                    return Ok(None);
                }
            } else {
                offset += start * strides[axis];
                if count > 1 {
                    if step != 1 {
                        return Ok(None);
                    }
                    block_started = true;
                }
            }
        }
        let dims = Dims::with_itemsize(out_shape, std::mem::size_of::<T>())?;
        Ok(Some(self.view(offset, dims, base())))
    }


    /// A C-contiguous window at `ptr` into the buffer owned by `base`.
    ///
    /// # Safety
    /// `ptr` must address `shape.product()` elements inside `base`'s buffer.
    pub unsafe fn raw_view(ptr: *mut u8, shape: &[usize], base: pyo3::Py<pyo3::PyAny>) -> Result<Array<T>> {
        let dims = Dims::with_itemsize(shape, std::mem::size_of::<T>())?;
        Ok(Array { data: Storage { ptr: ptr as *mut T, len: dims.size(), base: Some(base) }, dims })
    }


    /// Fill this array from a strided source: element `[i, j, ...]` is read
    /// at `ptr + i*strides[0] + j*strides[1] + ...` (byte strides).
    ///
    /// # Safety
    /// Every addressed element must be valid memory holding a `T`.
    pub unsafe fn gather_from(&mut self, ptr: *const u8, strides: &[isize]) {
        let dims = self.dims;
        let shape = dims.shape();
        let data: &mut [T] = &mut self.data;
        // SAFETY: the caller guarantees the addressed elements.
        unsafe {
            walk_strided(shape, strides, |k, offset| data[k] = *(ptr.offset(offset) as *const T));
        }
    }


    /// The inverse of `gather_from`: write every element to the strided target.
    ///
    /// # Safety
    /// As for `gather_from`, and the memory must be writable.
    pub unsafe fn scatter_to(&self, ptr: *mut u8, strides: &[isize]) {
        let data: &[T] = &self.data;
        // SAFETY: the caller guarantees the addressed elements.
        unsafe {
            walk_strided(self.shape(), strides, |k, offset| *(ptr.offset(offset) as *mut T) = data[k]);
        }
    }


    /// `a[i, j, ...]` with integers only, as a view of the sub-array.
    pub fn index_view(&self, indices: &[isize], base: impl FnOnce() -> pyo3::Py<pyo3::PyAny>) -> Result<Array<T>> {
        let (offset, dims) = self.locate(indices)?;
        Ok(self.view(offset, dims, base()))
    }


    /// The whole buffer under another shape, as a view.
    pub fn reshape_view(&self, shape: &[usize], base: impl FnOnce() -> pyo3::Py<pyo3::PyAny>) -> Result<Array<T>> {
        let dims = Dims::with_itemsize(shape, std::mem::size_of::<T>())?;
        if dims.size() != self.size() {
            return Err(ArrayError::SizeMismatch { expected: dims.size(), got: self.size() });
        }
        Ok(self.view(0, dims, base()))
    }


    /// `dims.size()` elements starting at `offset`, sharing this buffer.
    fn view(&self, offset: usize, dims: Dims, base: pyo3::Py<pyo3::PyAny>) -> Array<T> {
        let len = dims.size();
        assert!(offset + len <= self.data.len(), "view out of bounds");
        // SAFETY: the range was just checked; `base` keeps the buffer alive.
        let ptr = unsafe { self.data.as_ptr().add(offset) as *mut T };
        Array { data: Storage { ptr, len, base: Some(base) }, dims }
    }


    /// The base array for views, None for arrays owning their buffer.
    pub fn base(&self) -> Option<&pyo3::Py<pyo3::PyAny>> {
        self.data.base()
    }


    /// Address range of the buffer, to detect overlapping operands.
    pub fn span(&self) -> (usize, usize) {
        let start = self.data.as_ptr() as usize;
        (start, start + self.data.len() * std::mem::size_of::<T>())
    }


    /// Join arrays of equal ndim along `axis`; all other dimensions must match.
    pub fn concatenate(parts: &[&Array<T>], axis: isize) -> Result<Array<T>> {
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
    pub fn select_where(cond: &[bool], x: &Array<T>, y: &Array<T>) -> Result<Array<T>> {
        if x.dims != y.dims || cond.len() != x.size() {
            return Err(ArrayError::ShapeMismatch(x.shape().to_vec(), y.shape().to_vec()));
        }
        let data = cond.iter().zip(&x.data).zip(&y.data).map(|((&c, &a), &b)| if c { a } else { b }).collect();
        Ok(Array { data, dims: x.dims })
    }


    /// Reverse the axes (NumPy's `.T`). 0-d and 1-d arrays are returned as
    /// copies; 2-d is transposed directly; higher dims go through a general
    /// permutation.
    pub fn transpose(&self) -> Array<T> {
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
                Array { data: data.into(), dims: Dims::with_itemsize(&[cols, rows], std::mem::size_of::<T>()).expect("2-d") }
            }
            _ => {
                let shape = self.shape();
                let ndim = shape.len();
                let rev_shape: Vec<usize> = shape.iter().rev().copied().collect();
                let strides: Vec<usize> = self.dims.strides().iter().map(|&s| s as usize / std::mem::size_of::<T>()).collect();
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
                Array { data: data.into(), dims: Dims::with_itemsize(&rev_shape, std::mem::size_of::<T>()).expect("same ndim") }
            }
        }
    }


    /// `a[mask]` with a boolean mask over the whole array (mask shape equals
    /// the array shape): the selected elements as a 1-D array.
    pub fn compress_flat(&self, mask: &[bool]) -> Result<Array<T>> {
        if mask.len() != self.size() {
            return Err(ArrayError::Message(format!(
                "boolean index did not match indexed array; size {} but corresponding boolean size is {}",
                self.size(),
                mask.len()
            )));
        }
        let data: Vec<T> = self.data.iter().zip(mask).filter(|(_, &m)| m).map(|(&x, _)| x).collect();
        let n = data.len();
        Array::new(data, &[n])
    }


    /// `a[[i, j, ...]]`: rows along the leading axis, negative indices allowed.
    pub fn take_leading(&self, indices: &[isize]) -> Result<Array<T>> {
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


    pub fn reshape(&self, shape: &[usize]) -> Result<Array<T>> {
        Array::new(self.data.to_vec(), shape)
    }
}

/// Visit every element of a strided layout in C order: `f(k, byte_offset)`
/// with `k` the flat index. One and two dimensions avoid the odometer.
#[inline]
fn walk_strided(shape: &[usize], strides: &[isize], mut f: impl FnMut(usize, isize)) {
    match shape.len() {
        0 => f(0, 0),
        1 => {
            for i in 0..shape[0] {
                f(i, i as isize * strides[0]);
            }
        }
        2 => {
            let mut k = 0;
            for i in 0..shape[0] {
                let row = i as isize * strides[0];
                for j in 0..shape[1] {
                    f(k, row + j as isize * strides[1]);
                    k += 1;
                }
            }
        }
        n => {
            let size: usize = shape.iter().product();
            if size == 0 {
                return;
            }
            let mut idx = [0usize; crate::dims::MAX_NDIM];
            let mut offset = 0isize;
            for k in 0..size {
                f(k, offset);
                let mut axis = n;
                while axis > 0 {
                    axis -= 1;
                    idx[axis] += 1;
                    offset += strides[axis];
                    if idx[axis] < shape[axis] {
                        break;
                    }
                    offset -= strides[axis] * shape[axis] as isize;
                    idx[axis] = 0;
                }
            }
        }
    }
}

// ---- float64 kernels ----
impl Array<f64> {
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
            let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| f(a, b)).collect();
            return Ok(Array { data: data.into(), dims: self.dims });
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
        let data: Vec<f64> = self.data.iter().map(|&a| f(a)).collect();
        Array { data: data.into(), dims: self.dims }
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
        Array { data: data.into(), dims: Dims::from_shape(&[self.size()]).expect("1-d") }
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

}

// ---- dtype-erased array -----------------------------------------------------

/// A scalar of any supported dtype.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    F(f64),
    I(i64),
    B(bool),
}

impl Scalar {
    pub fn as_f64(self) -> f64 {
        match self {
            Scalar::F(v) => v,
            Scalar::I(v) => v as f64,
            Scalar::B(v) => v as u8 as f64,
        }
    }
}

/// An array of one of the supported dtypes. float64 is the fast path with
/// the full kernel set; int64 and bool have the structural operations plus
/// the few kernels that matter for them (comparisons, masks, counts, index
/// arrays) and promote to float64 for arithmetic with floats.
#[derive(Clone, Debug)]
pub enum AnyArray {
    F64(Array<f64>),
    I64(Array<i64>),
    Bool(Array<bool>),
}

macro_rules! each {
    ($any:expr, $a:ident => $body:expr) => {
        match $any {
            AnyArray::F64($a) => $body,
            AnyArray::I64($a) => $body,
            AnyArray::Bool($a) => $body,
        }
    };
}

macro_rules! map_each {
    ($any:expr, $a:ident => $body:expr) => {
        match $any {
            AnyArray::F64($a) => AnyArray::F64($body),
            AnyArray::I64($a) => AnyArray::I64($body),
            AnyArray::Bool($a) => AnyArray::Bool($body),
        }
    };
}

impl AnyArray {
    pub fn dims(&self) -> &Dims {
        each!(self, a => &a.dims)
    }
    pub fn shape(&self) -> &[usize] {
        self.dims().shape()
    }
    pub fn ndim(&self) -> usize {
        self.dims().ndim()
    }
    pub fn size(&self) -> usize {
        each!(self, a => a.data.len())
    }
    pub fn itemsize(&self) -> usize {
        self.dims().itemsize()
    }
    /// NumPy dtype name.
    pub fn dtype_name(&self) -> &'static str {
        match self {
            AnyArray::F64(_) => "float64",
            AnyArray::I64(_) => "int64",
            AnyArray::Bool(_) => "bool",
        }
    }
    /// Buffer-protocol format character.
    pub fn format(&self) -> &'static std::ffi::CStr {
        match self {
            AnyArray::F64(_) => c"d",
            // NumPy's default integer is C long where that is 8 bytes; using
            // its format code keeps reprs free of a spurious `dtype=int64`.
            AnyArray::I64(_) => {
                if std::mem::size_of::<std::ffi::c_long>() == 8 {
                    c"l"
                } else {
                    c"q"
                }
            }
            AnyArray::Bool(_) => c"?",
        }
    }
    pub fn data_ptr(&self) -> *const std::ffi::c_void {
        each!(self, a => a.data.as_ptr() as *const std::ffi::c_void)
    }
    pub fn as_f64(&self) -> Option<&Array<f64>> {
        match self {
            AnyArray::F64(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_f64_mut(&mut self) -> Option<&mut Array<f64>> {
        match self {
            AnyArray::F64(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<&Array<bool>> {
        match self {
            AnyArray::Bool(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<&Array<i64>> {
        match self {
            AnyArray::I64(a) => Some(a),
            _ => None,
        }
    }
    /// Promote to float64 (a copy for int64/bool, a clone for float64).
    pub fn to_f64(&self) -> Array<f64> {
        match self {
            AnyArray::F64(a) => a.clone(),
            AnyArray::I64(a) => Array { data: a.data.iter().map(|&v| v as f64).collect(), dims: a.dims.retyped(8) },
            AnyArray::Bool(a) => Array { data: a.data.iter().map(|&v| v as u8 as f64).collect(), dims: a.dims.retyped(8) },
        }
    }
    /// Promote bool to int64; None for float64.
    pub fn to_i64(&self) -> Option<Array<i64>> {
        match self {
            AnyArray::F64(_) => None,
            AnyArray::I64(a) => Some(a.clone()),
            AnyArray::Bool(a) => Some(Array { data: a.data.iter().map(|&v| v as i64).collect(), dims: a.dims.retyped(8) }),
        }
    }
    pub fn index(&self, indices: &[isize]) -> Result<AnyArray> {
        Ok(map_each!(self, a => a.index(indices)?))
    }
    /// Scalar element for a full integer index.
    pub fn get(&self, indices: &[isize]) -> Result<Option<Scalar>> {
        Ok(match self {
            AnyArray::F64(a) => a.get(indices)?.map(Scalar::F),
            AnyArray::I64(a) => a.get(indices)?.map(Scalar::I),
            AnyArray::Bool(a) => a.get(indices)?.map(Scalar::B),
        })
    }
    /// The single element of a size-1 array.
    pub fn item(&self) -> Option<Scalar> {
        if self.size() != 1 {
            return None;
        }
        Some(match self {
            AnyArray::F64(a) => Scalar::F(a.data[0]),
            AnyArray::I64(a) => Scalar::I(a.data[0]),
            AnyArray::Bool(a) => Scalar::B(a.data[0]),
        })
    }
    pub fn select(&self, sels: &[Selector]) -> Result<AnyArray> {
        Ok(map_each!(self, a => a.select(sels)?))
    }
    /// Basic indexing that shares memory; None when the selection is strided.
    pub fn select_view(&self, sels: &[Selector], base: impl FnOnce() -> pyo3::Py<pyo3::PyAny>) -> Result<Option<AnyArray>> {
        Ok(match self {
            AnyArray::F64(a) => a.select_view(sels, base)?.map(AnyArray::F64),
            AnyArray::I64(a) => a.select_view(sels, base)?.map(AnyArray::I64),
            AnyArray::Bool(a) => a.select_view(sels, base)?.map(AnyArray::Bool),
        })
    }
    /// Per axis `(start, step, count)` of a basic index, and the result shape.
    #[allow(clippy::type_complexity)]
    pub fn selection_plan(&self, sels: &[Selector]) -> Result<(Vec<(usize, isize, usize)>, Vec<usize>)> {
        each!(self, a => a.selection_plan(sels))
    }
    /// The contiguous window a plan selects; None when it is strided.
    pub fn plan_view(&self, plan: &[(usize, isize, usize)], out_shape: &[usize], base: impl FnOnce() -> pyo3::Py<pyo3::PyAny>) -> Result<Option<AnyArray>> {
        Ok(match self {
            AnyArray::F64(a) => a.plan_view(plan, out_shape, base)?.map(AnyArray::F64),
            AnyArray::I64(a) => a.plan_view(plan, out_shape, base)?.map(AnyArray::I64),
            AnyArray::Bool(a) => a.plan_view(plan, out_shape, base)?.map(AnyArray::Bool),
        })
    }
    /// A C-contiguous window of this dtype at `ptr`, kept alive by `base`.
    ///
    /// # Safety
    /// See `Array::raw_view`.
    pub unsafe fn raw_view_like(&self, ptr: *mut u8, shape: &[usize], base: pyo3::Py<pyo3::PyAny>) -> Result<AnyArray> {
        // SAFETY: forwarded to the caller.
        Ok(unsafe {
            match self {
                AnyArray::F64(_) => AnyArray::F64(Array::raw_view(ptr, shape, base)?),
                AnyArray::I64(_) => AnyArray::I64(Array::raw_view(ptr, shape, base)?),
                AnyArray::Bool(_) => AnyArray::Bool(Array::raw_view(ptr, shape, base)?),
            }
        })
    }
    /// A default-filled array of this dtype.
    pub fn zeros_like(&self, shape: &[usize]) -> Result<AnyArray> {
        Ok(match self {
            AnyArray::F64(_) => AnyArray::F64(Array::filled(shape, 0.0)?),
            AnyArray::I64(_) => AnyArray::I64(Array::filled(shape, 0)?),
            AnyArray::Bool(_) => AnyArray::Bool(Array::filled(shape, false)?),
        })
    }
    /// # Safety
    /// See `Array::gather_from`.
    pub unsafe fn gather_from(&mut self, ptr: *const u8, strides: &[isize]) {
        // SAFETY: forwarded to the caller.
        unsafe { each!(self, a => a.gather_from(ptr, strides)) }
    }
    /// # Safety
    /// See `Array::scatter_to`.
    pub unsafe fn scatter_to(&self, ptr: *mut u8, strides: &[isize]) {
        // SAFETY: forwarded to the caller.
        unsafe { each!(self, a => a.scatter_to(ptr, strides)) }
    }
    /// Integer indexing of leading axes, as a view of the sub-array.
    pub fn index_view(&self, indices: &[isize], base: impl FnOnce() -> pyo3::Py<pyo3::PyAny>) -> Result<AnyArray> {
        Ok(map_each!(self, a => a.index_view(indices, base)?))
    }
    /// The same buffer under another shape.
    pub fn reshape_view(&self, shape: &[usize], base: impl FnOnce() -> pyo3::Py<pyo3::PyAny>) -> Result<AnyArray> {
        Ok(map_each!(self, a => a.reshape_view(shape, base)?))
    }
    /// The array this one is a view of, if any.
    pub fn base(&self) -> Option<&pyo3::Py<pyo3::PyAny>> {
        each!(self, a => a.base())
    }
    /// Address range of the buffer.
    pub fn span(&self) -> (usize, usize) {
        each!(self, a => a.span())
    }
    /// In-place reshape (`a.shape = ...`): same data, new dims.
    pub fn set_shape(&mut self, shape: &[usize]) -> Result<()> {
        let size = self.size();
        if shape.iter().product::<usize>() != size {
            return Err(ArrayError::SizeMismatch { expected: shape.iter().product(), got: size });
        }
        let dims = Dims::with_itemsize(shape, self.itemsize())?;
        each!(self, a => a.dims = dims);
        Ok(())
    }
    pub fn reshape(&self, shape: &[usize]) -> Result<AnyArray> {
        Ok(map_each!(self, a => a.reshape(shape)?))
    }
    pub fn transpose(&self) -> AnyArray {
        map_each!(self, a => a.transpose())
    }
    pub fn take_leading(&self, indices: &[isize]) -> Result<AnyArray> {
        Ok(map_each!(self, a => a.take_leading(indices)?))
    }
    pub fn compress_flat(&self, mask: &[bool]) -> Result<AnyArray> {
        Ok(map_each!(self, a => a.compress_flat(mask)?))
    }
    /// Concatenate arrays of one dtype; None when the dtypes differ.
    pub fn concatenate(parts: &[&AnyArray], axis: isize) -> Result<Option<AnyArray>> {
        macro_rules! same {
            ($variant:ident) => {{
                let mut refs = Vec::with_capacity(parts.len());
                for p in parts {
                    match p {
                        AnyArray::$variant(a) => refs.push(a),
                        _ => return Ok(None),
                    }
                }
                Ok(Some(AnyArray::$variant(Array::concatenate(&refs, axis)?)))
            }};
        }
        match parts.first() {
            None => Err(ArrayError::Message("need at least one array to concatenate".into())),
            Some(AnyArray::F64(_)) => same!(F64),
            Some(AnyArray::I64(_)) => same!(I64),
            Some(AnyArray::Bool(_)) => same!(Bool),
        }
    }
}

impl Array<bool> {
    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&b| b).count()
    }
    pub fn not(&self) -> Array<bool> {
        Array { data: self.data.iter().map(|&b| !b).collect(), dims: self.dims }
    }
    /// Element-wise combination of two equal-shape masks.
    pub fn zip_bool<F: Fn(bool, bool) -> bool>(&self, other: &Array<bool>, f: F) -> Option<Array<bool>> {
        if self.dims != other.dims {
            return None;
        }
        Some(Array { data: self.data.iter().zip(&other.data).map(|(&a, &b)| f(a, b)).collect(), dims: self.dims })
    }
}

impl Array<i64> {
    pub fn sum(&self) -> i64 {
        self.data.iter().fold(0i64, |acc, &v| acc.wrapping_add(v))
    }
    pub fn min(&self) -> Option<i64> {
        self.data.iter().copied().min()
    }
    pub fn max(&self) -> Option<i64> {
        self.data.iter().copied().max()
    }
    /// Equal-shape element-wise op; None when shapes differ (caller falls back).
    pub fn zip_int<F: Fn(i64, i64) -> i64>(&self, other: &Array<i64>, f: F) -> Option<Array<i64>> {
        if self.dims != other.dims {
            return None;
        }
        Some(Array { data: self.data.iter().zip(&other.data).map(|(&a, &b)| f(a, b)).collect(), dims: self.dims })
    }
    pub fn map_int<F: Fn(i64) -> i64>(&self, f: F) -> Array<i64> {
        Array { data: self.data.iter().map(|&a| f(a)).collect(), dims: self.dims }
    }
}

impl AnyArray {
    /// Convert between the native dtypes like NumPy's `astype` (C casts:
    /// floats truncate toward zero, anything non-zero is True). None when a
    /// float is NaN or out of the int64 range, where NumPy's result is
    /// platform-defined and comes with a warning; the caller lets NumPy do it.
    pub fn cast(&self, target: &str) -> Option<AnyArray> {
        Some(match (self, target) {
            (a, t) if a.dtype_name() == t => a.clone(),
            (a, "float64") => AnyArray::F64(a.to_f64()),
            (AnyArray::F64(a), "int64") => {
                if a.data.iter().any(|v| !v.is_finite() || v.abs() >= 9.2e18) {
                    return None;
                }
                AnyArray::I64(Array { data: a.data.iter().map(|&v| v as i64).collect(), dims: a.dims })
            }
            (AnyArray::Bool(_), "int64") => AnyArray::I64(self.to_i64()?),
            (AnyArray::F64(a), "bool") => AnyArray::Bool(Array { data: a.data.iter().map(|&v| v != 0.0).collect(), dims: a.dims.retyped(1) }),
            (AnyArray::I64(a), "bool") => AnyArray::Bool(Array { data: a.data.iter().map(|&v| v != 0).collect(), dims: a.dims.retyped(1) }),
            _ => return None,
        })
    }

    /// Indices of the non-zero elements, one int64 array per dimension
    /// (NumPy's `nonzero`).
    pub fn nonzero(&self) -> Vec<Array<i64>> {
        let flat: Vec<usize> = match self {
            AnyArray::F64(a) => a.data.iter().enumerate().filter(|(_, &v)| v != 0.0).map(|(i, _)| i).collect(),
            AnyArray::I64(a) => a.data.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, _)| i).collect(),
            AnyArray::Bool(a) => a.data.iter().enumerate().filter(|(_, &v)| v).map(|(i, _)| i).collect(),
        };
        let shape = self.shape();
        let ndim = shape.len().max(1);
        let mut out: Vec<Vec<i64>> = vec![Vec::with_capacity(flat.len()); ndim];
        for &f in &flat {
            let mut rem = f;
            for k in (0..shape.len()).rev() {
                out[k].push((rem % shape[k]) as i64);
                rem /= shape[k];
            }
            if shape.is_empty() {
                out[0].push(0);
            }
        }
        out.into_iter().map(|v| { let n = v.len(); Array::new(v, &[n]).expect("1-d") }).collect()
    }
}

impl Array<f64> {
    /// Indices that sort a 1-D array (stable; NaN last like NumPy).
    pub fn argsort_1d(&self) -> Option<Array<i64>> {
        if self.ndim() != 1 {
            return None;
        }
        let mut idx: Vec<i64> = (0..self.size() as i64).collect();
        idx.sort_by(|&i, &j| {
            let (a, b) = (self.data[i as usize], self.data[j as usize]);
            match (a.is_nan(), b.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => a.partial_cmp(&b).expect("non-NaN floats compare"),
            }
        });
        let n = idx.len();
        Some(Array::new(idx, &[n]).expect("1-d"))
    }
}

/// Equal-shape comparison producing a mask; None when shapes differ.
pub fn compare<T: Element, F: Fn(T, T) -> bool>(a: &Array<T>, b: &Array<T>, f: F) -> Option<Array<bool>> {
    if a.shape() != b.shape() {
        return None;
    }
    Some(Array { data: a.data.iter().zip(&b.data).map(|(&x, &y)| f(x, y)).collect(), dims: a.dims.retyped(1) })
}

/// Comparison against a scalar.
pub fn compare_scalar<T: Element, F: Fn(T) -> bool>(a: &Array<T>, f: F) -> Array<bool> {
    Array { data: a.data.iter().map(|&x| f(x)).collect(), dims: a.dims.retyped(1) }
}

/// One axis of a basic index: an integer or a slice already resolved with
/// `slice.indices(len)` semantics.
#[derive(Debug, Clone, Copy)]
pub enum Selector {
    Int(isize),
    Slice { start: isize, stop: isize, step: isize },
}

fn gather<T: Copy>(data: &[T], plan: &[(usize, isize, usize)], strides: &[usize], axis: usize, offset: usize, out: &mut Vec<T>) {
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

impl<T: Element> fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array").field("shape", &self.shape()).field("data", &&self.data[..]).finish()
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
        assert_eq!(Array::<f64>::new(vec![], &[0]).unwrap().max(), None);
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
    fn any_array_structure_and_promotion() {
        let i = AnyArray::I64(Array::new(vec![1i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap());
        assert_eq!((i.dtype_name(), i.itemsize(), i.shape()), ("int64", 8, &[2, 3][..]));
        assert_eq!(i.get(&[1, -1]).unwrap(), Some(Scalar::I(6)));
        assert_eq!(i.transpose().shape(), &[3, 2]);
        assert_eq!(i.to_f64().data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = AnyArray::Bool(Array::new(vec![true, false, true], &[3]).unwrap());
        assert_eq!((b.dtype_name(), b.itemsize()), ("bool", 1));
        assert_eq!(b.dims().strides(), &[1]);
        assert_eq!(b.as_bool().unwrap().count(), 2);
        assert_eq!(b.to_i64().unwrap().data(), &[1, 0, 1]);
        assert!(AnyArray::concatenate(&[&i, &b], 0).unwrap().is_none());
        let f = Array::new(vec![1.0, 5.0, 3.0], &[3]).unwrap();
        let m = compare_scalar(&f, |x| x > 2.0);
        assert_eq!(m.data(), &[false, true, true]);
        assert_eq!(f.compress_flat(m.data()).unwrap().data(), &[5.0, 3.0]);
        assert_eq!(compare(&f, &f, |x, y| x == y).unwrap().count(), 3);
    }

    #[test]
    fn cast_nonzero_argsort() {
        let f = AnyArray::F64(Array::new(vec![1.9, -1.9, 0.0, 3.0], &[2, 2]).unwrap());
        assert_eq!(f.cast("int64").unwrap().as_i64().unwrap().data(), &[1, -1, 0, 3]);
        assert_eq!(f.cast("bool").unwrap().as_bool().unwrap().data(), &[true, true, false, true]);
        assert!(AnyArray::F64(Array::new(vec![f64::NAN], &[1]).unwrap()).cast("int64").is_none());
        let nz = f.nonzero();
        assert_eq!((nz[0].data(), nz[1].data()), (&[0i64, 0, 1][..], &[0i64, 1, 1][..]));
        let a = Array::new(vec![3.0, f64::NAN, 1.0, 2.0], &[4]).unwrap();
        assert_eq!(a.argsort_1d().unwrap().data(), &[2, 3, 0, 1]);
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
