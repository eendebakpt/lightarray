//! The Python-visible array type.
//!
//! `PyArray` is a frozen wrapper around `Array`. Everything that is cheap and
//! common is implemented natively; everything else is delegated to NumPy
//! through the helpers in `lightarray/_fallback.py`, with float64 results
//! wrapped back into `PyArray`.

use crate::array::{compare, compare_scalar, AnyArray, Array, ArrayError, Scalar, Selector};
use crate::dims::MAX_NDIM;
use pyo3::buffer::{PyBuffer, PyUntypedBuffer};
use pyo3::call::PyCallArgs;
use pyo3::exceptions::{PyBufferError, PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMemoryView, PyNotImplemented, PySlice, PyTuple};
use std::ffi::{c_int, c_void};
use std::ptr;

impl From<ArrayError> for PyErr {
    fn from(e: ArrayError) -> PyErr {
        match e {
            ArrayError::IndexOutOfBounds { .. } | ArrayError::TooManyIndices { .. } => {
                PyIndexError::new_err(e.to_string())
            }
            _ => PyValueError::new_err(e.to_string()),
        }
    }
}

/// Python wrapper for `Array`.
///
/// The class is `frozen` so PyO3 skips the borrow flag on every call (5 ns
/// each), and the array lives in an `UnsafeCell` so `__setitem__`, in-place
/// operators and writable buffer views can still mutate the data. All
/// mutation happens while holding the GIL, which serialises it against
/// every other access on the GIL build. On the free-threaded build the
/// guarantees are NumPy's: concurrent writes to one array are a data race
/// the caller must avoid. The shape never changes after construction, and
/// the data `Vec` is never reallocated, so pointers handed out through the
/// buffer protocol and DLPack stay valid for the object's lifetime.
#[pyclass(name = "ndarray", module = "lightarray", frozen, subclass)]
pub struct PyArray {
    inner: std::cell::UnsafeCell<AnyArray>,
    /// Set for strided views (`a[:, 0]`, `a[::2]`, `a.T`); see `Strided`.
    strided: Option<Box<Strided>>,
}

/// A strided view. The kernels only know contiguous buffers, so `inner` is a
/// contiguous cache of the view: every read access refreshes it from the
/// base's memory, every native write is followed by `commit`, which scatters
/// it back, and the buffer protocol exports the base's memory itself with
/// these strides, so NumPy reads and writes the real thing. Refreshing costs
/// one pass over the view's elements per access.
pub struct Strided {
    /// The array owning the memory.
    base: Py<PyAny>,
    /// Address of the view's first element, inside `base`'s buffer.
    ptr: *mut u8,
    /// Byte strides per axis of the view.
    strides: [isize; MAX_NDIM],
}

// SAFETY: see the type docs; mutation is confined to GIL-holding methods and
// NumPy views, matching NumPy's own thread-safety contract.
unsafe impl Send for PyArray {}
unsafe impl Sync for PyArray {}

// The frozen class is shared between threads on free-threaded Python without
// locks; this fails to compile if a future field breaks that.
const _: () = {
    fn assert_thread_safe<T: Send + Sync>() {}
    let _ = assert_thread_safe::<PyArray>;
};

impl PyArray {
    /// A float64 array (the fast path and by far the most common result).
    pub fn new(inner: Array) -> PyArray {
        PyArray::from_any(AnyArray::F64(inner))
    }

    pub fn from_any(inner: AnyArray) -> PyArray {
        PyArray { inner: std::cell::UnsafeCell::new(inner), strided: None }
    }

    /// Shared access to the array, whatever its dtype.
    #[inline]
    pub fn arr(&self) -> &AnyArray {
        if self.strided.is_some() {
            self.refresh();
        }
        // SAFETY: readers and the GIL-serialised writers never overlap on the
        // GIL build; see the type docs for the free-threaded contract.
        unsafe { &*self.inner.get() }
    }

    /// Shape, dtype and size without refreshing a strided view's cache.
    #[inline]
    pub fn meta(&self) -> &AnyArray {
        // SAFETY: as for `arr`.
        unsafe { &*self.inner.get() }
    }

    /// Reload a strided view's cache from the base's memory. Kept out of
    /// line so that `arr()` stays one branch for ordinary arrays, but not
    /// `#[cold]`: for a strided view this is the hot path, and cold functions
    /// are optimised for size.
    #[inline(never)]
    fn refresh(&self) {
        if let Some(s) = &self.strided {
            // SAFETY: `s.ptr`/`s.strides` address elements of `s.base`'s
            // buffer, which never moves; the cache has the view's shape.
            unsafe {
                let cache = &mut *self.inner.get();
                let ndim = cache.ndim();
                cache.gather_from(s.ptr, &s.strides[..ndim]);
            }
        }
    }

    /// Address of the element at a full integer index of a strided view, so
    /// scalar reads and writes skip the cache. None for other arrays and for
    /// partial indices.
    #[inline]
    fn strided_element(&self, idx: &[isize]) -> PyResult<Option<*mut u8>> {
        match &self.strided {
            None => Ok(None),
            Some(s) => self.strided_element_at(s, idx),
        }
    }

    #[cold]
    fn strided_element_at(&self, s: &Strided, idx: &[isize]) -> PyResult<Option<*mut u8>> {
        let shape = self.meta().shape();
        if idx.len() != shape.len() {
            return Ok(None);
        }
        let mut ptr = s.ptr;
        for (axis, (&i, &len)) in idx.iter().zip(shape).enumerate() {
            let k = if i < 0 { i + len as isize } else { i };
            if k < 0 || k >= len as isize {
                return Err(ArrayError::IndexOutOfBounds { index: i, axis, len }.into());
            }
            ptr = ptr.wrapping_offset(k * s.strides[axis]);
        }
        Ok(Some(ptr))
    }

    /// Where the elements really live: first element and element strides
    /// (the base's memory for strided views). Used by the DLPack export.
    pub fn element_layout(&self) -> (*mut u8, [i64; MAX_NDIM]) {
        let meta = self.meta();
        let mut strides = [0i64; MAX_NDIM];
        match &self.strided {
            Some(s) => {
                for k in 0..meta.ndim() {
                    strides[k] = (s.strides[k] / meta.itemsize() as isize) as i64;
                }
                (s.ptr, strides)
            }
            None => {
                for (k, stride) in strides.iter_mut().enumerate().take(meta.ndim()) {
                    *stride = meta.dims().elem_stride(k) as i64;
                }
                (meta.data_ptr() as *mut u8, strides)
            }
        }
    }

    /// Write a strided view's cache back to the base after a native write.
    #[inline]
    pub fn commit(&self) {
        if let Some(s) = &self.strided {
            // SAFETY: as for `refresh`.
            unsafe {
                let cache = &*self.inner.get();
                cache.scatter_to(s.ptr, &s.strides[..cache.ndim()]);
            }
        }
    }

    /// Mutable access to the data. Callers must hold the GIL and must not
    /// change the shape or reallocate the data (views point at it).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn arr_mut(&self) -> &mut AnyArray {
        if self.strided.is_some() {
            self.refresh();
        }
        // SAFETY: as above; only element values are written.
        unsafe { &mut *self.inner.get() }
    }

    /// The float64 array, or None for int64/bool arrays. Every float64
    /// kernel goes through this, so no kernel can see another dtype.
    #[inline]
    pub fn f64(&self) -> Option<&Array<f64>> {
        self.arr().as_f64()
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn f64_mut(&self) -> Option<&mut Array<f64>> {
        self.arr_mut().as_f64_mut()
    }

    pub fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, self)?.into_any())
    }

    /// Copy an object exposing a float64, int64 or bool buffer (NumPy arrays,
    /// memoryviews, `array.array`) into a new array of the same dtype; None
    /// for anything else. 0-d buffers carry a NULL shape that PyBuffer
    /// rejects, so they are read through the object's dtype instead.
    pub fn from_buffer_any(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Option<AnyArray>> {
        let ndim = obj.getattr("ndim").and_then(|n| n.extract::<usize>()).ok();
        if ndim == Some(0) {
            let Ok(dtype) = obj.getattr("dtype") else { return Ok(None) };
            let kind: String = dtype.getattr("kind")?.extract()?;
            let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
            return Ok(match (kind.as_str(), itemsize) {
                ("f", 8) => Some(AnyArray::F64(Array::scalar(obj.extract::<f64>()?))),
                ("i", 8) => Some(AnyArray::I64(Array::scalar(obj.call_method0("__int__")?.extract::<i64>()?))),
                ("b", 1) => Some(AnyArray::Bool(Array::scalar(obj.is_truthy()?))),
                _ => None,
            });
        }
        if ndim.map_or(false, |n| n > crate::dims::MAX_NDIM) {
            return Ok(None); // stays a NumPy array
        }
        if let Ok(buf) = PyBuffer::<f64>::get(obj) {
            let shape: Vec<usize> = buf.shape().to_vec();
            return Ok(Some(AnyArray::F64(Array::new(buf.to_vec(py)?, &shape)?)));
        }
        if let Ok(buf) = PyBuffer::<i64>::get(obj) {
            let shape: Vec<usize> = buf.shape().to_vec();
            return Ok(Some(AnyArray::I64(Array::new(buf.to_vec(py)?, &shape)?)));
        }
        if let Some((mask, shape)) = numpy_bool_buffer(obj) {
            return Ok(Some(AnyArray::Bool(Array::new(mask, &shape)?)));
        }
        Ok(None)
    }

    /// float64 view of things: copy a buffer as float64, coercing NumPy
    /// scalars of any numeric kind when `coerce_scalars` is set.
    pub fn from_f64_buffer(py: Python<'_>, obj: &Bound<'_, PyAny>, coerce_scalars: bool) -> PyResult<Option<Array>> {
        match PyArray::from_buffer_any(py, obj)? {
            Some(AnyArray::F64(a)) => Ok(Some(a)),
            Some(other) if coerce_scalars => Ok(Some(other.to_f64())),
            Some(_) => Ok(None),
            None => {
                let is_0d = obj.getattr("ndim").and_then(|n| n.extract::<usize>()).map_or(false, |n| n == 0);
                if is_0d && coerce_scalars {
                    return Ok(obj.extract::<f64>().ok().map(Array::scalar));
                }
                Ok(None)
            }
        }
    }

    /// Convert a NumPy result back into lightarray where possible
    /// (float64, int64 and bool arrays of at most MAX_NDIM dimensions).
    pub fn wrap_result(py: Python<'_>, obj: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if obj.is_instance_of::<PyArray>() {
            return Ok(obj.unbind());
        }
        if obj.hasattr("__array_interface__")? {
            if let Some(arr) = PyArray::from_buffer_any(py, &obj)? {
                return PyArray::from_any(arr).into_py(py);
            }
        }
        Ok(obj.unbind())
    }
}

/// The float64 array behind a Python object, if it is a float64 lightarray.
#[inline]
pub fn f64_of<'a>(obj: &'a Bound<'_, PyAny>) -> Option<&'a Array<f64>> {
    obj.cast::<PyArray>().ok().and_then(|a| a.get().f64())
}

/// Any lightarray array behind a Python object.
#[inline]
pub fn any_of<'a>(obj: &'a Bound<'_, PyAny>) -> Option<&'a AnyArray> {
    obj.cast::<PyArray>().ok().map(|a| a.get().arr())
}

static NP_FLOAT64: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NP_NDARRAY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NP_INTP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NP_BOOL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `PyArray_Scalar(data, descr, base)` from NumPy's C API: builds a NumPy
/// scalar straight from the bytes of the value. The Python constructors do
/// the same through argument parsing and a temporary 0-d array, which costs
/// 80 ns for `np.float64` and 250 ns for `np.int64`; this takes about 25.
type ScalarFn = unsafe extern "C" fn(*const c_void, *mut ffi::PyObject, *mut ffi::PyObject) -> *mut ffi::PyObject;

struct NumpyScalars {
    scalar: ScalarFn,
    /// `np.dtype` objects (a `PyArray_Descr` each) for the three native types.
    f64_descr: Py<PyAny>,
    i64_descr: Py<PyAny>,
    /// The `np.True_` / `np.False_` singletons.
    true_: Py<PyAny>,
    false_: Py<PyAny>,
}

// SAFETY: the function pointer is immutable and the Py<> handles are only
// used with the GIL (thread attached).
unsafe impl Send for NumpyScalars {}
unsafe impl Sync for NumpyScalars {}

static NP_SCALARS: PyOnceLock<Option<NumpyScalars>> = PyOnceLock::new();

/// Index of `PyArray_Scalar` in NumPy's C API table (`__multiarray_api.h`);
/// the table is append-only, so the index is stable across NumPy versions.
const PYARRAY_SCALAR_INDEX: usize = 60;

/// The fast scalar constructors, or None when the C API is not available or
/// does not behave as expected (then the Python constructors are used).
#[inline]
fn numpy_scalars(py: Python<'_>) -> Option<&'static NumpyScalars> {
    NP_SCALARS.get_or_init(py, || load_numpy_scalars(py)).as_ref()
}

/// Runs once; kept out of line so that the call sites (every reduction and
/// scalar index) stay small.
#[cold]
#[inline(never)]
fn load_numpy_scalars(py: Python<'_>) -> Option<NumpyScalars> {
    {
        {
            let load = || -> PyResult<NumpyScalars> {
                let np = py.import("numpy")?;
                let capsule = py.import("numpy._core._multiarray_umath")?.getattr("_ARRAY_API")?;
                // SAFETY: `_ARRAY_API` is the capsule NumPy publishes for its C API table,
                // an array of function pointers of which entry 60 is PyArray_Scalar.
                let scalar: ScalarFn = unsafe {
                    let table = ffi::PyCapsule_GetPointer(capsule.as_ptr(), ptr::null()) as *const *const c_void;
                    if table.is_null() {
                        return Err(PyValueError::new_err("no NumPy C API table"));
                    }
                    std::mem::transmute::<*const c_void, ScalarFn>(*table.add(PYARRAY_SCALAR_INDEX))
                };
                let dtype = np.getattr("dtype")?;
                let s = NumpyScalars {
                    scalar,
                    f64_descr: dtype.call1(("float64",))?.unbind(),
                    i64_descr: dtype.call1(("int64",))?.unbind(),
                    true_: np.getattr("True_")?.unbind(),
                    false_: np.getattr("False_")?.unbind(),
                };
                // Verify before trusting it: the right types and values must come back.
                let f = s.make_f64(py, -2.5)?;
                let i = s.make_i64(py, -7)?;
                let ok = f.bind(py).is_exact_instance(np.getattr("float64")?.cast::<pyo3::types::PyType>()?)
                    && f.bind(py).extract::<f64>()? == -2.5
                    && i.bind(py).is_exact_instance(np.getattr("int64")?.cast::<pyo3::types::PyType>()?)
                    && i.bind(py).extract::<i64>()? == -7
                    && s.make_i64(py, i64::MAX)?.bind(py).extract::<i64>()? == i64::MAX;
                if !ok {
                    return Err(PyValueError::new_err("PyArray_Scalar does not behave as expected"));
                }
                Ok(s)
            };
            load().ok()
        }
    }
}

impl NumpyScalars {
    #[inline]
    fn make(&self, py: Python<'_>, data: *const c_void, descr: &Py<PyAny>) -> PyResult<Py<PyAny>> {
        // SAFETY: `data` points at a value of the type `descr` describes; the
        // descr is a live `np.dtype` object; the result is a new reference.
        unsafe { Bound::from_owned_ptr_or_err(py, (self.scalar)(data, descr.as_ptr(), ptr::null_mut())).map(Bound::unbind) }
    }
    #[inline]
    fn make_f64(&self, py: Python<'_>, v: f64) -> PyResult<Py<PyAny>> {
        self.make(py, &v as *const f64 as *const c_void, &self.f64_descr)
    }
    #[inline]
    fn make_i64(&self, py: Python<'_>, v: i64) -> PyResult<Py<PyAny>> {
        self.make(py, &v as *const i64 as *const c_void, &self.i64_descr)
    }
}

/// A `numpy.float64` scalar, what NumPy returns from reductions and scalar
/// indexing (it carries `.dtype`, `.astype`, `.round`, ... unlike a Python
/// float).
pub fn np_float(py: Python<'_>, v: f64) -> PyResult<Py<PyAny>> {
    if let Some(s) = numpy_scalars(py) {
        return s.make_f64(py, v);
    }
    let ty = NP_FLOAT64.get_or_try_init(py, || -> PyResult<Py<PyAny>> { Ok(py.import("numpy")?.getattr("float64")?.unbind()) })?;
    Ok(ty.bind(py).call1((v,))?.unbind())
}

/// A `numpy.intp` scalar (what `argmax` returns in NumPy; int64 on every
/// platform lightarray builds for).
pub fn np_int(py: Python<'_>, v: usize) -> PyResult<Py<PyAny>> {
    np_i64(py, v as i64)
}

/// A `numpy.int64` scalar (integer reductions and integer indexing).
pub fn np_i64(py: Python<'_>, v: i64) -> PyResult<Py<PyAny>> {
    if let Some(s) = numpy_scalars(py) {
        return s.make_i64(py, v);
    }
    let ty = NP_INTP.get_or_try_init(py, || -> PyResult<Py<PyAny>> { Ok(py.import("numpy")?.getattr("int64")?.unbind()) })?;
    Ok(ty.bind(py).call1((v,))?.unbind())
}

/// A `numpy.bool_` scalar (what `any`/`all`/predicates return in NumPy):
/// one of the two singletons.
pub fn np_bool(py: Python<'_>, v: bool) -> PyResult<Py<PyAny>> {
    if let Some(s) = numpy_scalars(py) {
        return Ok(if v { s.true_.clone_ref(py) } else { s.false_.clone_ref(py) });
    }
    let ty = NP_BOOL.get_or_try_init(py, || -> PyResult<Py<PyAny>> { Ok(py.import("numpy")?.getattr("bool_")?.unbind()) })?;
    Ok(ty.bind(py).call1((v,))?.unbind())
}

fn not_implemented(py: Python<'_>) -> Py<PyAny> {
    PyNotImplemented::get(py).to_owned().into_any().unbind()
}

/// Classify a binary-operator operand: a float64 lightarray (the fast path),
/// a lightarray of another dtype, a Python or NumPy scalar, or something for
/// NumPy (or the other operand's reflected method) to handle.
pub enum Operand<'a> {
    F64(&'a Array<f64>),
    Alt(&'a AnyArray),
    Float(f64),
    Int(i64),
    Bool(bool),
    Other,
}

impl Operand<'_> {
    /// The operand as a float64 scalar, if it is a scalar at all.
    #[inline]
    pub fn scalar(&self) -> Option<f64> {
        match *self {
            Operand::Float(v) => Some(v),
            Operand::Int(v) => Some(v as f64),
            Operand::Bool(v) => Some(v as u8 as f64),
            _ => None,
        }
    }
}

pub fn classify<'a>(obj: &'a Bound<'_, PyAny>) -> Operand<'a> {
    if let Ok(a) = obj.cast::<PyArray>() {
        return match a.get().arr() {
            AnyArray::F64(f) => Operand::F64(f),
            other => Operand::Alt(other),
        };
    }
    if obj.is_instance_of::<PyFloat>() {
        if let Ok(v) = obj.extract::<f64>() {
            return Operand::Float(v);
        }
    }
    if obj.is_instance_of::<PyBool>() {
        return Operand::Bool(obj.is_truthy().unwrap_or(false));
    }
    if obj.is_instance_of::<PyInt>() {
        return match obj.extract::<i64>() {
            Ok(v) => Operand::Int(v),
            Err(_) => obj.extract::<f64>().map_or(Operand::Other, Operand::Float),
        };
    }
    // NumPy scalars and other numbers; arrays expose __len__ and go to NumPy.
    if !obj.hasattr("__len__").unwrap_or(true) {
        if obj.hasattr("__index__").unwrap_or(false) {
            if let Ok(v) = obj.extract::<i64>() {
                return Operand::Int(v);
            }
        }
        if let Ok(v) = obj.extract::<f64>() {
            return Operand::Float(v);
        }
    }
    Operand::Other
}

/// Does NumPy know how to combine with this object (arrays, lists, anything
/// with an array protocol)? If so the fallback handles it; otherwise we
/// return NotImplemented so Python tries the other operand.
fn numpy_handles(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyList>()
        || obj.is_instance_of::<PyTuple>()
        || obj.is_instance_of::<pyo3::types::PyComplex>()
        || obj.hasattr("__array_interface__").unwrap_or(false)
        || obj.hasattr("__array__").unwrap_or(false)
}

/// Apply `f` element-wise for `self OP other`, or `other OP self` when
/// `reflected` is set. float64 operands run natively; int64 and bool
/// arrays promote to float64 when a float is involved and have native
/// integer add/subtract/multiply; NumPy arrays and other array-likes go
/// through the NumPy ufunc `name`.
#[inline]
fn binary_op<F: Fn(f64, f64) -> f64>(
    slf: &Bound<'_, PyArray>,
    other: &Bound<'_, PyAny>,
    reflected: bool,
    name: &str,
    f: F,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let operand = classify(other);
    let AnyArray::F64(a) = slf.get().arr() else {
        return alt_binary(slf, other, operand, reflected, name, f);
    };
    let result = match operand {
        Operand::F64(b) => {
            if reflected {
                b.zip_map(a, &f)?
            } else {
                a.zip_map(b, &f)?
            }
        }
        Operand::Alt(b) => {
            let b = b.to_f64();
            if reflected {
                b.zip_map(a, &f)?
            } else {
                a.zip_map(&b, &f)?
            }
        }
        Operand::Other if numpy_handles(other) => return numpy_binary(py, slf, other, reflected, name),
        Operand::Other => return Ok(not_implemented(py)),
        scalar => {
            let s = scalar.scalar().expect("remaining variants are scalars");
            if reflected {
                a.map(|x| f(s, x))
            } else {
                a.map(|x| f(x, s))
            }
        }
    };
    PyArray::new(result).into_py(py)
}

fn numpy_binary(py: Python<'_>, slf: &Bound<'_, PyArray>, other: &Bound<'_, PyAny>, reflected: bool, name: &str) -> PyResult<Py<PyAny>> {
    if reflected {
        fallback(py, "call", (name, other.clone(), slf.clone()), None)
    } else {
        fallback(py, "call", (name, slf.clone(), other.clone()), None)
    }
}

/// Binary operation whose left operand is an int64 or bool array.
#[inline(never)]
fn alt_binary<F: Fn(f64, f64) -> f64>(
    slf: &Bound<'_, PyArray>,
    other: &Bound<'_, PyAny>,
    operand: Operand<'_>,
    reflected: bool,
    name: &str,
    f: F,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let me = slf.get().arr();
    if matches!(operand, Operand::Other) {
        return if numpy_handles(other) { numpy_binary(py, slf, other, reflected, name) } else { Ok(not_implemented(py)) };
    }
    // A float on the other side, or true division: the result is float64.
    if name == "divide" || matches!(operand, Operand::Float(_) | Operand::F64(_)) {
        let a = me.to_f64();
        let result = match operand {
            Operand::F64(b) => {
                if reflected {
                    b.zip_map(&a, &f)?
                } else {
                    a.zip_map(b, &f)?
                }
            }
            Operand::Alt(b) => {
                let b = b.to_f64();
                if reflected {
                    b.zip_map(&a, &f)?
                } else {
                    a.zip_map(&b, &f)?
                }
            }
            scalar => {
                let s = scalar.scalar().expect("remaining variants are scalars");
                if reflected {
                    a.map(|x| f(s, x))
                } else {
                    a.map(|x| f(x, s))
                }
            }
        };
        return PyArray::new(result).into_py(py);
    }
    // Integer arithmetic that cannot fail (wrapping, like NumPy's int64).
    let int_op: Option<fn(i64, i64) -> i64> = match name {
        "add" => Some(i64::wrapping_add),
        "subtract" => Some(i64::wrapping_sub),
        "multiply" => Some(i64::wrapping_mul),
        _ => None,
    };
    if let (Some(op), AnyArray::I64(a)) = (int_op, me) {
        let result = match operand {
            Operand::Int(i) => Some(a.map_int(|x| if reflected { op(i, x) } else { op(x, i) })),
            Operand::Alt(AnyArray::I64(b)) => {
                if reflected {
                    b.zip_int(a, op)
                } else {
                    a.zip_int(b, op)
                }
            }
            _ => None,
        };
        if let Some(r) = result {
            return PyArray::from_any(AnyArray::I64(r)).into_py(py);
        }
    }
    // Everything else (bool arithmetic, integer division and powers,
    // broadcasting between int arrays) follows NumPy exactly.
    numpy_binary(py, slf, other, reflected, name)
}

/// Round half to even (NumPy's `rint`/`round`).
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r }
}

/// The array methods with the uniform `(*args, **kwargs)` signature, for the
/// module-level functions (`np.sum(a)` is `a.sum()`): calling the Rust method
/// directly saves the Python-level method call, about 170 ns. None for other names.
pub fn call_native_method<'py>(
    slf: &Bound<'py, PyArray>,
    name: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> Option<PyResult<Py<PyAny>>> {
    Some(match name {
        "sum" => PyArray::sum(slf, args, kwargs),
        "prod" => PyArray::prod(slf, args, kwargs),
        "mean" => PyArray::mean(slf, args, kwargs),
        "max" => PyArray::max(slf, args, kwargs),
        "min" => PyArray::min(slf, args, kwargs),
        "var" => PyArray::var(slf, args, kwargs),
        "std" => PyArray::std(slf, args, kwargs),
        "argmax" => PyArray::argmax(slf, args, kwargs),
        "argmin" => PyArray::argmin(slf, args, kwargs),
        "any" => PyArray::any(slf, args, kwargs),
        "all" => PyArray::all(slf, args, kwargs),
        "cumsum" => PyArray::cumsum(slf, args, kwargs),
        "cumprod" => PyArray::cumprod(slf, args, kwargs),
        "clip" => PyArray::clip(slf, args, kwargs),
        "round" => PyArray::round(slf, args, kwargs),
        "reshape" => PyArray::reshape(slf, args, kwargs),
        "copy" => PyArray::copy(slf, args, kwargs),
        "ravel" => PyArray::ravel(slf, args, kwargs),
        "transpose" => PyArray::transpose(slf, args, kwargs),
        "dot" if args.len() == 1 && kwargs.map_or(true, |k| k.is_empty()) => match args.get_item(0) {
            Ok(other) => PyArray::dot(slf, &other),
            Err(e) => Err(e),
        },
        _ => return None,
    })
}

/// `add(x1, x2)` and the other operator functions: the operator itself,
/// called directly (going through the Python-level `__add__` costs 170 ns).
/// `name` is the ufunc's name. None when neither operand is a lightarray
/// array or the operator does not handle the pair.
pub fn operator_function(name: &str, x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    let f: fn(f64, f64) -> f64 = match name {
        "add" => |a, b| a + b,
        "subtract" => |a, b| a - b,
        "multiply" => |a, b| a * b,
        "divide" | "true_divide" => |a, b| a / b,
        "floor_divide" => float_floor_div,
        "remainder" | "mod" => float_mod,
        "power" => float_pow,
        _ => return Ok(None),
    };
    let ufunc = match name {
        "true_divide" => "divide",
        "mod" => "remainder",
        other => other,
    };
    let result = if let Ok(a) = x1.cast::<PyArray>() {
        binary_op(a, x2, false, ufunc, f)?
    } else if let Ok(b) = x2.cast::<PyArray>() {
        binary_op(b, x1, true, ufunc, f)?
    } else {
        return Ok(None);
    };
    if result.bind(x1.py()).is(&PyNotImplemented::get(x1.py())) {
        return Ok(None);
    }
    Ok(Some(result))
}

/// Module-level binary function (`maximum`, `arctan2`, ...): native for
/// array/array of equal shape and array/scalar either way, NumPy otherwise.
pub fn binary_native(
    py: Python<'_>,
    name: &str,
    x1: &Bound<'_, PyAny>,
    x2: &Bound<'_, PyAny>,
    f: fn(f64, f64) -> f64,
) -> PyResult<Py<PyAny>> {
    let result = match (classify(x1), classify(x2)) {
        (Operand::F64(a), Operand::F64(b)) => a.zip_map(b, f)?,
        (Operand::F64(a), Operand::Float(s)) => a.map(|x| f(x, s)),
        (Operand::Float(s), Operand::F64(b)) => b.map(|x| f(s, x)),
        (Operand::F64(a), Operand::Int(s)) => a.map(|x| f(x, s as f64)),
        (Operand::Int(s), Operand::F64(b)) => b.map(|x| f(s as f64, x)),
        // Python floats only: NumPy scalars keep their own type (np.float32, ...) in NumPy
        (Operand::Float(a), Operand::Float(b)) if x1.is_exact_instance_of::<PyFloat>() && x2.is_exact_instance_of::<PyFloat>() => {
            return Ok(f(a, b).into_pyobject(py)?.into_any().unbind())
        }
        _ => return fallback(py, "call", (name, x1.clone(), x2.clone()), None),
    };
    PyArray::new(result).into_py(py)
}



/// Read a contiguous NumPy bool array (format '?') as a Vec<bool> plus its
/// shape. bool is not a PyBuffer element type in PyO3, so this goes through
/// the untyped buffer. Non-buffers and other formats give None.
pub fn bool_mask(obj: &Bound<'_, PyAny>) -> Option<(Vec<bool>, Vec<usize>)> {
    if let Some(any) = any_of(obj) {
        return any.as_bool().map(|m| (m.data().to_vec(), m.shape().to_vec()));
    }
    numpy_bool_buffer(obj)
}

/// Read a contiguous NumPy bool buffer (format '?').
fn numpy_bool_buffer(obj: &Bound<'_, PyAny>) -> Option<(Vec<bool>, Vec<usize>)> {
    // SAFETY: `obj` is a valid object pointer for the duration of the call.
    if unsafe { ffi::PyObject_CheckBuffer(obj.as_ptr()) } != 1 {
        return None;
    }
    let raw = PyUntypedBuffer::get(obj).ok()?;
    if !raw.is_c_contiguous() || raw.format() != c"?" || raw.item_size() != 1 || raw.dimensions() == 0 {
        return None;
    }
    let shape = raw.shape().to_vec();
    // SAFETY: contiguous buffer of `item_count` one-byte items, alive while `raw` is.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(raw.buf_ptr() as *const u8, raw.item_count()) };
    Some((bytes.iter().map(|&b| b != 0).collect(), shape))
}

/// `where(cond, x, y)`: native when `cond` is a contiguous boolean buffer of
/// the operands' shape and `x`, `y` are arrays or scalars.
pub fn where_native(py: Python<'_>, cond: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let native = || -> PyResult<Option<Array>> {
        let Some((mask, shape)) = bool_mask(cond) else { return Ok(None) };
        let (ox, oy) = (classify(x), classify(y));
        // two integer scalars give an integer result in NumPy
        if !matches!(ox, Operand::F64(_) | Operand::Float(_)) && !matches!(oy, Operand::F64(_) | Operand::Float(_)) {
            return Ok(None);
        }
        let as_array = |o: &Operand<'_>| -> PyResult<Option<Array>> {
            Ok(match o {
                Operand::F64(a) if a.shape() == shape.as_slice() => Some((*a).clone()),
                Operand::Float(_) | Operand::Int(_) => Some(Array::filled(&shape, o.scalar().expect("scalar"))?),
                _ => None,
            })
        };
        let (Some(xa), Some(ya)) = (as_array(&ox)?, as_array(&oy)?) else { return Ok(None) };
        Ok(Some(Array::select_where(&mask, &xa, &ya)?))
    };
    match native()? {
        Some(arr) => PyArray::new(arr).into_py(py),
        None => fallback(py, "call", ("where", cond.clone(), x.clone(), y.clone()), None),
    }
}

/// NumPy's float `floor_divide` (`npy_divmod`), which follows Python rather
/// than the Array API: computed from `fmod`, so `inf // 2` is NaN and
/// `1.5 // -inf` is -1.0, and results are exact where `(a / b).floor()` rounds.
fn float_floor_div(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return a / b;
    }
    let rem = a % b;
    let mut div = (a - rem) / b;
    if rem != 0.0 && ((b < 0.0) != (rem < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let floored = div.floor();
        if div - floored > 0.5 {
            floored + 1.0
        } else {
            floored
        }
    } else {
        (0.0f64).copysign(a / b)
    }
}

/// Python/NumPy remainder: result has the sign of the divisor.
fn float_mod(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r == 0.0 {
        // a zero remainder takes the sign of the divisor (IEEE/Array API)
        return if b.is_nan() { r } else { 0.0f64.copysign(b) };
    }
    if (r < 0.0) != (b < 0.0) {
        r + b
    } else {
        r
    }
}

fn float_pow(a: f64, b: f64) -> f64 {
    if b == 2.0 {
        a * a
    } else {
        a.powf(b)
    }
}

/// Call `lightarray._fallback.<name>(*args, **kwargs)`.
pub fn fallback<'py>(
    py: Python<'py>,
    name: &str,
    args: impl PyCallArgs<'py>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let module = py.import("lightarray._fallback")?;
    module.getattr(name)?.call(args, kwargs).map(|r| r.unbind())
}

/// How a reduction was called: no axis, one integer axis, or something for NumPy.
enum Axis {
    None,
    Int(isize),
    Other,
}

fn parse_axis(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Axis> {
    let axis_obj = match (args.len(), kwargs) {
        (0, None) => return Ok(Axis::None),
        (0, Some(k)) if k.is_empty() => return Ok(Axis::None),
        (1, None) => args.get_item(0)?,
        (1, Some(k)) if k.is_empty() => args.get_item(0)?,
        (0, Some(k)) if k.len() == 1 => match k.get_item("axis")? {
            Some(a) => a,
            None => return Ok(Axis::Other),
        },
        _ => return Ok(Axis::Other),
    };
    if axis_obj.is_none() {
        return Ok(Axis::None);
    }
    if axis_obj.is_instance_of::<PyInt>() {
        return Ok(Axis::Int(axis_obj.extract()?));
    }
    Ok(Axis::Other)
}

/// Reduction that runs natively for the whole array or along one integer
/// axis, and falls back to NumPy for anything else (`keepdims`, `out`,
/// tuple axes, `dtype`, ...). Takes raw `*args/**kwargs` so the common
/// no-argument call pays no argument parsing.
fn reduce<'py>(
    slf: &Bound<'py, PyArray>,
    name: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    native: impl FnOnce(&Array) -> PyResult<f64>,
    along_axis: impl FnOnce(&Array, isize) -> PyResult<Array>,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let Some(a) = slf.get().f64() else {
        return alt_reduce(slf, name, args, kwargs);
    };
    match parse_axis(args, kwargs)? {
        Axis::None => np_float(py, native(a)?),
        Axis::Int(axis) => PyArray::new(along_axis(a, axis)?).into_py(py),
        Axis::Other => call_method_fallback(slf, name, args, kwargs),
    }
}

/// Reductions of int64 and bool arrays: the whole-array sum, extrema and
/// mean natively, everything else through NumPy.
#[inline(never)]
fn alt_reduce<'py>(
    slf: &Bound<'py, PyArray>,
    name: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    if matches!(parse_axis(args, kwargs)?, Axis::None) {
        match (slf.get().arr(), name) {
            (AnyArray::I64(a), "sum") => return np_i64(py, a.sum()),
            (AnyArray::Bool(a), "sum") => return np_i64(py, a.count() as i64),
            (AnyArray::I64(a), "max") => return np_i64(py, a.max().ok_or_else(|| no_identity("maximum"))?),
            (AnyArray::I64(a), "min") => return np_i64(py, a.min().ok_or_else(|| no_identity("minimum"))?),
            (any, "mean") if any.size() > 0 => return np_float(py, any.to_f64().mean()),
            _ => {}
        }
    }
    call_method_fallback(slf, name, args, kwargs)
}

/// Is the call bare, or only `order` in {C, A, K} (positional or keyword),
/// which is the native contiguous layout anyway?
fn c_order_only(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let order = match (args.len(), kwargs) {
        (0, None) => return Ok(true),
        (0, Some(k)) if k.is_empty() => return Ok(true),
        (1, None) => args.get_item(0)?,
        (1, Some(k)) if k.is_empty() => args.get_item(0)?,
        (0, Some(k)) if k.len() == 1 => match k.get_item("order")? {
            Some(o) => o,
            None => return Ok(false),
        },
        _ => return Ok(false),
    };
    Ok(order.extract::<&str>().map_or(false, |o| matches!(o, "C" | "A" | "K" | "c" | "a" | "k")))
}

/// Method with no native arguments: native when called bare, NumPy otherwise.
fn bare_or_fallback<'py>(
    slf: &Bound<'py, PyArray>,
    name: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    native: impl FnOnce(&Array) -> PyResult<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let bare = args.is_empty() && kwargs.map_or(true, |k| k.is_empty());
    if bare {
        match (slf.get().arr(), name) {
            (AnyArray::F64(a), _) => return native(a),
            (AnyArray::Bool(a), "any") => return np_bool(slf.py(), a.data().iter().any(|&b| b)),
            (AnyArray::Bool(a), "all") => return np_bool(slf.py(), a.data().iter().all(|&b| b)),
            (AnyArray::I64(a), "any") => return np_bool(slf.py(), a.data().iter().any(|&v| v != 0)),
            (AnyArray::I64(a), "all") => return np_bool(slf.py(), a.data().iter().all(|&v| v != 0)),
            _ => {}
        }
    }
    call_method_fallback(slf, name, args, kwargs)
}

fn no_identity(name: &str) -> PyErr {
    PyValueError::new_err(format!("zero-size array to reduction operation {name} which has no identity"))
}

fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}

fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
}

/// `lightarray._fallback.call_method(self, name, *args, **kwargs)`.
fn call_method_fallback<'py>(
    slf: &Bound<'py, PyArray>,
    name: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let mut full: Vec<Bound<'py, PyAny>> = Vec::with_capacity(args.len() + 2);
    full.push(slf.clone().into_any());
    full.push(name.into_pyobject(py)?.into_any());
    full.extend(args.iter());
    fallback(py, "call_method", PyTuple::new(py, full)?, kwargs)
}

// ---- the class ------------------------------------------------------------

#[pymethods]
impl PyArray {
    /// `ndarray(shape, dtype=None, buffer=None, offset=0, strides=None, order=None)`:
    /// NumPy's low-level constructor, for float64 without a buffer (zeros)
    /// or with a float64 buffer (copied). Other dtypes cannot be lightarray
    /// arrays; use `numpy.ndarray` for those.
    #[new]
    #[pyo3(signature = (shape, dtype=None, buffer=None, offset=0, strides=None, order=None))]
    fn py_new(
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        dtype: Option<&Bound<'_, PyAny>>,
        buffer: Option<&Bound<'_, PyAny>>,
        offset: usize,
        strides: Option<&Bound<'_, PyAny>>,
        order: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let _ = order;
        let (kind, itemsize) = match dtype {
            Some(d) if !d.is_none() => {
                let dt = py.import("numpy")?.getattr("dtype")?.call1((d,))?;
                (dt.getattr("kind")?.extract::<String>()?, dt.getattr("itemsize")?.extract::<usize>()?)
            }
            _ => ("f".to_string(), 8),
        };
        let unsupported = || {
            PyTypeError::new_err("lightarray.ndarray holds contiguous float64, int64 or bool data; use numpy.ndarray for other dtypes or strides")
        };
        if strides.map_or(false, |s| !s.is_none()) {
            return Err(unsupported());
        }
        let shape = extract_shape(shape)?;
        match (kind.as_str(), itemsize, buffer) {
            ("f", 8, None) => Ok(PyArray::new(Array::filled(&shape, 0.0)?)),
            ("i", 8, None) => Ok(PyArray::from_any(AnyArray::I64(Array::filled(&shape, 0i64)?))),
            ("b", 1, None) => Ok(PyArray::from_any(AnyArray::Bool(Array::filled(&shape, false)?))),
            ("f", 8, Some(buf)) => {
                let np = py.import("numpy")?;
                let kw = PyDict::new(py);
                kw.set_item("dtype", "float64")?;
                kw.set_item("offset", offset)?;
                let flat = np.getattr("frombuffer")?.call((buf,), Some(&kw))?;
                let n: usize = shape.iter().product();
                let taken = flat.get_item(PySlice::new(py, 0, n as isize, 1))?;
                let arr = PyArray::from_f64_buffer(py, &taken, true)?
                    .ok_or_else(|| PyTypeError::new_err("buffer must expose float64 data"))?;
                Ok(PyArray::new(arr.reshape(&shape)?))
            }
            _ => Err(unsupported()),
        }
    }

    // ---- attributes ----

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.meta().shape())
    }

    /// `a.shape = new_shape` reshapes in place, as in NumPy. Buffer views
    /// taken earlier keep the shape they were created with.
    #[setter]
    fn set_shape(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.strided.is_some() {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "Incompatible shape for in-place modification. Use `.reshape()` to make a copy with the desired shape.",
            ));
        }
        let size = self.meta().size();
        let raw: Vec<isize> = match value.extract::<isize>() {
            Ok(n) => vec![n],
            Err(_) => value.extract()?,
        };
        let known: usize = raw.iter().filter(|&&d| d >= 0).map(|&d| d as usize).product();
        let mut shape = Vec::with_capacity(raw.len());
        for &d in &raw {
            if d < 0 {
                if raw.iter().filter(|&&x| x < 0).count() > 1 || known == 0 || size % known != 0 {
                    return Err(PyValueError::new_err(format!("cannot reshape array of size {size} into shape {raw:?}")));
                }
                shape.push(size / known);
            } else {
                shape.push(d as usize);
            }
        }
        Ok(self.arr_mut().set_shape(&shape)?)
    }

    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        match &self.strided {
            Some(v) => PyTuple::new(py, &v.strides[..self.meta().ndim()]),
            None => PyTuple::new(py, self.meta().dims().strides()),
        }
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.meta().ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.meta().size()
    }

    #[getter]
    fn itemsize(&self) -> usize {
        self.meta().itemsize()
    }

    /// `numpy.ndarray`, so that `isinstance(a, numpy.ndarray)` is True.
    /// `isinstance` first looks at the real type (still `lightarray.ndarray`,
    /// which is what `type(a)`, NumPy's C code and protocol dispatch see) and
    /// then at `__class__`, the way mocks and proxies pass such checks.
    /// Library code that gates on `isinstance(x, np.ndarray)` takes its array
    /// branch, where lightarray offers ndarray's attributes and methods.
    #[getter(__class__)]
    fn class_for_isinstance(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ty = NP_NDARRAY.get_or_try_init(py, || -> PyResult<Py<PyAny>> { Ok(py.import("numpy")?.getattr("ndarray")?.unbind()) })?;
        Ok(ty.clone_ref(py))
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.meta().size() * self.meta().itemsize()
    }

    /// Makes NumPy arrays and scalars defer to our reflected operators, so
    /// `np_array + la_array` comes back as lightarray.
    #[classattr]
    #[pyo3(name = "__array_priority__")]
    fn array_priority() -> f64 {
        1000.0
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("numpy")?.getattr("dtype")?.call1((self.meta().dtype_name(),))
    }

    // ---- conversions ----

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let np = slf.py().import("numpy")?;
        np.call_method1("asarray", (slf,))?.repr()?.extract()
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let np = slf.py().import("numpy")?;
        np.call_method1("asarray", (slf,))?.str()?.extract()
    }

    fn __len__(&self) -> PyResult<usize> {
        match self.meta().shape().first() {
            Some(&n) => Ok(n),
            None => Err(PyTypeError::new_err("len() of unsized object")),
        }
    }

    fn __bool__(&self) -> PyResult<bool> {
        match self.arr().item() {
            Some(Scalar::F(v)) => Ok(v != 0.0),
            Some(Scalar::I(v)) => Ok(v != 0),
            Some(Scalar::B(v)) => Ok(v),
            None => Err(PyValueError::new_err(
                "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()",
            )),
        }
    }

    fn __float__(&self) -> PyResult<f64> {
        match self.arr().item() {
            Some(v) => Ok(v.as_f64()),
            None => Err(PyTypeError::new_err("only length-1 arrays can be converted to Python scalars")),
        }
    }

    fn __int__(&self) -> PyResult<i64> {
        match self.arr().item() {
            Some(Scalar::F(v)) => Ok(v as i64),
            Some(Scalar::I(v)) => Ok(v),
            Some(Scalar::B(v)) => Ok(v as i64),
            None => Err(PyTypeError::new_err("only length-1 arrays can be converted to Python scalars")),
        }
    }

    /// NumPy's `__array__` protocol. NumPy normally takes the zero-copy
    /// buffer-protocol route first; this exists for libraries that call
    /// `__array__` directly. `copy=False` is honoured (the view is writable).
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        slf: &Bound<'py, Self>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let np = py.import("numpy")?;
        let view = PyMemoryView::from(slf.as_any())?;
        let kw = PyDict::new(py);
        if let Some(d) = dtype {
            kw.set_item("dtype", d)?;
        }
        if let Some(c) = copy {
            kw.set_item("copy", c)?;
        }
        np.getattr("asarray")?.call((view,), Some(&kw))
    }

    /// Buffer protocol: exposes the data writable and zero-copy, so
    /// `np.asarray(a)` is a view and writes through it (including NumPy's
    /// in-place methods used by the fallbacks) change this array. The data,
    /// shape and strides live in this object and are never reallocated, so
    /// the pointers stay valid as long as the view holds a reference to us.
    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view is null"));
        }
        let arr = slf.get().meta();
        let ndim = arr.ndim();
        let itemsize = arr.itemsize();
        let strided = slf.get().strided.as_deref();
        if strided.is_some() && (flags & ffi::PyBUF_STRIDES) != ffi::PyBUF_STRIDES {
            return Err(PyBufferError::new_err("ndarray is not C-contiguous"));
        }
        if strided.is_some() && (flags & ffi::PyBUF_C_CONTIGUOUS) == ffi::PyBUF_C_CONTIGUOUS {
            return Err(PyBufferError::new_err("ndarray is not C-contiguous"));
        }
        // SAFETY: `view` is a valid Py_buffer; the pointers we store refer to
        // memory owned by `slf` (or its base, which it keeps alive), and
        // `view.obj` keeps `slf` alive.
        unsafe {
            (*view).obj = slf.clone().into_any().into_ptr();
            (*view).buf = match strided {
                Some(v) => v.ptr as *mut c_void,
                None => arr.data_ptr() as *mut c_void,
            };
            (*view).len = (arr.size() * itemsize) as isize;
            (*view).readonly = 0;
            (*view).itemsize = itemsize as isize;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                arr.format().as_ptr() as *mut _
            } else {
                ptr::null_mut()
            };
            // Without PyBUF_ND the consumer expects a flat contiguous buffer
            // (shape NULL, ndim 1), as PyBuffer_FillInfo provides.
            let nd_requested = (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND;
            (*view).ndim = if nd_requested { ndim as c_int } else { 1 };
            (*view).shape = if ndim > 0 && nd_requested {
                arr.dims().shape_ptr() as *mut ffi::Py_ssize_t
            } else {
                ptr::null_mut()
            };
            (*view).strides = if let (Some(v), true) = (strided, ndim > 0) {
                v.strides.as_ptr() as *mut ffi::Py_ssize_t
            } else if ndim > 0 && (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                arr.dims().strides_ptr() as *mut ffi::Py_ssize_t
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = ptr::null_mut();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // Nothing to free: format is a static string and shape/strides live in self.
    }

    // ---- arithmetic ----

    fn __add__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, false, "add", |a, b| a + b)
    }
    fn __radd__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, true, "add", |a, b| a + b)
    }
    fn __sub__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, false, "subtract", |a, b| a - b)
    }
    fn __rsub__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, true, "subtract", |a, b| a - b)
    }
    fn __mul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, false, "multiply", |a, b| a * b)
    }
    fn __rmul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, true, "multiply", |a, b| a * b)
    }
    fn __truediv__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, false, "divide", |a, b| a / b)
    }
    fn __rtruediv__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, true, "divide", |a, b| a / b)
    }
    fn __floordiv__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, false, "floor_divide", float_floor_div)
    }
    fn __rfloordiv__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, true, "floor_divide", float_floor_div)
    }
    fn __mod__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, false, "remainder", float_mod)
    }
    fn __rmod__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        binary_op(slf, other, true, "remainder", float_mod)
    }
    fn __pow__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err("pow() with modulus is not supported"));
        }
        binary_op(slf, other, false, "power", float_pow)
    }
    fn __rpow__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err("pow() with modulus is not supported"));
        }
        binary_op(slf, other, true, "power", float_pow)
    }
    fn __matmul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("matmul", slf.clone(), other.clone()), None)
    }
    fn __rmatmul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("matmul", other.clone(), slf.clone()), None)
    }

    /// Comparisons return boolean NumPy arrays (lightarray has no bool dtype
    /// yet). Array/array of equal shape and array/scalar are computed in Rust
    /// and handed to NumPy as a writable bytearray; other operands go
    /// through NumPy.
    fn __richcmp__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        macro_rules! cmp_fn {
            ($t:ty) => {
                match op {
                    CompareOp::Lt => (|x: $t, y: $t| x < y) as fn($t, $t) -> bool,
                    CompareOp::Le => |x: $t, y: $t| x <= y,
                    CompareOp::Eq => |x: $t, y: $t| x == y,
                    CompareOp::Ne => |x: $t, y: $t| x != y,
                    CompareOp::Gt => |x: $t, y: $t| x > y,
                    CompareOp::Ge => |x: $t, y: $t| x >= y,
                }
            };
        }
        let (fc, ic, bc) = (cmp_fn!(f64), cmp_fn!(i64), cmp_fn!(bool));
        let me = slf.get().arr();
        let operand = classify(other);
        let equality = matches!(op, CompareOp::Eq | CompareOp::Ne);
        let mask: Option<Array<bool>> = match (me, &operand) {
            (AnyArray::F64(a), Operand::F64(b)) => compare(a, b, fc),
            (AnyArray::F64(a), Operand::Alt(AnyArray::I64(_))) => {
                let Operand::Alt(b) = &operand else { unreachable!() };
                compare(a, &b.to_f64(), fc)
            }
            (AnyArray::F64(a), Operand::Float(_) | Operand::Int(_) | Operand::Bool(_)) => {
                let s = operand.scalar().expect("scalar");
                Some(compare_scalar(a, |x| fc(x, s)))
            }
            (AnyArray::I64(a), Operand::Int(i)) => Some(compare_scalar(a, |x| ic(x, *i))),
            (AnyArray::I64(a), Operand::Alt(AnyArray::I64(b))) => compare(a, b, ic),
            (AnyArray::I64(_), Operand::Float(s)) => Some(compare_scalar(&me.to_f64(), |x| fc(x, *s))),
            (AnyArray::I64(_), Operand::F64(b)) => compare(&me.to_f64(), b, fc),
            (AnyArray::Bool(a), Operand::Alt(AnyArray::Bool(b))) if equality => compare(a, b, bc),
            (AnyArray::Bool(a), Operand::Bool(v)) if equality => Some(compare_scalar(a, |x| bc(x, *v))),
            _ => None,
        };
        if let Some(mask) = mask {
            return PyArray::from_any(AnyArray::Bool(mask)).into_py(py);
        }
        if !matches!(operand, Operand::Other) || numpy_handles(other) {
            let name = match op {
                CompareOp::Lt => "less",
                CompareOp::Le => "less_equal",
                CompareOp::Eq => "equal",
                CompareOp::Ne => "not_equal",
                CompareOp::Gt => "greater",
                CompareOp::Ge => "greater_equal",
            };
            return fallback(py, "call", (name, slf.clone(), other.clone()), None);
        }
        Ok(not_implemented(py))
    }

    // ---- DLPack ----

    /// DLPack export: versioned (protocol 1.0) when the consumer passes
    /// `max_version`, the legacy unversioned capsule otherwise.
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        slf: &Bound<'_, Self>,
        stream: Option<&Bound<'_, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if stream.map_or(false, |s| !s.is_none()) {
            return Err(PyValueError::new_err("stream is not supported for CPU arrays"));
        }
        if let Some(dev) = dl_device {
            if dev != crate::dlpack::device() {
                return Err(PyBufferError::new_err("lightarray arrays live on the CPU; cannot export to another device"));
            }
        }
        // copy=True: export a fresh copy; the capsule keeps it alive.
        let owner = if copy == Some(true) {
            Py::new(slf.py(), PyArray::from_any(slf.get().arr().clone()))?.into_bound(slf.py())
        } else {
            slf.clone()
        };
        match max_version {
            Some((major, _)) if major >= 1 => crate::dlpack::export(&owner),
            _ => crate::dlpack::export_legacy(&owner),
        }
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        crate::dlpack::device()
    }

    // ---- bitwise operators: NumPy decides (TypeError for float arrays, like NumPy) ----

    fn __and__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        bitwise(slf, other, false, "bitwise_and", |x, y| x & y, |x, y| x & y)
    }
    fn __rand__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        bitwise(slf, other, true, "bitwise_and", |x, y| x & y, |x, y| x & y)
    }
    fn __or__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        bitwise(slf, other, false, "bitwise_or", |x, y| x | y, |x, y| x | y)
    }
    fn __ror__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        bitwise(slf, other, true, "bitwise_or", |x, y| x | y, |x, y| x | y)
    }
    fn __xor__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        bitwise(slf, other, false, "bitwise_xor", |x, y| x ^ y, |x, y| x ^ y)
    }
    fn __rxor__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        bitwise(slf, other, true, "bitwise_xor", |x, y| x ^ y, |x, y| x ^ y)
    }
    fn __lshift__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("left_shift", slf.clone(), other.clone()), None)
    }
    fn __rlshift__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("left_shift", other.clone(), slf.clone()), None)
    }
    fn __rshift__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("right_shift", slf.clone(), other.clone()), None)
    }
    fn __rrshift__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("right_shift", other.clone(), slf.clone()), None)
    }
    fn __invert__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        match slf.get().arr() {
            AnyArray::Bool(a) => PyArray::from_any(AnyArray::Bool(a.not())).into_py(py),
            AnyArray::I64(a) => PyArray::from_any(AnyArray::I64(a.map_int(|x| !x))).into_py(py),
            AnyArray::F64(_) => fallback(py, "call", ("invert", slf.clone()), None),
        }
    }

    fn __index__(&self) -> PyResult<i64> {
        match (self.arr(), self.arr().item()) {
            (AnyArray::I64(_), Some(Scalar::I(v))) => Ok(v),
            _ => Err(PyTypeError::new_err("only integer scalar arrays can be converted to a scalar index")),
        }
    }

    fn __complex__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyComplex>> {
        Ok(pyo3::types::PyComplex::from_doubles(py, self.__float__()?, 0.0))
    }

    // ---- copying and pickling (matplotlib calls copy.copy on its inputs) ----

    fn __copy__(&self) -> PyArray {
        PyArray::from_any(self.arr().clone())
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyArray {
        PyArray::from_any(self.arr().clone())
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyAny>,))> {
        let py = slf.py();
        let constructor = py.import("lightarray._core")?.getattr("array")?;
        let data = py.import("numpy")?.call_method1("array", (slf,))?; // an owned copy
        Ok((constructor, (data,)))
    }

    /// No-op used only to measure raw PyO3 method-call overhead in benchmarks.
    fn _noop(&self) {}

    fn __neg__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        match slf.get().arr() {
            AnyArray::F64(a) => PyArray::new(a.map(|x| -x)).into_py(py),
            AnyArray::I64(a) => PyArray::from_any(AnyArray::I64(a.map_int(i64::wrapping_neg))).into_py(py),
            AnyArray::Bool(_) => fallback(py, "call", ("negative", slf.clone()), None),
        }
    }
    fn __pos__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        match slf.get().arr() {
            AnyArray::Bool(_) => fallback(slf.py(), "call", ("positive", slf.clone()), None),
            other => PyArray::from_any(other.clone()).into_py(slf.py()),
        }
    }
    fn __abs__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        match slf.get().arr() {
            AnyArray::F64(a) => PyArray::new(a.map(f64::abs)).into_py(py),
            AnyArray::I64(a) => PyArray::from_any(AnyArray::I64(a.map_int(i64::wrapping_abs))).into_py(py),
            AnyArray::Bool(a) => PyArray::from_any(AnyArray::Bool(a.clone())).into_py(py),
        }
    }

    // ---- reductions ----

    #[pyo3(signature = (*args, **kwargs))]
    fn sum<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        reduce(slf, "sum", args, kwargs, |a| Ok(a.sum()), |a, ax| Ok(a.fold_axis(ax, Some(0.0), |x, y| x + y)?))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn prod<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        reduce(slf, "prod", args, kwargs, |a| Ok(a.prod()), |a, ax| Ok(a.fold_axis(ax, Some(1.0), |x, y| x * y)?))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn mean<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        reduce(slf, "mean", args, kwargs, |a| Ok(a.mean()), |a, ax| {
            let ndim = a.ndim() as isize;
            let len = a.shape()[((ax + ndim) % ndim) as usize] as f64;
            Ok(a.fold_axis(ax, Some(0.0), |x, y| x + y)?.map(|s| s / len))
        })
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn max<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        reduce(slf, "max", args, kwargs, |a| a.max().ok_or_else(|| no_identity("maximum")), |a, ax| Ok(a.fold_axis(ax, None, nan_max)?))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn min<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        reduce(slf, "min", args, kwargs, |a| a.min().ok_or_else(|| no_identity("minimum")), |a, ax| Ok(a.fold_axis(ax, None, nan_min)?))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn var<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        bare_or_fallback(slf, "var", args, kwargs, |a| np_float(py, a.var()))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn std<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        bare_or_fallback(slf, "std", args, kwargs, |a| np_float(py, a.var().sqrt()))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn argmax<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        bare_or_fallback(slf, "argmax", args, kwargs, |a| {
            np_int(py, a.argmax().ok_or_else(|| PyValueError::new_err("attempt to get argmax of an empty sequence"))?)
        })
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn argmin<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        bare_or_fallback(slf, "argmin", args, kwargs, |a| {
            np_int(py, a.argmin().ok_or_else(|| PyValueError::new_err("attempt to get argmin of an empty sequence"))?)
        })
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn any<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        bare_or_fallback(slf, "any", args, kwargs, |a| np_bool(py, a.any()))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn all<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        bare_or_fallback(slf, "all", args, kwargs, |a| np_bool(py, a.all()))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn cumsum<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        bare_or_fallback(slf, "cumsum", args, kwargs, |a| PyArray::new(a.scan(0.0, |x, y| x + y)).into_py(py))
    }
    #[pyo3(signature = (*args, **kwargs))]
    fn cumprod<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        bare_or_fallback(slf, "cumprod", args, kwargs, |a| PyArray::new(a.scan(1.0, |x, y| x * y)).into_py(py))
    }

    /// 1-D inner product natively; matrices and other operands through NumPy.
    fn dot(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if let (Some(a), Some(b)) = (slf.get().f64(), f64_of(other)) {
            if a.ndim() == 1 && b.ndim() == 1 {
                return np_float(py, a.dot1d(b)?);
            }
        }
        fallback(py, "call", ("dot", slf.clone(), other.clone()), None)
    }

    /// `clip(min, max)` with scalar bounds natively (one may be None).
    #[pyo3(signature = (*args, **kwargs))]
    fn clip<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let bound = |i: usize| -> PyResult<Option<Option<f64>>> {
            // Some(None) = explicitly None, Some(Some(v)) = scalar, None = not a scalar
            let item = args.get_item(i)?;
            if item.is_none() {
                return Ok(Some(None));
            }
            if item.is_instance_of::<PyFloat>() || item.is_instance_of::<PyInt>() {
                return Ok(Some(Some(item.extract()?)));
            }
            Ok(None)
        };
        if kwargs.map_or(true, |k| k.is_empty()) && args.len() == 2 {
            if let (Some(lo), Some(hi)) = (bound(0)?, bound(1)?) {
                if lo.is_none() && hi.is_none() {
                    return Err(PyValueError::new_err("One of max or min must be given"));
                }
                let Some(a) = slf.get().f64() else {
                    return call_method_fallback(slf, "clip", args, kwargs);
                };
                return PyArray::new(a.map(|x| {
                    let x = lo.map_or(x, |l| if x < l { l } else { x });
                    hi.map_or(x, |h| if x > h { h } else { x })
                }))
                .into_py(py);
            }
        }
        call_method_fallback(slf, "clip", args, kwargs)
    }

    /// `round(decimals=0)`, half to even like NumPy.
    #[pyo3(signature = (*args, **kwargs))]
    fn round<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let decimals: Option<i32> = match (args.len(), kwargs) {
            (0, None) => Some(0),
            (0, Some(k)) if k.is_empty() => Some(0),
            (1, None) => args.get_item(0)?.extract().ok(),
            (0, Some(k)) if k.len() == 1 => k.get_item("decimals")?.and_then(|d| d.extract().ok()),
            _ => None,
        };
        match decimals {
            Some(d) => {
                let scale = 10f64.powi(d);
                let Some(a) = slf.get().f64() else {
                    return call_method_fallback(slf, "round", args, kwargs);
                };
                PyArray::new(a.map(|x| round_half_even(x * scale) / scale)).into_py(py)
            }
            None => call_method_fallback(slf, "round", args, kwargs),
        }
    }

    // ---- shape ----

    #[pyo3(signature = (*shape, **kwargs))]
    fn reshape<'py>(slf: &Bound<'py, Self>, shape: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if kwargs.map_or(false, |k| !k.is_empty()) {
            return call_method_fallback(slf, "reshape", shape, kwargs);
        }
        let arr = slf.get().arr();
        let dims = if shape.len() == 1 { shape.get_item(0)?.len().unwrap_or(1) } else { shape.len() };
        if dims > MAX_NDIM {
            return call_method_fallback(slf, "reshape", shape, kwargs); // a NumPy array
        }
        let shape = extract_shape_args(shape, arr.size())?;
        if slf.get().strided.is_some() {
            return PyArray::from_any(arr.reshape(&shape)?).into_py(py);
        }
        PyArray::from_any(arr.reshape_view(&shape, || base_of(slf))?).into_py(py)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn copy<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        if c_order_only(args, kwargs)? {
            return PyArray::from_any(slf.get().arr().clone()).into_py(slf.py());
        }
        call_method_fallback(slf, "copy", args, kwargs)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn flatten<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        if c_order_only(args, kwargs)? {
            let arr = slf.get().arr();
            return PyArray::from_any(arr.reshape(&[arr.size()])?).into_py(slf.py());
        }
        call_method_fallback(slf, "flatten", args, kwargs)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn ravel<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        if c_order_only(args, kwargs)? {
            let arr = slf.get().arr();
            if slf.get().strided.is_some() {
                return PyArray::from_any(arr.reshape(&[arr.size()])?).into_py(slf.py());
            }
            return PyArray::from_any(arr.reshape_view(&[arr.size()], || base_of(slf))?).into_py(slf.py());
        }
        call_method_fallback(slf, "ravel", args, kwargs)
    }

    /// `astype(dtype)` between float64, int64 and bool natively; other
    /// targets, keywords and non-finite float-to-int casts through NumPy.
    #[pyo3(signature = (dtype, *args, **kwargs))]
    fn astype<'py>(slf: &Bound<'py, Self>, dtype: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if args.is_empty() && kwargs.map_or(true, |k| k.is_empty()) {
            let np = py.import("numpy")?;
            let dt = np.getattr("dtype")?.call1((dtype,))?;
            let kind: String = dt.getattr("kind")?.extract()?;
            let itemsize: usize = dt.getattr("itemsize")?.extract()?;
            let target = match (kind.as_str(), itemsize) {
                ("f", 8) => Some("float64"),
                ("i", 8) => Some("int64"),
                ("b", 1) => Some("bool"),
                _ => None,
            };
            if let Some(cast) = target.and_then(|t| slf.get().arr().cast(t)) {
                return PyArray::from_any(cast).into_py(py);
            }
        }
        let mut full: Vec<Bound<'py, PyAny>> = vec![slf.clone().into_any(), "astype".into_pyobject(py)?.into_any(), dtype.clone()];
        full.extend(args.iter());
        fallback(py, "call_method", PyTuple::new(py, full)?, kwargs)
    }

    /// `nonzero()`: a tuple of int64 index arrays, one per dimension.
    fn nonzero<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        if self.meta().ndim() == 0 {
            return Err(PyValueError::new_err(
                "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.",
            ));
        }
        let parts: Vec<Py<PyAny>> = self
            .arr()
            .nonzero()
            .into_iter()
            .map(|a| PyArray::from_any(AnyArray::I64(a)).into_py(py))
            .collect::<PyResult<_>>()?;
        PyTuple::new(py, parts)
    }

    /// `argsort()` of a 1-D float64 array natively (stable).
    #[pyo3(signature = (*args, **kwargs))]
    fn argsort<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        if args.is_empty() && kwargs.map_or(true, |k| k.is_empty()) {
            if let Some(order) = slf.get().f64().and_then(|a| a.argsort_1d()) {
                return PyArray::from_any(AnyArray::I64(order)).into_py(slf.py());
            }
        }
        call_method_fallback(slf, "argsort", args, stable_by_default(slf.py(), args, kwargs)?.as_ref())
    }

    /// The array whose memory this one shares, None when it owns its data.
    #[getter]
    fn base(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        match &self.strided {
            Some(v) => Some(v.base.clone_ref(py)),
            None => self.meta().base().map(|b| b.clone_ref(py)),
        }
    }

    #[getter]
    #[pyo3(name = "T")]
    fn transpose_property(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        transposed(slf)
    }

    /// `transpose()` with no axes natively; explicit axes through NumPy.
    #[pyo3(signature = (*args, **kwargs))]
    fn transpose<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let bare = kwargs.map_or(true, |k| k.is_empty()) && (args.is_empty() || (args.len() == 1 && args.get_item(0)?.is_none()));
        if bare {
            return transposed(slf);
        }
        call_method_fallback(slf, "transpose", args, kwargs)
    }

    // ---- mutation ----

    /// `a[key] = value`. Integer indices with a scalar value are set
    /// natively; every other form (slices, masks, lists, array values with
    /// broadcasting) is applied by NumPy through the writable view.
    fn __setitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        // Native only for a full integer index with a value of the array's
        // own kind; everything else (partial indices, slices, masks, casts)
        // is NumPy's assignment through the writable view.
        let ndim = slf.get().meta().ndim();
        let idx: Option<Vec<isize>> = if key.is_instance_of::<PyInt>() && ndim == 1 {
            Some(vec![key.extract::<isize>()?])
        } else if let Ok(tuple) = key.cast::<PyTuple>() {
            if tuple.len() == ndim && tuple.iter().all(|item| item.is_instance_of::<PyInt>()) {
                Some(tuple.extract()?)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(idx) = idx {
            if let Some(p) = slf.get().strided_element(&idx)? {
                // a strided view: write the element straight into the base
                // SAFETY: `p` addresses an element of the base's buffer.
                unsafe {
                    match (slf.get().meta(), classify(value)) {
                        (AnyArray::F64(_), v @ (Operand::Float(_) | Operand::Int(_) | Operand::Bool(_))) => {
                            *(p as *mut f64) = v.scalar().expect("scalar");
                            return Ok(());
                        }
                        (AnyArray::I64(_), Operand::Int(i)) => {
                            *(p as *mut i64) = i;
                            return Ok(());
                        }
                        (AnyArray::Bool(_), Operand::Bool(b)) => {
                            *(p as *mut bool) = b;
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                fallback(py, "setitem", (slf.clone(), key.clone(), value.clone()), None)?;
                return Ok(());
            }
            match (slf.get().arr_mut(), classify(value)) {
                (AnyArray::F64(a), v @ (Operand::Float(_) | Operand::Int(_) | Operand::Bool(_))) => {
                    a.set(&idx, v.scalar().expect("scalar"))?;
                    slf.get().commit();
                    return Ok(());
                }
                (AnyArray::I64(a), Operand::Int(i)) => {
                    a.set(&idx, i)?;
                    slf.get().commit();
                    return Ok(());
                }
                (AnyArray::Bool(a), Operand::Bool(b)) => {
                    a.set(&idx, b)?;
                    slf.get().commit();
                    return Ok(());
                }
                _ => {}
            }
        }
        fallback(py, "setitem", (slf.clone(), key.clone(), value.clone()), None)?;
        Ok(())
    }

    fn __iadd__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        inplace_op(slf, other, "add", |a, b| a + b)
    }
    fn __isub__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        inplace_op(slf, other, "subtract", |a, b| a - b)
    }
    fn __imul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        inplace_op(slf, other, "multiply", |a, b| a * b)
    }
    fn __itruediv__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        inplace_op(slf, other, "divide", |a, b| a / b)
    }
    fn __ifloordiv__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        inplace_op(slf, other, "floor_divide", float_floor_div)
    }
    fn __imod__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        inplace_op(slf, other, "remainder", float_mod)
    }
    fn __ipow__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        inplace_op(slf, other, "power", float_pow)
    }

    /// `a.fill(value)` in place.
    fn fill(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if let (Some(a), Some(v)) = (slf.get().f64_mut(), classify(value).scalar()) {
            a.map_inplace(|_| v);
            slf.get().commit();
            return Ok(());
        }
        fallback(slf.py(), "call_method", (slf.clone(), "fill", value.clone()), None).map(|_| ())
    }

    // ---- indexing ----

    /// Integers and slices (any mix, any number of axes) natively as copies;
    /// ellipsis, None, fancy and boolean indexing through NumPy.
    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        // basic indexing needs the shape only; a strided view's cache is
        // refreshed further down, where elements are gathered
        let arr = slf.get().meta();
        // Type checks first: a failed `extract` builds a Python exception,
        // which costs more than the whole slice.
        if key.is_instance_of::<PyInt>() {
            return index_result(slf, &[key.extract::<isize>()?]);
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            if arr.ndim() == 0 {
                return Err(PyIndexError::new_err("too many indices for array: array is 0-dimensional"));
            }
            return select_result(slf, &[slice_selector(slice, arr.shape()[0])?], &[]);
        }
        if let Ok(tuple) = key.cast::<PyTuple>() {
            let mut ints = Vec::with_capacity(tuple.len());
            let mut sels = Vec::with_capacity(tuple.len());
            // Output-axis positions where `None` inserts a length-1 axis.
            let mut new_axes: Vec<usize> = Vec::new();
            let mut all_ints = true;
            let mut axis = 0usize; // axis of the source array
            let mut out_axis = 0usize; // axis of the result
            for item in tuple.iter() {
                if item.is_none() {
                    all_ints = false;
                    new_axes.push(out_axis);
                    out_axis += 1;
                } else if item.is_instance_of::<PyInt>() {
                    let i = item.extract::<isize>()?;
                    ints.push(i);
                    sels.push(Selector::Int(i));
                    axis += 1;
                } else if let Ok(slice) = item.cast::<PySlice>() {
                    all_ints = false;
                    match arr.shape().get(axis) {
                        Some(&len) => sels.push(slice_selector(slice, len)?),
                        None => return Err(PyIndexError::new_err("too many indices for array")),
                    }
                    axis += 1;
                    out_axis += 1;
                } else {
                    return fallback(py, "getitem", (slf.clone(), key.clone()), None);
                }
            }
            if all_ints {
                return index_result(slf, &ints);
            }
            return select_result(slf, &sels, &new_axes);
        }
        if key.is_none() {
            return select_result(slf, &[], &[0]);
        }
        if key.is(py.Ellipsis()) {
            return select_result(slf, &[], &[]);
        }
        // A list of integers: rows along the leading axis.
        if let Ok(list) = key.cast::<PyList>() {
            let mut idx = Vec::with_capacity(list.len());
            for item in list.iter() {
                if item.is_instance_of::<PyInt>() {
                    idx.push(item.extract::<isize>()?);
                } else {
                    return fallback(py, "getitem", (slf.clone(), key.clone()), None);
                }
            }
            return PyArray::from_any(slf.get().arr().take_leading(&idx)?).into_py(py);
        }
        // An int64 lightarray used as an index array along the leading axis.
        if let Some(AnyArray::I64(index)) = any_of(key) {
            if index.ndim() == 1 {
                let idx: Vec<isize> = index.data().iter().map(|&i| i as isize).collect();
                return PyArray::from_any(slf.get().arr().take_leading(&idx)?).into_py(py);
            }
        }
        // A boolean mask of the array's shape (a lightarray or NumPy bool array).
        if let Some(mask) = bool_mask(key) {
            if mask.1 == arr.shape() {
                return PyArray::from_any(slf.get().arr().compress_flat(&mask.0)?).into_py(py);
            }
        }
        // NumPy integer scalars (and anything else with __index__) index too.
        if key.hasattr("__index__")? {
            if let Ok(i) = key.extract::<isize>() {
                return index_result(slf, &[i]);
            }
        }
        fallback(py, "getitem", (slf.clone(), key.clone()), None)
    }
}

fn slice_selector(slice: &Bound<'_, PySlice>, len: usize) -> PyResult<Selector> {
    let ix = slice.indices(len as isize)?;
    Ok(Selector::Slice { start: ix.start, stop: ix.stop, step: ix.step })
}

/// In-place operator `self OP= other`. Same-shape or broadcastable arrays
/// and scalars are applied natively; anything NumPy understands runs as
/// `np.<ufunc>(view, other, out=view)` on the writable view.
fn inplace_op<F: Fn(f64, f64) -> f64>(
    slf: &Bound<'_, PyArray>,
    other: &Bound<'_, PyAny>,
    name: &str,
    f: F,
) -> PyResult<()> {
    let py = slf.py();
    let operand = classify(other);
    let numpy_inplace = || fallback(py, "inplace", (slf.clone(), name, other.clone()), None).map(|_| ());
    // int64 and bool targets follow NumPy's casting rules through the view
    let Some(a) = slf.get().f64_mut() else {
        return numpy_inplace();
    };
    match operand {
        Operand::F64(b) => {
            if overlaps(a.span(), b.span()) {
                // the operand is the target itself or a view sharing its memory
                let copy = b.clone();
                a.zip_map_inplace(&copy, f)?;
            } else {
                a.zip_map_inplace(b, f)?;
            }
        }
        Operand::Alt(b) => a.zip_map_inplace(&b.to_f64(), f)?,
        Operand::Other if numpy_handles(other) => return numpy_inplace(),
        Operand::Other => {
            return Err(PyTypeError::new_err(format!(
                "unsupported operand type(s) for in-place {name}: 'lightarray.ndarray' and '{}'",
                other.get_type().name()?
            )))
        }
        scalar => {
            let v = scalar.scalar().expect("remaining variants are scalars");
            a.map_inplace(|x| f(x, v));
        }
    }
    slf.get().commit();
    Ok(())
}

/// The array owning the buffer of `slf`: its base when it is a view itself.
fn base_of(slf: &Bound<'_, PyArray>) -> Py<PyAny> {
    if let Some(v) = &slf.get().strided {
        return v.base.clone_ref(slf.py());
    }
    match slf.get().meta().base() {
        Some(base) => base.clone_ref(slf.py()),
        None => slf.clone().into_any().unbind(),
    }
}

/// Where the elements of an array live: first element, shape, byte strides.
/// Fixed-size, so composing layouts allocates nothing.
struct Layout {
    ptr: *mut u8,
    ndim: usize,
    shape: [usize; MAX_NDIM],
    strides: [isize; MAX_NDIM],
}

fn layout_of(slf: &Bound<'_, PyArray>) -> Layout {
    let this = slf.get();
    let meta = this.meta();
    let ndim = meta.ndim();
    let mut shape = [0usize; MAX_NDIM];
    shape[..ndim].copy_from_slice(meta.shape());
    let mut strides = [0isize; MAX_NDIM];
    let ptr = match &this.strided {
        Some(v) => {
            strides = v.strides;
            v.ptr
        }
        None => {
            strides[..ndim].copy_from_slice(meta.dims().strides());
            meta.data_ptr() as *mut u8
        }
    };
    Layout { ptr, ndim, shape, strides }
}

impl Layout {
    fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Apply a basic-indexing plan (`(start, step, count)` per axis; integer
    /// selectors are the entries of `sels` that drop their axis).
    fn select(&self, plan: &[(usize, isize, usize)], sels: &[Selector]) -> Layout {
        let mut out = Layout { ptr: self.ptr, ndim: 0, shape: [0; MAX_NDIM], strides: [0; MAX_NDIM] };
        for (axis, &(start, step, count)) in plan.iter().enumerate() {
            if count > 0 {
                out.ptr = out.ptr.wrapping_offset(start as isize * self.strides[axis]);
            }
            if !matches!(sels.get(axis), Some(Selector::Int(_))) {
                out.shape[out.ndim] = count;
                out.strides[out.ndim] = step * self.strides[axis];
                out.ndim += 1;
            }
        }
        out
    }

    /// Insert a length-1 axis at `pos` (`None` in an index).
    fn insert_axis(&mut self, pos: usize) -> PyResult<()> {
        if self.ndim == MAX_NDIM {
            return Err(PyValueError::new_err(format!("at most {MAX_NDIM} dimensions are supported")));
        }
        let pos = pos.min(self.ndim);
        self.shape.copy_within(pos..self.ndim, pos + 1);
        self.strides.copy_within(pos..self.ndim, pos + 1);
        self.shape[pos] = 1;
        self.strides[pos] = 0;
        self.ndim += 1;
        Ok(())
    }

    fn reversed(&self) -> Layout {
        let mut out = Layout { ptr: self.ptr, ndim: self.ndim, shape: [0; MAX_NDIM], strides: [0; MAX_NDIM] };
        for k in 0..self.ndim {
            out.shape[k] = self.shape[self.ndim - 1 - k];
            out.strides[k] = self.strides[self.ndim - 1 - k];
        }
        out
    }

    fn is_c_contiguous(&self, itemsize: usize) -> bool {
        let mut expected = itemsize as isize;
        for k in (0..self.ndim).rev() {
            if self.shape[k] != 1 && self.strides[k] != expected {
                return false;
            }
            expected *= self.shape[k] as isize;
        }
        true
    }
}

/// The array described by `layout` inside the memory `slf` lives in: a
/// contiguous window when the layout allows, a strided view otherwise.
fn view_from_layout(slf: &Bound<'_, PyArray>, layout: Layout) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let meta = slf.get().meta();
    if layout.shape().iter().product::<usize>() == 0 {
        return PyArray::from_any(meta.zeros_like(layout.shape())?).into_py(py);
    }
    if layout.is_c_contiguous(meta.itemsize()) {
        // SAFETY: the layout was derived from `slf`'s own by basic indexing,
        // so it addresses elements of the buffer `base_of(slf)` owns.
        let view = unsafe { meta.raw_view_like(layout.ptr, layout.shape(), base_of(slf))? };
        return PyArray::from_any(view).into_py(py);
    }
    let view = PyArray {
        inner: std::cell::UnsafeCell::new(meta.zeros_like(layout.shape())?),
        strided: Some(Box::new(Strided { base: base_of(slf), ptr: layout.ptr, strides: layout.strides })),
    };
    view.into_py(py)
}

/// A lightarray view for `numpy_view`, a NumPy array that is a view of
/// `source`'s memory (what NumPy functions such as `swapaxes`, `split` or
/// `a[..., 1]` return for the zero-copy array handed to them). None when the
/// dtype differs or the view does not lie inside the owner's buffer.
pub fn view_of_numpy(source: &Bound<'_, PyArray>, numpy_view: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    let py = source.py();
    let owner = base_of(source);
    let owner = owner.bind(py).cast::<PyArray>()?;
    let meta = owner.get().meta();
    let Ok(raw) = PyUntypedBuffer::get(numpy_view) else { return Ok(None) };
    let itemsize = meta.itemsize();
    let ndim = raw.dimensions();
    if raw.readonly() || raw.item_size() != itemsize || raw.format() != meta.format() || ndim > MAX_NDIM {
        return Ok(None);
    }
    let mut layout = Layout { ptr: raw.buf_ptr() as *mut u8, ndim, shape: [0; MAX_NDIM], strides: [0; MAX_NDIM] };
    layout.shape[..ndim].copy_from_slice(raw.shape());
    layout.strides[..ndim].copy_from_slice(raw.strides());
    if layout.shape().iter().product::<usize>() == 0 {
        return Ok(None);
    }
    // every addressed element must lie inside the owner's buffer, aligned
    let (mut lo, mut hi) = (layout.ptr as isize, layout.ptr as isize);
    for k in 0..ndim {
        let reach = (layout.shape[k] as isize - 1) * layout.strides[k];
        if layout.strides[k] % itemsize as isize != 0 {
            return Ok(None);
        }
        if reach < 0 {
            lo += reach;
        } else {
            hi += reach;
        }
    }
    let (start, end) = meta.span();
    if lo < start as isize || hi + itemsize as isize > end as isize || (lo - start as isize) % itemsize as isize != 0 {
        return Ok(None);
    }
    view_from_layout(owner, layout).map(Some)
}

/// Keywords for NumPy's `argsort` with a stable sort unless the caller chose
/// a `kind`: the Array API makes stable the default, and NumPy promises
/// nothing about ties, so this is valid for both.
pub fn stable_by_default<'py>(py: Python<'py>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Option<Bound<'py, PyDict>>> {
    let kw = match kwargs {
        Some(k) => k.copy()?,
        None => PyDict::new(py),
    };
    let stable_given = kw.get_item("stable")?.map_or(false, |v| !v.is_none());
    if args.len() < 2 && !kw.contains("kind")? && !stable_given {
        kw.del_item("stable").ok();
        kw.set_item("kind", "stable")?;
    }
    Ok(Some(kw))
}

/// `a.T`: the same memory with the axes reversed.
fn transposed(slf: &Bound<'_, PyArray>) -> PyResult<Py<PyAny>> {
    view_from_layout(slf, layout_of(slf).reversed())
}

/// Do two arrays share any memory?
fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// `a[i, j]`: a NumPy scalar for a full index, a view of the sub-array otherwise.
/// Out of line: inlined into `__getitem__` it grows that function's frame,
/// which cost the slice paths 10 ns.
#[inline(never)]
fn index_result(slf: &Bound<'_, PyArray>, idx: &[isize]) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    if let Some(p) = slf.get().strided_element(idx)? {
        // SAFETY: `p` addresses an element of the base's buffer.
        return unsafe {
            match slf.get().meta() {
                AnyArray::F64(_) => np_float(py, *(p as *const f64)),
                AnyArray::I64(_) => np_i64(py, *(p as *const i64)),
                AnyArray::Bool(_) => np_bool(py, *(p as *const bool)),
            }
        };
    }
    let arr = slf.get().meta();
    if slf.get().strided.is_some() {
        let sels: Vec<Selector> = idx.iter().map(|&i| Selector::Int(i)).collect();
        let (plan, _) = arr.selection_plan(&sels)?;
        return view_from_layout(slf, layout_of(slf).select(&plan, &sels));
    }
    match arr.get(idx)? {
        Some(Scalar::F(v)) => np_float(py, v),
        Some(Scalar::I(v)) => np_i64(py, v),
        Some(Scalar::B(v)) => np_bool(py, v),
        None => PyArray::from_any(arr.index_view(idx, || base_of(slf))?).into_py(py),
    }
}

/// Basic indexing with slices: a contiguous window when possible (the fast
/// path), a strided view otherwise. `new_axes` are result positions of `None`.
fn select_result(slf: &Bound<'_, PyArray>, sels: &[Selector], new_axes: &[usize]) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let arr = slf.get().meta();
    let (plan, out_shape) = arr.selection_plan(sels)?;
    if slf.get().strided.is_none() {
        if let Some(mut selected) = arr.plan_view(&plan, &out_shape, || base_of(slf))? {
            if !new_axes.is_empty() {
                let mut shape = out_shape;
                for &pos in new_axes {
                    shape.insert(pos.min(shape.len()), 1);
                }
                selected.set_shape(&shape)?;
            }
            return PyArray::from_any(selected).into_py(py);
        }
    }
    let mut layout = layout_of(slf).select(&plan, sels);
    for &pos in new_axes {
        layout.insert_axis(pos)?;
    }
    view_from_layout(slf, layout)
}

/// Bitwise operators: masks and integer arrays natively, NumPy otherwise
/// (float operands raise TypeError there, as NumPy does).
fn bitwise(
    slf: &Bound<'_, PyArray>,
    other: &Bound<'_, PyAny>,
    reflected: bool,
    name: &str,
    bf: fn(bool, bool) -> bool,
    intf: fn(i64, i64) -> i64,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let result = match (slf.get().arr(), classify(other)) {
        (AnyArray::Bool(a), Operand::Alt(AnyArray::Bool(b))) => a.zip_bool(b, bf).map(AnyArray::Bool),
        (AnyArray::Bool(a), Operand::Bool(v)) => Some(AnyArray::Bool(compare_scalar(a, |x| bf(x, v)))),
        (AnyArray::I64(a), Operand::Alt(AnyArray::I64(b))) => a.zip_int(b, intf).map(AnyArray::I64),
        (AnyArray::I64(a), Operand::Int(i)) => Some(AnyArray::I64(a.map_int(|x| intf(x, i)))),
        _ => None,
    };
    match result {
        Some(r) => PyArray::from_any(r).into_py(py),
        None => numpy_binary(py, slf, other, reflected, name),
    }
}

/// Accept `reshape(2, 3)`, `reshape((2, 3))`, `reshape([2, 3])`, `reshape(-1)`.
fn extract_shape_args(args: &Bound<'_, PyTuple>, size: usize) -> PyResult<Vec<usize>> {
    let raw: Vec<isize> = if args.len() == 1 && !args.get_item(0)?.is_instance_of::<PyInt>() {
        args.get_item(0)?.extract()?
    } else {
        args.extract()?
    };
    let unknown = raw.iter().filter(|&&d| d < 0).count();
    if unknown > 1 {
        return Err(PyValueError::new_err("can only specify one unknown dimension"));
    }
    let known: usize = raw.iter().filter(|&&d| d >= 0).map(|&d| d as usize).product();
    let mut shape = Vec::with_capacity(raw.len());
    for &d in &raw {
        if d < 0 {
            if known == 0 || size % known != 0 {
                return Err(PyValueError::new_err(format!("cannot reshape array of size {size} into shape {raw:?}")));
            }
            shape.push(size / known);
        } else {
            shape.push(d as usize);
        }
    }
    Ok(shape)
}

/// `shape` argument of creation functions: an int or a sequence of ints.
pub fn extract_shape(shape: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    // Type checks first: a failed `extract` builds and drops a Python
    // exception, which costs more than making the array.
    if shape.is_instance_of::<PyInt>() {
        return Ok(vec![shape.extract::<usize>()?]);
    }
    if let Ok(tuple) = shape.cast::<PyTuple>() {
        return tuple.iter().map(|d| d.extract::<usize>()).collect::<PyResult<_>>().map_err(|_| PyTypeError::new_err("shape must be an int or a sequence of ints"));
    }
    if let Ok(n) = shape.extract::<usize>() {
        return Ok(vec![n]);
    }
    shape.extract::<Vec<usize>>().map_err(|_| PyTypeError::new_err("shape must be an int or a sequence of ints"))
}

/// `array(obj)` / `asarray(obj)`: a lightarray array when the input maps to
/// float64, int64 or bool (nested Python sequences, numbers, NumPy arrays of
/// those dtypes); otherwise NumPy's array with its own dtype (complex,
/// int32, str, ...).
pub fn array_or_numpy(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Some(arr) = native_array(py, obj)? {
        return PyArray::from_any(arr).into_py(py);
    }
    let np = py.import("numpy")?;
    let converted = np.getattr("asarray")?.call1((obj,))?;
    PyArray::wrap_result(py, converted)
}

/// Native conversion only (lightarray arrays, numbers, nested sequences of
/// numbers, float64/int64/bool buffers); None for everything else. The
/// dtype follows NumPy's inference: any float makes float64, integers make
/// int64, only bools make bool.
fn native_array(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Option<AnyArray>> {
    if let Some(a) = any_of(obj) {
        return Ok(Some(a.clone()));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        return nested_sequence(obj);
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Some(AnyArray::F64(Array::scalar(obj.extract()?))));
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(Some(AnyArray::Bool(Array::scalar(obj.is_truthy()?))));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(obj.extract::<i64>().ok().map(|v| AnyArray::I64(Array::scalar(v))));
    }
    if has_buffer(obj) {
        return PyArray::from_buffer_any(py, obj);
    }
    Ok(None)
}

/// Build a float64 `Array` from anything NumPy's `array()` accepts, coercing
/// other dtypes (used where float64 is required).
pub fn array_from_any(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Array> {
    if let Some(any) = native_array(py, obj)? {
        return Ok(match any {
            AnyArray::F64(a) => a,
            other => other.to_f64(),
        });
    }
    let np = py.import("numpy")?;
    let kw = PyDict::new(py);
    kw.set_item("dtype", "float64")?;
    let converted = np.getattr("asarray")?.call((obj,), Some(&kw))?;
    PyArray::from_f64_buffer(py, &converted, true)?.ok_or_else(|| PyTypeError::new_err("could not convert input to a float64 array"))
}

/// Does the object support the buffer protocol at all? Cheaper than
/// attempting a `PyBuffer::get` and unwinding the TypeError.
fn has_buffer(obj: &Bound<'_, PyAny>) -> bool {
    // SAFETY: `obj` is a valid object pointer for the duration of the call.
    unsafe { ffi::PyObject_CheckBuffer(obj.as_ptr()) == 1 }
}

/// State of a walk over nested sequences; fixed-size so the walk allocates
/// nothing but the data. Integers are kept exactly (in `ints`) until a float
/// shows up, because float64 cannot hold every int64.
struct Walk {
    ndim: usize,
    dims: [usize; crate::dims::MAX_NDIM],
    floats: Vec<f64>,
    ints: Vec<i64>,
    saw_float: bool,
    saw_int: bool,
    saw_bool: bool,
}

/// Rectangular nested lists/tuples of numbers. Returns None when the input is
/// not such a structure (ragged, contains strings, contains arrays, ...).
// The inlining of this path is pinned: left to the optimiser it changes when
// unrelated code is added, which moved `np.array(list)` by 10 ns.
#[inline(never)]
fn nested_sequence(obj: &Bound<'_, PyAny>) -> PyResult<Option<AnyArray>> {
    let mut w = Walk { ndim: 0, dims: [0; crate::dims::MAX_NDIM], floats: Vec::new(), ints: Vec::new(), saw_float: false, saw_int: false, saw_bool: false };
    if !walk(obj, 0, &mut w)? {
        return Ok(None);
    }
    let shape = &w.dims[..w.ndim];
    // Mixed scalars and sequences at one level slip past `walk`; let NumPy
    // produce the proper error for those.
    if w.floats.len() != shape.iter().product::<usize>() {
        return Ok(None);
    }
    Ok(Some(if w.saw_float || !(w.saw_int || w.saw_bool) {
        AnyArray::F64(Array::new(w.floats, shape)?)
    } else if w.saw_int {
        AnyArray::I64(Array::new(w.ints, shape)?)
    } else {
        AnyArray::Bool(Array::new(w.ints.iter().map(|&v| v != 0).collect(), shape)?)
    }))
}

#[inline(never)]
fn walk(obj: &Bound<'_, PyAny>, depth: usize, w: &mut Walk) -> PyResult<bool> {
    if let Ok(list) = obj.cast::<PyList>() {
        walk_items(list.len(), list.iter(), depth, w)
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        walk_items(tuple.len(), tuple.iter(), depth, w)
    } else {
        Ok(false)
    }
}

#[inline(always)]
fn walk_items<'py>(len: usize, items: impl Iterator<Item = Bound<'py, PyAny>>, depth: usize, w: &mut Walk) -> PyResult<bool> {
    if depth == w.ndim {
        if depth + 1 > crate::dims::MAX_NDIM {
            return Ok(false);
        }
        w.dims[depth] = len;
        w.ndim += 1;
    } else if w.dims[depth] != len {
        return Ok(false);
    }
    if depth == 0 {
        w.floats.reserve(len);
    }
    for item in items {
        if let Ok(f) = item.cast_exact::<PyFloat>() {
            if depth + 1 != w.ndim {
                return Ok(false);
            }
            w.floats.push(f.value());
            w.saw_float = true;
        } else if item.is_instance_of::<PyBool>() {
            if depth + 1 != w.ndim {
                return Ok(false);
            }
            let v = item.is_truthy()?;
            w.floats.push(v as u8 as f64);
            if !w.saw_float {
                w.ints.push(v as i64);
            }
            w.saw_bool = true;
        } else if item.is_instance_of::<PyInt>() {
            if depth + 1 != w.ndim {
                return Ok(false);
            }
            let Ok(v) = item.extract::<i64>() else { return Ok(false) }; // beyond int64: NumPy decides
            w.floats.push(v as f64);
            if !w.saw_float {
                w.ints.push(v);
            }
            w.saw_int = true;
        } else if item.is_instance_of::<PyFloat>() {
            if depth + 1 != w.ndim {
                return Ok(false);
            }
            w.floats.push(item.extract()?);
            w.saw_float = true;
        } else if !walk(&item, depth + 1, w)? {
            return Ok(false);
        }
    }
    Ok(true)
}
