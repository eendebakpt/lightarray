//! The Python-visible array type.
//!
//! `PyArray` is a frozen wrapper around `Array`. Everything that is cheap and
//! common is implemented natively; everything else is delegated to NumPy
//! through the helpers in `lightarray/_fallback.py`, with float64 results
//! wrapped back into `PyArray`.

use crate::array::{Array, ArrayError, Selector};
use crate::dims::ITEMSIZE;
use pyo3::buffer::{PyBuffer, PyUntypedBuffer};
use pyo3::call::PyCallArgs;
use pyo3::exceptions::{PyBufferError, PyIndexError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList, PyMemoryView, PyNotImplemented, PySlice, PyTuple};
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
    inner: std::cell::UnsafeCell<Array>,
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
    pub fn new(inner: Array) -> PyArray {
        PyArray { inner: std::cell::UnsafeCell::new(inner) }
    }

    /// Shared access to the array.
    #[inline]
    pub fn inner(&self) -> &Array {
        // SAFETY: readers and the GIL-serialised writers never overlap on the
        // GIL build; see the type docs for the free-threaded contract.
        unsafe { &*self.inner.get() }
    }

    /// Mutable access to the data. Callers must hold the GIL and must not
    /// change the shape or reallocate the data (views point at it).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn inner_mut(&self) -> &mut Array {
        // SAFETY: as above; only element values are written.
        unsafe { &mut *self.inner.get() }
    }

    pub fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, self)?.into_any())
    }

    /// Copy any object exposing a float64 buffer (NumPy arrays, memoryviews,
    /// `array.array('d')`) into a new `Array`. 0-d buffers carry a NULL
    /// shape that PyBuffer rejects, so they are read separately: with
    /// `coerce_scalars` any numeric 0-d object (NumPy integer or bool
    /// scalars included) becomes a float64 scalar; without it only exact
    /// float64 0-d buffers qualify, so complex or integer results keep
    /// their NumPy dtype.
    pub fn from_f64_buffer(py: Python<'_>, obj: &Bound<'_, PyAny>, coerce_scalars: bool) -> PyResult<Option<Array>> {
        let buf = match PyBuffer::<f64>::get(obj) {
            Ok(b) => b,
            Err(_) => {
                let is_0d = obj.getattr("ndim").and_then(|n| n.extract::<usize>()).map_or(false, |n| n == 0);
                if !is_0d {
                    return Ok(None);
                }
                if coerce_scalars {
                    return Ok(obj.extract::<f64>().ok().map(Array::scalar));
                }
                // PyBuffer rejects 0-d buffers (NULL shape), so ask the dtype.
                let is_f64 = obj
                    .getattr("dtype")
                    .and_then(|d| d.getattr("num"))
                    .and_then(|n| n.extract::<i32>())
                    .map_or(false, |n| n == 12);
                return Ok(if is_f64 { Some(Array::scalar(obj.extract()?)) } else { None });
            }
        };
        if buf.dimensions() > crate::dims::MAX_NDIM {
            return Ok(None); // stays a NumPy array
        }
        let shape: Vec<usize> = buf.shape().to_vec();
        let data = buf.to_vec(py)?;
        Ok(Some(Array::new(data, &shape)?))
    }

    /// Convert a NumPy result back into lightarray where possible.
    pub fn wrap_result(py: Python<'_>, obj: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if obj.is_instance_of::<PyArray>() {
            return Ok(obj.unbind());
        }
        if obj.hasattr("__array_interface__")? {
            let ndim = obj.getattr("ndim").and_then(|n| n.extract::<usize>()).unwrap_or(0);
            if ndim <= crate::dims::MAX_NDIM {
                if let Some(arr) = PyArray::from_f64_buffer(py, &obj, false)? {
                    return PyArray::new(arr).into_py(py);
                }
            }
        }
        Ok(obj.unbind())
    }
}

// ---- helpers ------------------------------------------------------------

static NP_FLOAT64: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NP_INTP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NP_BOOL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// A `numpy.intp` scalar (what `argmax` returns in NumPy).
pub fn np_int(py: Python<'_>, v: usize) -> PyResult<Py<PyAny>> {
    let ty = NP_INTP.get_or_try_init(py, || -> PyResult<Py<PyAny>> { Ok(py.import("numpy")?.getattr("intp")?.unbind()) })?;
    Ok(ty.bind(py).call1((v,))?.unbind())
}

/// A `numpy.bool_` scalar (what `any`/`all`/predicates return in NumPy).
pub fn np_bool(py: Python<'_>, v: bool) -> PyResult<Py<PyAny>> {
    let ty = NP_BOOL.get_or_try_init(py, || -> PyResult<Py<PyAny>> { Ok(py.import("numpy")?.getattr("bool_")?.unbind()) })?;
    Ok(ty.bind(py).call1((v,))?.unbind())
}

/// A `numpy.float64` scalar, what NumPy returns from reductions and scalar
/// indexing (it carries `.dtype`, `.astype`, `.round`, ... unlike a Python
/// float). Costs about 80 ns; parity is worth it.
pub fn np_float(py: Python<'_>, v: f64) -> PyResult<Py<PyAny>> {
    let ty = NP_FLOAT64.get_or_try_init(py, || -> PyResult<Py<PyAny>> { Ok(py.import("numpy")?.getattr("float64")?.unbind()) })?;
    Ok(ty.bind(py).call1((v,))?.unbind())
}

fn not_implemented(py: Python<'_>) -> Py<PyAny> {
    PyNotImplemented::get(py).to_owned().into_any().unbind()
}

/// Classify a binary-operator operand: our own array, a scalar, or something
/// for NumPy (or the other operand's reflected method) to handle.
enum Operand<'a, 'py> {
    Array(&'a Bound<'py, PyArray>),
    Scalar(f64),
    Other,
}

fn classify<'a, 'py>(obj: &'a Bound<'py, PyAny>) -> Operand<'a, 'py> {
    if let Ok(a) = obj.cast::<PyArray>() {
        return Operand::Array(a);
    }
    if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>() {
        if let Ok(v) = obj.extract::<f64>() {
            return Operand::Scalar(v);
        }
    }
    // NumPy scalars and other numbers; arrays expose __len__ and go to NumPy.
    if !obj.hasattr("__len__").unwrap_or(true) {
        if let Ok(v) = obj.extract::<f64>() {
            return Operand::Scalar(v);
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
/// `reflected` is set. NumPy arrays and other array-likes go through the
/// NumPy ufunc `name` and come back as lightarray when float64.
#[inline]
fn binary_op<F: Fn(f64, f64) -> f64>(
    slf: &Bound<'_, PyArray>,
    other: &Bound<'_, PyAny>,
    reflected: bool,
    name: &str,
    f: F,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let a = slf.get().inner();
    let result = match classify(other) {
        Operand::Array(b) => {
            let b = b.get().inner();
            if reflected {
                b.zip_map(a, &f)?
            } else {
                a.zip_map(b, &f)?
            }
        }
        Operand::Scalar(s) => {
            if reflected {
                a.map(|x| f(s, x))
            } else {
                a.map(|x| f(x, s))
            }
        }
        Operand::Other if numpy_handles(other) => {
            return if reflected {
                fallback(py, "call", (name, other.clone(), slf.clone()), None)
            } else {
                fallback(py, "call", (name, slf.clone(), other.clone()), None)
            };
        }
        Operand::Other => return Ok(not_implemented(py)),
    };
    PyArray::new(result).into_py(py)
}

/// Round half to even (NumPy's `rint`/`round`).
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r }
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
        (Operand::Array(a), Operand::Array(b)) => a.get().inner().zip_map(b.get().inner(), f)?,
        (Operand::Array(a), Operand::Scalar(s)) => a.get().inner().map(|x| f(x, s)),
        (Operand::Scalar(s), Operand::Array(b)) => b.get().inner().map(|x| f(s, x)),
        (Operand::Scalar(a), Operand::Scalar(b)) => return Ok(f(a, b).into_pyobject(py)?.into_any().unbind()),
        _ => return fallback(py, "call", (name, x1.clone(), x2.clone()), None),
    };
    PyArray::new(result).into_py(py)
}

/// Build a writable NumPy bool array of `shape` from 0/1 bytes.
pub fn bool_array(py: Python<'_>, bytes: Vec<u8>, shape: &[usize]) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let buffer = pyo3::types::PyByteArray::new(py, &bytes);
    let flat = np.getattr("frombuffer")?.call1((buffer, "?"))?;
    if shape.len() == 1 {
        return Ok(flat.unbind());
    }
    Ok(flat.call_method1("reshape", (PyTuple::new(py, shape)?,))?.unbind())
}

/// Read a contiguous NumPy bool array (format '?') as a Vec<bool> plus its
/// shape. bool is not a PyBuffer element type in PyO3, so this goes through
/// the untyped buffer. Non-buffers and other formats give None.
pub fn bool_mask(obj: &Bound<'_, PyAny>) -> Option<(Vec<bool>, Vec<usize>)> {
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
        let as_array = |v: &Bound<'_, PyAny>| -> PyResult<Option<Array>> {
            Ok(match classify(v) {
                Operand::Array(a) if a.get().inner().shape() == shape.as_slice() => Some(a.get().inner().clone()),
                Operand::Scalar(s) => Some(Array::filled(&shape, s)?),
                _ => None,
            })
        };
        let (Some(xa), Some(ya)) = (as_array(x)?, as_array(y)?) else { return Ok(None) };
        Ok(Some(Array::select_where(&mask, &xa, &ya)?))
    };
    match native()? {
        Some(arr) => PyArray::new(arr).into_py(py),
        None => fallback(py, "call", ("where", cond.clone(), x.clone(), y.clone()), None),
    }
}

fn float_floor_div(a: f64, b: f64) -> f64 {
    (a / b).floor()
}

/// Python/NumPy remainder: result has the sign of the divisor.
fn float_mod(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && ((r < 0.0) != (b < 0.0)) {
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
    match parse_axis(args, kwargs)? {
        Axis::None => np_float(py, native(slf.get().inner())?),
        Axis::Int(axis) => PyArray::new(along_axis(slf.get().inner(), axis)?).into_py(py),
        Axis::Other => call_method_fallback(slf, name, args, kwargs),
    }
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
    if args.is_empty() && kwargs.map_or(true, |k| k.is_empty()) {
        return native(slf.get().inner());
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
        let float64 = match dtype {
            None => true,
            Some(d) if d.is_none() => true,
            Some(d) => {
                let np = py.import("numpy")?;
                np.getattr("dtype")?.call1((d,))?.getattr("num")?.extract::<i32>()? == 12
            }
        };
        if !float64 || strides.map_or(false, |s| !s.is_none()) {
            return Err(PyTypeError::new_err(
                "lightarray.ndarray holds contiguous float64 data only; use numpy.ndarray for other dtypes or strides",
            ));
        }
        let shape = extract_shape(shape)?;
        match buffer {
            None => Ok(PyArray::new(Array::filled(&shape, 0.0)?)),
            Some(buf) => {
                let np = py.import("numpy")?;
                let kw = PyDict::new(py);
                kw.set_item("dtype", "float64")?;
                kw.set_item("offset", offset)?;
                let flat = np.getattr("frombuffer")?.call((buf,), Some(&kw))?;
                let n: usize = shape.iter().product();
                let taken = flat.get_item(pyo3::types::PySlice::new(py, 0, n as isize, 1))?;
                let arr = PyArray::from_f64_buffer(py, &taken, true)?
                    .ok_or_else(|| PyTypeError::new_err("buffer must expose float64 data"))?;
                Ok(PyArray::new(arr.reshape(&shape)?))
            }
        }
    }

    // ---- attributes ----

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner().shape())
    }

    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner().dims.strides())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner().ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner().size()
    }

    #[getter]
    fn itemsize(&self) -> usize {
        ITEMSIZE as usize
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner().size() * ITEMSIZE as usize
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
        py.import("numpy")?.getattr("dtype")?.call1(("float64",))
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
        match self.inner().shape().first() {
            Some(&n) => Ok(n),
            None => Err(PyTypeError::new_err("len() of unsized object")),
        }
    }

    fn __bool__(&self) -> PyResult<bool> {
        match self.inner().size() {
            1 => Ok(self.inner().data()[0] != 0.0),
            _ => Err(PyValueError::new_err(
                "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()",
            )),
        }
    }

    fn __float__(&self) -> PyResult<f64> {
        match self.inner().size() {
            1 => Ok(self.inner().data()[0]),
            _ => Err(PyTypeError::new_err("only length-1 arrays can be converted to Python scalars")),
        }
    }

    fn __int__(&self) -> PyResult<i64> {
        Ok(self.__float__()? as i64)
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
        let arr = slf.get().inner();
        let ndim = arr.ndim();
        // SAFETY: `view` is a valid Py_buffer; the pointers we store refer to
        // memory owned by `slf`, which `view.obj` keeps alive.
        unsafe {
            (*view).obj = slf.clone().into_any().into_ptr();
            (*view).buf = arr.data().as_ptr() as *mut c_void;
            (*view).len = (arr.size() * ITEMSIZE as usize) as isize;
            (*view).readonly = 0;
            (*view).itemsize = ITEMSIZE;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                c"d".as_ptr() as *mut _
            } else {
                ptr::null_mut()
            };
            // Without PyBUF_ND the consumer expects a flat contiguous buffer
            // (shape NULL, ndim 1), as PyBuffer_FillInfo provides.
            let nd_requested = (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND;
            (*view).ndim = if nd_requested { ndim as c_int } else { 1 };
            (*view).shape = if ndim > 0 && nd_requested {
                arr.dims.shape_ptr() as *mut ffi::Py_ssize_t
            } else {
                ptr::null_mut()
            };
            (*view).strides = if ndim > 0 && (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                arr.dims.strides_ptr() as *mut ffi::Py_ssize_t
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
        let a = slf.get().inner();
        let cmp: fn(f64, f64) -> bool = match op {
            CompareOp::Lt => |x, y| x < y,
            CompareOp::Le => |x, y| x <= y,
            CompareOp::Eq => |x, y| x == y,
            CompareOp::Ne => |x, y| x != y,
            CompareOp::Gt => |x, y| x > y,
            CompareOp::Ge => |x, y| x >= y,
        };
        let bytes: Vec<u8> = match classify(other) {
            Operand::Array(b) if b.get().inner().dims == a.dims => {
                a.data().iter().zip(b.get().inner().data()).map(|(&x, &y)| cmp(x, y) as u8).collect()
            }
            Operand::Scalar(s) => a.data().iter().map(|&x| cmp(x, s) as u8).collect(),
            Operand::Array(_) | Operand::Other if numpy_handles(other) => {
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
            _ => return Ok(not_implemented(py)),
        };
        bool_array(py, bytes, a.shape())
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
            Py::new(slf.py(), PyArray::new(slf.get().inner().clone()))?.into_bound(slf.py())
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
        fallback(slf.py(), "call", ("bitwise_and", slf.clone(), other.clone()), None)
    }
    fn __rand__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("bitwise_and", other.clone(), slf.clone()), None)
    }
    fn __or__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("bitwise_or", slf.clone(), other.clone()), None)
    }
    fn __ror__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("bitwise_or", other.clone(), slf.clone()), None)
    }
    fn __xor__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("bitwise_xor", slf.clone(), other.clone()), None)
    }
    fn __rxor__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        fallback(slf.py(), "call", ("bitwise_xor", other.clone(), slf.clone()), None)
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
        fallback(slf.py(), "call", ("invert", slf.clone()), None)
    }

    fn __index__(&self) -> PyResult<isize> {
        Err(PyTypeError::new_err("only integer scalar arrays can be converted to a scalar index"))
    }

    fn __complex__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyComplex>> {
        Ok(pyo3::types::PyComplex::from_doubles(py, self.__float__()?, 0.0))
    }

    // ---- copying and pickling (matplotlib calls copy.copy on its inputs) ----

    fn __copy__(&self) -> PyArray {
        PyArray::new(self.inner().clone())
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyArray {
        PyArray::new(self.inner().clone())
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyAny>,))> {
        let py = slf.py();
        let constructor = py.import("lightarray._core")?.getattr("array")?;
        let data = py.import("numpy")?.call_method1("array", (slf,))?; // an owned copy
        Ok((constructor, (data,)))
    }

    /// No-op used only to measure raw PyO3 method-call overhead in benchmarks.
    fn _noop(&self) {}

    fn __neg__(&self) -> PyArray {
        PyArray::new(self.inner().map(|x| -x))
    }
    fn __pos__(&self) -> PyArray {
        PyArray::new(self.inner().clone())
    }
    fn __abs__(&self) -> PyArray {
        PyArray::new(self.inner().map(f64::abs))
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
        if let Ok(b) = other.cast::<PyArray>() {
            let (a, b) = (slf.get().inner(), b.get().inner());
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
                let a = slf.get().inner();
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
                let a = slf.get().inner();
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
        let arr = slf.get().inner();
        let shape = extract_shape_args(shape, arr.size())?;
        PyArray::new(arr.reshape(&shape)?).into_py(py)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn copy<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if c_order_only(args, kwargs)? {
            let native = |a: &Array| PyArray::new(a.clone()).into_py(py);
            return native(slf.get().inner());
        }
        call_method_fallback(slf, "copy", args, kwargs)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn flatten<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if c_order_only(args, kwargs)? {
            let native = |a: &Array| PyArray::new(a.reshape(&[a.size()])?).into_py(py);
            return native(slf.get().inner());
        }
        call_method_fallback(slf, "flatten", args, kwargs)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn ravel<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if c_order_only(args, kwargs)? {
            let native = |a: &Array| PyArray::new(a.reshape(&[a.size()])?).into_py(py);
            return native(slf.get().inner());
        }
        call_method_fallback(slf, "ravel", args, kwargs)
    }

    /// `.T`: reversed axes, as a copy.
    #[getter]
    #[pyo3(name = "T")]
    fn transpose_property(&self) -> PyArray {
        PyArray::new(self.inner().transpose())
    }

    /// `transpose()` with no axes natively; explicit axes through NumPy.
    #[pyo3(signature = (*args, **kwargs))]
    fn transpose<'py>(slf: &Bound<'py, Self>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let bare = kwargs.map_or(true, |k| k.is_empty()) && (args.is_empty() || (args.len() == 1 && args.get_item(0)?.is_none()));
        if bare {
            return PyArray::new(slf.get().inner().transpose()).into_py(py);
        }
        call_method_fallback(slf, "transpose", args, kwargs)
    }

    // ---- mutation ----

    /// `a[key] = value`. Integer indices with a scalar value are set
    /// natively; every other form (slices, masks, lists, array values with
    /// broadcasting) is applied by NumPy through the writable view.
    fn __setitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let scalar = match classify(value) {
            Operand::Scalar(v) => Some(v),
            _ => None,
        };
        // Native only for a full integer index; partial indices broadcast the
        // value into a sub-array, which NumPy does through the view.
        if let Some(v) = scalar {
            let ndim = slf.get().inner().ndim();
            if key.is_instance_of::<PyInt>() && ndim == 1 {
                return Ok(slf.get().inner_mut().set(&[key.extract::<isize>()?], v)?);
            }
            if let Ok(tuple) = key.cast::<PyTuple>() {
                if tuple.len() == ndim && tuple.iter().all(|item| item.is_instance_of::<PyInt>()) {
                    let idx: Vec<isize> = tuple.extract()?;
                    return Ok(slf.get().inner_mut().set(&idx, v)?);
                }
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
        match classify(value) {
            Operand::Scalar(v) => {
                slf.get().inner_mut().map_inplace(|_| v);
                Ok(())
            }
            _ => fallback(slf.py(), "call_method", (slf.clone(), "fill", value.clone()), None).map(|_| ()),
        }
    }

    // ---- indexing ----

    /// Integers and slices (any mix, any number of axes) natively as copies;
    /// ellipsis, None, fancy and boolean indexing through NumPy.
    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let arr = slf.get().inner();
        // Type checks first: a failed `extract` builds a Python exception,
        // which costs more than the whole slice.
        if key.is_instance_of::<PyInt>() {
            return index_result(py, arr, &[key.extract::<isize>()?]);
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            if arr.ndim() == 0 {
                return Err(PyIndexError::new_err("too many indices for array: array is 0-dimensional"));
            }
            return PyArray::new(arr.select(&[slice_selector(slice, arr.shape()[0])?])?).into_py(py);
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
                return index_result(py, arr, &ints);
            }
            let selected = arr.select(&sels)?;
            if new_axes.is_empty() {
                return PyArray::new(selected).into_py(py);
            }
            let mut shape: Vec<usize> = selected.shape().to_vec();
            for &pos in &new_axes {
                shape.insert(pos.min(shape.len()), 1);
            }
            return PyArray::new(selected.reshape(&shape)?).into_py(py);
        }
        if key.is_none() {
            let mut shape = vec![1];
            shape.extend_from_slice(arr.shape());
            return PyArray::new(arr.reshape(&shape)?).into_py(py);
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
            return PyArray::new(arr.take_leading(&idx)?).into_py(py);
        }
        // A boolean mask of the array's shape (a NumPy bool array from a comparison).
        if let Some(mask) = bool_mask(key) {
            if mask.1 == arr.shape() {
                return PyArray::new(arr.compress_flat(&mask.0)?).into_py(py);
            }
        }
        // NumPy integer scalars (and anything else with __index__) index too.
        if key.hasattr("__index__")? {
            if let Ok(i) = key.extract::<isize>() {
                return index_result(py, arr, &[i]);
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
    match classify(other) {
        Operand::Array(b) => {
            if b.is(slf) {
                let copy = b.get().inner().clone();
                slf.get().inner_mut().zip_map_inplace(&copy, f)?;
            } else {
                slf.get().inner_mut().zip_map_inplace(b.get().inner(), f)?;
            }
        }
        Operand::Scalar(v) => slf.get().inner_mut().map_inplace(|x| f(x, v)),
        Operand::Other if numpy_handles(other) => {
            fallback(py, "inplace", (slf.clone(), name, other.clone()), None)?;
        }
        Operand::Other => {
            return Err(PyTypeError::new_err(format!(
                "unsupported operand type(s) for in-place {name}: 'lightarray.ndarray' and '{}'",
                other.get_type().name()?
            )))
        }
    }
    Ok(())
}

fn index_result(py: Python<'_>, arr: &Array, idx: &[isize]) -> PyResult<Py<PyAny>> {
    match arr.get(idx)? {
        Some(v) => np_float(py, v),
        None => PyArray::new(arr.index(idx)?).into_py(py),
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
    if let Ok(n) = shape.extract::<usize>() {
        return Ok(vec![n]);
    }
    shape.extract::<Vec<usize>>().map_err(|_| PyTypeError::new_err("shape must be an int or a sequence of ints"))
}

/// `array(obj)` / `asarray(obj)`: a lightarray array when the input is
/// float64-representable natively or NumPy makes a float64 array of it;
/// otherwise NumPy's array with its own dtype (complex, int, str, ...), so
/// dtype semantics match NumPy for inputs lightarray cannot hold yet.
pub fn array_or_numpy(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Some(arr) = native_array(py, obj)? {
        return PyArray::new(arr).into_py(py);
    }
    let np = py.import("numpy")?;
    let converted = np.getattr("asarray")?.call1((obj,))?;
    PyArray::wrap_result(py, converted)
}

/// Native conversion only (lightarray arrays, numbers, nested sequences of
/// numbers, float64 buffers); None for everything else.
fn native_array(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Option<Array>> {
    if let Ok(a) = obj.cast::<PyArray>() {
        return Ok(Some(a.get().inner().clone()));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        return nested_sequence(obj);
    }
    if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>() {
        return Ok(Some(Array::scalar(obj.extract()?)));
    }
    if has_buffer(obj) {
        // NumPy scalars and 0-d arrays keep their own dtype (only float64
        // ones become lightarray); Python numbers above become float64.
        return PyArray::from_f64_buffer(py, obj, false);
    }
    Ok(None)
}

/// Build an `Array` from anything NumPy's `array()` accepts, coercing to
/// float64 through NumPy when needed (used where a float64 array is required).
pub fn array_from_any(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Array> {
    if let Ok(a) = obj.cast::<PyArray>() {
        return Ok(a.get().inner().clone());
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        if let Some(arr) = nested_sequence(obj)? {
            return Ok(arr);
        }
    } else if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>() {
        return Ok(Array::scalar(obj.extract()?));
    } else if has_buffer(obj) {
        if let Some(arr) = PyArray::from_f64_buffer(py, obj, true)? {
            return Ok(arr);
        }
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

/// Shape being discovered while walking nested sequences; fixed-size so the
/// walk allocates nothing but the data.
struct ShapeAcc {
    len: usize,
    dims: [usize; crate::dims::MAX_NDIM],
}

/// Rectangular nested lists/tuples of numbers. Returns None when the input is
/// not such a structure (ragged, contains strings, contains arrays, ...).
fn nested_sequence(obj: &Bound<'_, PyAny>) -> PyResult<Option<Array>> {
    let mut data = Vec::new();
    let mut shape = ShapeAcc { len: 0, dims: [0; crate::dims::MAX_NDIM] };
    if !walk(obj, 0, &mut shape, &mut data)? {
        return Ok(None);
    }
    let shape = &shape.dims[..shape.len];
    // Mixed scalars and sequences at one level slip past `walk`; let NumPy
    // produce the proper error for those.
    if data.len() != shape.iter().product::<usize>() {
        return Ok(None);
    }
    Ok(Some(Array::new(data, shape)?))
}

fn walk(obj: &Bound<'_, PyAny>, depth: usize, shape: &mut ShapeAcc, data: &mut Vec<f64>) -> PyResult<bool> {
    if let Ok(list) = obj.cast::<PyList>() {
        walk_items(list.len(), list.iter(), depth, shape, data)
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        walk_items(tuple.len(), tuple.iter(), depth, shape, data)
    } else {
        Ok(false)
    }
}

fn walk_items<'py>(
    len: usize,
    items: impl Iterator<Item = Bound<'py, PyAny>>,
    depth: usize,
    shape: &mut ShapeAcc,
    data: &mut Vec<f64>,
) -> PyResult<bool> {
    if depth == shape.len {
        if depth + 1 > crate::dims::MAX_NDIM {
            return Ok(false);
        }
        shape.dims[depth] = len;
        shape.len += 1;
    } else if shape.dims[depth] != len {
        return Ok(false);
    }
    if depth == 0 {
        data.reserve(len);
    }
    for item in items {
        if let Ok(f) = item.cast_exact::<PyFloat>() {
            if depth + 1 != shape.len {
                return Ok(false);
            }
            data.push(f.value());
        } else if item.is_instance_of::<PyFloat>() || item.is_instance_of::<PyInt>() {
            if depth + 1 != shape.len {
                return Ok(false);
            }
            data.push(item.extract()?);
        } else if !walk(&item, depth + 1, shape, data)? {
            return Ok(false);
        }
    }
    Ok(true)
}
