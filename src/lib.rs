//! lightarray native core. The public Python API lives in the `lightarray`
//! package (`python/lightarray/__init__.py`), which re-exports this module
//! and fills in everything else from NumPy.

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyTuple};

mod array;
mod dims;
mod dlpack;
mod dtype;
mod python;

pub use array::Array;
pub use dims::{Dims, MAX_NDIM};
pub use dtype::DType;
pub use python::PyArray;

use array::{compare_scalar, AnyArray};
use python::{any_of, array_from_any, array_or_numpy, binary_native, bool_mask, extract_shape, f64_of, fallback, np_bool, where_native};

/// No-op used only to measure raw PyO3 call overhead in benchmarks.
#[pyfunction]
fn _noop() {}

/// Set while `lightarray.patch_module` has rebound NumPy inside some package.
/// Only then does `array()` check whether it is being called from inside an
/// `__array__` method (which must return a NumPy array), so unpatched use
/// pays nothing.
static PATCH_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[pyfunction]
fn _set_patch_active(active: bool) {
    PATCH_ACTIVE.store(active, std::sync::atomic::Ordering::Relaxed);
}

/// Is the calling Python frame an `__array__` method? NumPy requires
/// `__array__` to return a real ndarray, and patched packages implement it
/// with what is now lightarray's `array()`.
fn called_from_dunder_array(py: Python<'_>) -> bool {
    if !PATCH_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    // SAFETY: PyEval_GetFrame borrows the current frame; PyFrame_GetCode
    // returns a new reference which we release.
    unsafe {
        let frame = pyo3::ffi::PyEval_GetFrame();
        if frame.is_null() {
            return false;
        }
        let code = pyo3::ffi::PyFrame_GetCode(frame);
        if code.is_null() {
            return false;
        }
        let code_obj = Bound::from_owned_ptr(py, code as *mut pyo3::ffi::PyObject);
        code_obj.getattr("co_name").and_then(|n| n.extract::<String>()).map_or(false, |n| n == "__array__")
    }
}

/// NumPy's own `asarray`, for results that must be real ndarrays.
fn numpy_array(py: Python<'_>, object: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let kw = PyDict::new(py);
    if let Some(d) = dtype {
        kw.set_item("dtype", d)?;
    }
    Ok(np.getattr("asarray")?.call((object,), Some(&kw))?.unbind())
}

// ---- creation ----------------------------------------------------------

/// `array(object, dtype=None, ...)`: float64 natively, other dtypes and
/// extra keywords via NumPy.
#[pyfunction]
#[pyo3(name = "array", signature = (object, dtype=None, **kwargs), text_signature = "(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)")]
fn array_(py: Python<'_>, object: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    if called_from_dunder_array(py) {
        return numpy_array(py, object, dtype);
    }
    let plain = kwargs.map_or(true, |k| k.is_empty());
    if plain && dtype.map_or(true, |d| d.is_none()) {
        return array_or_numpy(py, object);
    }
    if plain && is_float64_or_none(py, dtype)? {
        return PyArray::new(array_from_any(py, object)?).into_py(py);
    }
    let kw = kwargs.map(|k| k.copy()).transpose()?.unwrap_or_else(|| PyDict::new(py));
    if let Some(d) = dtype {
        kw.set_item("dtype", d)?;
    }
    fallback(py, "call", ("array", object.clone()), Some(&kw))
}

/// `asarray(object, dtype=None, *, copy=None, device=None)`: returns the
/// input itself when it already is a float64 lightarray (unless a copy is
/// requested).
#[pyfunction]
#[pyo3(signature = (object, dtype=None, *, copy=None, device=None, order=None), text_signature = "(a, dtype=None, order=None, *, device=None, copy=None, like=None)")]
fn asarray(py: Python<'_>, object: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, copy: Option<bool>, device: Option<&Bound<'_, PyAny>>, order: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    check_device(device)?;
    let _ = order; // contiguous C order is the only layout
    if called_from_dunder_array(py) {
        return numpy_array(py, object, dtype);
    }
    let passthrough = match any_of(object) {
        Some(AnyArray::F64(_)) => is_float64_or_none(py, dtype)?,
        Some(other) => match dtype {
            None => true,
            Some(d) if d.is_none() => true,
            // the array's own dtype (int64 or bool) asked for explicitly
            Some(d) => {
                let wanted = py.import("numpy")?.getattr("dtype")?.call1((d,))?;
                wanted.getattr("name")?.extract::<String>()? == other.dtype_name()
            }
        },
        None => false,
    };
    if passthrough {
        return match copy {
            Some(true) => object.call_method0("copy").map(|r| r.unbind()),
            _ => Ok(object.clone().unbind()),
        };
    }
    if copy == Some(false) && !object.is_instance_of::<PyArray>() && !has_f64_buffer(object) {
        return Err(pyo3::exceptions::PyValueError::new_err("Unable to avoid copy while creating an array as requested."));
    }
    if dtype.map_or(true, |d| d.is_none()) {
        let result = array_or_numpy(py, object)?;
        if copy == Some(true) && result.bind(py).is(object) {
            // a NumPy array of a dtype lightarray leaves to NumPy
            return object.call_method0("copy").map(|r| r.unbind());
        }
        return Ok(result);
    }
    if is_float64_or_none(py, dtype)? {
        return PyArray::new(array_from_any(py, object)?).into_py(py);
    }
    let kw = PyDict::new(py);
    kw.set_item("dtype", dtype)?;
    // `asarray` shares memory when the dtype already matches; `copy` decides otherwise
    kw.set_item("copy", copy)?;
    if let Some(order) = order {
        kw.set_item("order", order)?;
    }
    let np = py.import("numpy")?;
    let result = np.getattr("asarray")?.call((object,), Some(&kw))?;
    if result.is(object) {
        return Ok(result.unbind());
    }
    PyArray::wrap_result(py, result)
}

/// Only the CPU device exists.
fn check_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    match device {
        None => Ok(()),
        Some(d) if d.is_none() => Ok(()),
        Some(d) => {
            let name = d.str()?.to_string();
            if name == "cpu" {
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported device {name:?}: lightarray arrays live on the CPU")))
            }
        }
    }
}

fn has_f64_buffer(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("__array_interface__").unwrap_or(false)
}

fn is_float64_or_none(py: Python<'_>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    match dtype {
        None => Ok(true),
        Some(d) if d.is_none() => Ok(true),
        // Common spellings without building NumPy dtype objects.
        Some(d) if d.is_exact_instance_of::<pyo3::types::PyString>() => {
            let name: &str = &d.extract::<String>()?;
            if matches!(name, "float64" | "f8" | "d" | "double" | "float" | "<f8" | "=f8") {
                return Ok(true);
            }
            if matches!(name, "float32" | "f4" | "int64" | "i8" | "int32" | "i4" | "bool" | "int" | "uint8" | "complex128") {
                return Ok(false);
            }
            let np = py.import("numpy")?;
            let wanted = np.getattr("dtype")?.call1((d,))?;
            wanted.getattr("num")?.extract::<i32>().map(|n| n == 12)
        }
        Some(d) if d.is(&py.get_type::<PyFloat>()) => Ok(true),
        // Type objects such as np.float64 / np.int32: decide by name.
        Some(d) if d.is_instance_of::<pyo3::types::PyType>() => {
            let name = d.cast::<pyo3::types::PyType>()?.name()?.to_string();
            match name.as_str() {
                "float64" | "float" | "double" => Ok(true),
                "float32" | "float16" | "int64" | "int32" | "int16" | "int8" | "uint64" | "uint32" | "uint16" | "uint8" | "bool" | "bool_" | "int" | "complex128" | "complex64" | "complex" => Ok(false),
                _ => {
                    let np = py.import("numpy")?;
                    np.getattr("dtype")?.call1((d,))?.getattr("num")?.extract::<i32>().map(|n| n == 12)
                }
            }
        }
        Some(d) => {
            let np = py.import("numpy")?;
            let wanted = np.getattr("dtype")?.call1((d,))?;
            let float64 = np.getattr("dtype")?.call1(("float64",))?;
            wanted.eq(float64)
        }
    }
}

fn filled(py: Python<'_>, shape: &Bound<'_, PyAny>, value: f64, dtype: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<Py<PyAny>> {
    // More dimensions than lightarray holds natively: let NumPy make the array.
    let too_many_dims = !shape.is_instance_of::<pyo3::types::PyInt>() && shape.len().map_or(false, |n| n > MAX_NDIM);
    if too_many_dims || called_from_dunder_array(py) || !is_float64_or_none(py, dtype)? {
        let kw = PyDict::new(py);
        kw.set_item("dtype", dtype)?;
        return fallback(py, "call", (name, shape.clone()), Some(&kw));
    }
    PyArray::new(Array::filled(&extract_shape(shape)?, value)?).into_py(py)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None, order=None, *, device=None), text_signature = "(shape, dtype=float, order='C', *, device=None, like=None)")]
fn zeros(py: Python<'_>, shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, order: Option<&Bound<'_, PyAny>>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    check_device(device)?;
    let _ = order;
    filled(py, shape, 0.0, dtype, "zeros")
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None, order=None, *, device=None), text_signature = "(shape, dtype=None, order='C', *, device=None, like=None)")]
fn ones(py: Python<'_>, shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, order: Option<&Bound<'_, PyAny>>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    check_device(device)?;
    let _ = order;
    filled(py, shape, 1.0, dtype, "ones")
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None, order=None, *, device=None), text_signature = "(shape, dtype=float, order='C', *, device=None, like=None)")]
fn empty(py: Python<'_>, shape: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, order: Option<&Bound<'_, PyAny>>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    check_device(device)?;
    let _ = order;
    filled(py, shape, 0.0, dtype, "empty")
}

/// `full(shape, fill_value, dtype=None)`: float and int scalar fill values
/// natively; bools, array-valued fill values (broadcast) and other dtypes
/// through NumPy so the dtype follows NumPy's inference.
#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None, order=None, *, device=None), text_signature = "(shape, fill_value, dtype=None, order='C', *, device=None, like=None)")]
fn full(py: Python<'_>, shape: &Bound<'_, PyAny>, fill_value: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, order: Option<&Bound<'_, PyAny>>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    check_device(device)?;
    let _ = order;
    let too_many_dims = !shape.is_instance_of::<pyo3::types::PyInt>() && shape.len().map_or(false, |n| n > MAX_NDIM);
    let no_dtype = dtype.map_or(true, |d| d.is_none());
    if !too_many_dims && !called_from_dunder_array(py) {
        let dims = extract_shape(shape)?;
        if fill_value.is_instance_of::<pyo3::types::PyBool>() {
            if no_dtype {
                return PyArray::from_any(AnyArray::Bool(Array::filled(&dims, fill_value.is_truthy()?)?)).into_py(py);
            }
        } else if fill_value.is_instance_of::<pyo3::types::PyInt>() {
            if no_dtype {
                if let Ok(v) = fill_value.extract::<i64>() {
                    return PyArray::from_any(AnyArray::I64(Array::filled(&dims, v)?)).into_py(py);
                }
            } else if is_float64_or_none(py, dtype)? {
                return PyArray::new(Array::filled(&dims, fill_value.extract::<f64>()?)?).into_py(py);
            }
        } else if fill_value.is_instance_of::<PyFloat>() && is_float64_or_none(py, dtype)? {
            return PyArray::new(Array::filled(&dims, fill_value.extract::<f64>()?)?).into_py(py);
        }
    }
    let kw = PyDict::new(py);
    kw.set_item("dtype", dtype)?;
    fallback(py, "call", ("full", shape.clone(), fill_value.clone()), Some(&kw))
}

/// `arange([start,] stop[, step])`. Always float64 (NumPy would give int64
/// for integer arguments; that changes when integer dtypes arrive).
#[pyfunction]
#[pyo3(signature = (start, stop=None, step=None, dtype=None, *, device=None), text_signature = "(start, stop=None, step=1, dtype=None, *, device=None, like=None)")]
fn arange(py: Python<'_>, start: &Bound<'_, PyAny>, stop: Option<&Bound<'_, PyAny>>, step: Option<&Bound<'_, PyAny>>, dtype: Option<&Bound<'_, PyAny>>, device: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    check_device(device)?;
    let stop = stop.filter(|s| !s.is_none());
    let step = step.filter(|s| !s.is_none());
    let is_int = |v: &Bound<'_, PyAny>| v.is_instance_of::<pyo3::types::PyInt>() && !v.is_instance_of::<pyo3::types::PyBool>();
    let no_dtype = dtype.map_or(true, |d| d.is_none());
    let plain = !called_from_dunder_array(py);
    if plain && no_dtype && is_int(start) && stop.map_or(true, is_int) && step.map_or(true, is_int) {
        let (a, b) = match stop {
            Some(s) => (start.extract::<i64>()?, s.extract::<i64>()?),
            None => (0, start.extract::<i64>()?),
        };
        let st = step.map_or(Ok(1), |s| s.extract::<i64>())?;
        if st == 0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err("division by zero"));
        }
        let n = if (st > 0 && b > a) || (st < 0 && b < a) { ((b - a).abs() as u64).div_ceil(st.unsigned_abs()) as usize } else { 0 };
        let data: Vec<i64> = (0..n as i64).map(|i| a + i * st).collect();
        return PyArray::from_any(AnyArray::I64(Array::new(data, &[n])?)).into_py(py);
    }
    let as_f64 = |v: &Bound<'_, PyAny>| v.extract::<f64>();
    let numeric = as_f64(start).is_ok() && stop.map_or(true, |s| as_f64(s).is_ok()) && step.map_or(true, |s| as_f64(s).is_ok());
    if plain && numeric && is_float64_or_none(py, dtype)? {
        let (a, b) = match stop {
            Some(s) => (as_f64(start)?, as_f64(s)?),
            None => (0.0, as_f64(start)?),
        };
        let st = step.map_or(Ok(1.0), as_f64)?;
        if st == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err("division by zero"));
        }
        let n = ((b - a) / st).ceil().max(0.0) as usize;
        let data: Vec<f64> = (0..n).map(|i| a + i as f64 * st).collect();
        return PyArray::new(Array::new(data, &[n])?).into_py(py);
    }
    let kw = PyDict::new(py);
    kw.set_item("dtype", dtype)?;
    match (stop, step) {
        (Some(s), Some(t)) => fallback(py, "call", ("arange", start.clone(), s.clone(), t.clone()), Some(&kw)),
        (Some(s), None) => fallback(py, "call", ("arange", start.clone(), s.clone()), Some(&kw)),
        (None, Some(t)) => fallback(py, "call", ("arange", 0, start.clone(), t.clone()), Some(&kw)),
        (None, None) => fallback(py, "call", ("arange", start.clone()), Some(&kw)),
    }
}

/// `linspace(start, stop, num=50, endpoint=True)`. `retstep`, `axis` and
/// array-valued endpoints go through NumPy.
#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true, **kwargs), text_signature = "(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None)")]
fn linspace(py: Python<'_>, start: &Bound<'_, PyAny>, stop: &Bound<'_, PyAny>, num: usize, endpoint: bool, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    if let Some(k) = kwargs {
        if let Some(d) = k.get_item("device")? {
            check_device(Some(&d))?;
            k.del_item("device")?;
        }
    }
    let native = kwargs.map_or(true, |k| k.is_empty()) && !called_from_dunder_array(py);
    match (start.extract::<f64>(), stop.extract::<f64>()) {
        (Ok(a), Ok(b)) if native => {
            let div = if endpoint { num.saturating_sub(1) } else { num };
            let step = if div > 0 { (b - a) / div as f64 } else { 0.0 };
            let mut data: Vec<f64> = (0..num).map(|i| a + i as f64 * step).collect();
            if endpoint && num > 1 {
                data[num - 1] = b;
            }
            PyArray::new(Array::new(data, &[num])?).into_py(py)
        }
        _ => {
            let kw = kwargs.map(|k| k.copy()).transpose()?.unwrap_or_else(|| PyDict::new(py));
            kw.set_item("num", num)?;
            kw.set_item("endpoint", endpoint)?;
            fallback(py, "call", ("linspace", start.clone(), stop.clone()), Some(&kw))
        }
    }
}

// ---- element-wise math ----------------------------------------------------

/// Raw `METH_FASTCALL | METH_KEYWORDS` entry point shared by the native
/// element-wise functions. Called with exactly `arity` positional arguments
/// and no keywords it runs `native` with no argument parsing at all (PyO3's
/// keyword handling costs about 40 ns per call); every other call form
/// (`out=`, `where=`, `dtype=`, a positional `out`) is NumPy's.
///
/// # Safety
/// Must be called by CPython with the vectorcall argument layout.
unsafe fn elementwise_entry(
    name: &'static str,
    arity: usize,
    args: *const *mut ffi::PyObject,
    nargs: ffi::Py_ssize_t,
    kwnames: *mut ffi::PyObject,
    native: impl FnOnce(Python<'_>, &[Bound<'_, PyAny>]) -> PyResult<Py<PyAny>>,
) -> *mut ffi::PyObject {
    // SAFETY: CPython calls us with the GIL held (thread attached).
    let py = unsafe { Python::assume_attached() };
    let nargs = (nargs as usize) & !(1usize << (usize::BITS - 1)); // PyVectorcall_NARGS
    let run = || -> PyResult<Py<PyAny>> {
        // SAFETY: `args` holds `nargs` positional arguments followed by one
        // value per name in `kwnames`, all borrowed for the call.
        unsafe {
            if nargs == arity && kwnames.is_null() {
                let first = Bound::from_borrowed_ptr(py, *args);
                let second = if arity == 2 { Bound::from_borrowed_ptr(py, *args.add(1)) } else { first.clone() };
                let inputs = [first, second];
                return native(py, &inputs[..arity]);
            }
            let mut full: Vec<Bound<'_, PyAny>> = vec![name.into_pyobject(py)?.into_any()];
            for k in 0..nargs {
                full.push(Bound::from_borrowed_ptr(py, *args.add(k)));
            }
            let kw = PyDict::new(py);
            if !kwnames.is_null() {
                let names = Bound::from_borrowed_ptr(py, kwnames).cast_into::<PyTuple>()?;
                for (k, key) in names.iter().enumerate() {
                    kw.set_item(key, Bound::from_borrowed_ptr(py, *args.add(nargs + k)))?;
                }
            }
            fallback(py, "call", PyTuple::new(py, full)?, Some(&kw))
        }
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(result)) => result.into_ptr(),
        Ok(Err(err)) => {
            err.restore(py);
            std::ptr::null_mut()
        }
        Err(_) => {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lightarray.{name} panicked")).restore(py);
            std::ptr::null_mut()
        }
    }
}

type RawFunction = unsafe extern "C" fn(*mut ffi::PyObject, *const *mut ffi::PyObject, ffi::Py_ssize_t, *mut ffi::PyObject) -> *mut ffi::PyObject;

/// Add a raw fastcall function to the module. `name` and `doc` are
/// NUL-terminated; `doc` starts with the text signature.
fn register_raw(m: &Bound<'_, PyModule>, name: &'static str, doc: &'static str, function: RawFunction) -> PyResult<()> {
    let py = m.py();
    let c_name = std::ffi::CStr::from_bytes_with_nul(name.as_bytes()).expect("NUL-terminated name");
    let c_doc = std::ffi::CStr::from_bytes_with_nul(doc.as_bytes()).expect("NUL-terminated doc");
    // The definition must outlive the function object: leak it (once per function).
    let def = Box::leak(Box::new(ffi::PyMethodDef {
        ml_name: c_name.as_ptr(),
        ml_meth: ffi::PyMethodDefPointer { PyCFunctionFastWithKeywords: function },
        ml_flags: ffi::METH_FASTCALL | ffi::METH_KEYWORDS,
        ml_doc: c_doc.as_ptr(),
    }));
    let module_name = m.name()?;
    // SAFETY: `def` is 'static and the pointers it holds are 'static strings.
    let function = unsafe { Bound::from_owned_ptr_or_err(py, ffi::PyCFunction_NewEx(def, std::ptr::null_mut(), module_name.as_ptr()))? };
    m.add(c_name.to_str().expect("ASCII name"), function)
}

/// NumPy's name for a native function (`mod_` is `mod`).
fn numpy_name(name: &'static str) -> &'static str {
    name.trim_end_matches('_')
}

/// Apply `f` to an array natively, a Python number as a float, and anything
/// else through NumPy.
fn unary(py: Python<'_>, name: &str, x: &Bound<'_, PyAny>, f: fn(f64) -> f64) -> PyResult<Py<PyAny>> {
    if let Some(a) = f64_of(x) {
        return PyArray::new(a.map(f)).into_py(py);
    }
    if let Some(any) = any_of(x) {
        // int64/bool input: NumPy keeps the integer dtype for these, and
        // computes in float64 for the rest (sin, exp, sqrt, ...)
        const KEEPS_INT: [&str; 12] = ["floor", "ceil", "trunc", "rint", "negative", "positive", "absolute", "abs", "sign", "square", "reciprocal", "fabs"];
        // (bool input gives float16 in NumPy, so that goes to NumPy too)
        if KEEPS_INT.contains(&name) || matches!(any, AnyArray::Bool(_)) {
            return fallback(py, "call", (name, x.clone()), None);
        }
        return PyArray::new(any.to_f64().map(f)).into_py(py);
    }
    if x.is_exact_instance_of::<pyo3::types::PyFloat>() || x.is_exact_instance_of::<pyo3::types::PyInt>() {
        return Ok(f(x.extract()?).into_pyobject(py)?.into_any().unbind());
    }
    if x.is_instance_of::<pyo3::types::PyFloat>() {
        // np.float64 in, np.float64 out
        return crate::python::np_float(py, f(x.extract()?));
    }
    fallback(py, "call", (name, x.clone()), None)
}

macro_rules! unary_functions {
    ($( $name:ident => $f:expr ),* $(,)?) => {
        $(
            fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
                unary(py, numpy_name(stringify!($name)), x, $f)
            }
        )*
        fn register_unary(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $( {
                unsafe extern "C" fn raw(_module: *mut ffi::PyObject, args: *const *mut ffi::PyObject, nargs: ffi::Py_ssize_t, kwnames: *mut ffi::PyObject) -> *mut ffi::PyObject {
                    // SAFETY: called by CPython through the method definition below.
                    unsafe { elementwise_entry(numpy_name(stringify!($name)), 1, args, nargs, kwnames, |py, xs| $name(py, &xs[0])) }
                }
                register_raw(m, concat!(stringify!($name), "\0"), concat!(stringify!($name), "(x, /, out=None, *, where=True, dtype=None)\n--\n\n", "Element-wise `", stringify!($name), "` with NumPy semantics: native for lightarray arrays (returns lightarray) and Python numbers (returns float), NumPy for anything else.", "\0"), raw)?;
            } )*
            Ok(())
        }
    };
}

unary_functions! {
    sin => f64::sin, cos => f64::cos, tan => f64::tan,
    arcsin => f64::asin, arccos => f64::acos, arctan => f64::atan,
    sinh => f64::sinh, cosh => f64::cosh, tanh => f64::tanh,
    arcsinh => f64::asinh, arccosh => f64::acosh, arctanh => f64::atanh,
    exp => f64::exp, exp2 => f64::exp2, expm1 => f64::exp_m1,
    log => f64::ln, log2 => f64::log2, log10 => f64::log10, log1p => f64::ln_1p,
    sqrt => f64::sqrt, cbrt => f64::cbrt, square => |x| x * x, reciprocal => |x| 1.0 / x,
    absolute => f64::abs, fabs => f64::abs, negative => |x| -x, positive => |x| x,
    sign => |x: f64| if x.is_nan() { x } else if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
    floor => f64::floor, ceil => f64::ceil, trunc => f64::trunc,
    rint => |x: f64| { let r = x.round(); if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r } },
    deg2rad => f64::to_radians, rad2deg => f64::to_degrees,
    radians => f64::to_radians, degrees => f64::to_degrees,
}

/// Element-wise predicates (`isnan`, ...) return NumPy bool arrays, like
/// comparisons, since lightarray has no bool dtype yet.
macro_rules! predicate_functions {
    ($( $name:ident => $f:expr ),* $(,)?) => {
        $(
            fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
                let f: fn(f64) -> bool = $f;
                if let Some(a) = f64_of(x) {
                    return PyArray::from_any(AnyArray::Bool(compare_scalar(a, f))).into_py(py);
                }
                if x.is_instance_of::<PyFloat>() || x.is_instance_of::<pyo3::types::PyInt>() {
                    // NumPy returns np.bool_ for scalars; `~np.isfinite(v)` relies on it.
                    return np_bool(py, f(x.extract()?));
                }
                fallback(py, "call", (stringify!($name), x.clone()), None)
            }
        )*
        fn register_predicates(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $( {
                unsafe extern "C" fn raw(_module: *mut ffi::PyObject, args: *const *mut ffi::PyObject, nargs: ffi::Py_ssize_t, kwnames: *mut ffi::PyObject) -> *mut ffi::PyObject {
                    // SAFETY: called by CPython through the method definition below.
                    unsafe { elementwise_entry(numpy_name(stringify!($name)), 1, args, nargs, kwnames, |py, xs| $name(py, &xs[0])) }
                }
                register_raw(m, concat!(stringify!($name), "\0"), concat!(stringify!($name), "(x, /, out=None, *, where=True)\n--\n\n", "Element-wise `", stringify!($name), "` with NumPy semantics: native for float64 lightarray arrays (returns a bool lightarray) and Python numbers (returns np.bool_), NumPy otherwise.", "\0"), raw)?;
            } )*
            Ok(())
        }
    };
}

predicate_functions! {
    isnan => |v| v.is_nan(), isfinite => |v| v.is_finite(), isinf => |v| v.is_infinite(),
    signbit => |v| v.is_sign_negative(),
}

/// Binary element-wise functions are the operators under their NumPy names.
/// Whichever operand is a lightarray array handles the operation (through
/// its forward or reflected operator); otherwise NumPy does.
macro_rules! binary_functions {
    ($( $name:ident => ($dunder:literal, $rdunder:literal) ),* $(,)?) => {
        $(
            fn $name<'py>(py: Python<'py>, x1: &Bound<'py, PyAny>, x2: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
                let name = numpy_name(stringify!($name));
                match crate::python::operator_function(name, x1, x2)? {
                    Some(result) => Ok(result),
                    None => fallback(py, "call", (name, x1.clone(), x2.clone()), None),
                }
            }
        )*
        fn register_binary(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $( {
                unsafe extern "C" fn raw(_module: *mut ffi::PyObject, args: *const *mut ffi::PyObject, nargs: ffi::Py_ssize_t, kwnames: *mut ffi::PyObject) -> *mut ffi::PyObject {
                    // SAFETY: called by CPython through the method definition below.
                    unsafe { elementwise_entry(numpy_name(stringify!($name)), 2, args, nargs, kwnames, |py, xs| $name(py, &xs[0], &xs[1])) }
                }
                register_raw(m, concat!(stringify!($name), "\0"), concat!(stringify!($name), "(x1, x2, /, out=None, *, where=True, dtype=None)\n--\n\n", "Element-wise `", stringify!($name), "(x1, x2)` with NumPy semantics and broadcasting: native when either operand is a lightarray array, NumPy otherwise.", "\0"), raw)?;
            } )*
            Ok(())
        }
    };
}

binary_functions! {
    add => ("__add__", "__radd__"), subtract => ("__sub__", "__rsub__"),
    multiply => ("__mul__", "__rmul__"), divide => ("__truediv__", "__rtruediv__"),
    true_divide => ("__truediv__", "__rtruediv__"), floor_divide => ("__floordiv__", "__rfloordiv__"),
    remainder => ("__mod__", "__rmod__"), mod_ => ("__mod__", "__rmod__"), power => ("__pow__", "__rpow__"),
}

/// Module-level reductions: `la.sum(a, axis=...)` forwards to the method,
/// which decides between native and NumPy. Non-lightarray inputs go to NumPy.
macro_rules! method_functions {
    ($( $name:ident => ($py_name:literal, $sig:literal) ),* $(,)?) => {
        $(
            #[doc = concat!("`", $py_name, $sig, "`: the same as the `", $py_name, "` method for lightarray arrays (native when the arguments allow it), NumPy's function otherwise.")]
            #[pyfunction]
            #[pyo3(name = $py_name, signature = (a, *args, **kwargs), text_signature = $sig)]
            fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
                if let Ok(arr) = a.cast_exact::<PyArray>() {
                    if let Some(result) = crate::python::call_native_method(arr, $py_name, args, kwargs) {
                        return result;
                    }
                }
                if a.is_instance_of::<PyArray>() {
                    return a.call_method($py_name, args.clone(), kwargs).map(|r| r.unbind());
                }
                let mut full: Vec<Bound<'py, PyAny>> = vec![$py_name.into_pyobject(py)?.into_any(), a.clone()];
                full.extend(args.iter());
                fallback(py, "call", PyTuple::new(py, full)?, kwargs)
            }
        )*
        fn register_method_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $( m.add_function(wrap_pyfunction!($name, m)?)?; )*
            Ok(())
        }
    };
}

/// `any(a)` / `all(a)`: lightarray arrays through the method; NumPy bool
/// arrays (the results of comparisons) natively too, since `np.all(mask)`
/// is what code does with them; everything else NumPy.
macro_rules! bool_reductions {
    ($( $name:ident => ($py_name:literal, $fold:expr) ),* $(,)?) => {
        $(
            #[doc = concat!("`", $py_name, "(a, axis=None, out=None, keepdims=False, *, where=True)`: native for lightarray arrays and NumPy bool arrays, NumPy otherwise.")]
            #[pyfunction]
            #[pyo3(name = $py_name, signature = (a, *args, **kwargs), text_signature = "(a, axis=None, out=None, keepdims=False, *, where=True)")]
            fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
                if let Ok(arr) = a.cast_exact::<PyArray>() {
                    if let Some(result) = crate::python::call_native_method(arr, $py_name, args, kwargs) {
                        return result;
                    }
                }
                if a.is_instance_of::<PyArray>() {
                    return a.call_method($py_name, args.clone(), kwargs).map(|r| r.unbind());
                }
                if args.is_empty() && kwargs.map_or(true, |k| k.is_empty()) {
                    if let Some((mask, _)) = bool_mask(a) {
                        let fold: fn(&[bool]) -> bool = $fold;
                        return np_bool(py, fold(&mask));
                    }
                }
                let mut full: Vec<Bound<'py, PyAny>> = vec![$py_name.into_pyobject(py)?.into_any(), a.clone()];
                full.extend(args.iter());
                fallback(py, "call", PyTuple::new(py, full)?, kwargs)
            }
        )*
        fn register_bool_reductions(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $( m.add_function(wrap_pyfunction!($name, m)?)?; )*
            Ok(())
        }
    };
}

bool_reductions! { any_fn => ("any", |m| m.iter().any(|&b| b)), all_fn => ("all", |m| m.iter().all(|&b| b)) }

method_functions! {
    sum_fn => ("sum", "(a, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)"),
    prod_fn => ("prod", "(a, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)"),
    mean_fn => ("mean", "(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True)"),
    max_fn => ("max", "(a, axis=None, out=None, keepdims=False, initial=None, where=True)"),
    min_fn => ("min", "(a, axis=None, out=None, keepdims=False, initial=None, where=True)"),
    var_fn => ("var", "(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=None, correction=None)"),
    std_fn => ("std", "(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=None, correction=None)"),
    argmax_fn => ("argmax", "(a, axis=None, out=None, *, keepdims=False)"),
    argmin_fn => ("argmin", "(a, axis=None, out=None, *, keepdims=False)"),
    cumsum_fn => ("cumsum", "(a, axis=None, dtype=None, out=None)"),
    cumprod_fn => ("cumprod", "(a, axis=None, dtype=None, out=None)"),
    dot_fn => ("dot", "(a, b, out=None)"),
    clip_fn => ("clip", "(a, min=None, max=None, out=None, **kwargs)"),
    round_fn => ("round", "(a, decimals=0, out=None)"),
    reshape_fn => ("reshape", "(a, shape=None, order='C', *, newshape=None, copy=None)"),
    copy_fn => ("copy", "(a, order='K', subok=False)"),
    nonzero_fn => ("nonzero", "(a)"),
    ravel_fn => ("ravel", "(a, order='C')"),
    transpose_fn => ("transpose", "(a, axes=None)"),
}

macro_rules! binary_math_functions {
    ($( $name:ident => $f:expr ),* $(,)?) => {
        $(
            fn $name<'py>(py: Python<'py>, x1: &Bound<'py, PyAny>, x2: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
                binary_native(py, numpy_name(stringify!($name)), x1, x2, $f)
            }
        )*
        fn register_binary_math(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $( {
                unsafe extern "C" fn raw(_module: *mut ffi::PyObject, args: *const *mut ffi::PyObject, nargs: ffi::Py_ssize_t, kwnames: *mut ffi::PyObject) -> *mut ffi::PyObject {
                    // SAFETY: called by CPython through the method definition below.
                    unsafe { elementwise_entry(numpy_name(stringify!($name)), 2, args, nargs, kwnames, |py, xs| $name(py, &xs[0], &xs[1])) }
                }
                register_raw(m, concat!(stringify!($name), "\0"), concat!(stringify!($name), "(x1, x2, /, out=None, *, where=True, dtype=None)\n--\n\n", "Element-wise `", stringify!($name), "(x1, x2)` with NumPy semantics and broadcasting: native for lightarray arrays and scalars, NumPy otherwise.", "\0"), raw)?;
            } )*
            Ok(())
        }
    };
}

binary_math_functions! {
    maximum => |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) },
    minimum => |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) },
    fmax => f64::max, fmin => f64::min,
    arctan2 => f64::atan2, hypot => f64::hypot, copysign => f64::copysign,
    logaddexp => |a: f64, b: f64| {
        let m = a.max(b);
        if a.is_nan() || b.is_nan() { f64::NAN } else if m.is_infinite() { m } else { m + ((a - m).exp() + (b - m).exp()).ln() }
    },
}

/// `concatenate(arrays, axis=0)` natively for lightarray inputs; anything
/// else (mixed types, `out=`, `dtype=`) through NumPy.
#[pyfunction]
#[pyo3(signature = (arrays, axis=None, **kwargs))]
fn concatenate(py: Python<'_>, arrays: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    let axis_value: PyResult<isize> = axis.map_or(Ok(0), |a| a.extract::<isize>());
    if kwargs.map_or(true, |k| k.is_empty()) {
        if let (Ok(items), Ok(ax)) = (arrays.extract::<Vec<Bound<'_, PyAny>>>(), axis_value) {
            let parts: Vec<PyRef<'_, PyArray>> = match items.iter().map(|i| i.extract::<PyRef<'_, PyArray>>()).collect() {
                Ok(p) => p,
                Err(_) => Vec::new(),
            };
            if !parts.is_empty() && parts.len() == items.len() {
                let refs: Vec<&AnyArray> = parts.iter().map(|p| p.arr()).collect();
                if let Some(joined) = AnyArray::concatenate(&refs, ax)? {
                    return PyArray::from_any(joined).into_py(py);
                }
            }
        }
    }
    let kw = kwargs.map(|k| k.copy()).transpose()?.unwrap_or_else(|| PyDict::new(py));
    kw.set_item("axis", axis.map_or(0isize.into_pyobject(py)?.into_any(), |a| a.clone()))?;
    fallback(py, "call", ("concatenate", arrays.clone()), Some(&kw))
}

/// `stack(arrays, axis=0)`: insert a new axis of length one, then concatenate.
#[pyfunction]
#[pyo3(signature = (arrays, axis=None, **kwargs))]
fn stack(py: Python<'_>, arrays: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    let axis_value: PyResult<isize> = axis.map_or(Ok(0), |a| a.extract::<isize>());
    if kwargs.map_or(true, |k| k.is_empty()) {
        if let (Ok(items), Ok(ax)) = (arrays.extract::<Vec<Bound<'_, PyAny>>>(), axis_value) {
            let parts: Vec<PyRef<'_, PyArray>> = match items.iter().map(|i| i.extract::<PyRef<'_, PyArray>>()).collect() {
                Ok(p) => p,
                Err(_) => Vec::new(),
            };
            if !parts.is_empty() && parts.len() == items.len() && parts.iter().all(|p| p.arr().shape() == parts[0].arr().shape()) {
                let ndim = parts[0].arr().ndim() as isize + 1;
                if ax >= -ndim && ax < ndim {
                    let ax = if ax < 0 { (ax + ndim) as usize } else { ax as usize };
                    let mut shape = parts[0].arr().shape().to_vec();
                    shape.insert(ax, 1);
                    let expanded: Vec<AnyArray> = parts.iter().map(|p| p.arr().reshape(&shape)).collect::<Result<_, _>>()?;
                    let refs: Vec<&AnyArray> = expanded.iter().collect();
                    if let Some(joined) = AnyArray::concatenate(&refs, ax as isize)? {
                        return PyArray::from_any(joined).into_py(py);
                    }
                }
            }
        }
    }
    let kw = kwargs.map(|k| k.copy()).transpose()?.unwrap_or_else(|| PyDict::new(py));
    kw.set_item("axis", axis.map_or(0isize.into_pyobject(py)?.into_any(), |a| a.clone()))?;
    fallback(py, "call", ("stack", arrays.clone()), Some(&kw))
}

/// `isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)`: native for
/// lightarray arrays and scalars (result is a NumPy bool array or np.bool_),
/// NumPy otherwise.
#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-05, atol=1e-08, equal_nan=false))]
fn isclose(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, rtol: f64, atol: f64, equal_nan: bool) -> PyResult<Py<PyAny>> {
    let close = move |x: f64, y: f64| -> bool {
        if x.is_nan() || y.is_nan() {
            return equal_nan && x.is_nan() && y.is_nan();
        }
        if x.is_infinite() || y.is_infinite() {
            return x == y;
        }
        (x - y).abs() <= atol + rtol * y.abs()
    };
    let scalar = |v: &Bound<'_, PyAny>| -> Option<f64> {
        if v.is_instance_of::<PyFloat>() || v.is_instance_of::<pyo3::types::PyInt>() { v.extract::<f64>().ok() } else { None }
    };
    let mask = match (f64_of(a), f64_of(b)) {
        (Some(x), Some(y)) => array::compare(x, y, close),
        (Some(x), None) => scalar(b).map(|v| compare_scalar(x, |p| close(p, v))),
        (None, Some(y)) => scalar(a).map(|v| compare_scalar(y, |q| close(v, q))),
        (None, None) => {
            if let (Some(p), Some(q)) = (scalar(a), scalar(b)) {
                return np_bool(py, close(p, q));
            }
            None
        }
    };
    if let Some(mask) = mask {
        return PyArray::from_any(AnyArray::Bool(mask)).into_py(py);
    }
    let kw = PyDict::new(py);
    kw.set_item("rtol", rtol)?;
    kw.set_item("atol", atol)?;
    kw.set_item("equal_nan", equal_nan)?;
    fallback(py, "call", ("isclose", a.clone(), b.clone()), Some(&kw))
}

#[pyfunction]
#[pyo3(name = "where", signature = (condition, x=None, y=None))]
fn where_(py: Python<'_>, condition: &Bound<'_, PyAny>, x: Option<&Bound<'_, PyAny>>, y: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    match (x, y) {
        (Some(x), Some(y)) => where_native(py, condition, x, y),
        (None, None) => match any_of(condition) {
            Some(_) => condition.call_method0("nonzero").map(|r| r.unbind()),
            None => fallback(py, "call", ("where", condition.clone()), None),
        },
        _ => Err(pyo3::exceptions::PyValueError::new_err("either both or neither of x and y should be given")),
    }
}

/// `_view_of(source, numpy_view)`: a lightarray view sharing memory with
/// `source` that has the layout of `numpy_view`, or None (see `view_of_numpy`).
#[pyfunction]
fn _view_of(source: &Bound<'_, PyArray>, numpy_view: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    crate::python::view_of_numpy(source, numpy_view)
}

/// `argsort(a, axis=-1, kind=None, order=None, *, stable=None, descending=False)`:
/// the `argsort` method for lightarray arrays, NumPy's function otherwise.
/// `descending` is the Array API keyword NumPy lacks.
#[pyfunction]
#[pyo3(signature = (a, *args, **kwargs), text_signature = "(a, axis=-1, kind=None, order=None, *, stable=None, descending=False)")]
fn argsort<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
    if let Some(k) = kwargs {
        if let Some(d) = k.get_item("descending")? {
            let rest = k.copy()?;
            rest.del_item("descending")?;
            if d.is_truthy()? {
                let mut full: Vec<Bound<'py, PyAny>> = vec![a.clone()];
                full.extend(args.iter());
                return fallback(py, "argsort_descending", PyTuple::new(py, full)?, Some(&rest));
            }
            return argsort(py, a, args, Some(&rest));
        }
    }
    if a.is_instance_of::<PyArray>() {
        return a.call_method("argsort", args.clone(), kwargs).map(|r| r.unbind());
    }
    let mut full: Vec<Bound<'py, PyAny>> = vec!["argsort".into_pyobject(py)?.into_any(), a.clone()];
    full.extend(args.iter());
    fallback(py, "call", PyTuple::new(py, full)?, crate::python::stable_by_default(py, args, kwargs)?.as_ref())
}

/// `sort(a, axis=-1, kind=None, order=None, *, stable=None, descending=False)`:
/// native for 1-D lightarray input; NumPy otherwise (`descending` is the
/// Array API keyword NumPy lacks, applied by reversing along the axis).
#[pyfunction]
#[pyo3(signature = (a, *args, **kwargs), text_signature = "(a, axis=-1, kind=None, order=None, *, stable=None, descending=False)")]
fn sort<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>> {
    let mut descending = false;
    let kw = kwargs.map(|k| k.copy()).transpose()?;
    if let Some(k) = kw.as_ref() {
        if let Some(d) = k.get_item("descending")? {
            descending = d.is_truthy()?;
            k.del_item("descending")?;
        }
        if let Some(st) = k.get_item("stable")? {
            if st.is_none() {
                k.del_item("stable")?;
            }
        }
    }
    let plain = kw.as_ref().map_or(true, |k| k.is_empty());
    if let Some(inner) = f64_of(a) {
        let axis_ok = args.is_empty() || (args.len() == 1 && args.get_item(0)?.extract::<isize>().map_or(false, |ax| ax == -1 || ax == 0));
        if inner.ndim() == 1 && axis_ok && plain {
            let mut sorted = inner.sorted_1d()?;
            if descending {
                sorted = sorted.select(&[crate::array::Selector::Slice { start: sorted.size() as isize - 1, stop: -1, step: -1 }])?;
            }
            return PyArray::new(sorted).into_py(py);
        }
    }
    let mut full: Vec<Bound<'py, PyAny>> = vec!["sort".into_pyobject(py)?.into_any(), a.clone()];
    full.extend(args.iter());
    let result = fallback(py, "call", PyTuple::new(py, full)?, kw.as_ref())?;
    if descending {
        let axis = match (args.len(), kw.as_ref()) {
            (n, _) if n >= 1 => args.get_item(0)?,
            (_, Some(k)) => k.get_item("axis")?.unwrap_or_else(|| (-1isize).into_pyobject(py).unwrap().into_any()),
            _ => (-1isize).into_pyobject(py)?.into_any(),
        };
        return fallback(py, "call", ("flip", result.bind(py).clone(), axis), None);
    }
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (x, *args, **kwargs))]
fn abs(py: Python<'_>, x: &Bound<'_, PyAny>, args: &Bound<'_, pyo3::types::PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    if args.is_empty() && kwargs.map_or(true, |k| k.is_empty()) {
        return unary(py, "abs", x, f64::abs);
    }
    fallback(py, "call", ("abs", x.clone()), kwargs)
}

/// lightarray native core
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add("MAX_NDIM", MAX_NDIM)?;
    m.add_function(wrap_pyfunction!(_noop, m)?)?;
    m.add_function(wrap_pyfunction!(_set_patch_active, m)?)?;
    for f in [
        wrap_pyfunction!(array_, m)?,
        wrap_pyfunction!(asarray, m)?,
        wrap_pyfunction!(zeros, m)?,
        wrap_pyfunction!(ones, m)?,
        wrap_pyfunction!(empty, m)?,
        wrap_pyfunction!(full, m)?,
        wrap_pyfunction!(arange, m)?,
        wrap_pyfunction!(linspace, m)?,
        wrap_pyfunction!(abs, m)?,
        wrap_pyfunction!(concatenate, m)?,
        wrap_pyfunction!(stack, m)?,
        wrap_pyfunction!(where_, m)?,
        wrap_pyfunction!(sort, m)?,
        wrap_pyfunction!(argsort, m)?,
        wrap_pyfunction!(_view_of, m)?,
        wrap_pyfunction!(isclose, m)?,
    ] {
        m.add_function(f)?;
    }
    register_unary(m)?;
    register_binary(m)?;
    register_method_functions(m)?;
    register_binary_math(m)?;
    register_predicates(m)?;
    register_bool_reductions(m)?;
    Ok(())
}
