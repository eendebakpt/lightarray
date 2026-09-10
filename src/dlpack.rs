//! DLPack export (`__dlpack__`, `__dlpack_device__`), versioned protocol 1.0.
//!
//! Consumers (`numpy.from_dlpack`, torch, jax) receive a capsule holding a
//! `DLManagedTensorVersioned` that points straight at the array's buffer
//! (writable, like a NumPy view). The capsule keeps the Python array alive
//! until the consumer calls the deleter.

use crate::dims::ITEMSIZE;
use crate::python::PyArray;
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::{c_char, c_void};

const K_DL_CPU: i32 = 1;
const K_DL_FLOAT: u8 = 2;
const CAPSULE_NAME: &std::ffi::CStr = c"dltensor_versioned";

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

/// Everything the consumer's pointer must keep alive, in one allocation.
/// The managed tensor is the first field so a pointer to it is a pointer to
/// the container.
#[repr(C)]
struct Export {
    managed: DLManagedTensorVersioned,
    shape: [i64; crate::dims::MAX_NDIM],
    strides: [i64; crate::dims::MAX_NDIM],
    /// Strong reference to the array whose buffer `dl_tensor.data` points at.
    owner: *mut ffi::PyObject,
}

unsafe extern "C" fn deleter(managed: *mut DLManagedTensorVersioned) {
    // SAFETY: `managed` was produced by `export` below as the first field of
    // a `Box<Export>`; the consumer calls this exactly once.
    unsafe {
        let export = Box::from_raw(managed as *mut Export);
        Python::attach(|_py| ffi::Py_DECREF(export.owner));
        drop(export);
    }
}

unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // A consumer renames the capsule to "used_dltensor_versioned" and takes
    // over the deleter; only an unconsumed capsule still owns the tensor.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, CAPSULE_NAME.as_ptr()) == 1 {
            let ptr = ffi::PyCapsule_GetPointer(capsule, CAPSULE_NAME.as_ptr()) as *mut DLManagedTensorVersioned;
            if let Some(del) = (*ptr).deleter {
                del(ptr);
            }
        }
    }
}

/// Build the capsule for `array`.
pub fn export(array: &Bound<'_, PyArray>) -> PyResult<Py<PyAny>> {
    let py = array.py();
    let inner = &array.get().inner();
    let ndim = inner.ndim();
    let mut shape = [0i64; crate::dims::MAX_NDIM];
    let mut strides = [0i64; crate::dims::MAX_NDIM];
    for (k, &d) in inner.shape().iter().enumerate() {
        shape[k] = d as i64;
        strides[k] = (inner.dims.strides()[k] / ITEMSIZE) as i64; // DLPack strides are in elements
    }
    let mut export = Box::new(Export {
        managed: DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(deleter),
            flags: 0,
            dl_tensor: DLTensor {
                data: inner.data().as_ptr() as *mut c_void,
                device: DLDevice { device_type: K_DL_CPU, device_id: 0 },
                ndim: ndim as i32,
                dtype: DLDataType { code: K_DL_FLOAT, bits: 64, lanes: 1 },
                shape: std::ptr::null_mut(),
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
        },
        shape,
        strides,
        owner: array.clone().into_any().into_ptr(), // strong reference, released by `deleter`
    });
    export.managed.dl_tensor.shape = export.shape.as_mut_ptr();
    export.managed.dl_tensor.strides = export.strides.as_mut_ptr();
    export.managed.manager_ctx = &mut *export as *mut Export as *mut c_void;
    let raw = Box::into_raw(export) as *mut c_void;
    // SAFETY: `raw` is a valid heap pointer owned by the capsule until a
    // consumer takes it over or the capsule destructor runs.
    let capsule = unsafe { ffi::PyCapsule_New(raw, CAPSULE_NAME.as_ptr() as *const c_char, Some(capsule_destructor)) };
    if capsule.is_null() {
        // SAFETY: undo the export; nothing else references it yet.
        unsafe { deleter(raw as *mut DLManagedTensorVersioned) };
        return Err(PyBufferError::new_err("could not create DLPack capsule"));
    }
    // SAFETY: `capsule` is a new strong reference.
    Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.unbind())
}

/// Legacy (unversioned) `DLManagedTensor` in a "dltensor" capsule, for
/// consumers that do not pass `max_version`.
#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
struct LegacyExport {
    managed: DLManagedTensor,
    shape: [i64; crate::dims::MAX_NDIM],
    strides: [i64; crate::dims::MAX_NDIM],
    owner: *mut ffi::PyObject,
}

const LEGACY_CAPSULE_NAME: &std::ffi::CStr = c"dltensor";

unsafe extern "C" fn legacy_deleter(managed: *mut DLManagedTensor) {
    // SAFETY: produced by `export_legacy` as the first field of a Box<LegacyExport>.
    unsafe {
        let export = Box::from_raw(managed as *mut LegacyExport);
        Python::attach(|_py| ffi::Py_DECREF(export.owner));
        drop(export);
    }
}

unsafe extern "C" fn legacy_capsule_destructor(capsule: *mut ffi::PyObject) {
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, LEGACY_CAPSULE_NAME.as_ptr()) == 1 {
            let ptr = ffi::PyCapsule_GetPointer(capsule, LEGACY_CAPSULE_NAME.as_ptr()) as *mut DLManagedTensor;
            if let Some(del) = (*ptr).deleter {
                del(ptr);
            }
        }
    }
}

pub fn export_legacy(array: &Bound<'_, PyArray>) -> PyResult<Py<PyAny>> {
    let py = array.py();
    let inner = array.get().inner();
    let ndim = inner.ndim();
    let mut shape = [0i64; crate::dims::MAX_NDIM];
    let mut strides = [0i64; crate::dims::MAX_NDIM];
    for (k, &d) in inner.shape().iter().enumerate() {
        shape[k] = d as i64;
        strides[k] = (inner.dims.strides()[k] / ITEMSIZE) as i64;
    }
    let mut export = Box::new(LegacyExport {
        managed: DLManagedTensor {
            dl_tensor: DLTensor {
                data: inner.data().as_ptr() as *mut c_void,
                device: DLDevice { device_type: K_DL_CPU, device_id: 0 },
                ndim: ndim as i32,
                dtype: DLDataType { code: K_DL_FLOAT, bits: 64, lanes: 1 },
                shape: std::ptr::null_mut(),
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(legacy_deleter),
        },
        shape,
        strides,
        owner: array.clone().into_any().into_ptr(),
    });
    export.managed.dl_tensor.shape = export.shape.as_mut_ptr();
    export.managed.dl_tensor.strides = export.strides.as_mut_ptr();
    let raw = Box::into_raw(export) as *mut c_void;
    // SAFETY: as in `export`.
    let capsule = unsafe { ffi::PyCapsule_New(raw, LEGACY_CAPSULE_NAME.as_ptr() as *const c_char, Some(legacy_capsule_destructor)) };
    if capsule.is_null() {
        unsafe { legacy_deleter(raw as *mut DLManagedTensor) };
        return Err(PyBufferError::new_err("could not create DLPack capsule"));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.unbind())
}

/// (device_type, device_id) for `__dlpack_device__`: CPU.
pub fn device() -> (i32, i32) {
    (K_DL_CPU, 0)
}
