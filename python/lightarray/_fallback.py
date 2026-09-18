"""Delegation to NumPy for everything lightarray does not implement natively.

Arguments are converted to NumPy (zero-copy through the buffer protocol),
the NumPy function runs, and float64 array results are copied back into
lightarray arrays. Other results (integer, boolean or complex arrays,
scalars) pass through unchanged; tuples and lists are converted element-wise.

`install_ndarray_methods` gives the native ``ndarray`` class every method
and property of ``numpy.ndarray`` it does not implement itself, so the two
classes have the same attribute surface.
"""

from __future__ import annotations

import functools
import types

import numpy as np

from lightarray import _core

_ndarray = _core.ndarray


def to_numpy(obj):
    """lightarray -> read-only NumPy view; lists and tuples are converted recursively."""
    if isinstance(obj, _ndarray):
        return np.asarray(obj)
    t = type(obj)
    if t is list:
        return [to_numpy(x) for x in obj]
    if t is tuple:
        return tuple([to_numpy(x) for x in obj])
    return obj


def _args_to_numpy(args):
    """`to_numpy` over an argument tuple, with the common cases inlined."""
    out = []
    for x in args:
        if isinstance(x, _ndarray):
            out.append(np.asarray(x))
        elif type(x) is list or type(x) is tuple:
            out.append(to_numpy(x))
        else:
            out.append(x)
    return out


_FLOAT64_NUM = np.dtype("float64").num
#: dtypes lightarray stores natively; NumPy results of these come back as lightarray
_NATIVE_NUMS = frozenset({np.dtype("float64").num, np.dtype("int64").num, np.dtype("bool").num})


#: Set by lightarray.patch_module while any package is patched: then results
#: produced inside an `__array__` method must stay NumPy arrays.
patch_active = False


def _inside_dunder_array():
    import sys

    f = sys._getframe(2)
    for _ in range(6):  # wrapper, helpers and a possible ufunc proxy sit between us and __array__
        if f is None:
            return False
        if f.f_code.co_name == "__array__":
            return True
        f = f.f_back
    return False


def from_numpy(obj):
    """float64, int64 and bool ndarrays -> lightarray; other objects are returned unchanged."""
    t = type(obj)
    if t is np.ndarray:
        if obj.dtype.num in _NATIVE_NUMS and obj.ndim <= _core.MAX_NDIM and not (patch_active and _inside_dunder_array()):
            return _core.array(obj)
        return obj
    if t is tuple:
        return tuple([from_numpy(x) for x in obj])
    if t is list:
        return [from_numpy(x) for x in obj]
    if isinstance(obj, np.ndarray):  # subclasses (matrix, masked arrays) stay NumPy
        return obj
    return obj


def _kwargs_to_numpy(kwargs):
    return {k: to_numpy(v) for k, v in kwargs.items()}


#: Number of operations delegated to NumPy since import. Every fallback goes
#: through `invoke`, so this counts them all; useful to see how much of a
#: script runs natively (see tests/test_dropin.py).
calls = 0


def invoke(func, args, kwargs):
    """The single choke point for delegation: convert, call NumPy, convert back."""
    global calls
    calls += 1
    if kwargs:
        return from_numpy(func(*_args_to_numpy(args), **_kwargs_to_numpy(kwargs)))
    return from_numpy(func(*_args_to_numpy(args)))


def call(name, *args, **kwargs):
    """Call NumPy's `name` with converted arguments."""
    return invoke(getattr(np, name), args, kwargs)


def call_method(arr, name, *args, **kwargs):
    """Call ndarray method `name` on the NumPy view of `arr`."""
    if name in ("std", "var") and "correction" in kwargs:
        # Array API spelling; the ndarray methods only know ddof.
        kwargs["ddof"] = kwargs.pop("correction")
    return invoke(getattr(np.asarray(arr), name), args, kwargs)


def _index(key):
    """Integer-valued lightarray arrays used as indices become int64 arrays
    (lightarray has no integer dtype yet); everything else is converted as usual."""
    if isinstance(key, _ndarray):
        v = np.asarray(key)
        # float64 arrays holding whole numbers are meant as indices; int64 and
        # bool arrays (index arrays, masks) are passed through as they are
        if v.dtype.kind == "f" and (v.size == 0 or (v == np.floor(v)).all()):
            return v.astype(np.int64)
        return v
    if type(key) is tuple:
        return tuple(_index(k) for k in key)
    if type(key) is list:
        return [_index(k) for k in key]
    return key


def setitem(arr, key, value):
    """`arr[key] = value` through the writable NumPy view (all NumPy indexing forms)."""
    global calls
    calls += 1
    np.asarray(arr)[_index(key)] = to_numpy(value)


def inplace(arr, ufunc_name, other):
    """`arr OP= other` for operands only NumPy can handle, written into the view."""
    global calls
    calls += 1
    view = np.asarray(arr)
    getattr(np, ufunc_name)(view, to_numpy(other), out=view)


def getitem(arr, key):
    global calls
    calls += 1
    return from_numpy(np.asarray(arr)[_index(key)])


def wrap_function(func):
    """Make a NumPy function accept and return lightarray arrays."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return invoke(func, args, kwargs)

    wrapper.__module__ = "lightarray"
    return wrapper


class ModuleProxy:
    """A NumPy submodule (linalg, fft, random, ...) whose functions are wrapped."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        obj = getattr(self._module, name)
        if isinstance(obj, types.ModuleType):
            obj = ModuleProxy(obj)
        elif callable(obj) and not isinstance(obj, type):
            obj = wrap_function(obj)
        setattr(self, name, obj)
        return obj

    def __dir__(self):
        return dir(self._module)

    def __repr__(self):
        return f"<lightarray proxy of {self._module.__name__}>"


# Private NumPy names that other libraries (SciPy) reach for anyway.
_PRIVATE_PASSTHROUGH = ("_CopyMode", "_NoValue")


class UfuncProxy:
    """Stand-in for a NumPy ufunc inside a patched package: calling it runs
    lightarray's function, while the ufunc attributes (`reduce`,
    `accumulate`, `outer`, `at`, `nin`, ...) come from NumPy's ufunc, with
    lightarray arguments converted. Only patched packages see these objects,
    so direct lightarray calls keep their plain, faster functions."""

    def __init__(self, call, ufunc):
        self._call = call
        self._ufunc = ufunc
        self.__name__ = ufunc.__name__
        self.__doc__ = ufunc.__doc__
        self.__wrapped__ = ufunc

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __getattr__(self, name):
        attr = getattr(self._ufunc, name)
        return wrap_function(attr) if callable(attr) else attr

    def __repr__(self):
        return f"<lightarray proxy of ufunc '{self._ufunc.__name__}'>"


def populate_module(namespace):
    """Add every public NumPy name missing from `namespace`, wrapped."""
    for name in _PRIVATE_PASSTHROUGH:
        if hasattr(np, name):
            namespace[name] = getattr(np, name)
    for name in dir(np):
        if name.startswith("_") or name in namespace:
            continue
        obj = getattr(np, name)
        if isinstance(obj, types.ModuleType):
            obj = ModuleProxy(obj)
        elif callable(obj) and not isinstance(obj, type):
            obj = wrap_function(obj)
        namespace[name] = obj


def _make_method(name):
    def method(self, *args, **kwargs):
        return invoke(getattr(np.asarray(self), name), args, kwargs)

    method.__name__ = name
    method.__qualname__ = f"ndarray.{name}"
    method.__doc__ = getattr(np.ndarray, name).__doc__
    return method


def _make_property(name):
    def fget(self):
        global calls
        calls += 1
        return from_numpy(getattr(np.asarray(self), name))

    return property(fget, doc=getattr(np.ndarray, name).__doc__)


def install_ndarray_methods():
    """Give lightarray.ndarray every non-dunder attribute of numpy.ndarray."""
    native = set(dir(_ndarray))
    for name in dir(np.ndarray):
        if name.startswith("__") or name in native:
            continue
        descriptor = getattr(np.ndarray, name)
        if isinstance(descriptor, types.GetSetDescriptorType):
            setattr(_ndarray, name, _make_property(name))
        elif callable(descriptor):
            setattr(_ndarray, name, _make_method(name))


install_ndarray_methods()


# ---- NumPy dispatch protocols ------------------------------------------------
# With these installed, NumPy's own functions and ufuncs called on lightarray
# arrays (np.sin(a), np.concatenate([a, b]), np.sum(a)) return lightarray
# arrays instead of NumPy arrays, so code written against NumPy keeps its
# results in lightarray.

_NATIVE_UFUNCS = {}  # numpy ufunc -> lightarray native function, filled lazily


def _array_ufunc(self, ufunc, method, *inputs, **kwargs):
    if kwargs.get("out") is not None:
        # NumPy writes into `out` and returns it; hand it back untouched.
        return getattr(ufunc, method)(*_args_to_numpy(inputs), **_kwargs_to_numpy(kwargs))
    if method == "__call__" and not kwargs:
        native = _NATIVE_UFUNCS.get(ufunc)
        if native is not None:
            return native(*inputs)
    return invoke(getattr(ufunc, method), inputs, kwargs)


_NATIVE_FUNCTIONS = {}  # numpy function -> lightarray native function


def _array_function(self, func, types, args, kwargs):
    native = _NATIVE_FUNCTIONS.get(func)
    if native is not None:
        return native(*args, **kwargs)
    # `func._implementation` is the undispatched NumPy function
    return invoke(getattr(func, "_implementation", func), args, kwargs)


def install_numpy_protocols(native_namespace):
    """Attach __array_ufunc__/__array_function__ and register the native
    implementations of NumPy ufuncs that lightarray provides."""
    for name in dir(np):
        obj = getattr(np, name)
        if name not in native_namespace:
            continue
        if isinstance(obj, np.ufunc):
            _NATIVE_UFUNCS[obj] = native_namespace[name]
        elif callable(obj):
            _NATIVE_FUNCTIONS[obj] = native_namespace[name]
    _ndarray.__array_ufunc__ = _array_ufunc
    _ndarray.__array_function__ = _array_function
