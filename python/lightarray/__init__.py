"""lightarray: a fast small-array library with the NumPy module interface.

Operations implemented natively run in Rust on float64 data. Every other
public name of the ``numpy`` module is here too, wrapped so that lightarray
arrays go in and float64 array results come back out as lightarray arrays.
The same holds for the methods and properties of ``numpy.ndarray``.

    import lightarray as np
"""

from __future__ import annotations

import builtins as _builtins
import types as _types

import numpy as _np

from lightarray import _core, _fallback
from lightarray._core import *  # noqa: F401,F403  (native functions and ndarray)
from lightarray._core import _noop, ndarray  # _noop: benchmark hook

__version__ = "0.1.0"

# Note: this module deliberately shadows builtins (`sum`, `max`, `min`, `abs`,
# `round`, `any`, `all`, and `bool` copied from NumPy 2) with NumPy-compatible
# objects, like numpy does. Inside this file use `_builtins.<name>` for the
# Python builtin.

# NumPy exposes `mod` as well as `remainder`; the native function cannot be
# named `mod` in Rust.
mod = _core.mod_
del mod_  # noqa: F821


# ---- native *_like helpers ----------------------------------------------------
def _like(name, a, dtype, order, subok, shape, device, fill=None):
    """`*_like` helpers: lightarray input (or a plain shape) gives a native
    float64 array; NumPy input keeps NumPy's dtype inference."""
    if device not in (None, "cpu"):
        raise ValueError(f"Unsupported device {device!r}: lightarray arrays live on the CPU")
    native = (
        isinstance(a, ndarray) and a.dtype == _np.float64 and order in (None, "K", "C", "A") and subok in (True, False) and shape is None
    )
    if native:
        s = a.shape
        if name == "full_like":
            return full(s, fill, dtype=dtype if dtype is not None else a.dtype)  # noqa: F405
        return globals()[name[: -len("_like")]](s, dtype=dtype)
    kwargs = {"dtype": dtype, "order": order or "K", "subok": subok, "shape": shape}
    if name == "full_like":
        return _fallback.call("full_like", a, fill, **kwargs)
    return _fallback.call(name, a, **kwargs)


def zeros_like(a, dtype=None, order="K", subok=True, shape=None, *, device=None):
    return _like("zeros_like", a, dtype, order, subok, shape, device)


def ones_like(a, dtype=None, order="K", subok=True, shape=None, *, device=None):
    return _like("ones_like", a, dtype, order, subok, shape, device)


def empty_like(prototype, dtype=None, order="K", subok=True, shape=None, *, device=None):
    return _like("empty_like", prototype, dtype, order, subok, shape, device)


def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None, *, device=None):
    return _like("full_like", a, dtype, order, subok, shape, device, fill=fill_value)


def asanyarray(a, dtype=None, order=None, *, device=None, copy=None, like=None):
    return asarray(a, dtype=dtype, device=device, copy=copy)  # noqa: F405


def copy(a, order="K", subok=False):
    return asarray(a).copy(order=order)  # noqa: F405


def shape(a):
    return asarray(a).shape  # noqa: F405


def size(a, axis=None):
    return asarray(a).size if axis is None else asarray(a).shape[axis]  # noqa: F405


def ndim(a):
    return asarray(a).ndim  # noqa: F405


def finfo(dtype):
    """NumPy's finfo, also accepting arrays (Array API) and lightarray arrays.
    The values are NumPy scalars, as in NumPy (the Array API says Python
    floats; see data-apis/array-api#405)."""
    if isinstance(dtype, (ndarray, _np.ndarray)):
        dtype = dtype.dtype
    return _np.finfo(dtype)


def iinfo(int_type):
    if isinstance(int_type, (ndarray, _np.ndarray)):
        int_type = int_type.dtype
    return _np.iinfo(int_type)


def _as_index(obj):
    """Integer-valued lightarray arrays are meant as index arrays: hand NumPy
    an int64 array (lightarray has no integer dtype yet)."""
    if isinstance(obj, ndarray):
        v = _np.asarray(obj)
        # float64 arrays holding whole numbers are meant as indices; int64 and
        # bool arrays (index arrays, masks) are passed through as they are
        if v.dtype.kind == "f" and (v.size == 0 or (v == _np.floor(v)).all()):
            return v.astype(_np.int64)
        return v
    if type(obj) is tuple:
        return tuple(_as_index(o) for o in obj)
    if type(obj) is list:
        return [_as_index(o) for o in obj]
    return obj


def take(a, indices, axis=None, out=None, mode="raise"):
    return _fallback.call("take", a, _as_index(indices), axis=axis, out=out, mode=mode)


def take_along_axis(arr, indices, axis=-1):
    return _fallback.call("take_along_axis", arr, _as_index(indices), axis)


def put(a, ind, v, mode="raise"):
    return _fallback.call("put", a, _as_index(ind), v, mode=mode)


def delete(arr, obj, axis=None):
    return _fallback.call("delete", arr, _as_index(obj), axis=axis)


def insert(arr, obj, values, axis=None):
    return _fallback.call("insert", arr, _as_index(obj), values, axis=axis)


def bincount(x, weights=None, minlength=0):
    return _fallback.call("bincount", _as_index(x), weights=weights, minlength=minlength)


# ---- stacking helpers built on the native concatenate --------------------------
def atleast_1d(*arys):
    out = [a if getattr(a, "ndim", 1) >= 1 else a.reshape(1) for a in (asarray(x) for x in arys)]  # noqa: F405
    return out[0] if len(out) == 1 else out


def atleast_2d(*arys):
    out = []
    for a in (asarray(x) for x in arys):  # noqa: F405
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a.reshape(1, -1)
        out.append(a)
    return out[0] if len(out) == 1 else out


def hstack(tup, **kwargs):
    if kwargs:
        return _fallback.call("hstack", tup, **kwargs)
    arrs = [atleast_1d(a) for a in tup]
    return concatenate(arrs, axis=0 if arrs and arrs[0].ndim == 1 else 1)  # noqa: F405


def vstack(tup, **kwargs):
    if kwargs:
        return _fallback.call("vstack", tup, **kwargs)
    return concatenate([atleast_2d(a) for a in tup], axis=0)  # noqa: F405


def column_stack(tup):
    arrs = [a.reshape(-1, 1) if a.ndim < 2 else a for a in (asarray(x) for x in tup)]  # noqa: F405
    return concatenate(arrs, axis=1)  # noqa: F405


row_stack = vstack


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Trapezoidal integration; native for 1-D lightarray input."""
    if isinstance(y, ndarray) and y.ndim == 1 and axis in (-1, 0) and (x is None or (isinstance(x, ndarray) and x.shape == y.shape)):
        if y.size < 2:
            return 0.0
        widths = dx if x is None else x[1:] - x[:-1]
        return float(((y[1:] + y[:-1]) * widths).sum() * 0.5)
    return _fallback.invoke(_np.trapezoid, (y, x, dx, axis), {})


def squeeze(a, axis=None):
    a = asarray(a)  # noqa: F405
    if not isinstance(a, ndarray):
        return _fallback.call("squeeze", a, axis=axis)
    if axis is None:
        return a.reshape([d for d in a.shape if d != 1])
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(ax + a.ndim if ax < 0 else ax for ax in axes)
    if _builtins.any(a.shape[ax] != 1 for ax in axes):
        raise ValueError("cannot select an axis to squeeze out which has size not equal to one")
    return a.reshape([d for i, d in enumerate(a.shape) if i not in axes])


def expand_dims(a, axis):
    a = asarray(a)  # noqa: F405
    if not isinstance(a, ndarray):
        return _fallback.call("expand_dims", a, axis)
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    out_ndim = a.ndim + len(axes)
    for ax in axes:
        if not -out_ndim <= ax < out_ndim:
            raise _np.exceptions.AxisError(ax, out_ndim)
    axes = sorted(ax + out_ndim if ax < 0 else ax for ax in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("repeated axis")
    shape, it = [], iter(a.shape)
    for i in range(out_ndim):
        shape.append(1 if i in axes else next(it))
    return a.reshape(shape)


def flip(m, axis=None):
    m = asarray(m)  # noqa: F405
    if not isinstance(m, ndarray) or m.ndim == 0:
        return _fallback.call("flip", m, axis=axis)
    if axis is None:
        return m[(slice(None, None, -1),) * m.ndim]
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    key = [slice(None)] * m.ndim
    for ax in axes:
        key[ax] = slice(None, None, -1)
    return m[tuple(key)]


def diff(a, n=1, axis=-1, prepend=None, append=None):
    a = asarray(a)  # noqa: F405
    # (bool arrays difference with xor in NumPy, so they go there)
    if prepend is not None or append is not None or n != 1 or not isinstance(a, ndarray) or a.ndim == 0 or a.dtype == _np.bool_:
        kwargs = {k: v for k, v in (("prepend", prepend), ("append", append)) if v is not None}
        return _fallback.call("diff", a, n=n, axis=axis, **kwargs)
    ax = axis + a.ndim if axis < 0 else axis
    head = [slice(None)] * a.ndim
    tail = [slice(None)] * a.ndim
    head[ax], tail[ax] = slice(1, None), slice(None, -1)
    return a[tuple(head)] - a[tuple(tail)]


def count_nonzero(a, axis=None, *, keepdims=False):
    a = asarray(a)  # noqa: F405
    if axis is None and not keepdims and isinstance(a, ndarray):
        return _np.intp((a != 0).sum())
    return _fallback.call("count_nonzero", a, axis=axis, keepdims=keepdims)


def iscomplexobj(x):
    """lightarray arrays are float64, so never complex."""
    return False if isinstance(x, ndarray) else _builtins.bool(_np.iscomplexobj(_fallback.to_numpy(x)))


def isrealobj(x):
    return True if isinstance(x, ndarray) else _builtins.bool(_np.isrealobj(_fallback.to_numpy(x)))


def iscomplex(x):
    if isinstance(x, ndarray):
        return _np.zeros(x.shape, dtype=bool)
    return _fallback.call("iscomplex", x)


def isreal(x):
    if isinstance(x, ndarray):
        return _np.ones(x.shape, dtype=bool)
    return _fallback.call("isreal", x)


for _f in (
    atleast_1d,
    atleast_2d,
    hstack,
    vstack,
    column_stack,
    trapezoid,
    squeeze,
    expand_dims,
    flip,
    diff,
    count_nonzero,
    iscomplexobj,
    isrealobj,
    iscomplex,
    isreal,
    finfo,
    iinfo,
    take,
    take_along_axis,
    put,
    delete,
    insert,
    bincount,
    zeros_like,
    ones_like,
    empty_like,
    full_like,
    asanyarray,
    copy,
    shape,
    size,
    ndim,
):
    _f.__doc__ = getattr(_np, _f.__name__).__doc__


# ---- Array API entry points ---------------------------------------------------
__array_api_version__ = _np.__array_api_version__


def __array_namespace_info__():
    """Array API inspection namespace; lightarray shares NumPy's dtypes and devices."""
    return _np.__array_namespace_info__()


def _array_namespace(self, api_version=None):
    if api_version is not None and api_version not in ("2021.12", "2022.12", "2023.12", "2024.12"):
        raise ValueError(f"unsupported array API version {api_version!r}")
    if _numpy_conversions_active:
        # A package patched with conversions="numpy" (SciPy) must see the
        # NumPy namespace for its array-API dispatch, so that `xp.asarray`
        # hands its compiled kernels real ndarrays.
        return _np
    import sys as _sys

    return _sys.modules[__name__]


_numpy_conversions_active = False


ndarray.__array_namespace__ = _array_namespace
ndarray.__array_namespace__.__doc__ = "Return the array API namespace (the lightarray module)."


# ---- taking over NumPy inside other packages ----------------------------------
_patched = {}  # module -> {name: original NumPy object}


# NumPy functions that produce arrays for compiled code to consume. Packages
# with C/Cython kernels that insist on real ndarrays (SciPy) are patched with
# `conversions="numpy"`, which keeps these bound to NumPy while everything
# else (ufuncs, reductions, ...) runs on lightarray.
_CONVERSION_NAMES = frozenset(
    {
        "asarray",
        "asanyarray",
        "ascontiguousarray",
        "asfortranarray",
        "array",
        "require",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "asarray_chkfinite",
        "frombuffer",
        "fromiter",
    }
)


class _AnyNdarrayMeta(type(_np.ndarray)):
    def __instancecheck__(cls, obj):
        return isinstance(obj, (_np.ndarray, ndarray))

    def __subclasscheck__(cls, sub):
        return issubclass(sub, (_np.ndarray, ndarray))


class _AnyNdarray(_np.ndarray, metaclass=_AnyNdarrayMeta):
    """What `np.ndarray` means inside a patched package: `isinstance(x,
    np.ndarray)` is True for NumPy and lightarray arrays alike, so the
    package's array branches are taken for lightarray data too. Constructing
    or subclassing it behaves like `numpy.ndarray`."""


_AnyNdarray.__name__ = "ndarray"
_AnyNdarray.__qualname__ = "ndarray"


class _NamespaceForPackages(_types.ModuleType):
    """What a patched package sees as `np`: lightarray, with ufuncs wrapped in
    `UfuncProxy` so `np.add.reduce(...)` and friends keep working. Attributes
    are cached on first access, so lookups cost the same as on a module."""

    def __init__(self):
        super().__init__("lightarray")

    def __getattr__(self, name):
        import sys as _sys

        if name == "ndarray":
            obj = _AnyNdarray
        else:
            obj = _builtins.getattr(_sys.modules["lightarray"], name)
            np_obj = _builtins.getattr(_np, name, None)
            if isinstance(np_obj, _np.ufunc) and obj is not np_obj:
                obj = _fallback.UfuncProxy(obj, np_obj)
        setattr(self, name, obj)
        return obj

    def __repr__(self):
        return "<lightarray (namespace for patched packages)>"


_namespace_for_packages = _NamespaceForPackages()


class _TypePreservingUfunc:
    """A ufunc inside a `conversions="numpy"` package: calls dispatch on the
    argument types, attributes (`reduce`, `outer`, ...) are NumPy's."""

    def __init__(self, dispatch, ufunc):
        self._dispatch = dispatch
        self._ufunc = ufunc
        self.__name__ = ufunc.__name__
        self.__doc__ = ufunc.__doc__

    def __call__(self, *args, **kwargs):
        return self._dispatch(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):  # also stops copy/pickle probing from recursing
            raise AttributeError(name)
        return _builtins.getattr(self._ufunc, name)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_fallback._numpy_ufunc, (self._ufunc.__name__,))


class _ConversionsToNumpy(_types.ModuleType):
    """Stand-in for the `numpy` module alias inside packages with compiled
    kernels (SciPy): type-preserving dispatch. Conversion functions stay
    NumPy's; every other function runs on lightarray when it receives a
    lightarray argument and on NumPy otherwise, so such a package never
    manufactures lightarray arrays that its C/Cython code would reject,
    while user-supplied lightarray arrays are still processed natively."""

    def __init__(self):
        super().__init__("lightarray")

    def __getattr__(self, name):
        if name in _CONVERSION_NAMES:
            return _builtins.getattr(_np, name)
        import sys as _sys

        la_obj = _builtins.getattr(_sys.modules["lightarray"], name)
        np_obj = _builtins.getattr(_np, name, None)
        if callable(la_obj) and callable(np_obj) and not isinstance(np_obj, type) and not isinstance(la_obj, _types.ModuleType):

            def dispatch(*args, **kwargs):
                if _builtins.any(isinstance(a, ndarray) for a in args) or _builtins.any(isinstance(v, ndarray) for v in kwargs.values()):
                    return la_obj(*args, **kwargs)
                return np_obj(*args, **kwargs)

            dispatch.__name__ = name
            dispatch.__doc__ = getattr(np_obj, "__doc__", None)
            dispatch.__wrapped__ = np_obj
            if isinstance(np_obj, _np.ufunc):
                # keep np.add.reduce and friends; in this mode they stay NumPy's own
                dispatch = _TypePreservingUfunc(dispatch, np_obj)
            setattr(self, name, dispatch)  # cache
            return dispatch
        return la_obj

    def __repr__(self):
        return "<lightarray (type-preserving dispatch for packages with compiled kernels)>"


_conversions_to_numpy = _ConversionsToNumpy()


def patch_module(module, recursive=True, verbose=False, conversions="lightarray"):
    """Rebind NumPy references inside an already-imported module (or package)
    to lightarray, so code written as ``import numpy as np`` there runs on
    lightarray without editing it.

    Rebinds: the ``numpy`` module object itself (any alias name), NumPy
    ufuncs and functions bound by ``from numpy import ...``, and NumPy
    submodules (``numpy.linalg``, ...), and ``np.ndarray`` itself, which
    becomes a class whose ``isinstance`` check accepts NumPy and lightarray
    arrays alike. Leaves alone anything the module obtained from SciPy or
    other libraries, which keep their own NumPy.

    Returns the number of names rebound. With ``recursive`` every loaded
    submodule of a package is patched too. Idempotent: names already
    rebound are skipped. ``unpatch_module`` restores the originals;
    ``set_patched`` / ``patched`` / the ``LIGHTARRAY_PATCH`` environment
    variable are the toggles built on these two.

    ``conversions="numpy"`` keeps the array-conversion functions
    (``asarray``, ``array``, ``ascontiguousarray``, ...) bound to NumPy, for
    packages whose compiled kernels require real NumPy arrays (SciPy).
    """
    import sys as _sys

    this = _sys.modules[__name__]
    module_stand_in = _conversions_to_numpy if conversions == "numpy" else _namespace_for_packages
    count = 0
    for mod in _package_modules(module, recursive):
        for name, value in list(vars(mod).items()):
            replacement = None
            if value is _np:
                replacement = module_stand_in
            elif value is _np.ndarray and conversions != "numpy":
                replacement = _AnyNdarray  # isinstance checks accept lightarray arrays too
            elif conversions == "numpy" and callable(value) and not isinstance(value, type):
                # top-level NumPy functions imported by name get the same
                # type-preserving dispatcher; submodule functions stay NumPy
                if name in _CONVERSION_NAMES or str(getattr(value, "__module__", "")).count(".") > 0 and not hasattr(_np, name):
                    continue
                if hasattr(_np, name) and getattr(_np, name) is value:
                    replacement = _builtins.getattr(module_stand_in, name)
                else:
                    continue
            elif isinstance(value, _types.ModuleType) and value.__name__.startswith("numpy."):
                sub = value.__name__[len("numpy.") :]
                replacement = _builtins.getattr(this, sub, None) if "." not in sub else None
            elif isinstance(value, _np.ufunc) or (
                callable(value) and not isinstance(value, type) and str(getattr(value, "__module__", "")).startswith("numpy")
            ):
                candidate = _lightarray_equivalent(this, value)
                if candidate is not None and candidate is not value:
                    # ufuncs keep their attributes (np.add.reduce, ...) through a proxy
                    replacement = _fallback.UfuncProxy(candidate, value) if isinstance(value, _np.ufunc) else candidate
            if replacement is not None:
                _patched.setdefault(mod, {}).setdefault(name, value)
                setattr(mod, name, replacement)
                count += 1
                if verbose:
                    print(f"lightarray.patch_module: {mod.__name__}.{name} -> lightarray")
    if _patched:
        _core._set_patch_active(True)
        _fallback.patch_active = True
    if conversions == "numpy" and count:
        global _numpy_conversions_active
        _numpy_conversions_active = True
    return count


def _lightarray_equivalent(this, value):
    """The lightarray object standing in for a NumPy function: the native or
    wrapped top-level name, or for functions of a public NumPy submodule
    (`numpy.testing.assert_equal`, `numpy.linalg.norm`) the proxied
    submodule's attribute, which converts lightarray arguments on the way in."""
    name = getattr(value, "__name__", "")
    if not name:
        return None
    module_name = str(getattr(value, "__module__", ""))
    parts = module_name.split(".")
    if len(parts) > 1 and not parts[1].startswith("_"):
        sub = _builtins.getattr(this, parts[1], None)
        if isinstance(sub, _fallback.ModuleProxy):
            try:
                return _builtins.getattr(sub, name)
            except AttributeError:
                pass
    return _builtins.getattr(this, name, None)


def unpatch_module(module, recursive=True):
    """Undo ``patch_module``: restore the NumPy objects. Returns the count."""
    count = 0
    for mod in _package_modules(module, recursive):
        for name, original in _patched.pop(mod, {}).items():
            setattr(mod, name, original)
            count += 1
    if not _patched:
        _core._set_patch_active(False)
        _fallback.patch_active = False
        global _numpy_conversions_active
        _numpy_conversions_active = False
    return count


def is_patched(module):
    """True if ``patch_module`` is currently applied to ``module`` or any of its submodules."""
    if module in _patched:
        return True
    prefix = getattr(module, "__name__", "") + "."
    return _builtins.any(getattr(m, "__name__", "").startswith(prefix) for m in _patched)


def set_patched(module, enabled=True, recursive=True, conversions="lightarray"):
    """Toggle: run ``module``'s internals on lightarray (True) or NumPy (False).
    Idempotent; returns the new state."""
    if enabled:
        patch_module(module, recursive=recursive, conversions=conversions)
    elif is_patched(module):
        unpatch_module(module, recursive=recursive)
    return is_patched(module)


class patched:
    """Context manager: ``with lightarray.patched(lmfit): ...`` runs the block
    with lmfit's internals on lightarray and restores NumPy afterwards."""

    def __init__(self, module, recursive=True, conversions="lightarray"):
        self.module, self.recursive, self.conversions = module, recursive, conversions

    def __enter__(self):
        self._was = is_patched(self.module)
        set_patched(self.module, True, self.recursive, self.conversions)
        return self.module

    def __exit__(self, *exc):
        set_patched(self.module, self._was, self.recursive)


def _package_modules(module, recursive):
    import sys as _sys

    mods = [module]
    if recursive and hasattr(module, "__name__"):
        prefix = module.__name__ + "."
        mods += [m for name, m in list(_sys.modules.items()) if name.startswith(prefix) and m is not None]
    return mods


def _auto_patch_from_environment():
    """``LIGHTARRAY_PATCH=lmfit,scipy:numpy`` patches those packages as soon as
    they are imported (or right away if they already are), so the toggle
    needs no code change at all; ``:numpy`` selects the type-preserving mode
    for packages with compiled kernels. Unset or empty disables it."""
    import os as _os
    import sys as _sys

    # "lmfit,scipy:numpy" -> {"lmfit": "lightarray", "scipy": "numpy"}
    modes = {}
    for entry in _os.environ.get("LIGHTARRAY_PATCH", "").split(","):
        entry = entry.strip()
        if entry:
            name, _, mode = entry.partition(":")
            modes[name.strip()] = mode.strip() or "lightarray"
    names = list(modes)
    if not names:
        return

    def _patch_now(name):
        mod = _sys.modules.get(name)
        if mod is not None:
            patch_module(mod, conversions=modes[name])  # idempotent: already-rebound names are skipped

    for n in names:
        _patch_now(n)

    # After any import statement *outside* the package completes, (re)patch
    # the package: this catches the first import and later submodule imports,
    # without touching the package while its own __init__ is still running.
    _original_import = _builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _original_import(name, globals, locals, fromlist, level)
        root = name.partition(".")[0]
        if root in names:
            importer = (globals or {}).get("__name__", "")
            if importer.partition(".")[0] != root:
                _patch_now(root)
        return module

    _builtins.__import__ = _import


# ---- everything else in the numpy namespace, wrapped --------------------------
# Snapshot of everything implemented natively (Rust core and the helpers
# above) before the NumPy names are filled in; `__array_function__` uses it
# to route NumPy's own functions (`numpy.hstack(a, b)`, ...) to these.
_native_functions = {k: v for k, v in globals().items() if callable(v) and not k.startswith("_") and not isinstance(v, type)}

_fallback.populate_module(globals())


# ---- native overrides inside proxied submodules --------------------------------
def _norm(x, ord=None, axis=None, keepdims=False):
    """Euclidean norm of a 1-D lightarray natively; everything else NumPy."""
    if ord is None and axis is None and not keepdims and isinstance(x, ndarray) and x.ndim == 1:
        return _core.sqrt(x.dot(x))
    return _fallback.invoke(_np.linalg.norm, (x, ord, axis, keepdims), {})


_norm.__doc__ = _np.linalg.norm.__doc__
linalg.norm = _norm  # noqa: F821  (linalg is the proxy created by populate_module)


def _with_dtype(numpy_function):
    """`fft.fftfreq` / `fft.rfftfreq` with the Array API's `dtype` keyword, which NumPy lacks."""

    def function(n, d=1.0, *, dtype=None, device=None):
        result = numpy_function(n, d, device=device)
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return _fallback.from_numpy(result)

    function.__name__ = function.__qualname__ = numpy_function.__name__
    function.__doc__ = numpy_function.__doc__
    return function


fft.fftfreq = _with_dtype(_np.fft.fftfreq)  # noqa: F821  (fft is a proxy, like linalg)
fft.rfftfreq = _with_dtype(_np.fft.rfftfreq)  # noqa: F821
_fallback.install_numpy_protocols(_native_functions)
_auto_patch_from_environment()

__all__ = [name for name in globals() if not name.startswith("_")]
