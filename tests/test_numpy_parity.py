"""Every operation lightarray implements must agree with NumPy."""

import math

import lightarray as la
import numpy as np
import pytest
from lightarray import _fallback

SIZES = [0, 1, 7, 100]


def rng_data(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def check(got, expected):
    """`got` (lightarray or scalar) matches `expected` (NumPy) in value and shape."""
    if isinstance(expected, np.ndarray):
        assert isinstance(got, la.ndarray), type(got)
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-13, atol=1e-13)
    else:
        assert not isinstance(got, la.ndarray)
        np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)


# ---- creation -------------------------------------------------------------


@pytest.mark.parametrize("data", [[], [1.0], [1, 2, 3], [[1, 2], [3, 4]], [[[1.0]]], (1, 2), 3.5, 2])
def test_array_from_python_objects(data):
    check(la.array(data), np.array(data))  # dtype follows NumPy: ints -> int64, floats -> float64


@pytest.mark.parametrize("value", [np.array(3.0), np.float64(2.5), np.zeros(())])
def test_array_from_zero_dimensional_float64_numpy(value):
    got = la.array(value)
    assert isinstance(got, la.ndarray) and got.shape == ()
    assert float(got) == float(value)


@pytest.mark.parametrize("value", [np.int64(2), np.float32(1.5), np.bool_(True), np.array(1, dtype=np.int8)])
def test_array_from_other_numpy_scalars_keeps_dtype(value):
    got = la.array(value)
    expected = np.asarray(value)
    assert got.dtype == expected.dtype
    # int64 and bool are native dtypes; narrower ones stay NumPy arrays
    assert isinstance(got, la.ndarray if expected.dtype in (np.int64, np.bool_) else np.ndarray)


def test_array_from_numpy_keeps_shape():
    x = rng_data(24).reshape(2, 3, 4)
    a = la.array(x)
    check(a, x)
    assert a.strides == x.strides


def test_array_ragged_raises_like_numpy():
    with pytest.raises(ValueError):
        la.array([[1, 2], [3]])


def test_array_non_float_dtype_returns_numpy():
    a = la.array([1, 2, 3], dtype=np.int64)
    assert isinstance(a, la.ndarray) and a.dtype == np.int64
    b = la.array([1, 2, 3], dtype=np.int32)
    assert isinstance(b, np.ndarray) and b.dtype == np.int32


def test_asarray_is_identity_for_lightarray():
    a = la.array([1.0, 2.0])
    assert la.asarray(a) is a


@pytest.mark.parametrize("shape", [3, (2, 3), [4, 1, 2], ()])
def test_zeros_ones_full_empty(shape):
    check(la.zeros(shape), np.zeros(shape))
    check(la.ones(shape), np.ones(shape))
    check(la.full(shape, 2.5), np.full(shape, 2.5))
    assert la.empty(shape).shape == np.empty(shape).shape


def test_like_helpers():
    a = la.ones((2, 3))
    check(la.zeros_like(a), np.zeros((2, 3)))
    check(la.full_like(a, 7), np.full((2, 3), 7.0))


@pytest.mark.parametrize("args", [(5,), (2, 5), (0, 1, 0.25), (5, 0, -1), (1, 1)])
def test_arange(args):
    check(la.arange(*args), np.arange(*args))


@pytest.mark.parametrize("kw", [{}, {"num": 1}, {"num": 0}, {"num": 7, "endpoint": False}])
def test_linspace(kw):
    check(la.linspace(-1.0, 2.0, **kw), np.linspace(-1.0, 2.0, **kw))


def test_linspace_retstep_falls_back():
    got, step = la.linspace(0, 1, 5, retstep=True)
    exp, exp_step = np.linspace(0, 1, 5, retstep=True)
    check(got, exp)
    assert step == exp_step


# ---- attributes -----------------------------------------------------------


def test_attributes_match_numpy():
    x = rng_data(6).reshape(2, 3)
    a = la.array(x)
    for name in ["shape", "strides", "ndim", "size", "itemsize", "nbytes", "dtype"]:
        assert getattr(a, name) == getattr(x, name), name
    assert len(a) == len(x)
    assert repr(a) == repr(x)
    assert str(a) == str(x)


def test_attributes_via_fallback():
    x = rng_data(6).reshape(2, 3)
    a = la.array(x)
    check(a.T, x.T)
    assert a.tolist() == x.tolist()
    assert a.argmax() == x.argmax()
    with pytest.raises(AttributeError):
        a.no_such_attribute


def test_scalar_conversions():
    a = la.array([2.5])
    assert float(a) == 2.5 and int(a) == 2 and bool(a)
    assert not bool(la.zeros(1))
    with pytest.raises(ValueError):
        bool(la.ones(3))
    with pytest.raises(TypeError):
        float(la.ones(3))


# ---- arithmetic -----------------------------------------------------------

BINARY_OPS = ["add", "sub", "mul", "truediv", "floordiv", "mod", "pow"]


def _apply(opname, x, y):
    import operator

    return getattr(operator, opname)(x, y)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("op", BINARY_OPS)
def test_array_array_ops(op, n):
    x, y = rng_data(n, 1), rng_data(n, 2)
    if op == "pow":
        x = np.abs(x)
    with np.errstate(all="ignore"):
        expected = _apply(op, x, y)
    check(_apply(op, la.array(x), la.array(y)), expected)


@pytest.mark.parametrize("scalar", [2, 2.5, -0.5, np.float64(3.0), np.int64(2)])
@pytest.mark.parametrize("op", BINARY_OPS)
def test_array_scalar_ops_both_sides(op, scalar):
    x = np.abs(rng_data(9)) + 0.1
    with np.errstate(all="ignore"):
        check(_apply(op, la.array(x), scalar), _apply(op, x, scalar))
        check(_apply(op, scalar, la.array(x)), _apply(op, scalar, x))


def test_pow_two_is_square():
    x = rng_data(5)
    check(la.array(x) ** 2, x**2)


def test_division_by_zero_follows_ieee():
    with np.errstate(all="ignore"):
        got = la.array([1.0, -1.0, 0.0]) / la.array([0.0, 0.0, 0.0])
        assert np.asarray(got).tolist()[:2] == [math.inf, -math.inf]
        assert math.isnan(got[2])


def test_shape_mismatch_raises_like_numpy():
    with pytest.raises(ValueError, match=r"could not be broadcast together with shapes \(3,\) \(4,\)"):
        la.ones(3) + la.ones(4)
    with pytest.raises(ValueError):
        la.ones((2, 3)) * la.ones((3, 2))


@pytest.mark.parametrize(
    "sa, sb",
    [
        ((3,), (1,)),
        ((1,), (4,)),
        ((2, 3), (3,)),
        ((3,), (2, 3)),
        ((2, 3), (2, 1)),
        ((2, 1), (1, 3)),
        ((2, 3, 4), (3, 1)),
        ((2, 3, 4), (4,)),
        ((1, 3, 1), (2, 1, 4)),
        ((), (3,)),
        ((2, 2), ()),
        ((0,), (1,)),
        ((0, 3), (3,)),
    ],
)
def test_broadcasting_native(sa, sb):
    x = rng_data(int(np.prod(sa)), 1).reshape(sa)
    y = rng_data(int(np.prod(sb)), 2).reshape(sb) + 3.0
    a, b = la.array(x), la.array(y)
    check(a + b, x + y)
    check(b - a, y - x)
    check(a * b, x * y)
    check(a / b, x / y)
    check(la.maximum(a, b), np.maximum(x, y))
    check(la.arctan2(a, b), np.arctan2(x, y))
    check(la.add(a, b), np.add(x, y))


def test_centering_idiom_broadcasts_natively():
    x = rng_data(12).reshape(4, 3)
    a = la.array(x)
    before = _fallback.calls
    centred = a - a.mean(axis=0)
    check(centred, x - x.mean(axis=0))
    check(centred / a.std(axis=0), (x - x.mean(axis=0)) / x.std(axis=0))
    assert _fallback.calls - before == 1  # only std(axis=0) is delegated


def test_mixed_with_numpy_returns_lightarray():
    a, x = la.ones(3), np.ones(3) * 2
    check(a + x, np.ones(3) * 3)
    check(x - a, np.ones(3))
    check(a * [1, 2, 3], np.array([1.0, 2.0, 3.0]))
    check(np.float64(2.0) ** a, np.full(3, 2.0))
    check(a + np.ones((2, 3)), np.full((2, 3), 2.0))  # broadcasting through NumPy


def test_unsupported_operand_raises_type_error():
    with pytest.raises(TypeError):
        la.ones(3) + "text"


def test_unary_operators():
    x = rng_data(8)
    check(-la.array(x), -x)
    check(+la.array(x), +x)
    check(abs(la.array(x)), abs(x))


def test_matmul_falls_back():
    x = rng_data(6).reshape(2, 3)
    check(la.array(x) @ la.array(x.T), x @ x.T)


# ---- element-wise functions ---------------------------------------------

UNARY = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    "cbrt",
    "square",
    "reciprocal",
    "absolute",
    "fabs",
    "negative",
    "positive",
    "sign",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "deg2rad",
    "rad2deg",
    "radians",
    "degrees",
    "abs",
]


@pytest.mark.parametrize("name", UNARY)
def test_unary_functions(name):
    x = np.abs(rng_data(50)) % 0.9 + 0.05  # in (0.05, 0.95) for the inverse trig / log family
    if name == "rint":
        x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 2.4, 2.6])
    expected = getattr(np, name)(x)
    check(getattr(la, name)(la.array(x)), expected)
    # Python scalars come back as Python floats
    got = getattr(la, name)(float(x[0]))
    assert isinstance(got, float) and np.isclose(got, expected[0])


def test_unary_on_list_goes_through_numpy():
    check(la.sin([0.0, 1.0]), np.sin([0.0, 1.0]))


@pytest.mark.parametrize("name", ["add", "subtract", "multiply", "divide", "power", "remainder", "floor_divide", "mod"])
def test_binary_functions(name):
    x, y = np.abs(rng_data(10)) + 0.5, np.abs(rng_data(10, 3)) + 0.5
    check(getattr(la, name)(la.array(x), la.array(y)), getattr(np, name)(x, y))
    check(getattr(la, name)(la.array(x), 2.0), getattr(np, name)(x, 2.0))
    check(getattr(la, name)(x.tolist(), la.array(y)), getattr(np, name)(x, y))


# ---- reductions -----------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 100, 1003])
@pytest.mark.parametrize("name", ["sum", "prod", "mean", "max", "min"])
def test_full_reductions(name, n):
    x = rng_data(n) * 0.5 + 1.0
    a = la.array(x)
    expected = getattr(np, name)(x)
    got = getattr(a, name)()
    assert isinstance(got, np.float64)  # NumPy scalar, so .round(), .dtype, .astype work
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    np.testing.assert_allclose(getattr(la, name)(a), expected, rtol=1e-12)
    np.testing.assert_allclose(getattr(np, name)(a), expected, rtol=1e-12)


def test_reductions_with_axis_and_keepdims():
    x = rng_data(24).reshape(2, 3, 4)
    a = la.array(x)
    check(a.sum(axis=1), x.sum(axis=1))
    check(a.mean(axis=(0, 2)), x.mean(axis=(0, 2)))
    check(a.max(axis=0, keepdims=True), x.max(axis=0, keepdims=True))
    check(la.sum(a, axis=-1), np.sum(x, axis=-1))


def test_reductions_nan_and_empty():
    assert math.isnan(la.array([1.0, np.nan]).max())
    with pytest.raises(ValueError):
        la.zeros(0).max()
    assert la.zeros(0).sum() == 0.0


def test_std_var_via_fallback():
    x = rng_data(30)
    a = la.array(x)
    np.testing.assert_allclose(a.std(), x.std())
    np.testing.assert_allclose(la.var(a), np.var(x))


# ---- indexing -------------------------------------------------------------


def test_integer_indexing():
    x = rng_data(24).reshape(2, 3, 4)
    a = la.array(x)
    assert a[1, 2, 3] == x[1, 2, 3] and isinstance(a[1, 2, 3], np.float64)
    assert a[-1, -1, -1] == x[-1, -1, -1]
    assert a.mean().round(2) == x.mean().round(2)
    check(a[1], x[1])
    check(a[0, 2], x[0, 2])
    with pytest.raises(IndexError):
        a[2]
    with pytest.raises(IndexError):
        a[0, 0, 0, 0]


@pytest.mark.parametrize("sl", [slice(None), slice(1, 3), slice(None, None, 2), slice(None, None, -1), slice(10, 20), slice(-2, None)])
def test_slicing(sl):
    x = rng_data(5)
    check(la.array(x)[sl], x[sl])
    y = rng_data(12).reshape(4, 3)
    check(la.array(y)[sl], y[sl])


@pytest.mark.parametrize(
    "key",
    [
        (slice(None), 1),
        (1, slice(None)),
        (slice(1, None), slice(None, None, 2)),
        (slice(None, None, -1), slice(None, None, -1)),
        (0, slice(1, 3)),
        (slice(2, 0, -1), 2),
        (slice(0, 0), slice(None)),
        (slice(None), slice(None), 3),
        (slice(None), 1, slice(None, None, 2)),
        (-1, -1, slice(None)),
        (slice(10, 20), 0),
        (slice(None),),
    ],
)
def test_multi_axis_basic_indexing_native(key):
    x = rng_data(60).reshape(3, 4, 5)
    check(la.array(x)[key], x[key])


@pytest.mark.parametrize(
    "key",
    [
        (None, slice(None)),
        (slice(None), None),
        (None,),
        (1, None),
        (None, 1, None),
        (slice(1, None), None, slice(None, None, 2)),
        (None, None, 0, 0),
    ],
)
def test_newaxis_indexing_native(key):
    x = rng_data(12).reshape(3, 4)
    a = la.array(x)
    before = _fallback.calls
    check(a[key], x[key])
    assert _fallback.calls == before
    check(la.array(x[0])[None], x[0][None])
    check(la.array(x[0])[:, la.newaxis], x[0][:, np.newaxis])


def test_multi_axis_indexing_errors():
    a = la.ones((2, 3))
    with pytest.raises(IndexError):
        a[:, :, 0]
    with pytest.raises(IndexError):
        a[5, :]


def test_integer_list_and_boolean_mask_indexing_native():
    x = rng_data(6)
    a = la.array(x)
    m = rng_data(12).reshape(3, 4)
    b = la.array(m)
    mask_a, mask_b = a > 0, b > 0  # comparisons themselves are delegated (no bool dtype yet)
    before = _fallback.calls
    check(a[[0, 2, 4]], x[[0, 2, 4]])
    check(a[[-1, 0]], x[[-1, 0]])
    check(a[[]], x[[]])
    check(a[mask_a], x[x > 0])
    check(a[x > 0], x[x > 0])
    check(b[[2, 0]], m[[2, 0]])
    check(b[mask_b], m[m > 0])
    assert _fallback.calls == before
    with pytest.raises(IndexError):
        a[[6]]
    with pytest.raises(IndexError):
        a[np.array([True, False])]  # wrong mask length: NumPy raises IndexError
    check(a[np.array([0, 2, 4])], x[[0, 2, 4]])  # integer ndarray: NumPy
    check(b[:, [1, 3]], m[:, [1, 3]])


def test_numpy_functions_dispatch_to_native():
    x = rng_data(12).reshape(3, 4)
    a = la.array(x)
    mask = a > 0
    before = _fallback.calls
    np.testing.assert_allclose(np.sum(a), x.sum())
    check(np.sum(a, axis=0), x.sum(axis=0))
    check(np.concatenate([a, a]), np.concatenate([x, x]))
    check(np.where(mask, a, 0.0), np.where(x > 0, x, 0.0))
    check(np.clip(a, -1, 1), np.clip(x, -1, 1))
    check(np.transpose(a), x.T)
    assert _fallback.calls == before
    check(np.sort(a, axis=0), np.sort(x, axis=0))  # 2-D sort: delegated inside the native sort


def test_fancy_indexing_other_cases_fall_back():
    x = rng_data(6)
    a = la.array(x)
    check(a[None, :], x[None, :])
    check(a[...], x[...])
    z = rng_data(12).reshape(3, 4)
    check(la.array(z)[:, 1], z[:, 1])
    check(la.array(z)[1:, ::2], z[1:, ::2])


def test_iteration():
    x = rng_data(6).reshape(2, 3)
    rows = list(la.array(x))
    assert len(rows) == 2
    check(rows[1], x[1])
    assert list(la.array(x[0])) == x[0].tolist()


# ---- shape manipulation -------------------------------------------------


def test_reshape_variants():
    x = np.arange(6.0)
    a = la.array(x)
    check(a.reshape(2, 3), x.reshape(2, 3))
    check(a.reshape((3, 2)), x.reshape((3, 2)))
    check(a.reshape(-1, 2), x.reshape(-1, 2))
    check(la.reshape(a, (6,)), x.reshape(6))
    check(a.reshape(2, 3).flatten(), x)
    with pytest.raises(ValueError):
        a.reshape(4, 2)


def test_copy_is_independent_object():
    a = la.ones(3)
    b = a.copy()
    assert b is not a
    check(b, np.ones(3))


def test_comparisons_return_numpy_bool_arrays():
    x = rng_data(10)
    a = la.array(x)
    before = _fallback.calls
    for op in ["lt", "le", "gt", "ge", "eq", "ne"]:
        got = _apply(op, a, 0.0)
        assert isinstance(got, la.ndarray) and got.dtype == bool
        np.testing.assert_array_equal(got, _apply(op, x, 0.0))
        np.testing.assert_array_equal(_apply(op, a, la.array(-x)), _apply(op, x, -x))
        np.testing.assert_array_equal(_apply(op, 0.0, a), _apply(op, 0.0, x))
    assert _fallback.calls == before  # equal shapes and scalars are native
    m = rng_data(12).reshape(3, 4)
    b = la.array(m)
    got = b > 0
    assert got.shape == (3, 4)
    np.testing.assert_array_equal(got, m > 0)
    np.testing.assert_array_equal(b > la.array(m[0]), m > m[0])  # broadcast: NumPy
    np.testing.assert_array_equal(b == m, m == m)  # NumPy operand
    nan = la.array([np.nan, 1.0])
    np.testing.assert_array_equal(nan == nan, np.array([False, True]))
    assert (a == "text") is False


def test_copy_deepcopy_pickle():
    import copy
    import pickle

    x = rng_data(6).reshape(2, 3)
    a = la.array(x)
    check(copy.copy(a), x)
    check(copy.deepcopy(a), x)
    check(pickle.loads(pickle.dumps(a)), x)


# ---- NumPy interop ------------------------------------------------------


def test_asarray_is_zero_copy_writable_view():
    a = la.array([1.0, 2.0, 3.0])
    v = np.asarray(a)
    assert v.base is a or (v.base is not None and np.shares_memory(v, np.asarray(a)))
    assert v.flags.writeable
    v[0] = 5.0  # writes through, like a NumPy view
    assert a[0] == 5.0
    v2 = np.asarray(a, copy=False)
    assert np.shares_memory(v, v2)


def test_numpy_functions_accept_lightarray_and_return_it():
    """NumPy's own functions dispatch to lightarray through __array_ufunc__
    and __array_function__, so results stay in lightarray."""
    x = rng_data(12).reshape(3, 4)
    a = la.array(x)
    np.testing.assert_allclose(np.linalg.norm(a), np.linalg.norm(x))
    check(np.sin(a), np.sin(x))  # native ufunc
    check(np.arctan2(a, 2.0), np.arctan2(x, 2.0))  # ufunc without native version
    check(np.add(a, 1), x + 1)
    check(np.concatenate([a, a]), np.concatenate([x, x]))
    check(np.transpose(a), x.T)
    np.testing.assert_allclose(np.sum(a), np.sum(x))
    np.testing.assert_allclose(np.add.reduce(a, axis=0), x.sum(axis=0))
    assert isinstance(np.argmax(a), np.integer)
    out = np.empty_like(x)
    result = np.sin(a, out=out)
    assert result is out
    np.testing.assert_allclose(out, np.sin(x))


def test_ndarray_has_every_numpy_ndarray_attribute():
    missing = [n for n in dir(np.ndarray) if not n.startswith("__") and not hasattr(la.ndarray, n)]
    assert missing == []


def test_module_has_every_public_numpy_name():
    missing = [n for n in dir(np) if not n.startswith("_") and not hasattr(la, n)]
    assert missing == []


def test_numpy_in_place_methods_mutate_the_array():
    a = la.array([3.0, 1.0, 2.0])
    assert a.sort() is None
    check(a, np.array([1.0, 2.0, 3.0]))
    a.fill(7.0)
    check(a, np.full(3, 7.0))
    la.copyto(a, la.array([1.0, 2.0, 3.0]))
    check(a, np.array([1.0, 2.0, 3.0]))
    a.put([0], [9.0])
    assert a[0] == 9.0


def test_module_fallback_wraps_results():
    x = rng_data(12).reshape(3, 4)
    a = la.array(x)
    check(la.concatenate([a, a]), np.concatenate([x, x]))
    check(la.linalg.inv(la.array(x[:3, :3])), np.linalg.inv(x[:3, :3]))
    spectrum = la.fft.fft(a)  # complex results stay NumPy arrays
    assert isinstance(spectrum, np.ndarray)
    np.testing.assert_allclose(spectrum, np.fft.fft(x))
    got = la.where(a > 0, a, 0.0)
    check(got, np.where(x > 0, x, 0.0))
    assert la.argmax(a) == np.argmax(x)
    with pytest.raises(AttributeError):
        la.no_such_function


def test_constants_and_dtypes_are_numpys():
    assert la.pi is np.pi and la.float64 is np.float64
    assert la.ones(2).dtype == la.float64
    assert la.dtype("float64") == np.float64


# ---- native extras added in phase 2b -------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
@pytest.mark.parametrize("name", ["sum", "prod", "mean", "max", "min"])
def test_single_axis_reductions_native(name, axis):
    x = rng_data(24).reshape(2, 3, 4) * 0.5 + 1.0
    a = la.array(x)
    check(getattr(a, name)(axis), getattr(x, name)(axis))
    check(getattr(a, name)(axis=axis), getattr(x, name)(axis=axis))
    check(getattr(la, name)(a, axis=axis), getattr(np, name)(x, axis=axis))


def test_axis_out_of_range_and_empty():
    a = la.ones((2, 3))
    with pytest.raises(ValueError):
        a.sum(axis=2)
    with pytest.raises(ValueError):
        la.zeros((0, 3)).max(axis=0)
    check(la.zeros((0, 3)).sum(axis=0), np.zeros(3))


@pytest.mark.parametrize("name", ["var", "std", "argmax", "argmin", "any", "all", "cumsum", "cumprod"])
def test_bare_native_methods(name):
    x = rng_data(20) * 0.3 + 0.7
    a = la.array(x)
    for got in (getattr(a, name)(), getattr(la, name)(a)):
        expected = getattr(np, name)(x)
        if isinstance(expected, np.ndarray):
            check(got, expected)
        else:
            assert type(got) is type(expected), (type(got), type(expected))
            np.testing.assert_allclose(got, expected, rtol=1e-12)
    # with arguments, NumPy semantics still hold
    np.testing.assert_allclose(getattr(a, name)(axis=0), getattr(x, name)(axis=0), rtol=1e-12)


def test_argmax_nan_and_bool_reductions():
    assert la.array([1.0, np.nan, 5.0]).argmax() == 1
    assert la.array([0.0, 0.0]).any() is np.False_
    assert la.array([1.0, 2.0]).all() is np.True_
    assert la.array([0.0, 2.0]).all() is np.False_


def test_dot():
    x, y = rng_data(7, 1), rng_data(7, 2)
    np.testing.assert_allclose(la.array(x).dot(la.array(y)), x.dot(y))
    np.testing.assert_allclose(la.dot(la.array(x), la.array(y)), np.dot(x, y))
    m = rng_data(21).reshape(3, 7)
    check(la.array(m).dot(la.array(y)), m.dot(y))
    with pytest.raises(ValueError):
        la.ones(3).dot(la.ones(4))


def test_clip_and_round():
    x = rng_data(20) * 3
    a = la.array(x)
    check(a.clip(-1, 1), x.clip(-1, 1))
    check(a.clip(None, 0.5), x.clip(None, 0.5))
    check(a.clip(0.0, None), x.clip(0.0, None))
    check(la.clip(a, -1, 1), np.clip(x, -1, 1))
    check(a.clip(la.zeros(20), 1.0), x.clip(np.zeros(20), 1.0))  # array bound: NumPy
    check(a.round(), x.round())
    check(a.round(2), x.round(2))
    check(a.round(decimals=-1), x.round(decimals=-1))
    check(la.round(la.array([0.5, 1.5, 2.5, -0.5])), np.array([0.0, 2.0, 2.0, -0.0]))


def test_array_positional_dtype_and_extra_keywords():
    check(la.array([1, 2], la.float64), np.array([1.0, 2.0]))
    assert la.array([1, 2], np.int32).dtype == np.int32
    got = la.array([[1.0, 2.0]], ndmin=3)
    check(got, np.array([[1.0, 2.0]], ndmin=3))
    with pytest.raises(TypeError):
        la.array()


@pytest.mark.parametrize("name", ["maximum", "minimum", "fmax", "fmin", "arctan2", "hypot", "copysign", "logaddexp"])
def test_binary_math_functions_native(name):
    x, y = rng_data(20, 1), rng_data(20, 2)
    f, g = getattr(la, name), getattr(np, name)
    check(f(la.array(x), la.array(y)), g(x, y))
    check(f(la.array(x), 0.5), g(x, 0.5))
    check(f(-0.5, la.array(y)), g(-0.5, y))
    assert isinstance(f(1.0, 2.0), float) and f(1.0, 2.0) == g(1.0, 2.0)
    check(f(la.array(x), y), g(x, y))  # NumPy operand: through NumPy


def test_nan_semantics_of_maximum_and_fmax():
    a = la.array([1.0, np.nan])
    b = la.array([np.nan, 2.0])
    np.testing.assert_array_equal(np.asarray(la.maximum(a, b)), np.maximum(np.asarray(a), np.asarray(b)))
    np.testing.assert_array_equal(np.asarray(la.fmax(a, b)), np.fmax(np.asarray(a), np.asarray(b)))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_concatenate_and_stack_native(axis):
    x, y = rng_data(6).reshape(2, 3), rng_data(9, 3).reshape(3, 3)
    a, b = la.array(x), la.array(y)
    if axis in (0, -2):
        check(la.concatenate([a, b]), np.concatenate([x, y]))
    check(la.concatenate([a, a], axis=axis), np.concatenate([x, x], axis=axis))
    check(la.concatenate((a, a, a), axis), np.concatenate((x, x, x), axis))
    check(la.stack([a, a], axis=axis), np.stack([x, x], axis=axis))
    check(la.stack([a, a, a], axis), np.stack([x, x, x], axis))
    check(la.concatenate([a, y]), np.concatenate([x, y]))  # mixed input: NumPy
    check(la.stack([x.tolist(), a]), np.stack([x, x]))
    with pytest.raises(ValueError):
        la.concatenate([a, b], axis=1)
    with pytest.raises(ValueError):
        la.stack([a, b])
    with pytest.raises(ValueError):
        la.concatenate([la.array(1.0), la.array(2.0)])


def test_where_native_and_fallback():
    x = rng_data(12).reshape(3, 4)
    a = la.array(x)
    check(la.where(a > 0, a, 0.0), np.where(x > 0, x, 0.0))
    check(la.where(a > 0, 1.0, a), np.where(x > 0, 1.0, x))
    check(la.where(a > 0, a, -a), np.where(x > 0, x, -x))
    check(la.where(a > 0, a, la.ones(4)), np.where(x > 0, x, np.ones(4)))  # broadcast: NumPy
    check(la.where([True, False, True], la.array([1.0, 2.0, 3.0]), 0.0), np.array([1.0, 0.0, 3.0]))
    got = la.where(a > 0)  # tuple of index arrays
    assert all(isinstance(g, la.ndarray) and g.dtype == np.int64 for g in got)
    with pytest.raises(ValueError):
        la.where(a > 0, a)


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 3, 4), (2, 1, 3, 2)])
def test_transpose_native(shape):
    x = rng_data(int(np.prod(shape))).reshape(shape)
    a = la.array(x)
    check(a.T, x.T)
    check(a.transpose(), x.transpose())
    check(la.transpose(a), np.transpose(x))
    if len(shape) == 3:
        check(a.transpose(1, 0, 2), x.transpose(1, 0, 2))  # explicit axes: NumPy
        check(a.transpose((2, 0, 1)), x.transpose((2, 0, 1)))


def test_sort_native_and_fallback():
    x = rng_data(30)
    a = la.array(x)
    check(la.sort(a), np.sort(x))
    check(la.sort(la.array([2.0, np.nan, 1.0])), np.sort(np.array([2.0, np.nan, 1.0])))
    m = rng_data(12).reshape(3, 4)
    check(la.sort(la.array(m)), np.sort(m))  # 2-D: NumPy
    check(la.sort(la.array(m), axis=0), np.sort(m, axis=0))
    check(la.sort(x.tolist()), np.sort(x))
    assert la.sort(la.zeros(0)).shape == (0,)
    a.sort()  # in place, through the writable view
    check(a, np.sort(x))


def test_linalg_norm_native_for_vectors():
    x = rng_data(10)
    a = la.array(x)
    before = _fallback.calls
    np.testing.assert_allclose(la.linalg.norm(a), np.linalg.norm(x))
    assert _fallback.calls == before
    m = rng_data(12).reshape(3, 4)
    np.testing.assert_allclose(la.linalg.norm(la.array(m)), np.linalg.norm(m))
    check(la.linalg.norm(la.array(m), axis=0), np.linalg.norm(m, axis=0))
    np.testing.assert_allclose(la.linalg.norm(a, ord=1), np.linalg.norm(x, ord=1))


def test_stacking_helpers_native():
    x, y = rng_data(3), rng_data(3, 5)
    m = rng_data(6).reshape(2, 3)
    a, b, c = la.array(x), la.array(y), la.array(m)
    before = _fallback.calls
    check(la.hstack([a, b]), np.hstack([x, y]))
    check(la.hstack((c, c)), np.hstack((m, m)))
    check(la.vstack([a, b]), np.vstack([x, y]))
    check(la.vstack([c, a]), np.vstack([m, x]))
    check(la.column_stack([a, b]), np.column_stack([x, y]))
    two = rng_data(2, 9)
    check(la.column_stack([c, la.array(two)]), np.column_stack([m, two]))
    check(la.atleast_2d(a), np.atleast_2d(x))
    check(la.atleast_2d(la.array(2.0)), np.atleast_2d(2.0))
    check(la.atleast_1d(la.array(2.0)), np.atleast_1d(2.0))
    assert _fallback.calls == before
    check(la.hstack([x, y]), np.hstack([x, y]))  # NumPy inputs still fine
    check(la.vstack([a, b], dtype=np.float64), np.vstack([x, y]))


def test_trapezoid_native():
    y = rng_data(20)
    x = np.sort(rng_data(20, 7))
    a, t = la.array(y), la.array(x)
    before = _fallback.calls
    np.testing.assert_allclose(la.trapezoid(a), np.trapezoid(y))
    np.testing.assert_allclose(la.trapezoid(a, t), np.trapezoid(y, x))
    np.testing.assert_allclose(la.trapezoid(a, dx=0.5), np.trapezoid(y, dx=0.5))
    assert la.trapezoid(la.array([1.0])) == 0.0
    assert _fallback.calls == before
    m = rng_data(12).reshape(3, 4)
    check(la.trapezoid(la.array(m), axis=0), np.trapezoid(m, axis=0))


def test_dlpack_export_is_zero_copy_and_read_only():
    import gc

    x = rng_data(12).reshape(3, 4)
    a = la.array(x)
    d = np.from_dlpack(a)
    np.testing.assert_array_equal(d, x)
    assert d.shape == (3, 4) and d.strides == a.strides and d.dtype == np.float64
    assert d.flags.writeable
    assert np.shares_memory(d, np.asarray(a))
    d[0, 0] = 42.0
    assert a[0, 0] == 42.0
    x[0, 0] = 42.0
    assert a.__dlpack_device__() == (1, 0)
    # the consumer keeps the array alive
    del a
    gc.collect()
    np.testing.assert_array_equal(d, x)
    # an unconsumed capsule is released cleanly
    b = la.array(x)
    cap = b.__dlpack__(max_version=(1, 0))
    assert "dltensor_versioned" in repr(cap)
    del cap
    gc.collect()
    np.testing.assert_array_equal(np.from_dlpack(b), x)
    # 0-d and 1-d
    assert float(np.from_dlpack(la.array(2.5))) == 2.5
    np.testing.assert_array_equal(np.from_dlpack(la.arange(4)), np.arange(4.0))
    # refused variants
    with pytest.raises(BufferError):
        b.__dlpack__(max_version=(1, 0), dl_device=(2, 0))
    c = np.from_dlpack(b, copy=True)
    assert not np.shares_memory(c, np.asarray(b))
    np.testing.assert_array_equal(c, x)


def test_dlpack_roundtrip_many_times_has_no_leak():
    import sys

    a = la.ones(5)
    base = sys.getrefcount(a)
    for _ in range(1000):
        np.from_dlpack(a)
    assert sys.getrefcount(a) == base


def test_shape_helpers_native():
    x = rng_data(6).reshape(1, 2, 1, 3)
    a = la.array(x)
    before = _fallback.calls
    check(la.squeeze(a), np.squeeze(x))
    check(la.squeeze(a, axis=0), np.squeeze(x, axis=0))
    check(la.squeeze(a, axis=(0, 2)), np.squeeze(x, axis=(0, 2)))
    check(la.expand_dims(a, 0), np.expand_dims(x, 0))
    check(la.expand_dims(a, -1), np.expand_dims(x, -1))
    check(la.expand_dims(a, (0, 3)), np.expand_dims(x, (0, 3)))
    m = rng_data(12).reshape(3, 4)
    b = la.array(m)
    check(la.flip(b), np.flip(m))
    check(la.flip(b, 0), np.flip(m, 0))
    check(la.flip(b, axis=(0, 1)), np.flip(m, axis=(0, 1)))
    check(la.diff(b), np.diff(m))
    check(la.diff(b, axis=0), np.diff(m, axis=0))
    v = la.array(m[0])
    check(la.diff(v), np.diff(m[0]))
    assert _fallback.calls == before
    with pytest.raises(ValueError):
        la.squeeze(a, axis=1)
    check(la.diff(b, n=2), np.diff(m, n=2))  # NumPy
    z = la.array([0.0, 1.0, 0.0, 2.5])
    assert la.count_nonzero(z) == 2 and isinstance(la.count_nonzero(z), np.integer)
    np.testing.assert_array_equal(la.count_nonzero(b > 0, axis=0), np.count_nonzero(m > 0, axis=0))


def test_full_with_array_fill_value_and_bool():
    check(la.full((2, 3), la.array([1.0, 2.0, 3.0])), np.full((2, 3), np.array([1.0, 2.0, 3.0])))
    assert la.full(3, True).dtype == bool  # bool fill values follow NumPy's dtype inference
    check(la.full_like(la.ones(3), la.array([1.0, 2.0, 3.0])), np.array([1.0, 2.0, 3.0]))


def test_buffer_views_do_not_leak_references():
    import sys

    a = la.ones(5)
    base = sys.getrefcount(a)
    for _ in range(1000):
        v = np.asarray(a)
        m = memoryview(a)
        _ = v.sum() + m[0]
        del v, m
    assert sys.getrefcount(a) == base
    # a view outliving its Python name keeps the data alive
    b = la.array([1.0, 2.0, 3.0])
    v = np.asarray(b)
    del b
    import gc

    gc.collect()
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])


# ---- mutability -------------------------------------------------------------


def test_setitem_integer_indices_native():
    x = np.zeros((3, 4))
    a = la.zeros((3, 4))
    before = _fallback.calls
    a[2, 3] = 2
    x[2, 3] = 2
    a[0, 0] = 1.5
    x[0, 0] = 1.5
    a[-1, -1] = -3
    x[-1, -1] = -3
    v = la.zeros(5)
    v[1] = 2.5
    v[-1] = np.float64(4.0)
    assert _fallback.calls == before
    check(a, x)
    assert v[1] == 2.5 and v[4] == 4.0
    with pytest.raises(IndexError):
        a[3, 0] = 1.0
    with pytest.raises(IndexError):
        a[0, 0, 0] = 1.0


@pytest.mark.parametrize(
    "key, value",
    [
        (slice(None), 0.5),
        (slice(1, 3), np.array([[7.0], [8.0]])),
        ((slice(None), 1), 9.0),
        ((1, slice(None)), [1.0, 2.0, 3.0, 4.0]),
        ((slice(None), slice(None, None, 2)), np.ones((3, 2))),
        (1, 3.0),
        (1, [5.0, 6.0, 7.0, 8.0]),
        ([0, 2], 4.0),
        (None, 1.0),
    ],
)
def test_setitem_general_forms_match_numpy(key, value):
    x = rng_data(12).reshape(3, 4)
    a = la.array(x)
    v = la.array(value) if isinstance(value, np.ndarray) else value
    a[key] = v
    x[key] = value
    check(a, x)


def test_setitem_with_mask_and_broadcast_errors():
    x = rng_data(12).reshape(3, 4)
    a = la.array(x)
    mask = a > 0
    a[mask] = 0.0
    x[x > 0] = 0.0
    check(a, x)
    with pytest.raises(ValueError):
        a[:, 0] = [1.0, 2.0]  # shape mismatch, NumPy's error


def test_inplace_operators_keep_identity_and_aliases():
    x = rng_data(6).reshape(2, 3) + 3.0
    a = la.array(x)
    alias = a
    before = _fallback.calls
    a += 1.0
    x += 1.0
    a *= la.array(x[0])  # broadcast row
    x *= x[0]
    a -= a  # self operand
    x -= x
    a += 2.5
    x += 2.5
    a /= 2.0
    x /= 2.0
    a **= 2
    x **= 2
    a //= 0.3
    x //= 0.3
    a %= 0.7
    x %= 0.7
    assert _fallback.calls == before
    assert a is alias
    check(a, x)
    check(alias, x)
    a += np.ones((2, 3))  # NumPy operand: applied in place through the view
    x += 1.0
    check(a, x)
    assert a is alias
    with pytest.raises(ValueError):
        la.ones(3).__iadd__(la.ones((2, 3)))  # would change shape


def test_view_and_dlpack_writes_are_shared():
    a = la.zeros(4)
    v = np.asarray(a)
    d = np.from_dlpack(a)
    v[0] = 1.0
    d[1] = 2.0
    a[2] = 3.0
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0, 0.0])
    np.testing.assert_array_equal(np.asarray(a), [1.0, 2.0, 3.0, 0.0])


def test_arrays_are_unhashable_like_numpy():
    with pytest.raises(TypeError):
        hash(la.ones(2))


@pytest.mark.parametrize("name", ["isnan", "isfinite", "isinf", "signbit"])
def test_predicates_native(name):
    x = np.array([1.0, -0.0, np.nan, np.inf, -np.inf, -2.5])
    a = la.array(x)
    before = _fallback.calls
    got = getattr(la, name)(a)
    assert isinstance(got, la.ndarray) and got.dtype == bool
    np.testing.assert_array_equal(got, getattr(np, name)(x))
    np.testing.assert_array_equal(getattr(np, name)(a), getattr(np, name)(x))  # via __array_ufunc__
    assert getattr(la, name)(float("nan")) == bool(getattr(np, name)(float("nan")))
    assert _fallback.calls == before
    m = la.array(x[:6].reshape(2, 3))
    assert getattr(la, name)(m).shape == (2, 3)


def test_complexity_predicates_native():
    a = la.ones((2, 2))
    before = _fallback.calls
    assert la.iscomplexobj(a) is False and la.isrealobj(a) is True
    np.testing.assert_array_equal(la.iscomplex(a), np.zeros((2, 2), bool))
    np.testing.assert_array_equal(la.isreal(a), np.ones((2, 2), bool))
    assert _fallback.calls == before
    assert la.iscomplexobj(np.array([1j])) is True and la.isrealobj([1.0, 2.0]) is True
    np.testing.assert_array_equal(la.iscomplex(np.array([1j, 1.0])), [True, False])


def test_array_keeps_numpy_dtype_for_non_float_inputs():
    c = la.asarray(1j)
    assert isinstance(c, np.ndarray) and c.dtype == np.complex128
    assert la.array(["a", "b"]).dtype.kind == "U"
    check(la.array([True, False]), np.array([True, False]))
    check(la.array([1, 2]), np.array([1, 2]))
    check(la.array([1, 2.5]), np.array([1, 2.5]))
    check(la.array([True, 2]), np.array([True, 2]))


def test_shape_methods_accept_numpy_keywords():
    x = rng_data(6).reshape(2, 3)
    a = la.array(x)
    check(a.ravel(order="C"), x.ravel(order="C"))
    check(a.ravel(order="F"), x.ravel(order="F"))
    check(a.flatten("F"), x.flatten("F"))
    check(a.reshape(3, 2, order="F"), x.reshape(3, 2, order="F"))
    check(a.copy(order="C"), x)


def test_scalar_predicates_return_numpy_bool():
    r = la.isfinite(1.0)
    assert isinstance(r, np.bool_) and r
    assert (~la.isfinite(float("inf"))) is np.True_ or bool(~la.isfinite(float("inf")))
    assert not la.isnan(2)


def test_any_all_native_on_numpy_bool_masks():
    x = rng_data(10)
    a = la.array(x)
    mask = a > 0
    before = _fallback.calls
    assert la.all(mask) == np.all(x > 0) and la.any(mask) == np.any(x > 0)
    assert la.all(a > -100) is np.True_ and la.any(a > 100) is np.False_
    assert isinstance(la.any(mask), np.bool_)
    assert _fallback.calls == before
    assert la.all(mask, axis=0) == np.all(x > 0, axis=0)  # NumPy
    assert la.all([True, False]) is np.False_ or not la.all([True, False])


# ---- compatibility round: easy Array API and NumPy cases ---------------------


def test_asarray_copy_and_device_keywords():
    a = la.ones(3)
    assert la.asarray(a, copy=None) is a
    b = la.asarray(a, copy=True)
    assert b is not a and isinstance(b, la.ndarray)
    assert la.asarray(a, device="cpu") is a
    with pytest.raises(ValueError):
        la.asarray(a, device="cuda")
    with pytest.raises(ValueError):
        la.asarray([1.0, 2.0], copy=False)
    for f in (la.zeros, la.ones, la.empty):
        assert f((2,), device="cpu").shape == (2,)
    check(la.full((2,), 1.5, device="cpu"), np.full((2,), 1.5))
    check(la.arange(3, device="cpu"), np.arange(3))
    check(la.linspace(0, 1, 3, device="cpu"), np.linspace(0, 1, 3))
    assert la.__array_api_version__ == np.__array_api_version__


def test_like_helpers_keep_numpy_dtypes():
    b = np.array([True, False])
    assert la.zeros_like(b).dtype == bool and la.ones_like(b).dtype == bool
    assert la.full_like(b, True).dtype == bool and la.empty_like(b).dtype == bool
    i = np.arange(3)
    assert la.zeros_like(i).dtype == np.int64
    check(la.zeros_like(la.ones((2, 2))), np.zeros((2, 2)))
    assert la.zeros_like(la.ones(2), shape=(3,)).shape == (3,)
    assert isinstance(la.full((2,), False), la.ndarray) and la.full((2,), False).dtype == bool


def test_finfo_iinfo_accept_arrays():
    assert la.finfo(la.ones(2)).eps == np.finfo(np.float64).eps
    assert la.finfo(np.float32).bits == 32
    assert la.iinfo(np.array(1, dtype=np.int8)).max == 127
    assert la.iinfo(np.int64).min == np.iinfo(np.int64).min


def test_integer_valued_index_arrays():
    x = rng_data(6)
    a = la.array(x)
    idx = la.array([0.0, 2.0, 4.0])  # float64, but integral: meant as indices
    check(a[idx], x[[0, 2, 4]])
    check(la.take(a, idx), np.take(x, [0, 2, 4]))
    check(la.take(a, [1, 1]), np.take(x, [1, 1]))
    m = rng_data(12).reshape(3, 4)
    b = la.array(m)
    check(la.take_along_axis(b, la.array([[0.0], [1.0], [2.0]]), axis=1), np.take_along_axis(m, np.array([[0], [1], [2]]), axis=1))
    a[idx] = 0.0
    x[[0, 2, 4]] = 0.0
    check(a, x)


def test_bitwise_index_complex_dunders_match_numpy():
    a = la.ones(3)
    x = np.ones(3)
    for op in ("__and__", "__or__", "__xor__", "__lshift__", "__rshift__", "__invert__"):
        with pytest.raises(TypeError):
            getattr(a, op)(a) if op != "__invert__" else a.__invert__()
        with pytest.raises(TypeError):
            getattr(x, op)(x) if op != "__invert__" else x.__invert__()
    with pytest.raises(TypeError):
        [1, 2, 3][la.array([1.0])]
    assert complex(la.array([2.5])) == 2.5 + 0j


def test_legacy_dlpack_capsule():
    a = la.arange(4)
    cap = a.__dlpack__()
    assert "dltensor" in repr(cap)
    del cap
    d = np.from_dlpack(a)
    np.testing.assert_array_equal(d, np.arange(4.0))


def test_sort_descending_and_keywords():
    x = rng_data(9)
    a = la.array(x)
    check(la.sort(a, descending=True), np.sort(x)[::-1])
    check(la.sort(a, stable=True), np.sort(x))
    check(la.sort(a, axis=-1, descending=False), np.sort(x))
    m = rng_data(12).reshape(3, 4)
    check(la.sort(la.array(m), axis=0, descending=True), np.flip(np.sort(m, axis=0), axis=0))


def test_signatures_expose_named_parameters():
    import inspect

    assert "keepdims" in inspect.signature(la.sum).parameters
    assert "correction" in inspect.signature(la.std).parameters
    assert "descending" in inspect.signature(la.sort).parameters
    assert "device" in inspect.signature(la.zeros_like).parameters
    assert "copy" in inspect.signature(la.asarray).parameters


def test_std_var_accept_array_api_correction():
    x = rng_data(12).reshape(3, 4)
    a = la.array(x)
    np.testing.assert_allclose(a.std(correction=1), x.std(ddof=1))
    check(a.var(axis=0, correction=1, keepdims=True), x.var(axis=0, ddof=1, keepdims=True))
    np.testing.assert_allclose(la.std(a, correction=0.0), x.std())


def test_isclose_native():
    x = np.array([1.0, 1.0 + 1e-9, np.nan, np.inf, -np.inf, 2.0])
    y = np.array([1.0, 1.0, np.nan, np.inf, np.inf, 2.5])
    a, b = la.array(x), la.array(y)
    before = _fallback.calls
    for kw in ({}, {"equal_nan": True}, {"rtol": 0.5}, {"atol": 1.0}):
        np.testing.assert_array_equal(la.isclose(a, b, **kw), np.isclose(x, y, **kw))
    np.testing.assert_array_equal(la.isclose(a, 1.0), np.isclose(x, 1.0))
    np.testing.assert_array_equal(la.isclose(2.5, b), np.isclose(2.5, y))
    assert la.isclose(1.0, 1.0 + 1e-9) and not la.isclose(1.0, 1.1)
    assert _fallback.calls == before
    np.testing.assert_array_equal(la.isclose(a, y), np.isclose(x, y))  # NumPy operand
    np.testing.assert_array_equal(la.isclose(la.ones((2, 3)), la.ones(3)), np.ones((2, 3), bool))  # broadcast: NumPy


def test_more_than_eight_dimensions_go_to_numpy():
    shape = (1,) * 12
    z = la.zeros(shape)
    assert isinstance(z, np.ndarray) and z.shape == shape
    assert la.ones(shape).shape == shape and la.full(shape, 2.0).shape == shape
    nested = [[[[[[[[[[1.0]]]]]]]]]]  # 10 levels
    a = la.array(nested)
    assert isinstance(a, np.ndarray) and a.ndim == 10
    assert la.asarray(np.zeros((1,) * 9)).ndim == 9


def test_index_arguments_of_delete_insert_bincount():
    x = rng_data(6)
    a = la.array(x)
    idx = la.array([1.0, 3.0])
    check(la.delete(a, idx), np.delete(x, [1, 3]))
    check(la.insert(a, idx, 0.0), np.insert(x, [1, 3], 0.0))
    counts = la.bincount(la.array([0.0, 1.0, 1.0, 3.0]))
    np.testing.assert_array_equal(counts, np.bincount([0, 1, 1, 3]))


def test_complex_scalar_operands_go_to_numpy():
    x = rng_data(4)
    a = la.array(x)
    for got, expected in [(a * 1j, x * 1j), (1j * a, 1j * x), (a + (1 + 2j), x + (1 + 2j)), ((2 - 1j) - a, (2 - 1j) - x), (a / 2j, x / 2j)]:
        assert isinstance(got, np.ndarray) and got.dtype == np.complex128
        np.testing.assert_allclose(got, expected)


def test_ndarray_constructor():
    a = la.ndarray((2, 3))
    check(a, np.zeros((2, 3)))
    assert la.ndarray(3, dtype=np.float64).shape == (3,)
    src = np.arange(6.0)
    b = la.ndarray((2, 3), buffer=src)
    check(b, src.reshape(2, 3))
    c = la.ndarray((2,), buffer=src, offset=8)
    check(c, np.array([1.0, 2.0]))
    with pytest.raises(TypeError):
        la.ndarray((2,), dtype=np.int32)


def test_ndarray_can_be_subclassed():
    class Tagged(la.ndarray):
        tag = "t"

    t = Tagged((2, 2))
    assert isinstance(t, la.ndarray) and t.tag == "t" and t.shape == (2, 2)
    t[0, 0] = 1.5
    check(t + 1, np.array([[2.5, 1.0], [1.0, 1.0]]))  # results are plain lightarray arrays
    assert type(t + 1) is la.ndarray
    assert np.asarray(t).shape == (2, 2)


def test_signed_zero_and_infinity_special_cases():
    x = np.array([-0.0, 0.0, 3.0, -3.0])
    for d in (1.0, -1.0, 1.5):
        got, exp = np.asarray(la.array(x) % d), x % d
        np.testing.assert_array_equal(got, exp)
        np.testing.assert_array_equal(np.signbit(got), np.signbit(exp))
    a, b = np.array([np.inf, -np.inf, np.inf, 1.0, np.nan]), np.array([1.0, -np.inf, np.inf, -np.inf, 1.0])
    np.testing.assert_array_equal(np.asarray(la.logaddexp(la.array(a), la.array(b))), np.logaddexp(a, b))
    z = np.array([-0.0, 0.0, -0.5, 0.5])
    got = np.asarray(la.trunc(la.array(z)))
    np.testing.assert_array_equal(np.signbit(got), np.signbit(np.trunc(z)))


def test_expand_dims_rejects_bad_axes_like_numpy():
    a = la.ones((2, 3))
    with pytest.raises(ValueError, match="repeated axis"):
        la.expand_dims(a, (0, 0))
    with pytest.raises(np.exceptions.AxisError):
        la.expand_dims(a, 5)
    check(la.expand_dims(la.zeros((0, 0)), (0, -1)), np.expand_dims(np.zeros((0, 0)), (0, -1)))


def test_shape_assignment_reshapes_in_place():
    x = np.arange(6.0)
    a = la.array(x)
    view = np.asarray(a)
    a.shape = (2, 3)
    x.shape = (2, 3)
    check(a, x)
    a.shape = (3, -1)
    assert a.shape == (3, 2) and view.shape == (6,)
    a.shape = 6
    assert a.shape == (6,)
    with pytest.raises(ValueError):
        a.shape = (4, 2)
    i = la.arange(4)
    i.shape = (2, 2)
    assert i.shape == (2, 2) and i.strides == (16, 8)


def test_out_arguments_return_the_callers_object():
    x, y = rng_data(4), rng_data(4, 5)
    a, b = la.array(x), la.array(y)
    out = la.zeros(4)
    assert la.add(a, b, out=out) is out
    check(out, x + y)
    assert la.sin(a, out=out) is out
    check(out, np.sin(x))
    assert la.multiply(a, 2.0, out) is out  # positional out
    check(out, x * 2.0)
    assert np.add(a, b, out=out) is out  # NumPy's own ufunc with a lightarray out
    check(out, x + y)
    big = la.zeros(8)
    assert la.concatenate([a, b], out=big) is big
    check(big, np.concatenate([x, y]))
    nout = np.zeros(4)
    assert la.add(a, b, out=nout) is nout
    np.testing.assert_allclose(nout, x + y)
    check(la.sqrt(la.array([4.0, -1.0, 9.0]), where=np.array([True, False, True]), out=la.zeros(3)), np.array([2.0, 0.0, 3.0]))
    check(la.add(a, b, dtype=np.float64), x + y)


def test_argsort_descending_and_nonzero_of_a_scalar():
    x = np.array([3.0, 1.0, 3.0, 2.0, 1.0])
    a = la.array(x)
    order = la.argsort(a, descending=True, stable=True)
    assert np.asarray(order).tolist() == [0, 2, 3, 1, 4]
    assert np.asarray(la.argsort(a, descending=False)).tolist() == np.argsort(x, stable=True).tolist()
    m = rng_data(3, 4)
    np.testing.assert_array_equal(
        np.asarray(la.take_along_axis(la.array(m), la.argsort(la.array(m), axis=0, descending=True), axis=0)), -np.sort(-m, axis=0)
    )
    ties = np.zeros(40, dtype=np.uint8)
    assert np.asarray(la.argsort(ties)).tolist() == list(range(40))  # stable by default
    assert np.asarray(la.argsort(la.zeros((2, 40)))).tolist() == [list(range(40))] * 2
    assert np.asarray(la.argsort(ties, descending=True)).tolist() == list(range(40))
    nine = la.reshape(la.array(1.0), (1,) * 9)  # beyond the native eight dimensions: NumPy's
    assert nine.shape == (1,) * 9
    with pytest.raises(ValueError):
        la.nonzero(la.array(1.0))
    with pytest.raises(ValueError):
        la.array(1.0).nonzero()


def test_asarray_copy_and_numpy_scalars():
    u = np.array([1, 2], dtype=np.uint8)
    c = la.asarray(u, copy=True)
    assert c is not u and not np.shares_memory(c, u) and c.dtype == np.uint8
    assert la.asarray(u) is u
    assert la.asarray(u, dtype=np.uint8, copy=False) is u
    assert not np.shares_memory(la.asarray(u, dtype=np.uint8, copy=True), u)
    with pytest.raises(ValueError):
        la.asarray(u, dtype=np.uint16, copy=False)
    assert la.asarray(u, dtype=np.float64).dtype == np.float64 and isinstance(la.asarray(u, dtype=np.float64), la.ndarray)
    assert type(la.cos(np.float64(0.0))) is np.float64
    assert type(la.cos(0.0)) is float
    assert type(la.log(np.float64(1.0))) is np.float64


def test_raw_entry_points_keep_every_call_form():
    """The hot functions skip argument parsing for the plain call; keywords,
    extra arguments and errors behave as before."""
    import inspect

    from lightarray import _core

    v = [0.5, 1.5, 2.5]
    a = la.array(v)
    assert la.asarray(a) is a and la.asarray(a, dtype=float) is a and la.asarray(a, copy=True) is not a
    assert la.asarray(object=v).dtype == np.float64
    check(la.array(object=v, copy=True), np.array(v))
    assert la.array(v, dtype=np.int64).dtype == np.int64
    check(la.where(a > 1.0, a, 0.0), np.where(np.array(v) > 1.0, v, 0.0))
    assert np.asarray(la.where(condition=a > 1.0)[0]).tolist() == [1, 2]
    assert float(la.sum(a)) == float(la.sum(a=a)) == 4.5
    check(la.sum(la.ones((2, 3)), axis=0), np.full(3, 2.0))
    check(la.sum(la.ones((2, 3)), 1, keepdims=True), np.full((2, 1), 3.0))
    assert float(la.mean(v)) == 1.5 and bool(la.any(a > 2)) and float(la.dot(a, a)) == float(np.dot(v, v))
    for call in (lambda: la.asarray(), lambda: la.where(), lambda: la.sum(), lambda: la.sin(), lambda: la.add(a)):
        with pytest.raises(TypeError):
            call()
    with pytest.raises(TypeError):
        la.array(v, bogus=1)
    for name in ("asarray", "array", "where", "sum", "any", "sin", "add", "isnan", "maximum"):
        assert "(" in str(inspect.signature(getattr(la, name)))
        assert _core.__all__.count(name) == 1
