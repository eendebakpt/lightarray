"""int64 and bool arrays: dtype inference, promotion, native kernels, and
NumPy semantics for everything delegated."""

import pickle

import lightarray as la
import numpy as np
import pytest
from lightarray import _fallback


def check(got, expected):
    expected = np.asarray(expected)
    assert isinstance(got, la.ndarray), type(got)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(got), expected)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [[1, 2], [3, 4]],
        [True, False],
        [[True], [False]],
        [1, True],
        [1, 2.5],
        [True, 0.5],
        3,
        True,
        2.5,
        [],
        (1, 2),
        [2**40, -(2**40)],
    ],
)
def test_dtype_inference_matches_numpy(data):
    check(la.array(data), np.array(data))
    check(la.asarray(data), np.asarray(data))


def test_integers_beyond_int64_go_to_numpy():
    got = la.array([2**70, 1])
    assert isinstance(got, np.ndarray) and got.dtype == object


@pytest.mark.parametrize("x", [np.arange(6).reshape(2, 3), np.array([True, False, True]), np.int64(5), np.bool_(True), np.array(7)])
def test_numpy_int_and_bool_inputs_become_lightarray(x):
    check(la.array(x), np.asarray(x))


@pytest.mark.parametrize("x", [np.arange(3, dtype=np.int32), np.arange(3, dtype=np.uint8), np.float32(1.5), np.array([1j])])
def test_other_numpy_dtypes_stay_numpy(x):
    got = la.array(x)
    assert isinstance(got, np.ndarray) and got.dtype == np.asarray(x).dtype


def test_creation_functions_infer_or_take_dtype():
    check(la.arange(5), np.arange(5))
    check(la.arange(2, 11, 3), np.arange(2, 11, 3))
    check(la.arange(5, 0, -2), np.arange(5, 0, -2))
    check(la.arange(0), np.arange(0))
    check(la.arange(5.0), np.arange(5.0))
    check(la.arange(0, 1, 0.25), np.arange(0, 1, 0.25))
    check(la.full((2, 2), 7), np.full((2, 2), 7))
    check(la.full(3, True), np.full(3, True))
    check(la.full(3, 7, dtype=float), np.full(3, 7, dtype=float))
    check(la.zeros(3, dtype=int), np.zeros(3, dtype=int))
    check(la.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=bool))
    check(la.zeros_like(la.arange(4)), np.zeros_like(np.arange(4)))
    check(la.full_like(la.ones(3), 7), np.full_like(np.ones(3), 7))
    check(la.ndarray((2,), dtype=np.int64), np.zeros(2, dtype=np.int64))
    check(la.ndarray((2,), dtype=bool), np.zeros(2, dtype=bool))


def test_attributes_and_conversions():
    a = la.array([[1, 2, 3], [4, 5, 6]])
    x = np.array([[1, 2, 3], [4, 5, 6]])
    for name in ("shape", "strides", "ndim", "size", "itemsize", "nbytes", "dtype"):
        assert getattr(a, name) == getattr(x, name), name
    m = la.array([True, False, True])
    assert (m.itemsize, m.nbytes, m.strides) == (1, 3, (1,))
    assert repr(a) == repr(x) and repr(m) == repr(np.array([True, False, True]))
    assert int(la.array([7])) == 7 and float(la.array([7])) == 7.0 and bool(la.array([0])) is False
    assert [10, 20, 30][la.array(1)] == 20  # __index__ for integer scalars
    with pytest.raises(TypeError):
        [1, 2][la.array(1.0)]
    assert np.asarray(a).dtype == np.int64 and np.shares_memory(np.asarray(a), np.asarray(a))
    assert np.from_dlpack(a).dtype == np.int64 and np.from_dlpack(m).dtype == bool
    check(pickle.loads(pickle.dumps(a)), x)
    check(pickle.loads(pickle.dumps(m)), np.array([True, False, True]))


OPS = ["add", "sub", "mul", "truediv", "floordiv", "mod", "pow"]


@pytest.mark.parametrize("op", OPS)
def test_integer_arithmetic_matches_numpy(op):
    import operator

    f = getattr(operator, op)
    x, y = np.array([7, -3, 12, 5]), np.array([2, 5, 4, 3])
    a, b = la.array(x), la.array(y)
    check(f(a, b), f(x, y))
    check(f(a, 3), f(x, 3))
    check(f(3, b), f(3, y))
    check(f(a, 2.5), f(x, 2.5))  # float scalar: float64
    check(f(2.5, b), f(2.5, y))
    check(f(a, la.array(y.astype(float))), f(x, y.astype(float)))  # int op float array
    check(f(la.array(x.astype(float)), b), f(x.astype(float), y))


def test_add_subtract_multiply_and_float_mixes_are_native():
    a, b, f = la.array([1, 2, 3]), la.array([4, 5, 6]), la.array([0.5, 1.5, 2.5])
    before = _fallback.calls
    a + b, a - 1, 2 * a, a * f, f + a, a / b, a / 2, a * 0.5, -a, abs(a), a > 1, a == b, a >= 2.5
    assert _fallback.calls == before


def test_bool_arithmetic_follows_numpy():
    m, x = la.array([True, False, True]), np.array([True, False, True])
    check(m + m, x + x)
    check(m * 2, x * 2)
    check(m * 0.5, x * 0.5)
    check(m + la.array([1, 2, 3]), x + np.array([1, 2, 3]))
    with pytest.raises(TypeError):
        -m


def test_comparisons_are_bool_lightarrays():
    x = np.array([0.5, -1.0, 2.0, np.nan])
    a = la.array(x)
    before = _fallback.calls
    for got, exp in [
        (a > 0, x > 0),
        (a <= 0.5, x <= 0.5),
        (a == a, x == x),
        (a != 2, x != 2),
        (1 < a, 1 < x),
        (a > la.array([0, 0, 3, 0]), x > np.array([0, 0, 3, 0])),
    ]:
        check(got, exp)
    i = la.array([3, 1, 2])
    check(i > 1, np.array([3, 1, 2]) > 1)
    check(i == la.array([3, 0, 2]), np.array([True, False, True]))
    check(i < 2.5, np.array([3, 1, 2]) < 2.5)
    assert _fallback.calls == before
    check(a > la.ones((2, 4)), x > np.ones((2, 4)))  # broadcast: NumPy
    check(la.array([True, False]) == la.array([True, True]), np.array([True, False]))


def test_masks_combine_count_and_index_natively():
    x = np.linspace(-1, 1, 9)
    a = la.array(x)
    before = _fallback.calls
    m = (a > -0.5) & (a < 0.5)
    check(m, (x > -0.5) & (x < 0.5))
    check(~m, ~((x > -0.5) & (x < 0.5)))
    check((a < -0.5) | (a > 0.5), (x < -0.5) | (x > 0.5))
    check(m ^ (a > 0), ((x > -0.5) & (x < 0.5)) ^ (x > 0))
    check(a[m], x[(x > -0.5) & (x < 0.5)])
    assert m.sum() == 3 and isinstance(m.sum(), np.integer)
    assert m.any() and not m.all() and la.any(m) and not la.all(m)
    check(la.where(m, a, 0.0), np.where((x > -0.5) & (x < 0.5), x, 0.0))
    a2 = a.copy()
    assert _fallback.calls == before
    a2[m] = 0.0  # mask assignment goes through the NumPy view
    x2 = x.copy()
    x2[(x > -0.5) & (x < 0.5)] = 0.0
    np.testing.assert_array_equal(np.asarray(a2), x2)


def test_integer_index_arrays():
    x = np.arange(10.0) * 1.5
    a = la.array(x)
    idx = la.array([0, 3, -1])
    before = _fallback.calls
    check(a[idx], x[[0, 3, -1]])
    assert _fallback.calls == before
    order = la.argsort(la.array([3.0, 1.0, 2.0]))
    check(order, np.argsort([3.0, 1.0, 2.0]))
    check(la.array([3.0, 1.0, 2.0])[order], np.array([1.0, 2.0, 3.0]))
    m = la.array(np.arange(12.0).reshape(3, 4))
    check(m[la.array([2, 0])], np.arange(12.0).reshape(3, 4)[[2, 0]])
    check(m[:, la.array([1, 3])], np.arange(12.0).reshape(3, 4)[:, [1, 3]])
    check(la.take(a, idx), np.take(x, [0, 3, -1]))


def test_integer_reductions_and_indexing():
    x = np.array([[5, -2, 9], [4, 4, 0]])
    a = la.array(x)
    before = _fallback.calls
    for name in ("sum", "max", "min"):
        got = getattr(a, name)()
        assert got == getattr(x, name)() and isinstance(got, np.integer), name
    assert a.mean() == x.mean() and isinstance(a.mean(), np.float64)
    assert a[1, 0] == 4 and isinstance(a[1, 0], np.integer)
    check(a[0], x[0])
    check(a[:, 1:], x[:, 1:])
    check(a.T, x.T)
    check(a.reshape(3, 2), x.reshape(3, 2))
    check(a.ravel(), x.ravel())
    assert a.any() and not a.all()
    assert _fallback.calls == before
    check(a.sum(axis=0), x.sum(axis=0))  # NumPy
    check(a.cumsum(), x.cumsum())
    assert a.argmax() == x.argmax() and a.std() == x.std()
    b = la.array([True, False])
    assert b[0] and isinstance(b[0], np.bool_)


def test_assignment_keeps_the_array_dtype():
    a, x = la.array([1, 2, 3]), np.array([1, 2, 3])
    a[0] = 9
    x[0] = 9
    a[1] = 2.9  # NumPy truncates on assignment into an int array
    x[1] = 2.9
    a[1:] += 1
    x[1:] += 1
    a *= 2
    x *= 2
    check(a, x)
    with pytest.raises(TypeError):
        a /= 2  # cannot cast float result to int64, like NumPy
    m = la.array([True, False])
    m[1] = True
    check(m, np.array([True, True]))


def test_math_functions_on_integer_arrays():
    x = np.array([1, 4, 9])
    a = la.array(x)
    before = _fallback.calls
    check(la.sqrt(a), np.sqrt(x))
    check(la.sin(a), np.sin(x))
    assert _fallback.calls == before
    got = la.exp(la.array([True, False]))  # bool input: NumPy computes in float16
    assert got.dtype == np.exp(np.array([True, False])).dtype
    check(la.square(a), np.square(x))  # integer result: NumPy
    check(la.negative(a), np.negative(x))
    check(la.floor(a), np.floor(x))
    check(np.sqrt(a), np.sqrt(x))


def test_concatenate_and_stack_by_dtype():
    a, b = la.array([1, 2]), la.array([3, 4])
    before = _fallback.calls
    check(la.concatenate([a, b]), np.array([1, 2, 3, 4]))
    check(la.stack([a, b]), np.array([[1, 2], [3, 4]]))
    check(la.concatenate([a > 1, b > 3]), np.array([False, True, False, True]))
    assert _fallback.calls == before
    check(la.concatenate([a, la.array([0.5])]), np.array([1.0, 2.0, 0.5]))  # mixed dtypes: NumPy promotes
