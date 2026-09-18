"""Basic indexing, reshape, ravel and transpose share memory with the array
they come from, like NumPy. Selections that are one contiguous block are
windows into the base's buffer; strided ones (`a[:, 0]`, `a[::2]`, `a.T`) keep
a contiguous cache that is refreshed from the base on every read and written
back after every native write, and export the base's memory with the true
strides through the buffer protocol."""

import copy
import gc
import pickle

import lightarray as la
import numpy as np
import pytest


def pair(shape=(3, 4)):
    n = np.arange(float(np.prod(shape))).reshape(shape)
    return la.array(n), n


VIEW_EXPRESSIONS = [
    "a[1]",
    "a[-1]",
    "a[1:3]",
    "a[1:]",
    "a[:2]",
    "a[1, 1:3]",
    "a[1:3, :]",
    "a[0:1, 1:3]",
    "a[...]",
    "a[None]",
    "a[1:3][0]",
    "a[1][1:3]",
    "a.reshape(12)",
    "a.reshape(2, 6)",
    "a.reshape(4, 3)[1:3]",
    "a.ravel()",
    "la.reshape(a, (6, 2))",
    "la.ravel(a)",
    # strided
    "a[:, 0]",
    "a[:, -1]",
    "a[::2]",
    "a[::-1]",
    "a[1:3, 1:3]",
    "a[::2, ::-1]",
    "a[:, 1:3]",
    "a[None, :, 1]",
    "a[:, None, 2]",
    "a.T",
    "a.transpose()",
    "a.T[1:3]",
    "a.T[1:3, ::2]",
    "a.T.T",
    "a[:, 1][::2]",
    "a[::2][1]",
    "a[:, ::3][1:, 1]",
    "la.transpose(a)",
]


@pytest.mark.parametrize("expr", VIEW_EXPRESSIONS)
def test_views_share_memory_and_write_through(expr):
    a, n = pair()
    v = eval(expr, {"a": a, "la": la})
    w = eval(expr, {"a": n, "la": np})
    assert v.shape == w.shape
    np.testing.assert_array_equal(np.asarray(v), w)
    assert v.base is a
    # (the stride of a length-1 axis is arbitrary; NumPy and lightarray differ there)
    assert [s for s, k in zip(v.strides, v.shape, strict=True) if k > 1] == [s for s, k in zip(w.strides, w.shape, strict=True) if k > 1]
    assert np.shares_memory(np.asarray(v), np.asarray(a))
    # writes through the view reach the base ...
    v[...] = -1.0
    w[...] = -1.0
    np.testing.assert_array_equal(np.asarray(a), n)
    v += 3.0
    w += 3.0
    np.testing.assert_array_equal(np.asarray(a), n)
    v.fill(2.5)
    w.fill(2.5)
    np.testing.assert_array_equal(np.asarray(a), n)
    first = (0,) * v.ndim
    v[first] = 8.0
    w[first] = 8.0
    np.testing.assert_array_equal(np.asarray(a), n)
    # ... and writes to the base show up in the view
    a[...] = np.arange(12.0).reshape(3, 4) * 2
    n[...] = np.arange(12.0).reshape(3, 4) * 2
    np.testing.assert_array_equal(np.asarray(v), w)
    np.testing.assert_array_equal(np.asarray(v + 1.0), w + 1.0)
    assert float(v.sum()) == float(w.sum())


@pytest.mark.parametrize(
    "expr", ["a.flatten()", "a.copy()", "a[[0, 2]]", "a[a > 3]", "a.T.reshape(12)", "a[:, 0].ravel()", "a + 0", "a[:, 0] * 1", "a[1:1]"]
)
def test_copies_do_not_write_through(expr):
    a, n = pair()
    v = eval(expr, {"a": a})
    np.testing.assert_array_equal(np.asarray(v), eval(expr, {"a": n}))
    assert v.base is None
    v[...] = -1.0
    np.testing.assert_array_equal(np.asarray(a), n)


def test_element_assignment_through_a_row():
    a, n = pair()
    a[2][1] = 9.0
    n[2][1] = 9.0
    row, nrow = a[0], n[0]
    row[0] = 7.0
    nrow[0] = 7.0
    row *= 2.0
    nrow *= 2.0
    np.testing.assert_array_equal(np.asarray(a), n)


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
def test_views_for_every_native_dtype(dtype):
    n = (np.arange(12).reshape(3, 4) % 3).astype(dtype)
    a = la.array(n)
    assert a.dtype == dtype
    a[1][:] = 1
    n[1][:] = 1
    a[0, 1:3] = 0
    n[0, 1:3] = 0
    np.testing.assert_array_equal(np.asarray(a), n)
    assert a[1].dtype == dtype and a[1].base is a


def test_overlapping_operands_behave_like_numpy():
    a, n = la.arange(8.0), np.arange(8.0)
    a[1:] += a[:-1]
    n[1:] += n[:-1]
    np.testing.assert_array_equal(np.asarray(a), n)
    a[:-1] -= a[1:]
    n[:-1] -= n[1:]
    np.testing.assert_array_equal(np.asarray(a), n)
    a[2:6] = a[0:4]
    n[2:6] = n[0:4]
    np.testing.assert_array_equal(np.asarray(a), n)
    v, w = a[2:6], n[2:6]
    v *= v
    w *= w
    np.testing.assert_array_equal(np.asarray(a), n)
    i, m = la.arange(8), np.arange(8)
    i[1:] += i[:-1]
    m[1:] += m[:-1]
    np.testing.assert_array_equal(np.asarray(i), m)


def test_view_keeps_its_base_alive():
    row = la.arange(12.0).reshape(3, 4)[1]
    gc.collect()
    junk = [la.zeros(12) for _ in range(100)]  # reuse freed memory, if any was freed
    np.testing.assert_array_equal(np.asarray(row), [4.0, 5.0, 6.0, 7.0])
    assert row.base.shape == (12,)  # the arange the reshape was a view of
    del junk


def test_views_of_views_refer_to_the_owner():
    a, _ = pair()
    assert a[1:3][0][1:3].base is a
    assert a.base is None


def test_copy_and_pickle_give_independent_arrays():
    a, n = pair()
    row = a[1]
    for clone in (copy.copy(row), copy.deepcopy(row), pickle.loads(pickle.dumps(row)), row.copy(), la.array(row)):
        assert clone.base is None
        clone[0] = -5.0
        np.testing.assert_array_equal(np.asarray(clone)[1:], n[1][1:])
    np.testing.assert_array_equal(np.asarray(a), n)


def test_numpy_sees_views_as_views():
    a, n = pair()
    np.asarray(a[1])[2] = 42.0
    n[1][2] = 42.0
    np.testing.assert_array_equal(np.asarray(a), n)
    out = a[2]
    np.multiply(a[0], 2.0, out=out)
    np.multiply(n[0], 2.0, out=n[2])
    np.testing.assert_array_equal(np.asarray(a), n)


def test_shape_helpers_share_memory():
    a, n = pair()
    for name, args in [("squeeze", ()), ("expand_dims", (0,)), ("atleast_2d", ()), ("atleast_1d", ())]:
        v = getattr(la, name)(a[1], *args)
        w = getattr(np, name)(n[1], *args)
        assert v.shape == w.shape
        assert np.shares_memory(np.asarray(v), np.asarray(a)) == np.shares_memory(w, n), name


def test_filling_a_matrix_row_by_row():
    out, ref = la.zeros((4, 3)), np.zeros((4, 3))
    for k in range(4):
        out[k][:] = k
        ref[k][:] = k
        row = out[k]
        row += 0.5
        ref[k] += 0.5
    np.testing.assert_array_equal(np.asarray(out), ref)


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
def test_strided_views_for_every_native_dtype(dtype):
    n = (np.arange(12).reshape(3, 4) % 3).astype(dtype)
    a = la.array(n)
    col, ncol = a[:, 1], n[:, 1]
    assert col.dtype == dtype and col.base is a
    col[0] = 1
    ncol[0] = 1
    col[1:] = 0
    ncol[1:] = 0
    a.T[2][:] = 1
    n.T[2][:] = 1
    np.testing.assert_array_equal(np.asarray(a), n)
    np.testing.assert_array_equal(np.asarray(col), ncol)


def test_strided_views_in_three_dimensions():
    n = np.arange(60.0).reshape(3, 4, 5)
    a = la.array(n)
    for expr in ["a[:, 1, :]", "a[:, :, 2]", "a[::2, 1:, ::2]", "a.T", "a[1, :, ::2]", "a.T[1:4, ::3, 1]"]:
        v, w = eval(expr, {"a": a}), eval(expr, {"a": n})
        assert v.shape == w.shape and v.strides == w.strides, expr
        np.testing.assert_array_equal(np.asarray(v), w)
        v *= -1.0
        w *= -1.0
        np.testing.assert_array_equal(np.asarray(a), n)


def test_strided_view_overlap_and_swap():
    a, n = pair()
    a[:, 0] += a[:, 1]
    n[:, 0] += n[:, 1]
    a[::2] -= a[1]
    n[::2] -= n[1]
    a.T[0] *= a[:, 0]
    n.T[0] *= n[:, 0]
    a[:, [0, 1]] = a[:, [1, 0]]
    n[:, [0, 1]] = n[:, [1, 0]]
    a[:, ::-1] = a
    n[:, ::-1] = n.copy()  # NumPy would also get this right without the copy
    np.testing.assert_array_equal(np.asarray(a), n)


def test_numpy_functions_write_into_strided_views():
    a, n = pair()
    np.multiply(a[:, 1], 3.0, out=a[:, 2])
    np.multiply(n[:, 1], 3.0, out=n[:, 2])
    la.add(a[:, 0], 1.0, out=a[:, 3])
    np.add(n[:, 0], 1.0, out=n[:, 3])
    a.T[1].sort()
    n.T[1].sort()
    np.asarray(a[::2])[0, 0] = 99.0
    n[::2][0, 0] = 99.0
    np.testing.assert_array_equal(np.asarray(a), n)


def test_buffer_export_of_strided_views():
    a, _ = pair()
    col = a[:, 1]
    m = memoryview(col)
    assert not m.c_contiguous and m.strides == (32,) and m.shape == (3,)
    assert m.tolist() == [1.0, 5.0, 9.0]
    assert memoryview(a[1]).c_contiguous
    with pytest.raises((BufferError, TypeError)):
        import ctypes  # a consumer asking for a plain contiguous block cannot get one

        ctypes.c_double.from_buffer(col)
    flags = np.asarray(col).flags
    assert not flags.c_contiguous and flags.writeable
    assert np.asarray(a.T).flags.f_contiguous


def test_strided_view_shape_cannot_be_set_in_place():
    a, _ = pair()
    with pytest.raises(AttributeError):
        a.T.shape = (12,)
    a[1].shape = (2, 2)  # contiguous views can


def test_strided_view_survives_its_base_and_copies_cleanly():
    col = la.arange(12.0).reshape(3, 4)[:, 2]
    gc.collect()
    junk = [la.zeros(12) for _ in range(100)]
    np.testing.assert_array_equal(np.asarray(col), [2.0, 6.0, 10.0])
    for clone in (
        copy.copy(col),
        copy.deepcopy(col),
        pickle.loads(pickle.dumps(col)),
        col.copy(),
        la.array(col),
        la.ascontiguousarray(col),
    ):
        assert clone.base is None and clone.strides == (8,)
        np.testing.assert_array_equal(np.asarray(clone), [2.0, 6.0, 10.0])
    del junk


def test_iteration_and_scalar_access_on_strided_views():
    a, n = pair()
    assert [float(x) for x in a[:, 1]] == [float(x) for x in n[:, 1]]
    assert [list(map(float, r)) for r in a.T] == [list(map(float, r)) for r in n.T]
    assert float(a.T[3, 2]) == float(n.T[3, 2])
    assert a.T.tolist() == n.T.tolist()
    assert repr(a[:, 1]) == repr(n[:, 1])


def test_matrix_products_with_transposes():
    a, n = pair()
    np.testing.assert_allclose(np.asarray(a @ a.T), n @ n.T)
    np.testing.assert_allclose(np.asarray(a.T @ a), n.T @ n)
    np.testing.assert_allclose(np.asarray(la.dot(a.T, a[:, 0])), np.dot(n.T, n[:, 0]))
