"""Call every public NumPy function that accepts a single array on lightarray
inputs. The point is not that each call succeeds (many need more arguments)
but that nothing crashes the interpreter, and that whenever NumPy and
lightarray both succeed they agree. This exercises the buffer-protocol and
dispatch code paths across the whole NumPy namespace."""

import warnings

import lightarray as la
import numpy as np
import pytest

SKIP = {
    # interactive / side effects / not array functions
    "show_config",
    "show_runtime",
    "info",
    "source",
    "lookfor",
    "test",
    "get_include",
    "save",
    "savez",
    "savez_compressed",
    "savetxt",
    "load",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "fromregex",
    "fromstring",
    "frombuffer",
    "memmap",
    "seterr",
    "seterrcall",
    "geterr",
    "geterrcall",
    "errstate",
    "set_printoptions",
    "get_printoptions",
    "printoptions",
    "setbufsize",
    "getbufsize",
    "may_share_memory",
    "shares_memory",
    "busday_count",
    "busday_offset",
    "is_busday",
    "datetime_as_string",
    "datetime_data",
    "require",
    "bmat",
    "asmatrix",  # np.matrix is deprecated and not an ndarray in the usual sense
    "empty",
    "empty_like",  # uninitialised memory cannot be compared
}


MUTATES_FIRST_ARGUMENT = set()  # arrays are mutable; in-place NumPy functions write through the view


def _candidates():
    for name in sorted(dir(np)):
        if name.startswith("_") or name in SKIP:
            continue
        obj = getattr(np, name)
        if callable(obj) and not isinstance(obj, type):
            yield name


@pytest.mark.parametrize("name", list(_candidates()))
@pytest.mark.parametrize("shape", [(5,), (2, 3)])
def test_numpy_function_on_lightarray_does_not_crash(name, shape):
    x = np.random.default_rng(1).standard_normal(shape) + 1.5
    a = la.array(x)
    outcomes = []
    for arg in (x, a):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outcomes.append(("ok", getattr(np, name)(arg)))
        except Exception as e:  # noqa: BLE001 - any Python-level failure is acceptable
            outcomes.append(("err", type(e)))
    (kind_np, val_np), (kind_la, val_la) = outcomes
    if kind_np == "ok" and kind_la == "ok":
        if isinstance(val_np, np.ndarray) and val_np.dtype.kind == "f":
            np.testing.assert_allclose(np.asarray(val_la), val_np, rtol=1e-10, atol=1e-12, err_msg=name)
        elif isinstance(val_np, (np.ndarray, np.generic, float, int, bool)) and np.ndim(val_np) == 0:
            np.testing.assert_allclose(float(val_la), float(val_np), rtol=1e-10, atol=1e-12, err_msg=name)
    elif kind_np == "ok":
        pytest.fail(f"np.{name} works on NumPy input but raised {val_la.__name__} on lightarray")


@pytest.mark.parametrize("name", list(_candidates()))
def test_lightarray_function_on_lightarray_does_not_crash(name):
    """Same sweep through the lightarray namespace (wrapped or native)."""
    x = np.random.default_rng(2).standard_normal((3, 4)) + 1.5
    a = la.array(x)
    try:
        expected = getattr(np, name)(x)
    except Exception:  # noqa: BLE001
        pytest.skip("needs more arguments")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = getattr(la, name)(a)
    if isinstance(expected, np.ndarray) and expected.dtype.kind == "f":
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("name", list(_candidates()))
def test_two_argument_numpy_function_on_lightarray_does_not_crash(name):
    """Same sweep with two array arguments (binary ufuncs, dot, allclose, ...)."""
    x = np.random.default_rng(3).standard_normal((3, 3)) + 1.5
    y = np.random.default_rng(4).standard_normal((3, 3)) + 1.5
    a, b = la.array(x), la.array(y)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = getattr(np, name)(x, y)
    except Exception:  # noqa: BLE001
        pytest.skip("not a two-array function")
    outputs = expected if isinstance(expected, tuple) else (expected,)
    if any(e is y for e in outputs) or name in MUTATES_FIRST_ARGUMENT:
        pytest.skip("writes into an argument; lightarray arrays are immutable")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            got = getattr(np, name)(a, b)
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"np.{name} works on NumPy input but raised {type(e).__name__}: {e} on lightarray")
        got_la = getattr(la, name)(a, b)
    for result in (got, got_la):
        if isinstance(expected, np.ndarray) and expected.dtype.kind == "f":
            np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-10, atol=1e-12, err_msg=name)
        elif isinstance(expected, (float, np.floating)):
            np.testing.assert_allclose(float(result), float(expected), rtol=1e-10, err_msg=name)
