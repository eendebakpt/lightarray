"""Drop-in check against a real downstream library: lmfit's documentation
examples run with `import lightarray as np` and reach the same optimum as
with NumPy."""

import lightarray
import numpy
import pytest
from lightarray import _fallback

lmfit = pytest.importorskip("lmfit")


def model_fit(np, seed=7):
    """lmfit 'Fitting with Model' example (doc_model_gaussian.py)."""
    from lmfit import Model

    def gaussian(x, amp, cen, wid):
        return (amp / (np.sqrt(2 * np.pi) * wid)) * np.exp(-((x - cen) ** 2) / (2 * wid**2))

    x = np.linspace(-10, 10, 101)
    noise = numpy.random.default_rng(seed).normal(scale=0.1, size=101)
    y = gaussian(x, 2.33, 0.21, 1.51) + np.asarray(noise)
    result = Model(gaussian).fit(y, x=x, amp=5, cen=5, wid=1)
    return result


def minimize_example(np, seed=3):
    """lmfit 'Getting started' example: minimize() with a residual function."""
    from lmfit import Parameters, minimize

    def residual(params, x, data):
        amp = params["amp"]
        phase = params["phase"]
        freq = params["frequency"]
        decay = params["decay"]
        model = amp * np.sin(x * freq + phase) * np.exp(-x * x * decay)
        return model - data

    x = np.linspace(0, 15, 301)
    noise = numpy.random.default_rng(seed).normal(size=301, scale=0.2)
    data = 5.0 * np.sin(2 * x - 0.1) * np.exp(-x * x * 0.025) + np.asarray(noise)
    params = Parameters()
    params.add("amp", value=10)
    params.add("decay", value=0.1)
    params.add("phase", value=0.2)
    params.add("frequency", value=3.0)
    return minimize(residual, params, args=(x, data))


@pytest.mark.parametrize("example", [model_fit, minimize_example])
def test_lmfit_example_runs_on_lightarray_and_matches_numpy(example):
    expected = example(numpy)
    before = _fallback.calls
    got = example(lightarray)
    delegated = _fallback.calls - before
    assert got.success
    for name in expected.params:
        numpy.testing.assert_allclose(got.params[name].value, expected.params[name].value, rtol=1e-6, atol=1e-8)
    numpy.testing.assert_allclose(got.chisqr, expected.chisqr, rtol=1e-6)
    if hasattr(expected, "best_fit"):  # lmfit stores its own NumPy copy of the evaluated model
        numpy.testing.assert_allclose(numpy.asarray(got.best_fit), expected.best_fit, rtol=1e-6)
    print(f"{example.__name__}: {delegated} delegated calls over {got.nfev} function evaluations")


@pytest.mark.parametrize("example", [model_fit, minimize_example])
def test_lmfit_internals_rebound_to_lightarray(example):
    """`lightarray.patch_module(lmfit)` rebinds lmfit's own NumPy references
    (its `np` aliases, ufuncs and `from numpy import ...` names) to
    lightarray, so lmfit's internal array work runs on lightarray as well.
    SciPy underneath keeps NumPy, so every array crossing into the
    optimiser is still converted (zero-copy) at that boundary."""
    expected = example(numpy)
    rebound = lightarray.patch_module(lmfit)
    try:
        assert rebound > 0
        before = _fallback.calls
        got = example(lightarray)
        delegated = _fallback.calls - before
    finally:
        assert lightarray.unpatch_module(lmfit) == rebound
    assert got.success
    for name in expected.params:
        numpy.testing.assert_allclose(got.params[name].value, expected.params[name].value, rtol=1e-6, atol=1e-8)
    numpy.testing.assert_allclose(got.chisqr, expected.chisqr, rtol=1e-6)
    print(f"{example.__name__} with lmfit patched: {delegated} delegated calls over {got.nfev} function evaluations")


def test_patch_toggle_and_context_manager():
    assert not lightarray.is_patched(lmfit)
    assert lightarray.set_patched(lmfit, True) is True
    assert lightarray.set_patched(lmfit, True) is True  # idempotent
    assert lmfit.minimizer.np is not numpy and lmfit.minimizer.np.zeros is lightarray.zeros
    assert lightarray.set_patched(lmfit, False) is False
    assert lmfit.minimizer.np is numpy
    with lightarray.patched(lmfit):
        assert lightarray.is_patched(lmfit) and lmfit.model.np is not numpy
        result = minimize_example(lightarray)
        assert result.success
    assert not lightarray.is_patched(lmfit) and lmfit.model.np is numpy


def test_environment_toggle_patches_on_import(tmp_path):
    import os
    import subprocess
    import sys

    code = "import lightarray, lmfit, numpy\nprint(lightarray.is_patched(lmfit), lmfit.minimizer.np is not numpy)\n"
    env = {**os.environ, "LIGHTARRAY_PATCH": "lmfit"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.strip() == "True True"
    env["LIGHTARRAY_PATCH"] = ""
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False False"
    code2 = "import scipy.optimize, lightarray\nprint(lightarray.is_patched(scipy), repr(scipy.optimize._minpack_py.np))\n"
    env["LIGHTARRAY_PATCH"] = "scipy:numpy"
    out = subprocess.run([sys.executable, "-c", code2], env=env, capture_output=True, text=True, check=True).stdout
    assert out.startswith("True <lightarray (type-preserving")


def test_patched_lmfit_still_works_for_numpy_scripts():
    """A script that keeps using NumPy arrays must work while lmfit is patched:
    lmfit's Parameter.__array__ now calls lightarray's array(), and NumPy
    insists that __array__ returns a real ndarray."""
    with lightarray.patched(lmfit):
        p = lmfit.Parameters()
        p.add("f", value=2.0)
        out = numpy.ones(3) * p["f"]
        assert isinstance(out, numpy.ndarray) and out.tolist() == [2.0, 2.0, 2.0]
        assert isinstance(p["f"].__array__(), numpy.ndarray)
        result = minimize_example(numpy)
        assert result.success
        result = minimize_example(lightarray)
        assert result.success


def test_scipy_python_layer_rebound_to_lightarray():
    """Going further: rebinding every loaded SciPy Python module too (its
    compiled kernels keep NumPy). Both lmfit examples still converge, for a
    NumPy-based and a lightarray-based script."""
    import scipy
    import scipy.optimize

    expected = {ex.__name__: ex(numpy) for ex in (model_fit, minimize_example)}
    rebound_lmfit = lightarray.patch_module(lmfit)
    rebound_scipy = lightarray.patch_module(scipy)
    try:
        assert rebound_scipy > 100
        for ex in (model_fit, minimize_example):
            for lib in (numpy, lightarray):
                got = ex(lib)
                assert got.success
                for name in expected[ex.__name__].params:
                    numpy.testing.assert_allclose(got.params[name].value, expected[ex.__name__].params[name].value, rtol=1e-6, atol=1e-8)
    finally:
        assert lightarray.unpatch_module(scipy) == rebound_scipy
        assert lightarray.unpatch_module(lmfit) == rebound_lmfit


def test_dunder_array_implementations_return_numpy_while_patched():
    import types

    mod = types.ModuleType("fake_pkg")
    mod.np = numpy
    exec(
        "class A:\n"
        "    def __array__(self, dtype=None, copy=None): return np.arange(3.0)\n"
        "class B:\n"
        "    def __array__(self, dtype=None, copy=None): return np.array([1.0, 2.0])\n"
        "class C:\n"
        "    def __array__(self, dtype=None, copy=None): return np.sin(np.zeros(2))\n",
        mod.__dict__,
    )
    with lightarray.patched(mod):
        assert mod.np is not numpy
        for cls in (mod.A, mod.B, mod.C):
            assert isinstance(cls().__array__(), numpy.ndarray), cls.__name__
            numpy.testing.assert_array_equal(numpy.asarray(cls()), cls().__array__())
    assert isinstance(mod.A().__array__(), numpy.ndarray)


def test_rebound_ufuncs_keep_their_attributes():
    import types

    mod = types.ModuleType("fake_pkg2")
    mod.np = numpy
    mod.add = numpy.add
    exec("def total(x):\n    return np.add.reduce(x), add.reduce(x), np.maximum.accumulate(x), add.nin, add(x, 1)\n", mod.__dict__)
    for mode in ("lightarray", "numpy"):
        lightarray.patch_module(mod, conversions=mode)
        try:
            a, b, c, nin, d = mod.total(lightarray.array([1.0, 3.0, 2.0]))
            assert a == 6.0 and b == 6.0 and nin == 2
            numpy.testing.assert_array_equal(numpy.asarray(c), [1.0, 3.0, 3.0])
            numpy.testing.assert_array_equal(numpy.asarray(d), [2.0, 4.0, 3.0])
            assert isinstance(d, lightarray.ndarray)
            e = mod.total(numpy.array([1.0, 3.0, 2.0]))[4]
            assert isinstance(e, lightarray.ndarray if mode == "lightarray" else numpy.ndarray)
        finally:
            lightarray.unpatch_module(mod)


def test_isinstance_checks_inside_patched_packages_accept_lightarray():
    import types

    mod = types.ModuleType("fake_pkg3")
    mod.np = numpy
    mod.ndarray = numpy.ndarray
    exec(
        "def kind(x):\n"
        "    return isinstance(x, np.ndarray), isinstance(x, ndarray), issubclass(type(x), np.ndarray)\n"
        "def make():\n"
        "    return np.ndarray((2,), dtype=float)\n",
        mod.__dict__,
    )
    # unpatched: isinstance is True through `__class__`; the real type is lightarray's own
    assert mod.kind(lightarray.ones(2)) == (True, True, False)
    with lightarray.patched(mod):
        assert mod.kind(lightarray.ones(2)) == (True, True, True)
        assert mod.kind(numpy.ones(2)) == (True, True, True)
        assert mod.kind([1.0, 2.0]) == (False, False, False)
        assert isinstance(mod.make(), numpy.ndarray)
    assert mod.kind(lightarray.ones(2)) == (True, True, False)


def test_proxies_survive_copy_and_pickle():
    import copy
    import pickle
    import types

    mod = types.ModuleType("fake_pkg4")
    mod.add = numpy.add
    for mode in ("lightarray", "numpy"):
        lightarray.patch_module(mod, conversions=mode)
        try:
            proxy = mod.add
            assert proxy is not numpy.add
            assert copy.deepcopy(proxy) is proxy and copy.copy(proxy) is proxy
            assert pickle.loads(pickle.dumps(proxy)) is numpy.add
            assert copy.deepcopy({"f": proxy})["f"] is proxy
        finally:
            lightarray.unpatch_module(mod)


def test_patching_covers_sibling_modules_of_the_same_distribution(tmp_path, monkeypatch):
    """A distribution can install several top-level modules (OApackage ships
    `oapackage` and the SWIG module `oalib`); patching the package covers the
    pure-Python siblings the distribution owns, and only those."""
    import importlib
    import sys

    dist = tmp_path / "twomod-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: twomod\nVersion: 1.0\n")
    (dist / "top_level.txt").write_text("twomod\ntwomod_helper\ntests\n")
    (dist / "RECORD").write_text("twomod/__init__.py,,\ntwomod_helper.py,,\ntests/__init__.py,,\n")
    (tmp_path / "twomod").mkdir()
    (tmp_path / "twomod" / "__init__.py").write_text("import numpy as np\nimport twomod_helper\n")
    (tmp_path / "twomod_helper.py").write_text("import numpy as np\n\ndef is_array(x):\n    return isinstance(x, np.ndarray)\n")
    elsewhere = tmp_path / "elsewhere" / "tests"
    elsewhere.mkdir(parents=True)
    (elsewhere / "__init__.py").write_text("import numpy as np\n")  # somebody else's `tests` package
    monkeypatch.syspath_prepend(str(tmp_path / "elsewhere"))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    lightarray._sibling_names_cache.pop("twomod", None)
    for name in ("twomod", "twomod_helper", "tests"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    twomod, helper, other = (importlib.import_module(n) for n in ("twomod", "twomod_helper", "tests"))
    try:
        lightarray.patch_module(twomod)
        assert lightarray.is_patched(twomod) and helper.np is not numpy
        assert helper.is_array(lightarray.zeros(2)) and helper.is_array(numpy.zeros(2)) and not helper.is_array([1.0])
        assert other.np is numpy  # same name as a listed module, but not the distribution's file
        lightarray.unpatch_module(twomod)
        assert helper.np is numpy
    finally:
        lightarray.unpatch_module(twomod)
        lightarray._sibling_names_cache.pop("twomod", None)


def test_type_preserving_mode_accepts_both_array_types_in_isinstance(tmp_path, monkeypatch):
    import importlib
    import sys

    (tmp_path / "isinst_mod.py").write_text("import numpy as np\n\ndef is_array(x):\n    return isinstance(x, np.ndarray)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    monkeypatch.delitem(sys.modules, "isinst_mod", raising=False)
    mod = importlib.import_module("isinst_mod")
    for mode in ("lightarray", "numpy"):
        lightarray.patch_module(mod, conversions=mode)
        try:
            assert mod.is_array(numpy.zeros(2)) and mod.is_array(lightarray.zeros(2)) and not mod.is_array([0.0]), mode
        finally:
            lightarray.unpatch_module(mod)
