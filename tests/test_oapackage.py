"""OApackage (orthogonal arrays; a SWIG-wrapped C++ library) on lightarray:
lightarray arrays go into its C++ entry points and its Python layer runs on
lightarray when patched. Its two top-level modules, `oapackage` and the SWIG
module `oalib`, are patched together."""

import subprocess
import sys

import lightarray as la
import numpy as np
import pytest

oapackage = pytest.importorskip("oapackage")


def design():
    return (np.arange(40).reshape(8, 5) * 7 // 3) % 2


def test_array_link_from_lightarray_and_back():
    n = design()
    al_np, al_la = oapackage.array_link(n), oapackage.array_link(la.array(n))
    assert al_la.shape == (8, 5) and al_la == al_np
    back = la.array(al_la)  # OApackage arrays are 16- or 32-bit integers, dtypes lightarray leaves to NumPy
    assert back.dtype.kind == "i" and type(back) is np.ndarray
    np.testing.assert_array_equal(back, n)
    np.testing.assert_array_equal(np.asarray(la.array(al_la, dtype=np.int64)), n)
    assert oapackage.makearraylink(la.array(n)) == al_np  # gated on isinstance(x, np.ndarray)


def test_cpp_functions_accept_lightarray():
    n = design()
    al = oapackage.array_link(la.array(n))
    assert al.Defficiency() == pytest.approx(oapackage.array_link(n).Defficiency())
    assert list(al.GWLP()) == pytest.approx(list(oapackage.array_link(n).GWLP()))
    graph = np.zeros((5, 5), dtype=int)
    graph[1, 0] = graph[0, 1] = 1
    assert list(oapackage.reduceGraphNauty(la.array(graph), verbose=0)) == list(oapackage.reduceGraphNauty(graph, verbose=0))
    al2 = oapackage.array_link()
    oapackage.update_array_link(al2, la.array([[1, 2, 3], [4, 5, 6]]))
    assert al2.shape == (2, 3) and list(al2) == [1, 4, 2, 5, 3, 6]


@pytest.mark.parametrize("mode", ["lightarray", "numpy"])
def test_patching_oapackage_covers_the_swig_module(mode):
    import oalib

    assert not la.is_patched(oalib)
    with la.patched(oapackage, conversions=mode):
        assert la.is_patched(oalib) and oalib.np is not np
        n = design()
        assert oapackage.makearraylink(la.array(n)) == oapackage.makearraylink(n)
        dds = np.array([[0.5, 0.25, 0.75], [0.1, 0.9, 0.3]])
        scores = oapackage.Doptim.calcScore(la.array(dds), [1, 0.5, 0])
        np.testing.assert_allclose(np.asarray(scores), [0.625, 0.55])
    assert oalib.np is np and not la.is_patched(oalib)


def test_environment_toggle_runs_an_oapackage_script():
    script = (
        "import numpy, lightarray as np, oapackage, oalib\n"
        "assert np.is_patched(oapackage) and np.is_patched(oalib)\n"
        "al = oapackage.exampleArray(2, 0)\n"
        "x = np.array(al)\n"
        "assert isinstance(x, numpy.ndarray) and x.shape == al.shape\n"
        "scores = oapackage.Doptim.calcScore(np.array([[0.5, 0.25, 0.75]]), [1, 0.5, 0])\n"
        "print(type(scores).__module__, float(scores[0]) == 0.625)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env={**__import__("os").environ, "LIGHTARRAY_PATCH": "oapackage"}
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("lightarray True")
