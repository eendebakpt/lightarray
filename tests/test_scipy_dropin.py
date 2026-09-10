"""SciPy functions from several submodules called with lightarray inputs
(through the buffer protocol), compared with NumPy inputs; then the same with
SciPy's Python layer rebound to lightarray."""

import lightarray as la
import numpy
import pytest

pytest.importorskip("scipy")
import scipy  # noqa: E402
import scipy.fft  # noqa: E402
import scipy.integrate  # noqa: E402
import scipy.interpolate  # noqa: E402
import scipy.linalg  # noqa: E402
import scipy.optimize  # noqa: E402
import scipy.signal  # noqa: E402
import scipy.stats  # noqa: E402


def workload(np):
    x = np.linspace(0.0, 2 * np.pi, 64)
    y = np.sin(x) + 0.1 * np.cos(3 * x)
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    out = {
        "solve": scipy.linalg.solve(A, b),
        "eigvals": np.sort(scipy.linalg.eigvalsh(A)),
        "trapz": scipy.integrate.trapezoid(y, x),
        "simpson": scipy.integrate.simpson(y, x=x),
        "quad": scipy.integrate.quad(lambda t: float(np.exp(-t * t)), 0.0, 1.0)[0],
        "interp": scipy.interpolate.interp1d(x, y)(np.array([0.5, 1.5, 2.5])),
        "cubic": scipy.interpolate.CubicSpline(x, y)(1.234),
        "convolve": scipy.signal.convolve(y, np.ones(5) / 5, mode="same"),
        "fft_abs": np.abs(scipy.fft.fft(y))[:8],
        "norm_pdf": scipy.stats.norm.pdf(x[:5], loc=1.0, scale=2.0),
        "minimize_x": scipy.optimize.minimize(lambda p: float(((p - b) ** 2).sum()), np.zeros(3)).x,
        "curve_fit": scipy.optimize.curve_fit(lambda t, a, w: a * np.sin(w * t), x, y, p0=[1.0, 1.0])[0],
        "root": scipy.optimize.brentq(lambda t: float(np.cos(t) - 0.3), 0.0, 2.0),
        "pearson": scipy.stats.pearsonr(x, y)[0],
    }
    return out


def assert_same(got, expected):
    for k in expected:
        numpy.testing.assert_allclose(numpy.asarray(got[k]), numpy.asarray(expected[k]), rtol=1e-6, atol=1e-9, err_msg=k)


def test_scipy_accepts_lightarray_inputs():
    expected = workload(numpy)
    got = workload(la)
    assert_same(got, expected)


def test_scipy_python_layer_patched_workload():
    """SciPy's compiled kernels insist on real ndarrays, so SciPy is patched
    with the conversion functions kept on NumPy."""
    expected = workload(numpy)
    rebound = la.patch_module(scipy, conversions="numpy")
    try:
        assert rebound > 100
        assert_same(workload(la), expected)
        assert_same(workload(numpy), expected)
    finally:
        la.unpatch_module(scipy)
