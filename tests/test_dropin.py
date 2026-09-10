"""Run typical small-array NumPy code with `import lightarray as np` and
compare every result with real NumPy. This is the drop-in check: the script
must run unchanged and give the same numbers.

It also counts how many calls went through the NumPy fallback, so the
number of non-native operations in ordinary code is visible."""

import lightarray
import numpy
import pytest
from lightarray import _fallback

# Upper bound on NumPy fallbacks per script; lower it as natives are added.
FALLBACK_BUDGET = {"analysis_script": 12, "two_level_system": 0, "matrix_and_masking": 20}


def analysis_script(np):
    """A small data-analysis routine written against the NumPy API."""
    t = np.linspace(0.0, 2.0, 41)
    signal = np.exp(-t) * np.cos(2 * np.pi * 1.5 * t) + 0.1 * np.sin(20 * t)
    signal = signal - signal.mean()
    envelope = np.abs(signal)
    peaks = np.where(envelope > 0.5, envelope, 0.0)
    energy = np.sum(signal**2) * (t[1] - t[0])
    window = np.ones(5) / 5
    smoothed = np.convolve(signal, window, mode="same")
    stats = np.stack([signal, smoothed])
    return {
        "t_max": float(t[np.argmax(envelope)]),
        "energy": float(energy),
        "std": float(signal.std()),
        "clipped": np.clip(signal, -0.5, 0.5),
        "peaks": peaks,
        "cumulative": np.cumsum(envelope),
        "stats_mean": stats.mean(axis=1),
        "stats_max": stats.max(axis=0),
        "sorted_top3": np.sort(envelope)[-3:],
        "norm": float(np.linalg.norm(signal)),
        "slice": signal[5:15:2],
        "outer": np.outer(signal[:3], signal[:3]),
        "poly": np.polyfit(t, signal, 3),
        "hist": np.histogram(signal, bins=5)[0],
    }


def matrix_and_masking(np):
    """2-D data handling idioms: centring, masking, row/column statistics, reshaping."""
    data = np.arange(24.0).reshape(4, 6) / 7.0 - 1.0
    centred = data - data.mean(axis=0)
    scaled = centred / (data.std(axis=0) + 1e-9)
    positive = data[data > 0]
    rows = data[[3, 1]]
    top_left = data[:2, :3]
    col = data[:, 2]
    outer = col[:, None] * data[0][None, :]
    flat = data.ravel()
    return {
        "scaled_norm": float(np.sqrt((scaled**2).sum())),
        "positive_count": int(positive.size),
        "positive_mean": float(positive.mean()),
        "rows": rows,
        "top_left_T": top_left.T,
        "col_max_idx": int(col.argmax()),
        "outer_trace": float(np.trace(outer)),
        "flat_cumsum": np.cumsum(flat)[-3:],
        "any_negative": bool((data < 0).any()),
        "abs_max": float(np.abs(data).max()),
        "diag": np.diag(data[:4, :4]),
        "rounded": np.round(data[0], 2),
        "hstack": np.hstack([col, col]),
        "mesh": np.meshgrid(np.arange(3.0), np.arange(2.0))[0],
        "integral": float(np.trapezoid(np.sin(flat), flat)),
        "clipped_sum": float(np.clip(data, -0.5, 0.5).sum()),
        "log_of_positive": np.log(data[data > 0.1]),
        "fraction": float(np.count_nonzero(data > 0) / data.size),
        "interp": np.interp([0.5, 1.5], [0.0, 1.0, 2.0], [0.0, 10.0, 20.0]),
    }


def two_level_system(np):
    """The Rabi calculation from la.py, generalised over detuning."""
    omega = 2 * np.pi
    t = np.linspace(0.0, 3.0, 61)
    out = []
    for delta in np.linspace(0.0, 4.0, 5):
        omega_r = np.sqrt(omega**2 + delta**2)
        p = (omega / omega_r) ** 2 * np.sin(omega_r * t / 2) ** 2
        out.append((float(p.max()), float(p.mean()), int(p.argmax())))
    return out


class FallbackCounter:
    """Number of operations delegated to NumPy inside the block."""

    def __enter__(self):
        self._start = _fallback.calls
        return self

    def __exit__(self, *exc):
        self.count = _fallback.calls - self._start


def assert_same(got, expected):
    if isinstance(expected, dict):
        assert got.keys() == expected.keys()
        for k in expected:
            assert_same(got[k], expected[k])
    elif isinstance(expected, (list, tuple)):
        assert len(got) == len(expected)
        for g, e in zip(got, expected, strict=True):
            assert_same(g, e)
    elif isinstance(expected, numpy.ndarray):
        assert isinstance(got, (lightarray.ndarray, numpy.ndarray))
        numpy.testing.assert_allclose(numpy.asarray(got), expected, rtol=1e-10, atol=1e-12)
    else:
        numpy.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("script", [analysis_script, matrix_and_masking, two_level_system])
def test_script_runs_unchanged_with_lightarray(script):
    expected = script(numpy)
    with FallbackCounter() as counter:
        got = script(lightarray)
    assert_same(got, expected)
    print(f"{script.__name__}: {counter.count} fallback calls")
    assert counter.count <= FALLBACK_BUDGET[script.__name__]


def test_rabi_script_has_no_fallbacks():
    with FallbackCounter() as counter:
        two_level_system(lightarray)
    assert counter.count == 0, f"{counter.count} calls went through NumPy"
