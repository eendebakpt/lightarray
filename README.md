# lightarray

A small-array library for Python with the NumPy interface. Arrays of up to a
few thousand elements run faster than NumPy because the
per-operation overhead smaller.
Everything lightarray does not implement itself is delegated to NumPy.

```python
import lightarray as np  # instead of: import numpy as np

t = np.linspace(0.0, 3.0, 601)
omega = 2 * np.pi
p = np.sin(omega * t / 2) ** 2  # runs in Rust
print(p.mean(), p.argmax(), p.std())  # native reductions
print(np.polyfit(t, p, 3))  # NumPy, transparently
```

- **Same names as NumPy.** `lightarray` exposes every public name of the
  `numpy` module and `lightarray.ndarray` every method and property of
  `numpy.ndarray`. Names without a native implementation call NumPy on a
  zero-copy view and return lightarray arrays for float64, int64 and bool
  results.
- **NumPy interop both ways.** The buffer protocol, `__array__`,
  `__array_ufunc__`, `__array_function__` and DLPack are implemented, so
  matplotlib, SciPy and NumPy itself accept lightarray arrays, and NumPy
  functions called on them hand back lightarray arrays.
- **Mutable, like NumPy.** `x[2, 3] = 2`, `x[:, 0] = row`, `x[mask] = 0`,
  `x += 1` and in-place methods such as `x.sort()` all work; `np.asarray(x)`
  is a writable zero-copy view.
- **Rust core, PyO3 bindings.** Contiguous float64, int64 and bool buffers
  with inline shape and strides; the binding overhead was measured against a
  hand-written C extension (`benchmarks/carray_reference`) before choosing
  Rust.

## Using it in place of NumPy

Change the import and nothing else:

```python
import lightarray as np  # was: import numpy as np
```

Downstream libraries keep their own `numpy` import; they receive lightarray
arrays through the buffer protocol and hand back NumPy arrays, which
lightarray accepts everywhere. [examples/lmfit_model_fit.py](examples/lmfit_model_fit.py)
is lmfit's "Fitting with Model" documentation example with only the import
changed: the model evaluation, noise and residuals run in lightarray and
lmfit/SciPy perform the optimisation. `tests/test_lmfit.py` checks that it
reaches the same optimum as with NumPy.

To see how much of a script runs natively, read `lightarray._fallback.calls`
before and after: it counts the operations delegated to NumPy.

### Taking over NumPy inside a library

A library such as lmfit keeps its own `import numpy as np`; its functions
look that name up at call time. `lightarray.patch_module(lmfit)` rebinds the
NumPy references in a loaded package (the `np` alias, ufuncs, names from
`from numpy import ...`, submodules) to lightarray, so the library's own
array work runs on lightarray without editing it. Classes such as
`np.ndarray` used in `isinstance` checks stay NumPy, and compiled code
receives lightarray arrays through the buffer protocol.

```python
import lmfit
import lightarray as np

np.set_patched(lmfit, True)  # on; False restores NumPy; is_patched() queries
with np.patched(lmfit):  # on for a block
    result = model.fit(y, x=x, amp=5, cen=5, wid=1)
```

```bash
LIGHTARRAY_PATCH=lmfit,scipy:numpy python my_script.py   # no code change at all
```

For packages whose compiled kernels require real NumPy arrays (SciPy) use
`conversions="numpy"` (the `:numpy` suffix above): conversion functions
stay NumPy's and every other function runs on lightarray only when it
receives a lightarray argument. With both lmfit and SciPy patched, lmfit's
documentation examples reach the same optimum as with NumPy and the
per-evaluation path delegates nothing ([examples/lmfit_internals.py](examples/lmfit_internals.py),
`tests/test_lmfit.py`, `tests/test_scipy_dropin.py`).

## Speed

Total time per operation in nanoseconds, best of 7 runs, benchmarked against
NumPy 2.5.3 on Python 3.14.2.

| Operation | lightarray | NumPy |
|---|---|---|
| `a + b` | 80 | 345 |
| `a * 2.0` | 80 | 533 |
| `np.sin(a)` | 103 | 378 |
| `a.sum()` | 65 | 499 |
| `a.std()` | 66 | 6529 |
| `np.sum(a)` | 62 | 1550 |
| `a > 0.5` | 90 | 535 |
| `a[a > 0.5]` | 206 | 844 |
| `(a > 0.2) & (a < 0.8)` | 251 | 1427 |
| `np.where(a > 0.5, a, 0.0)` | 275 | 1411 |
| `i * 2` | 99 | 629 |
| `a[idx]` | 116 | 144 |
| `m[1]` | 74 | 78 |
| `m[:, 1]` | 195 | 86 |
| `np.array(values)` | 136 | 476 |
| `np.arange(10)` | 113 | 401 |
| `np.zeros(10)` | 114 | 176 |
| `a[3] = 2.0` | 40 | 39 |
| `a += 1.0` | 26 | 495 |

`a` and `b` are float64 arrays of 10 elements, `m` is a 10 x 10 float64
array, `i` is `np.arange(10)`, `idx` an int64 array of 3 indices and `values`
a list of 10 floats (`python benchmarks/readme_table.py` regenerates the table).
Reductions return NumPy's scalar types (`np.float64`, `np.int64`), built
through NumPy's C API in about 25 ns. Creating a strided view (`m[:, 1]`) is the one
operation slower than NumPy: it allocates the contiguous cache the kernels
work on. Above roughly 10000 elements the two libraries converge; lightarray
is not a large-array library.

## Status

Version 0.3.0. float64, int64 and bool arrays are native, with NumPy's
dtype inference (`np.array([1, 2])` is int64, comparisons give bool arrays,
int and float mix to float64); every other dtype and the long tail of NumPy
functions go through NumPy at NumPy speed plus about 1 µs and come back as
NumPy arrays.

`isinstance(a, numpy.ndarray)` is True for a lightarray array, so library code
that checks for arrays (SciPy's root finders, OApackage's converters) takes
its array branch. The real type is still `lightarray.ndarray`:
`type(a) is numpy.ndarray` is False, and compiled code that demands an actual
NumPy array (Cython's typed arguments) converts through the buffer protocol
or rejects it. The reverse does not hold in a script that does
`import lightarray as np`: a NumPy array that lightarray hands back for a
dtype it does not hold is not an instance of `np.ndarray` there.

Indexing with integers and slices,
`reshape`, `ravel` and `.T` return views that share memory with the array, as
in NumPy (`row = a[0]; row[:] = 0` and `a[:, 1] *= 2` change `a`). The
kernels work on contiguous buffers: a contiguous selection is a window into
the base's buffer at no extra cost, while a strided one (`a[:, 0]`, `a[::2]`,
the transpose of a matrix) is gathered from the base when it is used, which
is cheap for small arrays and one extra pass over the data for large ones.
Views that NumPy returns for delegated operations (`swapaxes`, `split`,
`a[..., 1]`) stay views as well.

Where NumPy and the Array API disagree, lightarray follows NumPy. Against the
official Array API test suite it passes 1346 of 1374 tests; 27 of the 28
failures are tests NumPy lists as expected failures of its own
(`tools/ci/array-api-xfails.txt`): `finfo` returning NumPy scalars, complex
`expm1` at infinities (numpy#21746) and `floor_divide` of infinities, where
NumPy follows Python. The last one is indexing a NumPy array with an *empty*
lightarray bool mask, which NumPy casts to an integer index. lightarray does
add the Array API keywords NumPy lacks (`sort(descending=)`,
`fft.fftfreq(dtype=)`), since they change nothing NumPy does.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin "numpy>=2.1" pytest matplotlib lmfit
maturin develop --release
pytest                          # parity tests against NumPy
cargo test --release --lib      # kernel tests
benchmarks/carray_reference/build.sh && python benchmarks/bench_overhead.py
python benchmarks/gate.py       # coarse performance gate with absolute limits (CI)
python benchmarks/perf_check.py --save   # record this machine's timings of 70 key operations ...
python benchmarks/perf_check.py          # ... and fail when a later build is slower (1% overall, 6% per operation)
python examples/lmfit_model_fit.py   # lmfit example running on lightarray
python examples/lmfit_internals.py   # lmfit's own internals rebound to lightarray
```

Tested with the test suites of lmfit (650 of 650 with lmfit's internals on
lightarray), OApackage (115 of 115, including its SWIG-wrapped C++ entry
points) and parts of SciPy's and NumPy's.

New features must not cost the hot paths anything: run `perf_check.py --save`
before starting on a change and `perf_check.py` after rebuilding. It measures
in several fresh processes pinned to one core and compares with the saved
baseline, so it notices a few nanoseconds where `gate.py` only catches gross
regressions.

Requires Python 3.10+, NumPy 2.1+, and a stable Rust toolchain.
