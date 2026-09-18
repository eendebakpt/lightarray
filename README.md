# lightarray

A small-array library for Python with the NumPy interface. Arrays of up to a
few thousand elements run several times faster than NumPy because the
per-operation overhead is 30 to 70 ns instead of NumPy's 350 to 500 ns.
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

Total time per operation in nanoseconds, best of 7 runs, Python 3.14,
NumPy 2.5, one core of an i7-13650HX.

| Operation | lightarray | NumPy |
|---|---|---|
| `a + b` | 83 | 387 |
| `a * 2.0` | 81 | 592 |
| `np.sin(a)` | 106 | 373 |
| `a.sum()` | 126 | 467 |
| `a.std()` | 127 | 6596 |
| `np.sum(a)` | 121 | 1507 |
| `a > 0.5` | 97 | 574 |
| `a[a > 0.5]` | 217 | 887 |
| `(a > 0.2) & (a < 0.8)` | 270 | 1574 |
| `np.where(a > 0.5, a, 0.0)` | 307 | 1424 |
| `i * 2` | 98 | 665 |
| `a[idx]` | 122 | 146 |
| `m[1]` | 77 | 79 |
| `m[:, 1]` | 187 | 86 |
| `np.array(values)` | 198 | 478 |
| `np.arange(10)` | 115 | 405 |
| `np.zeros(10)` | 118 | 178 |
| `a[3] = 2.0` | 44 | 39 |
| `a += 1.0` | 27 | 549 |

`a` and `b` are float64 arrays of 10 elements, `m` is a 10 x 10 float64
array, `i` is `np.arange(10)`, `idx` an int64 array of 3 indices and `values`
a list of 10 floats (`python benchmarks/readme_table.py` regenerates the table).
Reductions include building the `np.float64` result, which is most
of `a.sum()`'s time. Creating a strided view (`m[:, 1]`) is the one
operation slower than NumPy: it allocates the contiguous cache the kernels
work on. Above roughly 10000 elements the two libraries converge; lightarray
is not a large-array library.

## Status

Version 0.1.0-dev. float64, int64 and bool arrays are native, with NumPy's
dtype inference (`np.array([1, 2])` is int64, comparisons give bool arrays,
int and float mix to float64); every other dtype and the long tail of NumPy
functions go through NumPy at NumPy speed plus about 1 µs and come back as
NumPy arrays. Known limit: `isinstance(x, numpy.ndarray)` is False for a
lightarray array and cannot be made True.

Indexing with integers and slices,
`reshape`, `ravel` and `.T` return views that share memory with the array, as
in NumPy (`row = a[0]; row[:] = 0` and `a[:, 1] *= 2` change `a`). The
kernels work on contiguous buffers: a contiguous selection is a window into
the base's buffer at no extra cost, while a strided one (`a[:, 0]`, `a[::2]`,
the transpose of a matrix) is gathered from the base when it is used, which
is cheap for small arrays and one extra pass over the data for large ones.
Views that NumPy returns for delegated operations (`swapaxes`, `split`,
`a[..., 1]`) stay views as well.

Against the official Array API test suite lightarray passes about 1360 of
1374 tests (the suite is randomised); the remaining ones are complex-number
special cases, `finfo` and `fft.fftfreq(dtype=)`, which NumPy itself does
not pass.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin "numpy>=2.1" pytest matplotlib lmfit
maturin develop --release
pytest                          # parity tests against NumPy
cargo test --release --lib      # kernel tests
benchmarks/carray_reference/build.sh && python benchmarks/bench_overhead.py
python benchmarks/gate.py       # performance regression gate
python examples/lmfit_model_fit.py   # lmfit example running on lightarray
python examples/lmfit_internals.py   # lmfit's own internals rebound to lightarray
```

Requires Python 3.10+, NumPy 2.1+, and a stable Rust toolchain.
