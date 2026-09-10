"""Run lmfit itself on lightarray.

`lightarray.patch_module(lmfit)` rebinds the NumPy references inside the
already-imported lmfit package (its `np` aliases, ufuncs and names imported
with `from numpy import ...`) to lightarray. lmfit's own array work then runs
on lightarray as well; SciPy underneath keeps NumPy, and arrays crossing
into the optimiser are converted at that boundary through the zero-copy
buffer protocol.

`lightarray._fallback.calls` counts the operations delegated to NumPy.
"""

import lightarray as np
import lmfit
from lightarray import _fallback


def gaussian(x, amp, cen, wid):
    return (amp / (np.sqrt(2 * np.pi) * wid)) * np.exp(-((x - cen) ** 2) / (2 * wid**2))


np.random.seed(2)
x = np.linspace(-10, 10, 101)
y = gaussian(x, 2.33, 0.21, 1.51) + np.random.normal(scale=0.1, size=x.size)
model = lmfit.Model(gaussian)

for label, enabled in (("lmfit on NumPy internally", False), ("lmfit internals on lightarray", True)):
    np.set_patched(lmfit, enabled)  # the toggle; `with np.patched(lmfit):` or LIGHTARRAY_PATCH=lmfit work too
    before = _fallback.calls
    result = model.fit(y, x=x, amp=5, cen=5, wid=1)
    delegated = _fallback.calls - before
    print(
        f"{label}: amp={result.params['amp'].value:.4f} cen={result.params['cen'].value:.4f} "
        f"wid={result.params['wid'].value:.4f}, {delegated} calls delegated to NumPy over {result.nfev} evaluations"
    )
np.set_patched(lmfit, False)
