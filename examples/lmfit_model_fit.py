"""lmfit's "Fitting with Model" documentation example, unchanged except for
the import: the array work (model evaluation, noise, residuals) runs in
lightarray while lmfit and SciPy drive the optimisation.

Original: https://lmfit.github.io/lmfit-py/model.html (doc_model_gaussian.py)
"""

import lightarray as np  # was: import numpy as np
from lmfit import Model


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2 * np.pi) * wid)) * np.exp(-((x - cen) ** 2) / (2 * wid**2))


x = np.linspace(-10, 10, 101)
y = gaussian(x, 2.33, 0.21, 1.51) + np.random.normal(scale=0.1, size=x.size)

gmodel = Model(gaussian)
result = gmodel.fit(y, x=x, amp=5, cen=5, wid=1)

print(result.fit_report())

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.plot(x, y, "o")
    plt.plot(x, result.init_fit, "--", label="initial fit")
    plt.plot(x, result.best_fit, "-", label="best fit")
    plt.legend()
    plt.savefig("lmfit_model_fit.png", dpi=100)
    print(
        f"x is a {type(x).__module__}.{type(x).__name__}; best_fit is a {type(result.best_fit).__module__}.{type(result.best_fit).__name__}"
    )
