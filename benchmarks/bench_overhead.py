"""Benchmark the real lightarray.PyArray (PyO3) against a minimal C-API array
(build it first with carray_reference/build.sh)
and NumPy, on the small-array workloads the project plan targets."""

import os
import sys
import timeit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "carray_reference"))
import carray
import lightarray as la
import numpy as np


def best_ns(stmt, glb, number=None):
    """Best-of-7 per-call time in ns."""
    t = timeit.Timer(stmt, globals=glb)
    if number is None:
        number, _ = t.autorange()
    reps = t.repeat(repeat=7, number=number)
    return min(reps) / number * 1e9


def row(label, *vals):
    print(f"{label:34s}" + "".join(f"{v:>12s}" for v in vals))


print(f"Python {sys.version.split()[0]}, numpy {np.__version__}\n")

# --- 1. raw call overhead -------------------------------------------------
print("1. Raw call overhead (ns/call, lower is better)")


def pynoop():
    pass


class PyObj:
    def _noop(self):
        pass


pyo = PyObj()
la1 = la.array([1.0])
ca1 = carray.make_array([1.0])
np1 = np.array([1.0])
g = dict(globals())
row("", "python", "C-API", "PyO3", "numpy")
row("module func no-op", f"{best_ns('pynoop()', g):.0f}", f"{best_ns('carray._noop()', g):.0f}", f"{best_ns('la._noop()', g):.0f}", "-")
row("method no-op", f"{best_ns('pyo._noop()', g):.0f}", f"{best_ns('ca1._noop()', g):.0f}", f"{best_ns('la1._noop()', g):.0f}", "-")
row("attr .size / .shape", "-", f"{best_ns('ca1.size', g):.0f}", f"{best_ns('la1.size', g):.0f}", f"{best_ns('np1.size', g):.0f}")
row("getitem a[0]", "-", f"{best_ns('ca1[0]', g):.0f}", f"{best_ns('la1[0]', g):.0f}", f"{best_ns('np1[0]', g):.0f}")
print()

# --- 2. a + b and a.sum() vs size ----------------------------------------
print("2. Element-wise add a+b (ns/op)")
row("n", "C-API", "PyO3", "numpy", "PyO3/C", "PyO3/np")
sizes = [1, 3, 10, 100, 1000, 10000]
add = {}
for n in sizes:
    data = [float(i) for i in range(n)]
    g.update(
        a_la=la.array(data),
        b_la=la.array(data),
        a_c=carray.make_array(data),
        b_c=carray.make_array(data),
        a_np=np.array(data),
        b_np=np.array(data),
    )
    c = best_ns("a_c + b_c", g)
    p = best_ns("a_la + b_la", g)
    q = best_ns("a_np + b_np", g)
    add[n] = (c, p, q)
    row(str(n), f"{c:.0f}", f"{p:.0f}", f"{q:.0f}", f"{p / c:.2f}", f"{p / q:.2f}")
print()
print("3. Full reduction a.sum() (ns/op)")
row("n", "C-API", "PyO3", "numpy", "PyO3/C", "PyO3/np")
for n in sizes:
    data = [float(i) for i in range(n)]
    g.update(a_la=la.array(data), a_c=carray.make_array(data), a_np=np.array(data))
    c = best_ns("a_c.sum()", g)
    p = best_ns("a_la.sum()", g)
    q = best_ns("a_np.sum()", g)
    row(str(n), f"{c:.0f}", f"{p:.0f}", f"{q:.0f}", f"{p / c:.2f}", f"{p / q:.2f}")
print()
print("4. Creation from a Python list (ns/op)")
row("n", "C-API", "PyO3", "numpy")
for n in [3, 10, 100, 1000]:
    g["data"] = [float(i) for i in range(n)]
    row(
        str(n), f"{best_ns('carray.make_array(data)', g):.0f}", f"{best_ns('la.array(data)', g):.0f}", f"{best_ns('np.array(data)', g):.0f}"
    )
print()
print("5. Chained expression (a+b)+(a+b) then .sum(), n=10 (ns/op)")
data = [float(i) for i in range(10)]
g.update(
    a_la=la.array(data),
    b_la=la.array(data),
    a_c=carray.make_array(data),
    b_c=carray.make_array(data),
    a_np=np.array(data),
    b_np=np.array(data),
)
row("", "C-API", "PyO3", "numpy")
row(
    "((a+b)+(a+b)).sum()",
    f"{best_ns('((a_c+b_c)+(a_c+b_c)).sum()', g):.0f}",
    f"{best_ns('((a_la+b_la)+(a_la+b_la)).sum()', g):.0f}",
    f"{best_ns('((a_np+b_np)+(a_np+b_np)).sum()', g):.0f}",
)
