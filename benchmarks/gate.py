"""Performance gate: the native hot paths must stay fast.

Runs the key micro-benchmarks and fails (exit 1) when any exceeds its
threshold. Thresholds are 1.5x the measured numbers so that noisy
machines do not trip it; a real regression (like the `__getattr__` slot that
once cost 65 ns per call) does. Run it after every change to src/.
For a check that notices a few percent on your own machine, see perf_check.py.

    python benchmarks/gate.py [--quiet]
"""

import sys
import timeit

import lightarray as la
import numpy as np

THRESHOLDS_NS = {
    # name: (statement, setup, threshold)
    "method call": ("a._noop()", "a = la.ones(10)", 30),
    "a + b (n=10)": ("a + b", "a = la.ones(10); b = la.ones(10)", 105),
    "a * 2.0 (n=10)": ("a * 2.0", "a = la.ones(10)", 105),
    "a.sum() (n=10)": ("a.sum()", "a = la.ones(10)", 200),  # includes building the np.float64 result
    "a.std() (n=10)": ("a.std()", "a = la.ones(10)", 240),
    "la.sin(a) (n=10)": ("la.sin(a)", "a = la.ones(10)", 190),
    "array(list) (n=10)": ("la.array(d)", "d = [float(i) for i in range(10)]", 280),
    "a[2:8:2] (n=10)": ("a[2:8:2]", "a = la.ones(10)", 220),
    "a.sum(axis=0) (10x10)": ("a.sum(axis=0)", "a = la.ones((10, 10))", 450),
    "a + row (4,3)+(3,)": ("a + r", "a = la.ones((4, 3)); r = la.ones(3)", 260),
    "np.asarray(a) (n=3)": ("np.asarray(a)", "a = la.ones(3)", 450),
}


def best_ns(stmt, setup):
    timer = timeit.Timer(stmt, setup=setup, globals={"la": la, "np": np})
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=7, number=number)) / number * 1e9


def main(quiet=False):
    failed = []
    for name, (stmt, setup, limit) in THRESHOLDS_NS.items():
        ns = best_ns(stmt, setup)
        status = "ok" if ns <= limit else "SLOW"
        if ns > limit:
            failed.append(name)
        if not quiet or ns > limit:
            print(f"{name:26s} {ns:7.0f} ns  (limit {limit})  {status}")
    if failed:
        print(f"\n{len(failed)} gate(s) exceeded: {', '.join(failed)}")
        return 1
    print("\nall performance gates pass")
    return 0


if __name__ == "__main__":
    sys.exit(main(quiet="--quiet" in sys.argv))
