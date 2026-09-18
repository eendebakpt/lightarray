"""Regenerate the Speed table of the README: total time per operation in
nanoseconds for lightarray and NumPy, best of 7 runs."""

import timeit

import lightarray
import numpy

OPERATIONS = [
    "a + b",
    "a * 2.0",
    "np.sin(a)",
    "a.sum()",
    "a.std()",
    "a > 0.5",
    "a[a > 0.5]",
    "(a > 0.2) & (a < 0.8)",
    "np.where(a > 0.5, a, 0.0)",
    "i * 2",
    "a[idx]",
    "m[1]",
    "m[:, 1]",
    "np.array(values)",
    "np.arange(10)",
    "a[3] = 2.0",
    "a += 1.0",
]


def namespace(np):
    values = [0.1 * k for k in range(10)]
    return {
        "np": np,
        "values": values,
        "a": np.array(values),
        "b": np.array(values) + 1.0,
        "i": np.arange(10),
        "idx": np.array([1, 4, 7]),
        "m": np.zeros((10, 10)),
    }


def best(stmt, scope, number=200_000, repeat=7):
    return min(timeit.repeat(stmt, globals=scope, number=number, repeat=repeat)) / number * 1e9


if __name__ == "__main__":
    print("| Operation | lightarray | NumPy |")
    print("|---|---|---|")
    for stmt in OPERATIONS:
        fast, ref = best(stmt, namespace(lightarray)), best(stmt, namespace(numpy))
        print(f"| `{stmt}` | {fast:.0f} | {ref:.0f} |")
