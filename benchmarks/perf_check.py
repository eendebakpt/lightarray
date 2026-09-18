"""Local performance regression check: no feature may cost the hot paths anything.

`gate.py` has loose absolute limits so that shared CI runners pass; a 30%
regression slips through it. This script compares against a baseline recorded
on *this* machine and fails on a few percent:

    python benchmarks/perf_check.py --save     # record the baseline (on a commit you trust)
    python benchmarks/perf_check.py            # compare; exit 1 on any regression
    python benchmarks/perf_check.py -k sum     # only the operations whose name contains "sum"
    python benchmarks/perf_check.py --numpy    # also show NumPy's time for the same statement

Workflow: save a baseline before starting on a feature, rebuild with
`maturin develop --release`, run the check. After an intended speed-up,
`--save` again.

How it stays quiet on a noisy machine:
- timings are taken in several fresh processes (`--processes`, 3), each pinned
  to one core (`--cpu`, Linux); the minimum over all processes and rounds
  counts. Separate processes matter: heap and address-space layout differ per
  process and shift allocation-heavy operations by up to 8%, which no number
  of rounds inside one process averages out. Rounds are interleaved across
  the operations;
- a pure-Python reference statement is timed alongside; when the machine as a
  whole is slower or faster than at baseline time, timings are judged after
  scaling by that ratio as well as raw, and only a regression in both counts;
  when the reference is more than 5% off (`--max-drift`) the machine is busy
  or throttled and the script refuses to judge (exit code 2);
- an operation that looks regressed is re-measured in further fresh
  processes before it is reported;
- a regression must exceed both the relative tolerance (`--tolerance`, 4%)
  and an absolute one (`--min-ns`, 2.5 ns).

What it cannot remove is the code-layout effect between two *builds*: adding
any code, even an unused method, moves the hot functions and changes inlining,
which shifts single operations by 3-8% in both directions. `.cargo/config.toml`
aligns all functions to 64 bytes to dampen this, and the summary line reports
the overall (geometric mean) change, so a layout shift (overall about 0, as
many operations faster as slower) can be told from a real regression (the
operations touched by the change slower, nothing gained).

The baseline is machine-specific and therefore not committed
(`benchmarks/.perf_baseline.json`).
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import timeit
from pathlib import Path

import lightarray
import numpy

BASELINE = Path(__file__).with_name(".perf_baseline.json")

# The reference: interpreter-only work, independent of lightarray.
REFERENCE = ("x * y + z", "x = 1.5; y = 2.5; z = 0.5")

# (group, statement, setup). `np` is lightarray in the main column (NumPy in the
# --numpy column), `numpy` is always the real NumPy. Names bound in COMMON are available.
COMMON = """
values = [0.1 * k for k in range(10)]
a = np.array(values); b = np.array(values) + 1.0
i = np.arange(10); idx = np.array([1, 4, 7])
m = np.zeros((10, 10)); row = np.ones(10)
mask = a > 0.5
col = m[:, 1]
A = np.linspace(0.0, 1.0, 1000); B = A + 1.0
"""

OPERATIONS = [
    # call overhead
    ("call", "len(a)", ""),
    ("call", "a.shape", ""),
    ("call", "a.dtype", ""),
    # operators
    ("operators", "a + b", ""),
    ("operators", "a - b", ""),
    ("operators", "a * 2.0", ""),
    ("operators", "a / b", ""),
    ("operators", "a ** 2", ""),
    ("operators", "-a", ""),
    ("operators", "m + row", ""),
    ("operators", "i * 2", ""),
    ("operators", "i + i", ""),
    ("operators", "a += 1.0", "a = np.array(values)"),
    ("operators", "a *= b", "a = np.array(values); b = np.ones(10)"),
    # element-wise functions
    ("functions", "np.sin(a)", ""),
    ("functions", "np.sqrt(a)", ""),
    ("functions", "np.exp(a)", ""),
    ("functions", "np.abs(a)", ""),
    ("functions", "np.add(a, b)", ""),
    ("functions", "np.maximum(a, 0.5)", ""),
    ("functions", "np.isnan(a)", ""),
    ("functions", "np.sin(1.5)", ""),
    # reductions
    ("reductions", "a.sum()", ""),
    ("reductions", "a.mean()", ""),
    ("reductions", "a.std()", ""),
    ("reductions", "a.max()", ""),
    ("reductions", "a.argmax()", ""),
    ("reductions", "a.dot(b)", ""),
    ("reductions", "np.sum(a)", ""),
    ("reductions", "np.mean(a)", ""),
    ("reductions", "np.any(mask)", ""),
    ("reductions", "m.sum(axis=0)", ""),
    ("reductions", "i.sum()", ""),
    # comparisons and masks
    ("masks", "a > 0.5", ""),
    ("masks", "(a > 0.2) & (a < 0.8)", ""),
    ("masks", "a[mask]", ""),
    ("masks", "np.where(mask, a, 0.0)", ""),
    ("masks", "np.isclose(a, b)", ""),
    # creation and conversion
    ("creation", "np.array(values)", ""),
    ("creation", "np.asarray(values)", ""),
    ("creation", "np.asarray(a)", ""),
    ("creation", "np.zeros(10)", ""),
    ("creation", "np.zeros((3, 3))", ""),
    ("creation", "np.ones(10)", ""),
    ("creation", "np.arange(10)", ""),
    ("creation", "np.linspace(0.0, 1.0, 10)", ""),
    ("creation", "np.concatenate([a, b])", ""),
    ("creation", "a.copy()", ""),
    # indexing and views
    ("indexing", "a[3]", ""),
    ("indexing", "m[1, 2]", ""),
    ("indexing", "m[1]", ""),
    ("indexing", "a[2:8]", ""),
    ("indexing", "a[2:8:2]", ""),
    ("indexing", "m[:, 1]", ""),
    ("indexing", "m.T", ""),
    ("indexing", "a.reshape(2, 5)", ""),
    ("indexing", "a[idx]", ""),
    ("indexing", "col + 1.0", ""),
    ("indexing", "col.sum()", ""),
    # assignment
    ("assignment", "a[3] = 2.0", ""),
    ("assignment", "m[1, 2] = 2.0", ""),
    ("assignment", "col[3] = 2.0", ""),
    ("assignment", "a[2:5] = 0.0", ""),
    # interoperability with the real NumPy
    ("interop", "numpy.asarray(a)", ""),
    ("interop", "numpy.add(a, b)", ""),
    # kernel throughput (n = 1000): catches lost vectorisation, not call overhead
    ("n=1000", "A + B", ""),
    ("n=1000", "A * 2.0", ""),
    ("n=1000", "A.sum()", ""),
    ("n=1000", "np.sin(A)", ""),
    ("n=1000", "A > 0.5", ""),
]


def make_timer(statement, setup, np_module):
    scope = {"np": np_module, "numpy": numpy}
    exec(COMMON, scope)
    return timeit.Timer(statement, setup=setup or "pass", globals=scope)


def calibrate(timer, target_seconds):
    """Loops per measurement so that one measurement takes about `target_seconds`."""
    number = 200
    while True:
        elapsed = timer.timeit(number)
        if elapsed >= target_seconds / 4:
            return max(1, int(number * target_seconds / elapsed))
        number *= 4


class Bench:
    def __init__(self, name, group, statement, setup, np_module, target_seconds):
        self.name, self.group = name, group
        self.timer = make_timer(statement, setup, np_module)
        self.number = calibrate(self.timer, target_seconds)
        self.best = float("inf")

    def measure(self):
        ns = self.timer.timeit(self.number) / self.number * 1e9
        self.best = min(self.best, ns)
        return ns


def run_rounds(benches, rounds):
    """Round-robin, so slow drift of the machine hits every operation alike."""
    for _ in range(rounds):
        for bench in benches:
            bench.measure()


def pin_to_cpu(cpu):
    if cpu is None or not hasattr(os, "sched_setaffinity"):
        return None
    try:
        os.sched_setaffinity(0, {cpu})
        return cpu
    except OSError:
        return None


def git_commit():
    try:
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment(cpu):
    return {
        "lightarray": lightarray.__version__,
        "commit": git_commit(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "machine": platform.node(),
        "cpu": cpu,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }


def worker(request):
    """Measure in this process; print {"reference": ns, "timings": {...}, "numpy": {...}} as JSON."""
    cpu = pin_to_cpu(request["cpu"])
    target = 0.02  # seconds per measurement
    chosen = [op for op in OPERATIONS if op[1] in set(request["statements"])]
    reference = Bench("reference", "", *REFERENCE, lightarray, target)
    benches = [Bench(s, g, s, setup, lightarray, target) for g, s, setup in chosen]
    run_rounds([reference, *benches], request["rounds"])
    result = {"cpu": cpu, "reference": reference.best, "timings": {b.name: b.best for b in benches}, "numpy": {}}
    if request["numpy"]:
        others = [Bench(s, g, s, setup, numpy, target) for g, s, setup in chosen]
        run_rounds(others, 3)
        result["numpy"] = {b.name: b.best for b in others}
    print(json.dumps(result))


class Measurements:
    """Minimum per operation over fresh worker processes."""

    def __init__(self, cpu, rounds):
        self.cpu, self.rounds = cpu, rounds
        self.reference = float("inf")
        self.timings, self.numpy = {}, {}

    def run(self, statements, processes, with_numpy=False):
        for k in range(processes):
            request = {"cpu": self.cpu, "rounds": self.rounds, "statements": statements, "numpy": with_numpy and k == 0}
            out = subprocess.run([sys.executable, __file__, "--worker", json.dumps(request)], capture_output=True, text=True)
            if out.returncode != 0:
                sys.exit(f"measurement process failed:\n{out.stderr}")
            result = json.loads(out.stdout.strip().splitlines()[-1])
            self.cpu_used = result["cpu"]
            self.reference = min(self.reference, result["reference"])
            for name, ns in result["timings"].items():
                self.timings[name] = min(self.timings.get(name, float("inf")), ns)
            self.numpy.update(result["numpy"])


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return worker(json.loads(sys.argv[2]))

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", action="store_true", help="record the current timings as the baseline")
    parser.add_argument("--baseline", type=Path, default=BASELINE, help="baseline file (default: %(default)s)")
    parser.add_argument("-k", dest="select", default="", help="only operations whose statement or group contains this text")
    parser.add_argument("--tolerance", type=float, default=0.04, help="relative slowdown that counts as a regression (default 0.04)")
    parser.add_argument("--min-ns", type=float, default=2.5, help="absolute slowdown a regression must also exceed (default 2.5 ns)")
    parser.add_argument("--processes", type=int, default=3, help="fresh processes to measure in; the minimum counts (default 3)")
    parser.add_argument("--rounds", type=int, default=5, help="measurements per operation and process (default 5)")
    parser.add_argument("--cpu", type=int, default=2, help="core to pin the measurements to on Linux (default 2); -1 to not pin")
    parser.add_argument(
        "--max-drift",
        type=float,
        default=0.05,
        help="give up when the interpreter reference differs more than this from the baseline's (default 0.05)",
    )
    parser.add_argument("--numpy", action="store_true", help="also time the statements with NumPy, for orientation")
    args = parser.parse_args()

    selected = [(g, s) for g, s, _ in OPERATIONS if args.select in s or args.select in g]
    if not selected:
        sys.exit(f"no operation matches {args.select!r}")
    statements = [s for _, s in selected]
    measured = Measurements(None if args.cpu < 0 else args.cpu, args.rounds)
    measured.run(statements, args.processes, with_numpy=args.numpy)
    cpu = measured.cpu_used

    if args.save:
        if args.select:
            sys.exit("--save records every operation; drop -k")
        data = {"environment": environment(cpu), "reference_ns": measured.reference, "timings_ns": measured.timings}
        args.baseline.write_text(json.dumps(data, indent=1) + "\n")
        for group, statement in selected:
            print(f"{group:11s} {statement:28s} {measured.timings[statement]:8.1f} ns")
        print(f"\nbaseline with {len(selected)} operations saved to {args.baseline}")
        return 0

    if not args.baseline.exists():
        sys.exit(f"no baseline at {args.baseline}: run with --save on a commit you trust first")
    base = json.loads(args.baseline.read_text())
    env, now = base["environment"], environment(cpu)
    for key in ("python", "numpy", "machine", "cpu"):
        if env.get(key) != now[key]:
            print(f"warning: baseline was recorded with {key}={env.get(key)}, this run has {key}={now[key]}")

    def verdict(statement):
        """(regressed, improved, raw change) against the baseline, machine state taken into account."""
        old = base["timings_ns"].get(statement)
        if old is None:
            return False, False, None
        best = measured.timings[statement]
        scale = measured.reference / base["reference_ns"]  # > 1: the machine is slower than at baseline time
        raw, scaled = best - old, best / scale - old
        slower, faster = min(raw, scaled), max(raw, scaled)
        regressed = slower > args.min_ns and slower > args.tolerance * old
        improved = faster < -args.min_ns and faster < -args.tolerance * old
        return regressed, improved, raw / old

    # A regression has to survive more fresh processes.
    for _ in range(2):
        suspects = [s for s in statements if verdict(s)[0]]
        if not suspects:
            break
        measured.run(suspects, 2)

    scale = measured.reference / base["reference_ns"]
    if abs(scale - 1) > args.max_drift:
        print(
            f"the interpreter reference is {scale - 1:+.0%} off the baseline's: the machine is busy, throttled or on another\n"
            f"kind of core (load average {os.getloadavg()[0]:.1f}, pinned to cpu {cpu}). Timings taken now mean nothing; try again\n"
            "when it is quiet, or pick another core with --cpu."
        )
        return 2
    print(f"baseline: {env.get('commit')} ({env.get('date')}); now: {now['commit']}; interpreter reference {scale - 1:+.1%}\n")
    print(f"{'':11s} {'operation':28s} {'baseline':>9s} {'now':>9s} {'change':>8s}" + (f" {'NumPy':>9s}" if args.numpy else ""))
    regressions, improvements, new = [], [], []
    previous = None
    for group, statement in selected:
        regressed, improved, change = verdict(statement)
        label, previous = (group if group != previous else ""), group
        best = measured.timings[statement]
        extra = f" {measured.numpy[statement]:9.1f}" if args.numpy else ""
        if change is None:
            new.append(statement)
            print(f"{label:11s} {statement:28s} {'-':>9s} {best:9.1f} {'new':>8s}{extra}")
            continue
        mark = "  REGRESSION" if regressed else ("  faster" if improved else "")
        print(f"{label:11s} {statement:28s} {base['timings_ns'][statement]:9.1f} {best:9.1f} {change:+8.1%}{extra}{mark}")
        if regressed:
            regressions.append(statement)
        if improved:
            improvements.append(statement)

    ratios = [measured.timings[s] / base["timings_ns"][s] for s in statements if s in base["timings_ns"]]
    if ratios:
        overall = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) - 1
        slower, faster = sum(r > 1.03 for r in ratios), sum(r < 0.97 for r in ratios)
        print(f"\noverall (geometric mean) {overall:+.2%}; {slower} operation(s) more than 3% slower, {faster} more than 3% faster")
        if regressions and abs(overall) < 0.01 and faster >= slower:
            print(
                "note: the build as a whole is not slower and as many operations gained as lost. That is the\n"
                "signature of a code-layout shift (any added code moves the hot functions), not of the change\n"
                "itself; check by building the same change under another name, or pin the inlining of the\n"
                "affected path (#[inline(never)] / #[inline(always)])."
            )
    print()
    if new:
        print(f"{len(new)} operation(s) not in the baseline (run --save to include them): {', '.join(new)}")
    if improvements:
        print(f"{len(improvements)} faster than the baseline: {', '.join(improvements)} (run --save to lock that in)")
    if regressions:
        print(f"{len(regressions)} REGRESSION(S): {', '.join(regressions)}")
        return 1
    print(f"no regressions in {len(selected)} operations (tolerance {args.tolerance:.0%} and {args.min_ns} ns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
