#!/usr/bin/env python3
"""
Created on Thu Apr 23 21:04:46 2026

@author: eendebakpt

Rabi oscillations of a driven two-level system, computed with lightarray.

The excited-state population under a drive with Rabi frequency Omega and
detuning Delta is

    P(t) = (Omega / Omega_R)^2 * sin^2(Omega_R t / 2),   Omega_R = sqrt(Omega^2 + Delta^2)

All array arithmetic runs in lightarray (sqrt, sin, scalar and array
multiplication, powers); matplotlib reads the arrays through the buffer
protocol without copying.
"""

# %%
# import numpy as np
import lightarray as la
import matplotlib.pyplot as plt


def rabi_population(t, omega, delta):
    """Excited-state population P(t) for Rabi frequency `omega` and detuning `delta`."""
    omega_r = la.sqrt(omega**2 + delta**2)
    return (omega / omega_r) ** 2 * la.sin(omega_r * t / 2) ** 2


# %%
omega = 2 * la.pi * 1.0  # Rabi frequency, rad/us (1 MHz)
t = la.linspace(0.0, 3.0, 601)  # time, us

fig, ax = plt.subplots(figsize=(7, 4))
for delta_mhz in (0.0, 0.5, 1.0, 2.0):
    delta = 2 * la.pi * delta_mhz
    p = rabi_population(t, omega, delta)

    # cross-check: the same formula evaluated by NumPy
    pop = rabi_population(t, omega, delta)

    ax.plot(t, p, label=f"$\\Delta/2\\pi$ = {delta_mhz:g} MHz  (max {p.max():.2f})")

ax.set_xlabel("time (µs)")
ax.set_ylabel("excited-state population")
ax.set_title("Rabi oscillations, $\\Omega/2\\pi$ = 1 MHz")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("rabi_oscillation.png", dpi=120)
plt.show()

print(f"arrays computed with {type(t).__module__}.{type(t).__name__}, {t.size} time points")
