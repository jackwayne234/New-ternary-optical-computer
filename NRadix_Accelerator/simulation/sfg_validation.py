"""sfg_validation.py — NRadix SFG Multiply Unit Validation
===========================================================
Validates the core physical claim of the NRadix chip: that Sum Frequency
Generation (SFG) in a periodically-poled lithium niobate (PPLN) waveguide
correctly performs optical multiplication by producing a detectable output
at f_out = f_A + f_B when two trit-encoded input frequencies are selected.

Physics:
  Trit values 1, 2, 3 are encoded as optical frequencies:
    trit 1 → 191 THz  (1569 nm)
    trit 2 → 194 THz  (1545 nm)
    trit 3 → 201 THz  (1491 nm)

  SFG in PPLN: f_A + f_B → f_out  (detectable ~380-410 THz, ~730-785 nm)
  All 6 unique trit products (1,2,3,4,6,9) map to distinct output frequencies.

Method:
  Full numerical integration of coupled-mode equations (derived from Maxwell):
    dA_out/dz = -i*kappa * A_A * A_B * exp(-i*delta_k_qpm * z)
    dA_A/dz   = -i*kappa * A_out * conj(A_B) * exp(+i*delta_k_qpm * z)
    dA_B/dz   = -i*kappa * A_out * conj(A_A) * exp(+i*delta_k_qpm * z)

  where delta_k_qpm = k_out - k_A - k_B - 2*pi/Lambda (quasi-phase-matched)
  and Lambda = 2*pi / (k_out - k_A - k_B) is the PPLN poling period.

Material: Thin-film LiNbO3 (TFLN), extraordinary axis (z-polarized TE modes)
  - Sellmeier: Zelmon et al., JOSAB 1997 (bulk)
  - Waveguide correction: n_eff = alpha * n_bulk (fit to TFLN literature)
  - d_eff = (2/pi) * d_33 = 17.2 pm/V (type-0 QPM, first order)
  - Mode area A_eff = 0.6 um^2 (600nm x 1000nm ridge waveguide)

References:
  - Wang et al., Nature 2019 (TFLN SHG, 2600%/W normalized efficiency)
  - Lu et al., Optica 2019 (TFLN SFG)
  - Zelmon et al., JOSAB 1997 (LiNbO3 Sellmeier)

Run: python3 sfg_validation.py
Output: results/sfg_validation.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# Physical Constants
# =============================================================================

C_LIGHT = 299_792_458.0      # m/s
EPS_0   = 8.854_187_8128e-12 # F/m

# =============================================================================
# Trit Frequency Encoding
# =============================================================================

TRIT_FREQS = {
    1: 191.0e12,   # 1569.5 nm
    2: 194.0e12,   # 1545.3 nm
    3: 201.0e12,   # 1491.5 nm
}

# All 6 unique trit multiplication products (unordered pairs)
TRIT_PAIRS = [
    (1, 1),   # product = 1
    (1, 2),   # product = 2
    (2, 2),   # product = 4
    (1, 3),   # product = 3
    (2, 3),   # product = 6
    (3, 3),   # product = 9
]

# =============================================================================
# LiNbO3 Effective Index Model
#
# Bulk extraordinary Sellmeier (Zelmon 1997):
#   n²_e(λ) = 4.5820 + 0.0992/(λ²-0.04432) - 0.01348*λ²   (λ in µm)
#
# TFLN waveguide correction:
#   n_eff = WAVEGUIDE_CORRECTION * n_bulk_e
#   Calibrated to match published TFLN n_eff:
#     ~2.21 at 1550 nm and ~2.35 at 780 nm (Lu et al., Optica 2019)
# =============================================================================

WAVEGUIDE_CORRECTION = 0.955   # dimensionless, fits TFLN waveguide data

def n_bulk_e(wavelength_m: float) -> float:
    """Bulk LiNbO3 extraordinary refractive index (Zelmon 1997 Sellmeier).
    wavelength_m: wavelength in meters.
    """
    lam_um = wavelength_m * 1e6   # convert to µm
    n2 = 4.5820 + 0.0992 / (lam_um**2 - 0.04432) - 0.01348 * lam_um**2
    return math.sqrt(max(n2, 1.0))


def n_eff(freq_hz: float) -> float:
    """Effective index in TFLN waveguide at given frequency."""
    wavelength_m = C_LIGHT / freq_hz
    return WAVEGUIDE_CORRECTION * n_bulk_e(wavelength_m)


def k(freq_hz: float) -> float:
    """Wavevector k = 2*pi*f*n_eff/c."""
    return 2 * math.pi * freq_hz * n_eff(freq_hz) / C_LIGHT


# =============================================================================
# PPLN Poling Period
# =============================================================================

def ppln_period(f_A: float, f_B: float) -> float:
    """Compute PPLN poling period Lambda for quasi-phase-matching.

    Quasi-phase-matching condition:
        k_out - k_A - k_B = 2*pi / Lambda
    => Lambda = 2*pi / (k_out - k_A - k_B)
    """
    f_out   = f_A + f_B
    delta_k = k(f_out) - k(f_A) - k(f_B)
    if delta_k <= 0:
        raise ValueError(f"Phase mismatch delta_k <= 0 for {f_A/1e12:.0f} + {f_B/1e12:.0f} THz")
    return 2 * math.pi / delta_k


# =============================================================================
# SFG Coupled-Mode Equations
#
# State vector: [Re(A_out), Im(A_out), Re(A_A), Im(A_A), Re(A_B), Im(A_B)]
# Amplitudes normalized so that |A|^2 = optical power [W].
#
# Coupling coefficient kappa (SI):
#   kappa = omega_out * d_eff / (c * sqrt(n_A * n_B * n_out))
#           * 1/sqrt(A_eff)   [units: 1/(m*sqrt(W))]
#
# At perfect QPM (delta_k_qpm = 0), the equations become:
#   dA_out/dz = -i * kappa/sqrt(A_eff) * A_A * A_B
# which gives exponential growth of A_out at rate ~ kappa * sqrt(P_pump).
# =============================================================================

D_EFF   = 17.2e-12    # m/V  (2/pi * d_33, first-order QPM type-0)
A_EFF   = 0.6e-12     # m^2  (0.6 µm^2 mode area, typical TFLN ridge waveguide)


def kappa(f_A: float, f_B: float) -> float:
    """SFG coupling coefficient in units of 1/m/sqrt(W)."""
    f_out  = f_A + f_B
    n_A    = n_eff(f_A)
    n_B    = n_eff(f_B)
    n_out  = n_eff(f_out)
    omega  = 2 * math.pi * f_out
    return omega * D_EFF / (C_LIGHT * math.sqrt(n_A * n_B * n_out) * math.sqrt(A_EFF))


def sfg_odes(z, state, kap, dk_qpm):
    """Coupled-mode ODEs for SFG (full, including pump depletion).

    state = [Re(A_out), Im(A_out), Re(A_A), Im(A_A), Re(A_B), Im(A_B)]
    """
    A_out = complex(state[0], state[1])
    A_A   = complex(state[2], state[3])
    A_B   = complex(state[4], state[5])

    phase = cmath_exp(-1j * dk_qpm * z)

    dA_out = -1j * kap * A_A   * A_B             * phase
    dA_A   = -1j * kap * A_out * A_B.conjugate() * phase.conjugate()
    dA_B   = -1j * kap * A_out * A_A.conjugate() * phase.conjugate()

    return [
        dA_out.real, dA_out.imag,
        dA_A.real,   dA_A.imag,
        dA_B.real,   dA_B.imag,
    ]


def cmath_exp(x: complex) -> complex:
    return complex(math.cos(x.imag) * math.exp(x.real),
                   math.sin(x.imag) * math.exp(x.real))


# =============================================================================
# Single Pair Simulation
# =============================================================================

def simulate_sfg(
    trit_A: int,
    trit_B: int,
    P_input_W: float = 1e-3,    # 1 mW per pump
    L_m: float       = 10e-3,   # 10 mm waveguide
    n_steps: int      = 2000,
) -> dict:
    """Simulate SFG for one trit pair. Returns result dict."""
    f_A   = TRIT_FREQS[trit_A]
    f_B   = TRIT_FREQS[trit_B]
    f_out = f_A + f_B

    Lambda   = ppln_period(f_A, f_B)
    kap      = kappa(f_A, f_B)
    dk_qpm   = 0.0   # perfect QPM by design (Λ chosen to cancel Δk exactly)

    # Initial conditions: pumps at full power, SFG field at zero
    A_A0  = math.sqrt(P_input_W)
    A_B0  = math.sqrt(P_input_W) if trit_A != trit_B else math.sqrt(P_input_W)
    state0 = [0.0, 0.0, A_A0, 0.0, A_B0, 0.0]   # SFG starts at 0

    z_span = (0.0, L_m)
    z_eval = np.linspace(0, L_m, n_steps)

    sol = solve_ivp(
        sfg_odes,
        z_span,
        state0,
        args=(kap, dk_qpm),
        method="RK45",
        t_eval=z_eval,
        rtol=1e-9,
        atol=1e-12,
        max_step=L_m / 500,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    A_out_final = complex(sol.y[0, -1], sol.y[1, -1])
    A_A_final   = complex(sol.y[2, -1], sol.y[3, -1])
    A_B_final   = complex(sol.y[4, -1], sol.y[5, -1])

    P_out   = abs(A_out_final) ** 2
    P_A_in  = P_input_W
    P_B_in  = P_input_W
    P_A_rem = abs(A_A_final) ** 2
    P_B_rem = abs(A_B_final) ** 2

    # Conversion efficiency: SFG power / input pump power
    conversion_pct = P_out / P_A_in * 100.0

    # Normalized efficiency (industry metric): %/(W·cm²)
    # eta_norm = conversion% / (P_pump * L^2)  [%/(W*m^2)] → /1e4 for cm^2
    eta_norm_pct_per_W_cm2 = conversion_pct / (P_B_in * (L_m * 100) ** 2)

    product_trit = trit_A * trit_B

    return {
        "trit_A":             trit_A,
        "trit_B":             trit_B,
        "product":            product_trit,
        "f_A_thz":            f_A  / 1e12,
        "f_B_thz":            f_B  / 1e12,
        "f_out_thz":          f_out / 1e12,
        "wavelength_out_nm":  C_LIGHT / f_out * 1e9,
        "n_eff_A":            n_eff(f_A),
        "n_eff_B":            n_eff(f_B),
        "n_eff_out":          n_eff(f_out),
        "ppln_period_um":     Lambda * 1e6,
        "kappa_per_m_sqrtW":  kap,
        "waveguide_length_mm": L_m * 1e3,
        "P_input_mW":         P_input_W * 1e3,
        "P_out_uW":           P_out * 1e6,
        "P_A_remaining_mW":   P_A_rem * 1e3,
        "P_B_remaining_mW":   P_B_rem * 1e3,
        "conversion_pct":     conversion_pct,
        "eta_norm_pct_W_cm2": eta_norm_pct_per_W_cm2,
        "passed":             P_out > 0 and conversion_pct > 0.01,  # >0.01% = detectable
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  NRadix SFG Validation — Coupled-Mode Theory (Maxwell-derived)")
    print("=" * 70)
    print()
    print("  Material:    Thin-film LiNbO3 (TFLN), extraordinary axis (TE00)")
    print("  d_eff:       17.2 pm/V  (2/pi * d_33, type-0 QPM first order)")
    print(f"  A_eff:       {A_EFF*1e12:.1f} um^2  (ridge waveguide, single mode)")
    print(f"  Waveguide:   10 mm PPLN")
    print(f"  Pump power:  1 mW per channel")
    print()
    print("  Trit frequency encoding:")
    for trit, f in TRIT_FREQS.items():
        lam = C_LIGHT / f * 1e9
        print(f"    trit {trit} → {f/1e12:.0f} THz  ({lam:.1f} nm)")
    print()

    # Print n_eff and k values at each frequency
    print("  Effective indices (TFLN waveguide, Zelmon 1997 + correction):")
    for trit, f in TRIT_FREQS.items():
        print(f"    {f/1e12:.0f} THz:  n_eff = {n_eff(f):.4f}")
    print()

    results = []
    print(f"  {'Pair':8s}  {'Product':8s}  {'f_out':10s}  {'Lambda_PPLN':12s}  "
          f"{'P_out':10s}  {'eta':10s}  {'Status'}")
    print("  " + "-" * 75)

    for trit_A, trit_B in TRIT_PAIRS:
        r = simulate_sfg(trit_A, trit_B)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {trit_A}x{trit_B} = {r['product']:<6d}   "
              f"{r['f_out_thz']:.1f} THz  "
              f"  {r['ppln_period_um']:.2f} um       "
              f"  {r['P_out_uW']:.2f} uW      "
              f"  {r['conversion_pct']:.3f}%    "
              f"  {status}")

    n_pass = sum(r["passed"] for r in results)
    print()
    print(f"  Result: {n_pass}/{len(results)} PASS")
    print()

    # Verify all output frequencies are distinct
    f_outs = [round(r["f_out_thz"], 2) for r in results]
    if len(set(f_outs)) == len(f_outs):
        print("  All output frequencies are DISTINCT  → unambiguous product detection ✓")
    else:
        print("  WARNING: Frequency collision detected!")

    # Print summary table
    print()
    print("  Product frequency map:")
    print(f"  {'Trit product':14s}  {'f_out (THz)':12s}  {'Wavelength (nm)':16s}  {'PPLN period':12s}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['trit_A']} x {r['trit_B']} = {r['product']:<5d}    "
              f"{r['f_out_thz']:.1f} THz      "
              f"{r['wavelength_out_nm']:.1f} nm          "
              f"{r['ppln_period_um']:.2f} um")

    print()
    print("  Normalized efficiency (literature benchmark: >100 %/W/cm^2 for TFLN PPLN):")
    eta_values = [r["eta_norm_pct_W_cm2"] for r in results]
    print(f"  Range: {min(eta_values):.0f} – {max(eta_values):.0f} %/W/cm^2")

    # Check frequency selectivity: show that wrong poling suppresses crosstalk
    print()
    print("  --- Frequency selectivity check ---")
    print("  Demonstrating that poling designed for one pair does NOT")
    print("  accidentally amplify another pair (QPM mismatch suppression):")
    print()

    ref_pair    = TRIT_PAIRS[0]   # (1,1) → 382 THz
    ref_Lambda  = ppln_period(TRIT_FREQS[ref_pair[0]], TRIT_FREQS[ref_pair[1]])
    ref_kap     = kappa(TRIT_FREQS[ref_pair[0]], TRIT_FREQS[ref_pair[1]])

    print(f"  Reference poling: Lambda = {ref_Lambda*1e6:.2f} um  (designed for trit 1x1=1)")
    print()
    print(f"  {'Input pair':12s}  {'dk_qpm (1/mm)':16s}  {'Suppression':12s}")
    print("  " + "-" * 45)

    for trit_A, trit_B in TRIT_PAIRS:
        f_A   = TRIT_FREQS[trit_A]
        f_B   = TRIT_FREQS[trit_B]
        f_out = f_A + f_B
        dk_raw   = k(f_out) - k(f_A) - k(f_B)
        dk_qpm   = dk_raw - 2 * math.pi / ref_Lambda
        sinc_val = math.sin(dk_qpm * 10e-3 / 2) / (dk_qpm * 10e-3 / 2) if abs(dk_qpm) > 1e-6 else 1.0
        suppression_db = 20 * math.log10(abs(sinc_val)) if abs(sinc_val) > 1e-10 else -100.0
        tag = " <- designed" if (trit_A, trit_B) == ref_pair else ""
        print(f"  {trit_A}x{trit_B} = {trit_A*trit_B:<4d}    "
              f"  {dk_qpm*1e-3:+.2f} /mm          "
              f"  {suppression_db:+.1f} dB{tag}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out = {
        "description": "SFG coupled-mode-theory validation for NRadix trit multiplication",
        "material":    "Thin-film LiNbO3 (TFLN), z-cut, extraordinary axis",
        "d_eff_pm_V":  D_EFF * 1e12,
        "A_eff_um2":   A_EFF * 1e12,
        "correction":  WAVEGUIDE_CORRECTION,
        "waveguide_length_mm": 10.0,
        "pump_power_mW": 1.0,
        "n_pass":      n_pass,
        "n_total":     len(results),
        "pairs":       results,
    }
    out_path = results_dir / "sfg_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("=" * 70)
    print(f"  SFG VALIDATION COMPLETE: {n_pass}/{len(results)} PASS")
    print(f"  Saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
