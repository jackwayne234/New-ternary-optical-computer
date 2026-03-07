"""e2e_simulation.py — NRadix End-to-End Signal Chain Simulation
================================================================
Simulates the complete signal chain for a single trit×trit multiply:

  trit_A, trit_B (unbalanced ternary {1,2,3})
    → Laser sources: encode as telecom C-band frequencies
         trit 1 → 191.0 THz (1569.59 nm)
         trit 2 → 194.0 THz (1545.32 nm)
         trit 3 → 201.0 THz (1491.54 nm)
    → SFG in TFLN PPLN waveguide (coupled-mode theory, full ODE)
         f_out = f_A + f_B  (visible range, 382–402 THz)
    → 6-channel visible-band AWG demux
         routes each unique product frequency to its port
         port 0 → product 1 (382 THz), port 1 → product 2 (385 THz),
         port 2 → product 4 (388 THz), port 3 → product 3 (392 THz),
         port 4 → product 6 (395 THz), port 5 → product 9 (402 THz)
    → Photodetector: which port has signal = product value

Physics:
  SFG: A_A(z), A_B(z), A_out(z) coupled via d_eff, quasi-phase-matched PPLN
  AWG: Fourier-optics phased-array model (same as awg_design.py / mul_awg_design.py)

Run:
  pip install numpy scipy
  python3 e2e_simulation.py

Output:
  results/e2e_simulation.json  — full per-combination results
"""

from __future__ import annotations

import json
import math
import cmath
import numpy as np
from pathlib import Path

try:
    from scipy.integrate import solve_ivp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("  [WARNING] scipy not found — SFG ODE will use simple Euler stepping.")
    print("  For full accuracy: pip install scipy")

# =============================================================================
# Physical Constants
# =============================================================================

C_LIGHT   = 299_792_458.0    # m/s
EPS_0     = 8.854187817e-12  # F/m

# =============================================================================
# Stage 1: Trit Frequency Encoding (telecom C-band inputs)
# =============================================================================

TRIT_FREQ_HZ = {
    1: 191.0e12,
    2: 194.0e12,
    3: 201.0e12,
}

# Expected SFG output frequency and product value for each trit pair
SFG_PRODUCTS = {
    (1, 1): {"product": 1, "freq_thz": 382.0, "port": 0},
    (1, 2): {"product": 2, "freq_thz": 385.0, "port": 1},
    (2, 1): {"product": 2, "freq_thz": 385.0, "port": 1},
    (2, 2): {"product": 4, "freq_thz": 388.0, "port": 2},
    (1, 3): {"product": 3, "freq_thz": 392.0, "port": 3},
    (3, 1): {"product": 3, "freq_thz": 392.0, "port": 3},
    (2, 3): {"product": 6, "freq_thz": 395.0, "port": 4},
    (3, 2): {"product": 6, "freq_thz": 395.0, "port": 4},
    (3, 3): {"product": 9, "freq_thz": 402.0, "port": 5},
}

# Unique (trit_a, trit_b) pairs ordered by product value
UNIQUE_PAIRS = [
    (1, 1),  # 1×1 = 1
    (1, 2),  # 1×2 = 2
    (2, 2),  # 2×2 = 4
    (1, 3),  # 1×3 = 3
    (2, 3),  # 2×3 = 6
    (3, 3),  # 3×3 = 9
]

# =============================================================================
# Stage 2: SFG Physics — TFLN PPLN Coupled-Mode Theory
# =============================================================================

# Material: Thin-Film Lithium Niobate (TFLN), z-cut
D_EFF     = 17.2e-12   # effective nonlinear coefficient, m/V   (d_33 z-cut TFLN)
A_EFF     = 0.6e-12    # effective mode area, m²  (0.6 µm²)
WG_LENGTH = 10e-3      # PPLN waveguide length, m  (10 mm)
P_IN_W    = 1e-3       # input pump power per channel, W  (1 mW)

# =============================================================================
# TFLN Effective Index Model (Zelmon 1997 Sellmeier + waveguide correction)
# Identical to sfg_validation.py — ensures kappa and PPLN periods match.
# =============================================================================

WAVEGUIDE_CORRECTION = 0.955  # fits TFLN ridge data: ~2.21 at 1550nm, ~2.35 at 780nm


def _n_bulk_e(wavelength_m: float) -> float:
    """Bulk LiNbO3 extraordinary index (Zelmon 1997 Sellmeier)."""
    lam_um = wavelength_m * 1e6
    n2 = 4.5820 + 0.0992 / (lam_um**2 - 0.04432) - 0.01348 * lam_um**2
    return math.sqrt(max(n2, 1.0))


def _n_eff(freq_hz: float) -> float:
    """TFLN waveguide effective index at given frequency."""
    return WAVEGUIDE_CORRECTION * _n_bulk_e(C_LIGHT / freq_hz)


def ppln_period(f_A_hz: float, f_B_hz: float) -> float:
    """Quasi-phase-matching period Lambda = 2*pi / delta_k.

    Uses Zelmon Sellmeier + waveguide correction for accurate dispersion.
    QPM condition: k_out - k_A - k_B = 2*pi / Lambda.
    """
    f_out  = f_A_hz + f_B_hz
    dk0    = (2 * math.pi / C_LIGHT) * (
        _n_eff(f_out) * f_out - _n_eff(f_A_hz) * f_A_hz - _n_eff(f_B_hz) * f_B_hz
    )
    if dk0 <= 0:
        raise ValueError(f"Phase mismatch dk <= 0 for {f_A_hz/1e12:.0f}+{f_B_hz/1e12:.0f} THz")
    return 2 * math.pi / dk0


def coupling_coeff(f_A_hz: float, f_B_hz: float) -> float:
    """SFG coupling coefficient kappa [1/(m*sqrt(W))].

    Correct SI form for power-normalized field amplitudes (|A|^2 = power in W):
      kappa = omega_out * d_eff / (c * sqrt(n_A * n_B * n_out) * sqrt(A_eff))

    Matches sfg_validation.py exactly (sfg_validation.py: 6/6 PASS, 0.036-0.040%).
    """
    f_out     = f_A_hz + f_B_hz
    omega_out = 2 * math.pi * f_out
    n_A       = _n_eff(f_A_hz)
    n_B       = _n_eff(f_B_hz)
    n_out     = _n_eff(f_out)
    return omega_out * D_EFF / (C_LIGHT * math.sqrt(n_A * n_B * n_out) * math.sqrt(A_EFF))


def run_sfg_ode(f_A_hz: float, f_B_hz: float, p_A_w: float, p_B_w: float,
                length_m: float) -> dict:
    """Numerically integrate SFG coupled-mode equations over waveguide length.

    State: [Re(A_out), Im(A_out), Re(A_A), Im(A_A), Re(A_B), Im(A_B)]
    where |A|² = power in W.

    QPM condition: PPLN poled at Λ such that dk_qpm = dk0 - 2π/Λ = 0.
    """
    kappa  = coupling_coeff(f_A_hz, f_B_hz)
    Lam    = ppln_period(f_A_hz, f_B_hz)
    dk0    = 2 * math.pi / Lam if Lam != float('inf') else 0.0
    dk_qpm = 0.0  # QPM zeroes residual phase mismatch by design

    # Initial conditions: no SFG output, full pump power
    A_A0  = math.sqrt(p_A_w)  # real initial amplitude
    A_B0  = math.sqrt(p_B_w)
    y0    = [0.0, 0.0, A_A0, 0.0, A_B0, 0.0]

    def sfg_odes(z, state):
        A_out = complex(state[0], state[1])
        A_A   = complex(state[2], state[3])
        A_B   = complex(state[4], state[5])
        phase = cmath.exp(-1j * dk_qpm * z)

        dA_out = -1j * kappa * A_A   * A_B             * phase
        dA_A   = -1j * kappa * A_out * A_B.conjugate() * phase.conjugate()
        dA_B   = -1j * kappa * A_out * A_A.conjugate() * phase.conjugate()

        return [dA_out.real, dA_out.imag,
                dA_A.real,   dA_A.imag,
                dA_B.real,   dA_B.imag]

    if HAS_SCIPY:
        sol = solve_ivp(sfg_odes, [0, length_m], y0,
                        method='RK45', rtol=1e-8, atol=1e-12,
                        dense_output=False)
        y_final = sol.y[:, -1]
    else:
        # Fallback: simple RK4 Euler stepping
        N_STEPS = 2000
        dz = length_m / N_STEPS
        y  = np.array(y0, dtype=float)
        for _ in range(N_STEPS):
            z_cur = _ * dz
            k1 = np.array(sfg_odes(z_cur,        y))
            k2 = np.array(sfg_odes(z_cur + dz/2, y + dz/2 * k1))
            k3 = np.array(sfg_odes(z_cur + dz/2, y + dz/2 * k2))
            k4 = np.array(sfg_odes(z_cur + dz,   y + dz   * k3))
            y = y + (dz / 6) * (k1 + 2*k2 + 2*k3 + k4)
        y_final = y

    p_out   = y_final[0]**2 + y_final[1]**2
    p_A_out = y_final[2]**2 + y_final[3]**2
    p_B_out = y_final[4]**2 + y_final[5]**2

    conv_eff  = p_out / (p_A_w + p_B_w)
    norm_eff  = conv_eff / (p_A_w * p_B_w * length_m**2) * 1e4  # %/W/cm²

    return {
        "p_sfg_out_mw":       p_out   * 1e3,
        "p_pump_A_out_mw":    p_A_out * 1e3,
        "p_pump_B_out_mw":    p_B_out * 1e3,
        "conversion_eff_pct": conv_eff * 100,
        "norm_eff_pct_W_cm2": norm_eff,
        "ppln_period_um":     Lam * 1e6,
        "kappa":              kappa,
        "freq_sfg_thz":       (f_A_hz + f_B_hz) * 1e-12,
    }


# =============================================================================
# Stage 3: 6-Channel Visible-Band AWG Demux (Fourier-optics model)
# =============================================================================
# Routes SFG output frequencies 382/385/388/392/395/402 THz → ports 0–5
#
# Material: SiN waveguides at visible (~765 nm center wavelength)
#   n_eff = 2.08  (SiN ridge, TE, 400×200 nm at 765 nm — slightly higher than C-band)
#   n_g   = 2.25  (higher group index at visible)
#   n_s   = 1.88  (FPR slab effective index at visible)
# =============================================================================

AWG_N_EFF_WG   = 2.08    # phase effective index in array waveguides
AWG_N_GROUP    = 2.25    # group index
AWG_N_EFF_SLAB = 1.88    # slab index in FPR
AWG_M_ORDER    = 10      # grating order (low → large FSR)
AWG_N_ARMS     = 22      # array arm count (resolves 3 THz spacing at FSR≈36 THz)
AWG_R_FPR      = 200e-6  # FPR radius, m
AWG_D_ARM      = 2e-6    # arm pitch at FPR, m
AWG_R_BEND     = 12e-6   # bend radius, m

# 6 output channel frequencies (SFG products)
AWG_CHANNELS = [
    {"freq_thz": 382.0, "product": 1, "port": 0},
    {"freq_thz": 385.0, "product": 2, "port": 1},
    {"freq_thz": 388.0, "product": 4, "port": 2},
    {"freq_thz": 392.0, "product": 3, "port": 3},
    {"freq_thz": 395.0, "product": 6, "port": 4},
    {"freq_thz": 402.0, "product": 9, "port": 5},
]
AWG_N_CH    = len(AWG_CHANNELS)
AWG_FREQS   = np.array([ch["freq_thz"] * 1e12 for ch in AWG_CHANNELS])
AWG_F_C     = AWG_FREQS.mean()
AWG_LAM_C   = C_LIGHT / AWG_F_C


def build_awg_model() -> dict:
    """Pre-compute AWG parameters (arm lengths, port positions)."""
    delta_L = AWG_M_ORDER * AWG_LAM_C / AWG_N_EFF_WG
    fsr_hz  = C_LIGHT * AWG_N_EFF_WG / (AWG_N_GROUP * AWG_M_ORDER * AWG_LAM_C)
    freq_span = AWG_FREQS.max() - AWG_FREQS.min()

    assert fsr_hz > freq_span * 1.2, (
        f"AWG FSR ({fsr_hz*1e-12:.2f} THz) must exceed 1.2× span "
        f"({freq_span*1e-12:.1f} THz). Reduce M_ORDER."
    )

    # Output port positions (non-uniform — computed per channel from dispersion)
    dispersion_per_hz = AWG_R_FPR * AWG_M_ORDER * AWG_LAM_C**2 / (
        AWG_N_EFF_SLAB * AWG_D_ARM * C_LIGHT
    )
    y_out = dispersion_per_hz * (AWG_FREQS - AWG_F_C)

    # Minimum port separation check
    seps = np.abs(np.diff(y_out))
    assert seps.min() >= 2e-6, (
        f"Min port sep {seps.min()*1e6:.2f} µm < 2 µm. Increase R_FPR."
    )

    # Arm lengths
    L_base = 2 * math.pi * AWG_R_BEND + 10e-6
    k_arms = np.arange(AWG_N_ARMS) - (AWG_N_ARMS - 1) / 2.0
    arm_lengths = AWG_M_ORDER * AWG_LAM_C / AWG_N_EFF_WG * np.arange(AWG_N_ARMS) + L_base

    return {
        "delta_L_um":      delta_L * 1e6,
        "fsr_thz":         fsr_hz * 1e-12,
        "freq_span_thz":   freq_span * 1e-12,
        "y_out_um":        y_out * 1e6,
        "port_seps_um":    seps * 1e6,
        "arm_lengths":     arm_lengths,
        "k_arms":          k_arms,
        "L_base_m":        L_base,
    }


def awg_route(freq_hz: float, awg: dict) -> dict:
    """Fourier-optics routing: given input frequency, return port powers."""
    arm_lengths = awg["arm_lengths"]
    k_arms      = awg["k_arms"]
    y_out_m     = awg["y_out_um"] * 1e-6

    k0 = 2 * math.pi * freq_hz / C_LIGHT

    # Gaussian aperture amplitude envelope
    y_arms  = k_arms * AWG_D_ARM
    w_in    = AWG_N_ARMS * AWG_D_ARM / 3.0
    A_in    = np.exp(-0.5 * (y_arms / w_in) ** 2)

    # Phase in array arms
    phi_arm = k0 * AWG_N_EFF_WG * arm_lengths

    # Steering phase at output FPR (Fraunhofer)
    theta_k   = np.arcsin(np.clip(y_arms / AWG_R_FPR, -1, 1))
    phi_steer = -k0 * AWG_N_EFF_SLAB * np.outer(np.sin(theta_k), y_out_m)

    # Output field at each port
    E_out = np.array([
        np.sum(A_in * np.exp(1j * (phi_arm + phi_steer[:, j])))
        for j in range(AWG_N_CH)
    ])
    P_out   = np.abs(E_out) ** 2
    P_total = P_out.sum()

    detected  = int(np.argmax(P_out))
    P_det     = P_out[detected]
    P_other   = np.delete(P_out, detected)
    P_max_alt = P_other.max() if len(P_other) else 1e-30
    er_db     = 10 * math.log10(P_det / max(P_max_alt, 1e-30))

    return {
        "detected_port":  detected,
        "power_frac":     float(P_det / P_total),
        "er_db":          float(er_db),
        "port_powers_norm": (P_out / P_total).tolist(),
    }


# =============================================================================
# End-to-End: Run All 9 Trit×Trit Combinations
# =============================================================================

def run_e2e():
    print("=" * 68)
    print("  NRadix End-to-End Signal Chain — Trit×Trit Multiply")
    print("  SFG (TFLN CMT) → 6-Channel Visible AWG → Output Port")
    print("=" * 68)

    # Pre-compute AWG model (same for all inputs)
    print("\n  Building 6-channel visible AWG model...")
    awg = build_awg_model()

    print(f"  Center wavelength:   {AWG_LAM_C*1e9:.1f} nm  ({AWG_F_C*1e-12:.1f} THz)")
    print(f"  Grating order:       m = {AWG_M_ORDER}")
    print(f"  Array arms:          {AWG_N_ARMS}")
    print(f"  ΔL per arm:          {awg['delta_L_um']:.4f} µm")
    print(f"  FSR:                 {awg['fsr_thz']:.2f} THz  (span: {awg['freq_span_thz']:.0f} THz)")
    print(f"  Min port separation: {min(awg['port_seps_um']):.2f} µm")

    print("\n  Output port → SFG product frequency mapping:")
    for ch in AWG_CHANNELS:
        print(f"    Port {ch['port']} → {ch['freq_thz']:.0f} THz  "
              f"(product = {ch['product']}, λ = {C_LIGHT/(ch['freq_thz']*1e12)*1e9:.1f} nm)")

    print()
    print(f"  {'Pair':>6}  {'Prod':>4}  {'SFG (THz)':>10}  "
          f"{'Conv%':>6}  {'PPLN(µm)':>9}  "
          f"{'AWG Port':>8}  {'ER(dB)':>7}  {'Pwr%':>6}  Status")
    print(f"  {'-'*80}")

    results   = []
    n_pass    = 0
    n_total   = 0

    for trit_a, trit_b in [(a, b) for a in [1, 2, 3] for b in [1, 2, 3]]:
        expected = SFG_PRODUCTS[(trit_a, trit_b)]
        f_A = TRIT_FREQ_HZ[trit_a]
        f_B = TRIT_FREQ_HZ[trit_b]

        # Stage 1: SFG — generate output frequency
        sfg = run_sfg_ode(f_A, f_B, P_IN_W, P_IN_W, WG_LENGTH)
        f_sfg_hz = (f_A + f_B)  # output frequency is exactly f_A + f_B

        # Stage 2: AWG routing
        routing = awg_route(f_sfg_hz, awg)

        expected_port = expected["port"]
        detected_port = routing["detected_port"]
        passed        = (detected_port == expected_port)

        if passed:
            n_pass += 1
        n_total += 1

        status = "PASS" if passed else "FAIL"
        print(f"  {trit_a}×{trit_b:>1} → {expected['product']:>2}  "
              f"  {sfg['freq_sfg_thz']:>8.1f} THz  "
              f"  {sfg['conversion_eff_pct']:>5.3f}%"
              f"  {sfg['ppln_period_um']:>8.2f} µm"
              f"  port {detected_port:>1} (exp {expected_port})"
              f"  {routing['er_db']:>6.1f} dB"
              f"  {routing['power_frac']*100:>5.1f}%"
              f"  {status}")

        results.append({
            "trit_a":              trit_a,
            "trit_b":              trit_b,
            "product":             expected["product"],
            "sfg_freq_thz":        sfg["freq_sfg_thz"],
            "sfg_conv_eff_pct":    sfg["conversion_eff_pct"],
            "sfg_norm_eff":        sfg["norm_eff_pct_W_cm2"],
            "ppln_period_um":      sfg["ppln_period_um"],
            "expected_port":       expected_port,
            "detected_port":       detected_port,
            "awg_er_db":           routing["er_db"],
            "awg_power_frac":      routing["power_frac"],
            "awg_port_powers_norm": routing["port_powers_norm"],
            "passed":              passed,
        })

    print(f"\n  {'='*68}")
    print(f"  End-to-End Result: {n_pass}/{n_total} PASS")

    er_vals   = [r["awg_er_db"] for r in results]
    conv_vals = [r["sfg_conv_eff_pct"] for r in results]
    print(f"  SFG conversion:    {min(conv_vals):.4f}% – {max(conv_vals):.4f}%")
    print(f"  AWG extinction:    {min(er_vals):.1f} – {max(er_vals):.1f} dB")

    if n_pass == n_total:
        print(f"  ALL {n_total} TRIT×TRIT COMBINATIONS PASS")
        print(f"  Signal chain: trit input → SFG → visible AWG → correct output port")
    else:
        failures = [(r["trit_a"], r["trit_b"]) for r in results if not r["passed"]]
        print(f"  FAILING pairs: {failures}")
    print(f"  {'='*68}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out = {
        "component":      "e2e_trit_multiply",
        "description":    "End-to-end NRadix trit×trit multiply: SFG + AWG demux",
        "n_pass":         n_pass,
        "n_total":        n_total,
        "pass_rate":      n_pass / n_total,
        "sfg_material":   "TFLN z-cut, d_eff=17.2 pm/V, A_eff=0.6 µm², L=10mm",
        "sfg_pump_power_mw": P_IN_W * 1e3,
        "awg_params": {
            "n_channels":    AWG_N_CH,
            "m_order":       AWG_M_ORDER,
            "n_arms":        AWG_N_ARMS,
            "r_fpr_um":      AWG_R_FPR * 1e6,
            "fsr_thz":       awg["fsr_thz"],
            "freq_span_thz": awg["freq_span_thz"],
            "min_port_sep_um": float(min(awg["port_seps_um"])),
        },
        "combinations":   results,
    }
    out_path = results_dir / "e2e_simulation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    run_e2e()
