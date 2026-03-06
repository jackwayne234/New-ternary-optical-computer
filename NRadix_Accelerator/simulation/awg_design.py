"""awg_design.py — NRadix AWG (Arrayed Waveguide Grating) Demux Design
=======================================================================

Analytical design + GDS layout + Fourier-optics channel response for the
NRadix 19-channel C-band demux.

Replaces the BPM inverse-designed free-form demux, which failed FDTD
validation (1/19 PASS). AWGs are analytically designed, foundry-proven,
and routinely achieve sub-0.3 THz channel spacing on SiN/SiO2 platforms.

Physics:
  Each array arm k accumulates phase φ_k = (2πf/c) · n_eff · L_k
  where L_k = L_0 + k·ΔL and ΔL = m·λ_c / n_eff (grating order condition).
  At f_c, adjacent arms differ by exactly 2πm → constructive at y=0.
  At f_c + i·Δf, the extra phase per arm steers the focus to y_i = i·D_out.
  The output demux is a spatial Fourier transform done in free space.

Design parameters:
  19 channels · 0.3 THz spacing · center 195 THz (1537 nm)
  Platform: SiN (n=2.2) / SiO2 (n=1.44)
  Grating order m=30, FPR radius R=370 µm, 30 array arms

Output (saved to results/):
  awg_design.json    — all design parameters
  awg_response.json  — analytical per-channel ER and pass/fail
  awg_demux.gds      — foundry-ready GDS layout (requires gdstk)

Run:
  cd NRadix_Accelerator/simulation
  python3 awg_design.py
"""

from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path

try:
    import gdstk
    HAS_GDSTK = True
except ImportError:
    HAS_GDSTK = False
    print("  WARNING: gdstk not installed — GDS export skipped.")
    print("  Install: pip install gdstk")

# ============================================================================
# Physical constants
# ============================================================================

C_LIGHT = 299_792_458.0  # m/s

# ============================================================================
# Material parameters  (SiN/SiO2 platform, 500 nm waveguide at C-band)
# ============================================================================

N_CORE       = 2.2      # SiN core index
N_CLAD       = 1.44     # SiO2 cladding index
N_EFF_WG     = 1.95     # array waveguide phase effective index
N_GROUP      = 2.05     # group index (n_eff + f·dn/df — drives FSR)
N_EFF_SLAB   = 1.75     # FPR slab effective index (2D effective medium)
WG_WIDTH     = 0.5e-6   # 500 nm waveguide width [m]

# ============================================================================
# AWG design parameters
# ============================================================================

M_ORDER  = 30        # grating order
N_ARMS   = 30        # number of array waveguides
R_FPR    = 370e-6    # Rowland circle radius [m]
D_ARM    = 3.0e-6    # array arm pitch at FPR aperture [m]
D_OUT    = 5.0e-6    # output waveguide pitch [m]
R_BEND   = 15e-6     # minimum bend radius for S-bends [m]

# GDS layers
LAYER_WG    = 1
LAYER_SLAB  = 2
LAYER_LABEL = 10


# ============================================================================
# AWG design calculator
# ============================================================================

def design_awg(fa: dict) -> dict:
    """Compute all AWG parameters from the NRadix frequency assignment.

    Verifies that the design equations are self-consistent:
      D_out = R_FPR · m · Δλ / (n_slab · D_arm)   [channel spacing → output pitch]
      ΔL    = m · λ_c / n_eff                       [grating order → path increment]
      FSR   > total bandwidth                        [no aliasing]
    """
    vtc = fa["value_to_channel"]
    ctf = fa["channel_to_freq"]

    values_sorted = sorted(
        int(v) for v in vtc.keys()
        if float(ctf.get(f"{vtc[v][0]},{vtc[v][1]}", 0)) > 0
    )
    n_ch = len(values_sorted)

    freqs_hz = []
    for v in values_sorted:
        port, sub = vtc[str(v)]
        freqs_hz.append(float(ctf[f"{port},{sub}"]) * 1e12)

    f_center  = float(np.mean(freqs_hz))
    lambda_c  = C_LIGHT / f_center
    delta_f   = freqs_hz[1] - freqs_hz[0] if n_ch > 1 else 0.3e12
    delta_lam = (C_LIGHT / f_center**2) * delta_f

    delta_L     = M_ORDER * lambda_c / N_EFF_WG
    # FSR = c·n_eff / (n_g·m·λ_c)  — group index governs how fast phase shifts with freq
    fsr_hz      = C_LIGHT * N_EFF_WG / (N_GROUP * M_ORDER * lambda_c)
    total_bw    = (n_ch - 1) * delta_f
    d_out_check = R_FPR * M_ORDER * delta_lam / (N_EFF_SLAB * D_ARM)
    pitch_error = abs(d_out_check - D_OUT) / D_OUT

    theta_arms = np.array([
        (k - N_ARMS / 2 + 0.5) * D_ARM / R_FPR
        for k in range(N_ARMS)
    ])

    y_out = np.array([
        (i - n_ch / 2 + 0.5) * D_OUT
        for i in range(n_ch)
    ])

    L_base      = 2 * math.pi * R_BEND + 100e-6
    arm_lengths = np.array([L_base + k * delta_L for k in range(N_ARMS)])

    print(f"\n{'='*62}")
    print(f"  NRadix AWG Demux — Design Summary")
    print(f"{'='*62}")
    print(f"  Channels:          {n_ch}")
    print(f"  Center freq:       {f_center*1e-12:.3f} THz  ({lambda_c*1e9:.2f} nm)")
    print(f"  Channel spacing:   {delta_f*1e-12:.4f} THz  ({delta_lam*1e9:.4f} nm)")
    print(f"  Total bandwidth:   {total_bw*1e-12:.2f} THz")
    print(f"  Grating order:     m = {M_ORDER}")
    print(f"  Array arms:        {N_ARMS}")
    print(f"  FPR radius:        {R_FPR*1e6:.0f} µm")
    print(f"  Arm pitch at FPR:  {D_ARM*1e6:.1f} µm")
    print(f"  Output pitch:      {D_OUT*1e6:.2f} µm  (design eq → {d_out_check*1e6:.2f} µm, err {pitch_error*100:.1f}%)")
    print(f"  Path increment ΔL: {delta_L*1e6:.3f} µm / arm")
    print(f"  Total ΔL:          {(N_ARMS-1)*delta_L*1e6:.2f} µm")
    print(f"  FSR:               {fsr_hz*1e-12:.2f} THz")
    print(f"  FSR / bandwidth:   {fsr_hz/total_bw:.2f}x  (need > 1.0)")
    print(f"  Arm lengths:       {arm_lengths[0]*1e6:.1f} – {arm_lengths[-1]*1e6:.1f} µm")
    print(f"{'='*62}\n")

    assert fsr_hz > total_bw, (
        f"FSR {fsr_hz*1e-12:.2f} THz < total BW {total_bw*1e-12:.2f} THz"
    )

    return {
        "n_channels":     n_ch,
        "values_sorted":  values_sorted,
        "freqs_hz":       freqs_hz,
        "f_center_hz":    f_center,
        "lambda_c_m":     lambda_c,
        "delta_f_hz":     delta_f,
        "delta_lam_m":    delta_lam,
        "delta_L_m":      delta_L,
        "fsr_hz":         fsr_hz,
        "M_ORDER":        M_ORDER,
        "N_ARMS":         N_ARMS,
        "R_FPR_m":        R_FPR,
        "D_ARM_m":        D_ARM,
        "D_OUT_m":        D_OUT,
        "R_BEND_m":       R_BEND,
        "theta_arms":     theta_arms.tolist(),
        "arm_lengths_m":  arm_lengths.tolist(),
        "y_out_m":        y_out.tolist(),
        "d_out_check_m":  d_out_check,
        "pitch_error":    pitch_error,
    }


# ============================================================================
# Analytical channel response  (Fourier-optics phased array model)
# ============================================================================

def compute_channel_response(design: dict) -> dict:
    """Fourier-optics transfer matrix for the AWG.

    For each input channel frequency f_i:
      1. Gaussian amplitude envelope across array aperture
      2. Phase in arm k: φ_k = (2πf/c) · n_eff · L_k
      3. Steering phase: φ_steer(k,j) = -(2πf/c) · n_slab · sin(θ_k) · y_j
         (Fraunhofer far-field: arm k at angle θ_k focuses to y_j at range R_fpr)
      4. T(i,j) = |Σ_k A_k · exp(j·(φ_k + φ_steer(k,j)))|²  normalized

    At f = f_c + i·Δf the incremental phase-per-arm exactly steers to y_i = i·D_out,
    confirmed by the self-consistency check: D_out = R·m·Δλ / (n_s·D_arm).
    """
    freqs_hz    = np.array(design["freqs_hz"])
    arm_lengths = np.array(design["arm_lengths_m"])
    theta_arms  = np.array(design["theta_arms"])
    y_out       = np.array(design["y_out_m"])
    n_ch        = design["n_channels"]
    n_arms      = len(arm_lengths)

    sigma_k = n_arms / 4.0
    k_idx   = np.arange(n_arms)
    A_in    = np.exp(-0.5 * ((k_idx - (n_arms - 1) / 2.0) / sigma_k) ** 2)
    A_in   /= np.linalg.norm(A_in)

    results  = []
    n_pass   = 0
    T_matrix = np.zeros((n_ch, n_ch))

    print("  Analytical AWG channel response (Fourier-optics):")
    print(f"  {'MAC':>5}  {'f (THz)':>8}  {'Port':>5}  {'Det':>5}  {'ER (dB)':>8}  {'Frac':>6}  Status")
    print(f"  {'-'*60}")

    for i, f in enumerate(freqs_hz):
        k0 = 2.0 * math.pi * f / C_LIGHT

        phi_arm   = k0 * N_EFF_WG * arm_lengths
        # Fraunhofer: φ=-k0·n_s·sin(θ_k)·y_j  (no R_FPR division — sin already encodes angle)
        # Gradient per arm = -k0·n_s·D_arm/R_fpr·y_j matches D_out = R·m·Δλ/(n_s·D_arm)
        phi_steer = -k0 * N_EFF_SLAB * np.outer(
            np.sin(theta_arms), y_out
        )

        E_out = np.array([
            np.sum(A_in * np.exp(1j * (phi_arm + phi_steer[:, j])))
            for j in range(n_ch)
        ])

        powers = np.abs(E_out) ** 2
        total  = powers.sum() + 1e-60
        T_matrix[i, :] = powers / total

        correct_port = i
        correct_pwr  = float(powers[correct_port])
        wrong_pwr    = float(total - correct_pwr) + 1e-60
        er_db        = 10.0 * math.log10(max(correct_pwr, 1e-60) / wrong_pwr)
        frac         = correct_pwr / total
        detected     = int(np.argmax(powers))
        passed       = detected == correct_port

        if passed:
            n_pass += 1

        status = "PASS" if passed else "FAIL"
        print(f"  {design['values_sorted'][i]:>+5d}  "
              f"{f*1e-12:>8.3f}  "
              f"{correct_port:>5d}  "
              f"{detected:>5d}  "
              f"{er_db:>8.1f}  "
              f"{frac:>6.3f}  {status}")

        results.append({
            "mac_value":           design["values_sorted"][i],
            "freq_thz":            f * 1e-12,
            "expected_port":       correct_port,
            "detected_port":       detected,
            "extinction_ratio_db": er_db,
            "power_fraction":      frac,
            "passed":              passed,
        })

    print(f"\n  AWG analytical: {n_pass}/{n_ch} PASS")
    if n_pass < n_ch:
        fails = [r["mac_value"] for r in results if not r["passed"]]
        print(f"  Failing channels: {fails}")
        print("  Tip: increase N_ARMS or R_FPR for better channel isolation.")

    return {
        "n_pass":    n_pass,
        "n_total":   n_ch,
        "pass_rate": n_pass / n_ch,
        "T_matrix":  T_matrix.tolist(),
        "results":   results,
    }


# ============================================================================
# GDS layout
# ============================================================================

def build_gds(design: dict, output_path: Path) -> None:
    """Generate foundry-ready GDS layout for the AWG demux.

    Layout (propagation in +x):
      input wg → FPR1 slab → 30 array arms (varying ΔL) → FPR2 slab → 19 output wgs

    Array arm routing strategy:
      Each arm k exits FPR1 at angle θ_k, routes through a dogleg region
      where arm k has an additional straight section of k·ΔL, then enters
      FPR2 at the mirror angle -θ_k.  gdstk FlexPath handles all bends
      automatically at bend_radius = R_BEND.
    """
    if not HAS_GDSTK:
        print("  Skipping GDS export (gdstk not available).")
        return

    lib  = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("NRadix_AWG_Demux")

    def um(x):
        return float(x) * 1e6

    arm_lengths = np.array(design["arm_lengths_m"])
    theta_arms  = np.array(design["theta_arms"])
    y_out       = np.array(design["y_out_m"])
    n_ch        = design["n_channels"]
    delta_L     = design["delta_L_m"]
    n_arms      = len(arm_lengths)

    ww      = um(WG_WIDTH)
    rb      = um(R_BEND)
    R_fpr   = um(R_FPR)

    # Key x-coordinates
    x_fpr1_in  = 0.0
    x_fpr1_out = R_fpr
    x_mid_base = x_fpr1_out + rb * 2 + 30.0
    x_mid_span = um((n_arms - 1) * delta_L)
    x_fpr2_in  = x_mid_base + x_mid_span + rb * 2 + 30.0
    x_fpr2_out = x_fpr2_in + R_fpr
    x_out_end  = x_fpr2_out + 20.0

    # ---- Input waveguide ----
    cell.add(gdstk.FlexPath([(-20.0, 0.0), (x_fpr1_in, 0.0)], ww, layer=LAYER_WG))

    # ---- FPR1 slab (wedge) ----
    half_ang = float(abs(theta_arms[-1])) + 0.05
    fpr_h    = R_fpr * math.sin(half_ang) + ww
    cell.add(gdstk.Polygon(
        [(x_fpr1_in, 0.0), (x_fpr1_out, fpr_h), (x_fpr1_out, -fpr_h)],
        layer=LAYER_SLAB
    ))

    # ---- Array arms ----
    for k in range(n_arms):
        theta_k = float(theta_arms[k])
        y_s     = R_fpr * math.sin(theta_k)    # start y at FPR1 aperture
        y_e     = -y_s                           # end y at FPR2 aperture (mirrored)
        x_extra = um(k * delta_L)               # incremental path length as x offset

        p0 = (x_fpr1_out, y_s)
        p1 = (x_mid_base + x_extra, y_s * 0.15)
        p2 = (x_mid_base + x_extra, y_e * 0.15)
        p3 = (x_fpr2_in,  y_e)

        cell.add(gdstk.FlexPath([p0, p1, p2, p3], ww, bend_radius=rb, layer=LAYER_WG))

    # ---- FPR2 slab (mirror) ----
    cell.add(gdstk.Polygon(
        [(x_fpr2_in, fpr_h), (x_fpr2_in, -fpr_h), (x_fpr2_out, 0.0)],
        layer=LAYER_SLAB
    ))

    # ---- Output waveguides ----
    for j, (mac_val, y_j) in enumerate(zip(design["values_sorted"], y_out)):
        y_j_um = um(y_j)
        cell.add(gdstk.FlexPath(
            [(x_fpr2_out, y_j_um), (x_out_end, y_j_um)],
            ww, layer=LAYER_WG
        ))
        cell.add(gdstk.Label(f"MAC{mac_val:+d}", (x_out_end + 3.0, y_j_um), layer=LAYER_LABEL))

    # ---- Labels ----
    cell.add(gdstk.Label(
        "NRadix AWG Demux  19ch 0.3THz  SiN/SiO2",
        (x_fpr1_out, fpr_h + 10.0), layer=LAYER_LABEL
    ))
    cell.add(gdstk.Label(
        f"m={M_ORDER}  N={N_ARMS}  R={R_FPR*1e6:.0f}um  dL={design['delta_L_m']*1e6:.2f}um",
        (x_fpr1_out, fpr_h + 5.0), layer=LAYER_LABEL
    ))

    lib.write_gds(str(output_path))

    bb = cell.bounding_box()
    if bb:
        w = bb[1][0] - bb[0][0]
        h = bb[1][1] - bb[0][1]
        print(f"  GDS saved: {output_path}")
        print(f"  Chip size: {w:.0f} x {h:.0f} um")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 62)
    print("  NRadix AWG Demux Design")
    print("  Analytical design + Fourier-optics validation + GDS")
    print("=" * 62)

    results_dir = Path("results")
    if not results_dir.exists():
        print("ERROR: results/ not found. Run mac_inverse_design.py first.")
        return

    fa_path = results_dir / "frequency_assignment.json"
    if not fa_path.exists():
        print("ERROR: frequency_assignment.json not found.")
        return

    with open(fa_path) as fh:
        fa = json.load(fh)

    design = design_awg(fa)

    out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in design.items()}
    with open(results_dir / "awg_design.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("  Saved: results/awg_design.json")

    print("\nComputing Fourier-optics channel response...")
    response = compute_channel_response(design)

    with open(results_dir / "awg_response.json", "w") as fh:
        json.dump(response, fh, indent=2)
    print("  Saved: results/awg_response.json")

    print("\nGenerating GDS layout...")
    build_gds(design, results_dir / "awg_demux.gds")

    n_pass  = response["n_pass"]
    n_total = response["n_total"]
    ers     = [r["extinction_ratio_db"] for r in response["results"]]

    print(f"\n{'='*62}")
    print(f"  AWG Design Complete")
    print(f"  Analytical validation: {n_pass}/{n_total} PASS")
    print(f"  ER range: {min(ers):.1f} – {max(ers):.1f} dB")
    if n_pass == n_total:
        print(f"  Design is analytically valid.")
        print(f"  Next: DRC check against target foundry PDK.")
    else:
        print(f"  Adjust N_ARMS or R_FPR to fix failing channels.")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
