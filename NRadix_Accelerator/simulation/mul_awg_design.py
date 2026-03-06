"""mul_awg_design.py — NRadix Multiply Unit AWG Router
=======================================================
Analytically designs a compact 3-channel Arrayed Waveguide Grating (AWG)
to route the three possible SFG product frequencies from the NRadix multiply
unit to three output ports.

Product frequency → port mapping:
  198 THz  →  port 0  (logical value -1)
  201 THz  →  port 1  (logical value  0)
  208 THz  →  port 2  (logical value +1)

Channel spacing is NON-UNIFORM: 3 THz (ports 0→1) and 7 THz (ports 1→2).
The AWG handles this naturally: each output port position is computed
individually from y_j = D_per_THz × (f_j − f_center).

Design strategy:
  - Low grating order (m=5) → large FSR (>>10 THz span needed)
  - Fewer arms (N_ARMS=10) → compact chip
  - Larger R_FPR than needed for uniform spacing → gives ≥3.7 µm port separation
    at the tightest 3 THz gap

Run on RunPod (or locally with numpy/gdstk):
  pip install numpy gdstk
  python3 mul_awg_design.py

Output:
  results/mul_awg_design.json   — design parameters + validation
  results/mul_awg_demux.gds     — foundry-ready GDS layout
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

try:
    import gdstk
    HAS_GDSTK = True
except ImportError:
    HAS_GDSTK = False
    print("  [WARNING] gdstk not found — GDS export skipped. pip install gdstk")

# =============================================================================
# Physical Constants
# =============================================================================

C_LIGHT = 299_792_458.0  # m/s

# =============================================================================
# Waveguide Material Parameters (SiN on SiO2, matches demux AWG)
# =============================================================================

N_EFF_WG   = 1.95   # effective index in array waveguides
N_GROUP    = 2.05   # group index (for FSR calculation)
N_EFF_SLAB = 1.75   # effective index in FPR slab

# =============================================================================
# Channel Frequencies
# =============================================================================

CHANNELS = [
    {"freq_thz": 198.0, "logical": -1, "port": 0},
    {"freq_thz": 201.0, "logical":  0, "port": 1},
    {"freq_thz": 208.0, "logical": +1, "port": 2},
]
N_CH    = len(CHANNELS)
FREQS   = np.array([ch["freq_thz"] * 1e12 for ch in CHANNELS])  # Hz
F_C     = FREQS.mean()                                            # center freq Hz
LAM_C   = C_LIGHT / F_C                                          # center wavelength m

# =============================================================================
# AWG Design Parameters
# =============================================================================

M_ORDER = 5       # grating order (low → large FSR)
N_ARMS  = 10      # number of array waveguides
R_FPR   = 120e-6  # FPR radius, m  (120 µm)
D_ARM   = 2e-6    # arm pitch at FPR interface, m
R_BEND  = 10e-6   # minimum bend radius for arms, m
WG_W    = 0.5e-6  # waveguide width, m

# =============================================================================
# Step 1: Verify FSR covers full channel span
# =============================================================================

print("=" * 65)
print("  NRadix Multiply Unit — 3-Channel AWG Design")
print("=" * 65)

delta_L    = M_ORDER * LAM_C / N_EFF_WG          # path-length increment per arm, m
fsr_hz     = C_LIGHT * N_EFF_WG / (N_GROUP * M_ORDER * LAM_C)  # FSR in Hz
freq_span  = FREQS.max() - FREQS.min()            # channel span, Hz

print(f"\n  Center wavelength: {LAM_C*1e9:.2f} nm  ({F_C*1e-12:.1f} THz)")
print(f"  Grating order:     m = {M_ORDER}")
print(f"  Arms:              N = {N_ARMS}")
print(f"  ΔL per arm:        {delta_L*1e6:.4f} µm")
print(f"  FSR:               {fsr_hz*1e-12:.2f} THz")
print(f"  Channel span:      {freq_span*1e-12:.1f} THz")

assert fsr_hz > freq_span * 1.2, (
    f"FSR ({fsr_hz*1e-12:.2f} THz) must be >1.2x channel span ({freq_span*1e-12:.1f} THz). "
    "Reduce M_ORDER or increase N_EFF_WG."
)
print(f"  FSR/span ratio:    {fsr_hz/freq_span:.2f}x  [need >1.2] PASS")

# =============================================================================
# Step 2: Output port positions (non-uniform — computed per channel)
# =============================================================================

# Spatial dispersion: position at output FPR for frequency f_j
#   y_j = -R_FPR * m * lam_c^2 / (N_EFF_SLAB * D_ARM * c) * (f_j - f_c)
dispersion_per_hz = R_FPR * M_ORDER * LAM_C**2 / (N_EFF_SLAB * D_ARM * C_LIGHT)
y_out = dispersion_per_hz * (FREQS - F_C)  # position of each output port, m

print(f"\n  Output port positions (relative to center):")
for i, (ch, y) in enumerate(zip(CHANNELS, y_out)):
    print(f"    port {i} ({ch['freq_thz']:.0f} THz, "
          f"logical {ch['logical']:+d}):  y = {y*1e6:+.2f} um")

separations = np.diff(y_out)
print(f"\n  Port separations: {[f'{abs(s)*1e6:.2f} um' for s in separations]}")
assert all(abs(s) >= 3e-6 for s in separations), (
    "Port separation < 3 um. Increase R_FPR or decrease D_ARM."
)
print(f"  Min separation check (>=3 um): PASS")

# =============================================================================
# Step 3: Arm lengths
# =============================================================================

k_arms      = np.arange(N_ARMS) - (N_ARMS - 1) / 2.0   # symmetric around 0
arm_lengths = M_ORDER * LAM_C / N_EFF_WG * np.arange(N_ARMS)
L_base      = 2 * math.pi * R_BEND + 10e-6  # minimum routing length
arm_lengths = arm_lengths + L_base

# =============================================================================
# Step 4: Fourier-optics validation
# =============================================================================

print("\n  --- Fourier-optics validation ---")

PASS_THRESH_ER = 10.0   # dB
results = []

for freq_idx, (ch, f_j, y_j) in enumerate(zip(CHANNELS, FREQS, y_out)):
    k0 = 2 * math.pi * f_j / C_LIGHT

    # Gaussian amplitude coupling from input FPR to arm k
    y_arms    = k_arms * D_ARM
    w_in      = N_ARMS * D_ARM / 3.0
    A_in      = np.exp(-0.5 * (y_arms / w_in) ** 2)

    # Phase accumulated in array
    phi_arm   = k0 * N_EFF_WG * arm_lengths

    # Steering phase at output FPR (Fraunhofer far-field)
    theta_k   = np.arcsin(np.clip(y_arms / R_FPR, -1, 1))
    phi_steer = -k0 * N_EFF_SLAB * np.outer(np.sin(theta_k), y_out)

    # Output field at each port
    E_out = np.array([
        np.sum(A_in * np.exp(1j * (phi_arm + phi_steer[:, j])))
        for j in range(N_CH)
    ])
    P_out = np.abs(E_out) ** 2
    P_total = P_out.sum()

    target_port  = ch["port"]
    P_target     = P_out[target_port]
    P_other      = np.delete(P_out, target_port)
    P_max_other  = P_other.max() if len(P_other) else 1e-30

    er_db        = 10 * math.log10(P_target / P_max_other) if P_max_other > 0 else 99.0
    power_frac   = P_target / P_total if P_total > 0 else 0.0
    passed       = er_db >= PASS_THRESH_ER

    results.append({
        "freq_thz":            ch["freq_thz"],
        "logical":             ch["logical"],
        "target_port":         target_port,
        "detected_port":       int(np.argmax(P_out)),
        "power_fraction":      float(power_frac),
        "extinction_ratio_db": float(er_db),
        "passed":              passed,
    })

    status = "PASS" if passed else "FAIL"
    print(f"    {ch['freq_thz']:.0f} THz -> port {target_port}  "
          f"ER={er_db:.1f} dB  pwr={power_frac:.1%}  {status}")

n_pass = sum(r["passed"] for r in results)
print(f"\n  Result: {n_pass}/{N_CH} PASS")
assert n_pass == N_CH, f"Only {n_pass}/{N_CH} channels pass — check AWG parameters."

# =============================================================================
# Step 5: Chip size estimate
# =============================================================================

L_base_m    = L_base
chip_width_m  = 2 * R_FPR + N_ARMS * D_ARM + 2 * L_base_m
chip_height_m = N_ARMS * D_ARM + 4 * R_BEND + 20e-6

print(f"\n  Chip estimate: {chip_width_m*1e6:.0f} x {chip_height_m*1e6:.0f} um")

# =============================================================================
# Step 6: Save design summary
# =============================================================================

design = {
    "component":      "multiply_unit_awg",
    "description":    "3-channel AWG router for NRadix SFG product frequencies",
    "channels":       CHANNELS,
    "n_ch":           N_CH,
    "f_center_thz":   float(F_C * 1e-12),
    "lam_center_nm":  float(LAM_C * 1e9),
    "m_order":        M_ORDER,
    "n_arms":         N_ARMS,
    "r_fpr_um":       float(R_FPR * 1e6),
    "d_arm_um":       float(D_ARM * 1e6),
    "delta_l_um":     float(delta_L * 1e6),
    "fsr_thz":        float(fsr_hz * 1e-12),
    "n_eff_wg":       N_EFF_WG,
    "n_group":        N_GROUP,
    "n_eff_slab":     N_EFF_SLAB,
    "y_out_um":       [float(y * 1e6) for y in y_out],
    "port_sep_um":    [float(abs(s) * 1e6) for s in separations],
    "chip_width_um":  float(chip_width_m * 1e6),
    "chip_height_um": float(chip_height_m * 1e6),
    "validation":     results,
    "n_pass":         n_pass,
    "n_total":        N_CH,
}

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
out_json = results_dir / "mul_awg_design.json"
with open(out_json, "w") as f:
    json.dump(design, f, indent=2)
print(f"\n  Saved: {out_json}")

# =============================================================================
# Step 7: GDS Layout
# =============================================================================

if HAS_GDSTK:
    print("\n  Generating GDS layout...")

    lib  = gdstk.Library()
    cell = lib.new_cell("MUL_AWG_ROUTER")

    WG_LAYER    = 1
    SLAB_LAYER  = 2
    LABEL_LAYER = 10

    x_out_fpr = chip_width_m * 1e6  # in um for GDS

    def um(x): return float(x * 1e6)

    # Input FPR slab
    n_arc = 60
    fpr_pts = []
    for i in range(n_arc + 1):
        theta = -math.pi / 2 + math.pi * i / n_arc
        fpr_pts.append((um(R_FPR) * math.cos(theta), um(R_FPR) * math.sin(theta)))
    fpr_pts.append((0.0, 0.0))
    cell.add(gdstk.Polygon(fpr_pts, layer=SLAB_LAYER))

    # Output FPR slab
    fpr_pts2 = []
    for i in range(n_arc + 1):
        theta = math.pi / 2 + math.pi * i / n_arc
        fpr_pts2.append((x_out_fpr + um(R_FPR) * math.cos(theta),
                         um(R_FPR) * math.sin(theta)))
    fpr_pts2.append((x_out_fpr, 0.0))
    cell.add(gdstk.Polygon(fpr_pts2, layer=SLAB_LAYER))

    # Array waveguides
    x_arm_start = um(R_FPR)
    x_arm_end   = x_out_fpr - um(R_FPR)
    y_arms_um   = k_arms * D_ARM * 1e6

    for idx, y_a in enumerate(y_arms_um):
        L_k = um(arm_lengths[idx])
        cell.add(gdstk.rectangle(
            (x_arm_start, y_a - um(WG_W) / 2),
            (x_arm_start + min(L_k, x_arm_end - x_arm_start), y_a + um(WG_W) / 2),
            layer=WG_LAYER,
        ))

    # Output waveguides (non-uniform positions)
    y_center_out = sum(y_out) / N_CH * 1e6
    for ch, y_j in zip(CHANNELS, y_out):
        y_um = y_j * 1e6
        cell.add(gdstk.rectangle(
            (x_out_fpr, y_um - um(WG_W) / 2),
            (x_out_fpr + 20.0, y_um + um(WG_W) / 2),
            layer=WG_LAYER,
        ))
        cell.add(gdstk.Label(
            f"{ch['freq_thz']:.0f}THz_p{ch['port']}",
            (x_out_fpr + 22.0, y_um),
            layer=LABEL_LAYER,
        ))

    # Input waveguide
    cell.add(gdstk.rectangle((-30.0, -um(WG_W)/2), (0.0, um(WG_W)/2), layer=WG_LAYER))
    cell.add(gdstk.Label("SFG_IN", (-35.0, 0.0), layer=LABEL_LAYER))

    out_gds = results_dir / "mul_awg_demux.gds"
    lib.write_gds(str(out_gds))
    print(f"  Saved: {out_gds}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 65)
print("  MULTIPLY UNIT AWG COMPLETE")
print("=" * 65)
print(f"  Channels:    {n_pass}/{N_CH} PASS")
print(f"  ER range:    {min(r['extinction_ratio_db'] for r in results):.1f} - "
      f"{max(r['extinction_ratio_db'] for r in results):.1f} dB")
print(f"  FSR:         {fsr_hz*1e-12:.2f} THz  (span: {freq_span*1e-12:.1f} THz)")
print(f"  Chip size:   {chip_width_m*1e6:.0f} x {chip_height_m*1e6:.0f} um")
print(f"  Output:      results/mul_awg_design.json")
if HAS_GDSTK:
    print(f"               results/mul_awg_demux.gds")
print("=" * 65)
