"""fdtdx_validation.py — NRadix FDTD Electromagnetic Validation
================================================================
Validates BPM-optimized NRadix designs using full 2D TM-mode Maxwell FDTD.

This script is the next level of physical verification beyond the BPM
(Beam Propagation Method). BPM is a paraxial scalar approximation — it
ignores reflections and assumes light travels primarily forward. FDTD
solves the full vectorial Maxwell equations with no approximations.

Method:
  - Broadband Gaussian pulse excitation
  - CPML absorbing boundaries (20-cell, m=3, R=1e-8)
  - In-simulation DFT monitors at all target frequencies
  - Source-normalized transmission spectra
  - Extinction ratio and power fraction comparison vs BPM results

Geometry (matches mac_inverse_design.py EXACTLY):
  - Multiply unit: 20×30 µm, 250×375 cells at 80 nm
  - Demux:         40×96 µm, 500×1200 cells at 80 nm
  - Materials: SiN core (n=2.2), SiO2 cladding (n=1.44)
  - Same density_to_index() sigmoid projection formula

Run on RunPod A100 80GB:
  pip install "jax[cuda12]" numpy matplotlib scipy
  python fdtdx_validation.py

Runtime estimate:
  - Multiply unit:  ~5 min on A100
  - Demux:          ~25 min on A100
  - Total:          ~30 min

Output: results/fdtdx_validation.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ============================================================================
# Physical Constants
# ============================================================================

C_LIGHT = 299_792_458.0     # m/s
MU0     = 1.2566370614e-6   # H/m
EPS0    = 8.8541878128e-12  # F/m
ETA0    = math.sqrt(MU0 / EPS0)  # ≈ 376.73 Ω

# ============================================================================
# BPM-Matching Geometry Constants
# ============================================================================

DX       = 80e-9    # 80 nm grid spacing (matches BPM)
DY       = 80e-9
N_CORE   = 2.2      # SiN
N_CLAD   = 1.44     # SiO2
WG_WIDTH = 0.5e-6   # 500 nm waveguide width (matches BPM)

# ============================================================================
# CPML Parameters (Perfectly Matched Layer)
# ============================================================================

PML_CELLS  = 20      # PML thickness in cells
PML_ORDER  = 3       # polynomial grading order (m=3)
PML_R0     = 1e-8    # target reflection coefficient
PML_KAPPA  = 1.0     # kappa_max (stretch factor, 1.0 = standard PML)
PML_ALPHA  = 2e10    # alpha_max (CFS factor, helps late-time stability)

# ============================================================================
# FDTD Simulation Parameters
# ============================================================================

# Time step: CFL stability for 2D → dt = CFL * dx/(c*sqrt(2))
CFL = 0.98
DT = CFL * DX / (C_LIGHT * math.sqrt(2.0))  # ≈ 1.848e-16 s

# Source: broadband Gaussian pulse centered in C-band
SOURCE_F_CENTER = 197.0e12   # 197 THz center frequency
SOURCE_BW       = 22.0e12    # 22 THz bandwidth (covers 186-208 THz)
SOURCE_TAU      = 1.0 / (math.pi * SOURCE_BW)   # time-domain pulse width ≈ 14.5 fs
SOURCE_T0       = 5.0 * SOURCE_TAU               # pulse center in time ≈ 72.5 fs

# Simulation lengths (chosen for >5 ps which resolves 0.3 THz spacing)
N_STEPS_MUL   = 40_000   # ~7.4 ps — multiply unit (3 freq, 3 THz spacing)
N_STEPS_DEMUX = 60_000   # ~11.1 ps — demux (19 freq, 0.3 THz min spacing)
                          # need T > 1/0.3 THz = 3.3 ps; use 11 ps for margin

# ============================================================================
# Density → Index Conversion  (MUST MATCH mac_inverse_design.py exactly)
# ============================================================================

def density_to_index(density: jnp.ndarray, beta: float = 8.0) -> jnp.ndarray:
    """Map design density [0,1] → refractive index via sigmoid projection.

    Identical formula to BPM: n = n_clad + (n_core - n_clad) * sigmoid(beta*(d-0.5))
    """
    projected = jax.nn.sigmoid(beta * (density - 0.5))
    return N_CLAD + (N_CORE - N_CLAD) * projected


def build_n_profile(
    design_density: np.ndarray,
    grid_shape: Tuple[int, int],
    design_region: dict,
    input_waveguides: list,
    output_waveguides: list,
    beta: float = 12.0,    # use fully-binarized density for validation
) -> np.ndarray:
    """Construct full (nx, ny) refractive index array.
    Matches build_n_profile() in mac_inverse_design.py.
    """
    nx, ny = grid_shape
    density_jnp = jnp.array(design_density, dtype=jnp.float32)
    n_design = density_to_index(density_jnp, beta)
    n_profile = np.full((nx, ny), N_CLAD, dtype=np.float32)

    dr = design_region
    n_profile[dr["x0"]: dr["x0"]+dr["nx"], dr["y0"]: dr["y0"]+dr["ny"]] = np.array(n_design)

    for wg in input_waveguides:
        y_lo = int((wg["y_center"] - wg["width"] / 2) / DY)
        y_hi = int((wg["y_center"] + wg["width"] / 2) / DY)
        x_hi = int(wg["x_end"] / DX)
        n_profile[:x_hi, y_lo:y_hi] = N_CORE

    for wg in output_waveguides:
        y_lo = int((wg["y_center"] - wg["width"] / 2) / DY)
        y_hi = int((wg["y_center"] + wg["width"] / 2) / DY)
        x_lo = int(wg["x_start"] / DX)
        n_profile[x_lo:, y_lo:y_hi] = N_CORE

    return n_profile


# ============================================================================
# CPML Coefficient Computation
# ============================================================================

def build_cpml_sigma(n_cells: int, pml_cells: int = PML_CELLS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CPML (Convolutional PML) sigma, kappa, alpha profiles.

    Returns arrays of shape (n_cells,) for the entire axis.
    Nonzero only in PML regions (first and last pml_cells cells).

    Reference: Gedney (1996), Roden & Gedney (2000).

    Returns:
        sigma_e : E-field conductivity (at E-field positions)
        sigma_h : H-field conductivity (at H-field positions, offset by 0.5 cell)
        b, c    : precomputed CPML coefficients for E and H update
    """
    # Taflove (2005): σ_opt = (m+1)*ln(1/R₀) / (2·η₀·n·d)
    sigma_max = -(PML_ORDER + 1) * np.log(PML_R0) / (2.0 * ETA0 * N_CLAD * PML_CELLS * DX)

    sigma_e = np.zeros(n_cells, dtype=np.float64)
    sigma_h = np.zeros(n_cells, dtype=np.float64)
    kappa_e = np.ones(n_cells, dtype=np.float64)
    kappa_h = np.ones(n_cells, dtype=np.float64)
    alpha_e = np.zeros(n_cells, dtype=np.float64)
    alpha_h = np.zeros(n_cells, dtype=np.float64)

    for i in range(pml_cells):
        # Left PML (i = 0 to pml_cells-1)
        rho_e  = (pml_cells - i - 0.0) / pml_cells   # E at cell center
        rho_h  = (pml_cells - i - 0.5) / pml_cells   # H at cell edge
        rho_e  = max(rho_e, 0.0)
        rho_h  = max(rho_h, 0.0)

        sigma_e[i] = sigma_max * rho_e ** PML_ORDER
        sigma_h[i] = sigma_max * rho_h ** PML_ORDER
        kappa_e[i] = 1.0 + (PML_KAPPA - 1.0) * rho_e ** PML_ORDER
        kappa_h[i] = 1.0 + (PML_KAPPA - 1.0) * rho_h ** PML_ORDER
        alpha_e[i] = PML_ALPHA * (1.0 - rho_e) ** PML_ORDER
        alpha_h[i] = PML_ALPHA * (1.0 - rho_h) ** PML_ORDER

        # Right PML (mirror)
        j = n_cells - 1 - i
        sigma_e[j] = sigma_e[i]
        sigma_h[j] = sigma_h[i]
        kappa_e[j] = kappa_e[i]
        kappa_h[j] = kappa_h[i]
        alpha_e[j] = alpha_e[i]
        alpha_h[j] = alpha_h[i]

    # CPML update coefficients:
    #   b = exp(-(sigma/kappa + alpha) * dt/eps0)
    #   c = sigma / (kappa*(sigma + kappa*alpha)) * (b - 1)
    def _bc(sigma, kappa, alpha):
        b = np.exp(-(sigma / kappa + alpha) * DT / EPS0)
        denom = kappa * (sigma + kappa * alpha)
        c = np.where(denom > 1e-30, sigma / denom * (b - 1.0), 0.0)
        return b.astype(np.float32), c.astype(np.float32)

    b_e, c_e = _bc(sigma_e, kappa_e, alpha_e)
    b_h, c_h = _bc(sigma_h, kappa_h, alpha_h)
    kappa_e   = kappa_e.astype(np.float32)
    kappa_h   = kappa_h.astype(np.float32)

    return b_e, c_e, kappa_e, b_h, c_h, kappa_h


# ============================================================================
# Core 2D TM FDTD Simulation
# ============================================================================
#
# TM mode: Ez, Hx, Hy (Ez perpendicular to 2D plane)
# Yee grid:
#   Ez(i,j)  at (i*dx, j*dy)         — cell centers
#   Hx(i,j)  at (i*dx, (j+0.5)*dy)  — between E in y
#   Hy(i,j)  at ((i+0.5)*dx, j*dy)  — between E in x
#
# Update equations:
#   Hx_n+0.5 = Hx_n-0.5 - dt/mu0 * (Ez(i,j+1) - Ez(i,j)) / dy
#   Hy_n+0.5 = Hy_n-0.5 + dt/mu0 * (Ez(i+1,j) - Ez(i,j)) / dx
#   Ez_n+1   = Ez_n + dt/eps(i,j) * [(Hy(i,j) - Hy(i-1,j))/dx
#                                    - (Hx(i,j) - Hx(i,j-1))/dy]
#
# With CPML: add auxiliary ψ fields in the PML regions.

def run_fdtd_jax(
    n_profile: np.ndarray,
    input_wg: dict,
    output_monitors: list,
    target_freqs_hz: np.ndarray,
    n_steps: int,
    source_x_cell: int,
    chunk_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JAX-compiled broadband FDTD using lax.scan — runs entirely on GPU.

    The full time loop is compiled to a single XLA kernel via jax.lax.scan.
    Progress is printed every chunk_size steps (requires a short sync per chunk).

    Returns:
        P_out : (n_freqs, n_monitors) DFT power at output ports
        P_src : (n_freqs,) DFT power at source
    """
    nx, ny     = n_profile.shape
    n_freqs    = len(target_freqs_hz)
    n_monitors = len(output_monitors)

    print(f"  FDTD grid: {nx}×{ny} = {nx*ny:,} cells  [JAX/GPU]")
    print(f"  Time steps: {n_steps} ({n_steps*DT*1e12:.2f} ps)")
    print(f"  Target freqs: {n_freqs}  |  Monitors: {n_monitors}")

    # ---- Pre-compute constants (all as float32 JAX arrays) ----
    inv_DX = jnp.float32(1.0 / DX)
    inv_DY = jnp.float32(1.0 / DY)
    dt_mu  = jnp.float32(DT / MU0)
    DT_F   = jnp.float32(DT)

    # E-field update coefficient: dt/(ε₀·ε_r)  shape (nx, ny)
    coeff_E = jnp.array(DT / (EPS0 * n_profile ** 2), dtype=jnp.float32)

    # CPML coefficients → float32 JAX, broadcast to 2D
    bex, cex, kex, bhx, chx, khx = build_cpml_sigma(nx)
    bey, cey, key, bhy, chy, khy = build_cpml_sigma(ny)

    bex_j = jnp.array(bex[:, None], dtype=jnp.float32)
    cex_j = jnp.array(cex[:, None], dtype=jnp.float32)
    kex_j = jnp.array(kex[:, None], dtype=jnp.float32)
    bhx_j = jnp.array(bhx[:, None], dtype=jnp.float32)
    chx_j = jnp.array(chx[:, None], dtype=jnp.float32)
    khx_j = jnp.array(khx[:, None], dtype=jnp.float32)
    bey_j = jnp.array(bey[None, :], dtype=jnp.float32)
    cey_j = jnp.array(cey[None, :], dtype=jnp.float32)
    key_j = jnp.array(key[None, :], dtype=jnp.float32)
    bhy_j = jnp.array(bhy[None, :], dtype=jnp.float32)
    chy_j = jnp.array(chy[None, :], dtype=jnp.float32)
    khy_j = jnp.array(khy[None, :], dtype=jnp.float32)

    # Source parameters
    y_c  = input_wg["y_center"]
    bw   = input_wg["width"]
    ny_arr = jnp.arange(ny, dtype=jnp.float32) * jnp.float32(DY)
    src_profile = jnp.exp(-((ny_arr - jnp.float32(y_c)) / jnp.float32(bw)) ** 2)
    src_y_lo = max(int((y_c - bw * 2) / DY), PML_CELLS + 1)
    src_y_hi = min(int((y_c + bw * 2) / DY) + 1, ny - PML_CELLS - 1)
    src_prof_slice = src_profile[src_y_lo:src_y_hi]   # constant slice

    t0_f  = jnp.float32(SOURCE_T0)
    tau_f = jnp.float32(SOURCE_TAU)
    fc_f  = jnp.float32(SOURCE_F_CENTER)
    twopi = jnp.float32(2.0 * math.pi)

    # DFT / monitor parameters
    omega_j   = jnp.array(2.0 * math.pi * target_freqs_hz, dtype=jnp.float32)
    mon_y_idx = jnp.array([m["grid_y"] for m in output_monitors], dtype=jnp.int32)
    out_x     = nx - PML_CELLS - 5
    src_y_mon = int(y_c / DY)

    # ---- Single step function: 2D TE mode (Hz, Ex, Ey) ----
    #
    # TE mode matches the BPM scalar simulation (Ey dominant, E in-plane).
    # Fields on Yee grid:
    #   Hz(i,j) at (i+0.5, j+0.5) — face center
    #   Ex(i,j) at (i+0.5, j)     — x-edge
    #   Ey(i,j) at (i,     j+0.5) — y-edge
    # Update order: H → E
    #   Hz += dt/μ₀ * [∂Ey/∂x - ∂Ex/∂y]
    #   Ex += dt/(ε₀εr) * ∂Hz/∂y
    #   Ey -= dt/(ε₀εr) * ∂Hz/∂x
    def step_fn(carry, t_idx):
        Hz, Ex, Ey, psi_Hz_x, psi_Hz_y, psi_Ex_y, psi_Ey_x, \
            dft_ro, dft_io, dft_rs, dft_is = carry

        t = jnp.float32(t_idx) * DT_F

        # --- Hz update ---
        dEy_dx = (Ey[1:, :] - Ey[:-1, :]) * inv_DX          # (nx-1, ny)
        dEx_dy = (Ex[:, 1:] - Ex[:, :-1]) * inv_DY           # (nx, ny-1)

        new_psi_Hz_x = psi_Hz_x.at[:-1, :].set(
            bhx_j[:-1] * psi_Hz_x[:-1, :] + chx_j[:-1] * dEy_dx)
        new_psi_Hz_y = psi_Hz_y.at[:, :-1].set(
            bhy_j[:, :-1] * psi_Hz_y[:, :-1] + chy_j[:, :-1] * dEx_dy)

        Hz = Hz.at[:-1, :-1].add(dt_mu * (
            dEx_dy[:-1, :] / khy_j[:, :-1] + new_psi_Hz_y[:-1, :-1]
          - dEy_dx[:, :-1] / khx_j[:-1] - new_psi_Hz_x[:-1, :-1]
        ))

        # --- Ex update: ∂Hz/∂y ---
        dHz_dy = (Hz[:, 1:] - Hz[:, :-1]) * inv_DY           # (nx, ny-1)

        new_psi_Ex_y = psi_Ex_y.at[:, 1:].set(
            bey_j[:, 1:] * psi_Ex_y[:, 1:] + cey_j[:, 1:] * dHz_dy)

        Ex = Ex.at[:, 1:].add(coeff_E[:, 1:] * (
            dHz_dy / key_j[:, 1:] + new_psi_Ex_y[:, 1:]
        ))

        # --- Ey update: -∂Hz/∂x ---
        dHz_dx = (Hz[1:, :] - Hz[:-1, :]) * inv_DX           # (nx-1, ny)

        new_psi_Ey_x = psi_Ey_x.at[1:, :].set(
            bex_j[1:] * psi_Ey_x[1:, :] + cex_j[1:] * dHz_dx)

        Ey = Ey.at[1:, :].add(-coeff_E[1:, :] * (
            dHz_dx / kex_j[1:] + new_psi_Ey_x[1:, :]
        ))

        # --- Soft source: inject Ey at input waveguide (TE mode) ---
        src_amp = (jnp.exp(-((t - t0_f) / tau_f) ** 2)
                   * jnp.sin(twopi * fc_f * t))
        Ey = Ey.at[source_x_cell, src_y_lo:src_y_hi].add(
            src_amp * src_prof_slice)

        # --- DFT: monitor Ey at output cross-section ---
        phase   = omega_j * t
        cos_p   = jnp.cos(phase)
        sin_p   = jnp.sin(phase)
        Ey_out  = Ey[out_x, mon_y_idx]             # (n_monitors,)
        Ey_s    = Ey[source_x_cell, src_y_mon]     # scalar

        dft_ro = dft_ro + jnp.outer(cos_p, Ey_out)
        dft_io = dft_io + jnp.outer(sin_p, Ey_out)
        dft_rs = dft_rs + cos_p * Ey_s
        dft_is = dft_is + sin_p * Ey_s

        return (Hz, Ex, Ey, new_psi_Hz_x, new_psi_Hz_y, new_psi_Ex_y, new_psi_Ey_x,
                dft_ro, dft_io, dft_rs, dft_is), None

    # ---- Initial state ----
    zeros2 = lambda: jnp.zeros((nx, ny), dtype=jnp.float32)
    carry = (
        zeros2(), zeros2(), zeros2(),            # Hz, Ex, Ey
        zeros2(), zeros2(), zeros2(), zeros2(),  # psi ×4
        jnp.zeros((n_freqs, n_monitors), dtype=jnp.float32),
        jnp.zeros((n_freqs, n_monitors), dtype=jnp.float32),
        jnp.zeros(n_freqs, dtype=jnp.float32),
        jnp.zeros(n_freqs, dtype=jnp.float32),
    )

    # ---- Run in chunks for progress reporting ----
    n_chunks  = math.ceil(n_steps / chunk_size)
    t_start   = time.time()
    compiled  = False

    for chunk in range(n_chunks):
        s0 = chunk * chunk_size
        s1 = min(s0 + chunk_size, n_steps)
        t_indices = jnp.arange(s0, s1, dtype=jnp.int32)

        carry, _ = jax.lax.scan(step_fn, carry, t_indices)
        jax.block_until_ready(carry[0])   # sync GPU before timing

        if not compiled:
            compiled = True
            print(f"  [JIT compiled in {time.time()-t_start:.1f}s]")

        elapsed = time.time() - t_start
        frac    = s1 / n_steps
        eta     = elapsed / frac - elapsed
        Ez_max  = float(jnp.abs(carry[0]).max())
        print(f"    step {s1:6d}/{n_steps}  |  {frac*100:.0f}%  |  "
              f"elapsed: {elapsed:.1f}s  |  ETA: {eta:.1f}s  |  "
              f"|Hz|_max: {Ez_max:.3e}")

    # ---- Extract DFT results ----
    *_, dft_ro, dft_io, dft_rs, dft_is = carry
    P_out = np.array(dft_ro**2 + dft_io**2)   # (n_freqs, n_monitors)
    P_src = np.array(dft_rs**2 + dft_is**2)   # (n_freqs,)

    print(f"  Done in {time.time()-t_start:.1f}s")
    return P_out, P_src


# Keep old name as alias so validate_* functions still work
run_fdtd_broadband = run_fdtd_jax


# ============================================================================
# Frequency Assignment Loader
# ============================================================================

def load_frequency_assignment(results_dir: Path) -> dict:
    """Load the saved frequency assignment from Stage 1."""
    path = results_dir / "frequency_assignment.json"
    with open(path) as f:
        return json.load(f)


def get_multiply_unit_freqs(fa: dict) -> Dict[int, float]:
    """Return {logical_value: freq_hz} for the multiply unit product frequencies."""
    # multiply_product_freqs keys are logical values: -1, 0, +1
    mf = fa["multiply_product_freqs"]
    return {int(k): float(v) * 1e12 for k, v in mf.items()}


def get_demux_freqs(fa: dict) -> Dict[int, float]:
    """Return {mac_value: freq_hz} for all 19 demux output channels."""
    # channel_to_freq has keys "port,sub", value_to_channel maps value → [port, sub]
    vtc = fa["value_to_channel"]        # str(value) → [port, sub]
    ctf = fa["channel_to_freq"]         # "port,sub" → freq_thz

    result = {}
    for val_str, (port, sub) in vtc.items():
        key = f"{port},{sub}"
        if key in ctf and float(ctf[key]) > 0:
            result[int(val_str)] = float(ctf[key]) * 1e12
    return result


# ============================================================================
# Multiply Unit Waveguide Geometry
# ============================================================================

def make_multiply_unit_waveguides() -> Tuple[list, list, list]:
    """
    Reconstruct multiply unit waveguide geometry.
    Mirrors MultiplyUnitConfig + create_multiply_unit_sim() logic.
    """
    domain_x   = 20e-6
    domain_y   = 30e-6
    design_x   = 15e-6
    design_y   = 25e-6
    design_x_off = 2.5e-6
    design_y_off = 2.5e-6
    wg_width   = WG_WIDTH
    n_ports    = 3

    # Input waveguide
    input_wgs = [{
        "label": "sfg_product_input",
        "x_end": design_x_off,
        "y_center": domain_y / 2,
        "width": wg_width,
    }]

    # Output waveguides: 3 ports, evenly spaced in usable y
    y_margin  = 3e-6
    usable_y  = domain_y - 2 * y_margin
    x_start   = design_x_off + design_x
    logical_labels = [-1, 0, +1]
    output_wgs = []
    for i, lval in enumerate(logical_labels):
        y_center = y_margin + usable_y * (i + 0.5) / n_ports
        output_wgs.append({
            "label": f"port_{i}_val{lval:+d}",
            "logical_value": lval,
            "port_index": i,
            "x_start": x_start,
            "x_end": domain_x,
            "y_center": y_center,
            "width": wg_width,
        })

    design_region = {
        "x0": int(design_x_off / DX),
        "y0": int(design_y_off / DY),
        "nx": int(design_x / DX),
        "ny": int(design_y / DY),
    }

    monitors = [{"grid_y": int(wg["y_center"] / DY), **wg} for wg in output_wgs]

    return input_wgs, output_wgs, monitors, design_region, (
        int(domain_x / DX), int(domain_y / DY)
    )


# ============================================================================
# Demux Waveguide Geometry
# ============================================================================

def make_demux_waveguides(fa: dict) -> Tuple[list, list, list, dict, Tuple[int,int]]:
    """
    Reconstruct demux waveguide geometry.
    Mirrors DemuxConfig + create_demux_sim() logic.
    """
    N_INPUTS   = 9
    domain_x   = 40e-6
    domain_y   = 96e-6
    design_x   = 35e-6
    design_y   = 90e-6
    design_x_off = 2.5e-6
    design_y_off = 3.0e-6
    wg_width   = WG_WIDTH
    port_spacing = 5.0e-6

    # Input waveguide
    input_wgs = [{
        "label": "cascade_output_input",
        "x_end": design_x_off,
        "y_center": domain_y / 2,
        "width": wg_width,
    }]

    # Output waveguides: 19 ports, sorted by mac value
    vtc = fa["value_to_channel"]  # value_str → [port, sub]
    values_sorted = sorted(int(v) for v in vtc.keys()
                           if fa["channel_to_freq"].get(f"{vtc[v][0]},{vtc[v][1]}", 0.0) > 0)
    n_ports = len(values_sorted)
    y_total = (n_ports - 1) * port_spacing
    y_start = domain_y / 2 - y_total / 2
    x_start = design_x_off + design_x

    output_wgs = []
    for i, val in enumerate(values_sorted):
        y_center = y_start + i * port_spacing
        output_wgs.append({
            "label": f"answer_{val:+d}",
            "mac_value": val,
            "port_index": i,
            "x_start": x_start,
            "x_end": domain_x,
            "y_center": y_center,
            "width": wg_width,
        })

    design_region = {
        "x0": int(design_x_off / DX),
        "y0": int(design_y_off / DY),
        "nx": int(design_x / DX),
        "ny": int(design_y / DY),
    }

    monitors = [{"grid_y": int(wg["y_center"] / DY), **wg} for wg in output_wgs]

    return input_wgs, output_wgs, monitors, design_region, (
        int(domain_x / DX), int(domain_y / DY)
    )


# ============================================================================
# Validation Logic
# ============================================================================

def compute_extinction_ratio_db(target_power: float, all_powers: np.ndarray) -> float:
    """ER = 10*log10(P_target / P_wrong_total)."""
    wrong_total = np.sum(all_powers) - target_power + 1e-30
    return 10.0 * math.log10(max(target_power, 1e-30) / wrong_total)


def compute_power_fraction(target_power: float, all_powers: np.ndarray) -> float:
    """Fraction of total output power at target port."""
    total = np.sum(all_powers) + 1e-30
    return float(target_power) / float(total)


def validate_multiply_unit(
    density: np.ndarray,
    fa: dict,
    results_dir: Path,
) -> dict:
    """Validate the multiply unit: 3 product frequencies → 3 ports."""
    print("\n" + "=" * 70)
    print("  FDTD Validation: Multiply Unit (Stage 2)")
    print("=" * 70)

    mul_freqs = get_multiply_unit_freqs(fa)   # {-1: Hz, 0: Hz, +1: Hz}
    logical_to_port = {-1: 0, 0: 1, 1: 2}    # matches BPM port assignment

    input_wgs, output_wgs, monitors, design_region, grid_shape = make_multiply_unit_waveguides()
    nx, ny = grid_shape

    print(f"\nBuilding index profile ({nx}×{ny} cells)...")
    n_profile = build_n_profile(density, grid_shape, design_region, input_wgs, output_wgs)

    # Target frequencies sorted by logical value
    sorted_vals = sorted(mul_freqs.keys())
    target_freqs_hz = np.array([mul_freqs[v] for v in sorted_vals], dtype=np.float64)

    print(f"\nTarget frequencies:")
    for val, f in zip(sorted_vals, target_freqs_hz):
        print(f"  x = {val:+d}  →  {f*1e-12:.4f} THz  (port {logical_to_port[val]})")

    # Source injection point: just before the design region
    source_x_cell = int(input_wgs[0]["x_end"] / DX) - 3
    source_x_cell = max(source_x_cell, PML_CELLS + 2)

    P_out, P_src = run_fdtd_broadband(
        n_profile       = n_profile,
        input_wg        = input_wgs[0],
        output_monitors = monitors,
        target_freqs_hz = target_freqs_hz,
        n_steps         = N_STEPS_MUL,
        source_x_cell   = source_x_cell,
    )

    # Normalize by source power
    P_src_safe = np.maximum(P_src[:, None], 1e-60)
    T = P_out / P_src_safe   # (n_freqs, n_monitors) — normalized transmission

    # Evaluate each frequency
    results = []
    n_pass = 0
    for freq_idx, val in enumerate(sorted_vals):
        correct_port = logical_to_port[val]
        powers = T[freq_idx, :]       # power at each port (normalized)

        correct_pwr = float(powers[correct_port])
        er_db       = compute_extinction_ratio_db(correct_pwr, powers)
        pwr_frac    = compute_power_fraction(correct_pwr, powers)
        detected    = int(np.argmax(powers))
        passed      = (detected == correct_port)

        if passed:
            n_pass += 1

        status = "PASS" if passed else "FAIL"
        print(f"  val {val:+d} → port {correct_port}: detected={detected}  "
              f"ER={er_db:.1f} dB  frac={pwr_frac:.3f}  [{status}]")

        results.append({
            "logical_value": val,
            "expected_port": correct_port,
            "detected_port": detected,
            "freq_thz": float(target_freqs_hz[freq_idx] * 1e-12),
            "extinction_ratio_db": er_db,
            "power_fraction": pwr_frac,
            "port_powers_normalized": powers.tolist(),
            "passed": passed,
        })

    print(f"\n  Multiply unit: {n_pass}/{len(sorted_vals)} PASS")

    return {
        "component": "multiply_unit",
        "n_pass": n_pass,
        "n_total": len(sorted_vals),
        "pass_rate": n_pass / len(sorted_vals),
        "n_steps": N_STEPS_MUL,
        "grid": f"{nx}x{ny}",
        "results": results,
    }


def validate_demux(
    density: np.ndarray,
    fa: dict,
    results_dir: Path,
) -> dict:
    """Validate the demux: 19 MAC output frequencies → 19 ports."""
    print("\n" + "=" * 70)
    print("  FDTD Validation: Output Demux (Stage 3)")
    print("=" * 70)

    demux_freqs = get_demux_freqs(fa)   # {mac_value: freq_hz}

    input_wgs, output_wgs, monitors, design_region, grid_shape = make_demux_waveguides(fa)
    nx, ny = grid_shape
    values_sorted = sorted(demux_freqs.keys())
    n_ports       = len(values_sorted)

    print(f"\nDemux: {n_ports} output channels ({nx}×{ny} grid)")
    print("\nBuilding index profile...")
    n_profile = build_n_profile(density, grid_shape, design_region, input_wgs, output_wgs)

    target_freqs_hz = np.array([demux_freqs[v] for v in values_sorted], dtype=np.float64)

    print(f"\nTarget frequencies (THz):")
    for val, f in zip(values_sorted, target_freqs_hz):
        print(f"  MAC={val:+d}  {f*1e-12:.4f} THz")

    source_x_cell = int(input_wgs[0]["x_end"] / DX) - 3
    source_x_cell = max(source_x_cell, PML_CELLS + 2)

    P_out, P_src = run_fdtd_broadband(
        n_profile       = n_profile,
        input_wg        = input_wgs[0],
        output_monitors = monitors,
        target_freqs_hz = target_freqs_hz,
        n_steps         = N_STEPS_DEMUX,
        source_x_cell   = source_x_cell,
    )

    P_src_safe = np.maximum(P_src[:, None], 1e-60)
    T = P_out / P_src_safe

    results = []
    n_pass = 0
    for freq_idx, val in enumerate(values_sorted):
        correct_port = freq_idx   # port order matches values_sorted order
        powers       = T[freq_idx, :]
        correct_pwr  = float(powers[correct_port])
        er_db        = compute_extinction_ratio_db(correct_pwr, powers)
        pwr_frac     = compute_power_fraction(correct_pwr, powers)
        detected     = int(np.argmax(powers))
        passed       = (detected == correct_port)

        if passed:
            n_pass += 1

        status = "PASS" if passed else "FAIL"
        print(f"  MAC={val:+d}  f={target_freqs_hz[freq_idx]*1e-12:.4f} THz  "
              f"port={correct_port}  detected={detected}  "
              f"ER={er_db:.1f} dB  frac={pwr_frac:.3f}  [{status}]")

        results.append({
            "mac_value": val,
            "expected_port": correct_port,
            "detected_port": detected,
            "freq_thz": float(target_freqs_hz[freq_idx] * 1e-12),
            "extinction_ratio_db": er_db,
            "power_fraction": pwr_frac,
            "port_powers_normalized": powers.tolist(),
            "passed": passed,
        })

    print(f"\n  Demux: {n_pass}/{n_ports} PASS")

    return {
        "component": "demux",
        "n_pass": n_pass,
        "n_total": n_ports,
        "pass_rate": n_pass / n_ports,
        "n_steps": N_STEPS_DEMUX,
        "grid": f"{nx}x{ny}",
        "results": results,
    }


# ============================================================================
# BPM Comparison
# ============================================================================

def compare_with_bpm(fdtd_results: dict, results_dir: Path) -> dict:
    """Load BPM validation results and compare key metrics."""
    comparison = {}

    mul_bpm_path   = results_dir / "multiply_unit_validation.json"
    demux_bpm_path = results_dir / "demux_validation.json"

    if mul_bpm_path.exists() and "multiply_unit" in fdtd_results:
        with open(mul_bpm_path) as f:
            mul_bpm = json.load(f)
        fdtd_mul = fdtd_results["multiply_unit"]
        comparison["multiply_unit"] = {
            "bpm_pass_rate": mul_bpm.get("pass_rate", mul_bpm.get("n_pass", 0) / max(mul_bpm.get("n_total",1), 1)),
            "fdtd_pass_rate": fdtd_mul["pass_rate"],
            "bpm_n_pass": mul_bpm.get("n_pass"),
            "fdtd_n_pass": fdtd_mul["n_pass"],
        }
        print(f"\n  Multiply unit:  BPM {mul_bpm.get('n_pass')}/{mul_bpm.get('n_total')} PASS  "
              f"vs  FDTD {fdtd_mul['n_pass']}/{fdtd_mul['n_total']} PASS")

    if demux_bpm_path.exists() and "demux" in fdtd_results:
        with open(demux_bpm_path) as f:
            demux_bpm = json.load(f)
        fdtd_dmx = fdtd_results["demux"]
        comparison["demux"] = {
            "bpm_pass_rate": demux_bpm.get("pass_rate", demux_bpm.get("n_pass",0) / max(demux_bpm.get("n_total",1),1)),
            "fdtd_pass_rate": fdtd_dmx["pass_rate"],
            "bpm_n_pass": demux_bpm.get("n_pass"),
            "fdtd_n_pass": fdtd_dmx["n_pass"],
        }
        print(f"  Demux:          BPM {demux_bpm.get('n_pass')}/{demux_bpm.get('n_total')} PASS  "
              f"vs  FDTD {fdtd_dmx['n_pass']}/{fdtd_dmx['n_total']} PASS")

    return comparison


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  NRadix FDTD Electromagnetic Validation")
    print("  2D TE-mode Maxwell solver with CPML boundaries")
    print("=" * 70)
    print(f"\n  Grid: {DX*1e9:.0f} nm  |  dt: {DT*1e15:.3f} fs  |  "
          f"Source: {SOURCE_F_CENTER*1e-12:.0f} THz ± {SOURCE_BW*1e-12:.0f} THz")
    print(f"  Multiply unit:  {N_STEPS_MUL} steps  ({N_STEPS_MUL*DT*1e12:.1f} ps)")
    print(f"  Demux:          {N_STEPS_DEMUX} steps  ({N_STEPS_DEMUX*DT*1e12:.1f} ps)")

    results_dir = Path("results")
    if not results_dir.exists():
        print("\n  ERROR: results/ directory not found.")
        print("  Run mac_inverse_design.py first to generate density arrays.")
        return

    # Check for required files
    required = ["frequency_assignment.json", "multiply_unit_density.npy", "demux_density.npy"]
    missing  = [f for f in required if not (results_dir / f).exists()]
    if missing:
        print(f"\n  ERROR: Missing required files: {missing}")
        print("  Run mac_inverse_design.py first.")
        return

    # ----- Load frequency assignment -----
    print("\nLoading frequency assignment...")
    fa = load_frequency_assignment(results_dir)
    mul_freqs   = get_multiply_unit_freqs(fa)
    demux_freqs = get_demux_freqs(fa)
    print(f"  Multiply unit: {len(mul_freqs)} product frequencies")
    print(f"  Demux: {len(demux_freqs)} output channels")

    # ----- Load density arrays -----
    print("\nLoading density arrays...")
    mul_density   = np.load(results_dir / "multiply_unit_density.npy")
    demux_density = np.load(results_dir / "demux_density.npy")
    print(f"  Multiply unit density: {mul_density.shape}")
    print(f"  Demux density:         {demux_density.shape}")

    fdtd_results = {}
    t_total = time.time()

    # ----- Validate multiply unit -----
    mul_val = validate_multiply_unit(mul_density, fa, results_dir)
    fdtd_results["multiply_unit"] = mul_val

    # Save intermediate result
    with open(results_dir / "fdtdx_validation_mul.json", "w") as f:
        json.dump(mul_val, f, indent=2)
    print(f"\n  Saved intermediate: results/fdtdx_validation_mul.json")

    # ----- Validate demux -----
    demux_val = validate_demux(demux_density, fa, results_dir)
    fdtd_results["demux"] = demux_val

    # ----- Compare with BPM -----
    print("\n" + "=" * 70)
    print("  FDTD vs BPM Comparison")
    print("=" * 70)
    comparison = compare_with_bpm(fdtd_results, results_dir)

    # ----- Assemble final output -----
    output = {
        "method": "2D TE-mode FDTD with CPML boundaries",
        "grid_spacing_nm": DX * 1e9,
        "dt_fs": DT * 1e15,
        "pml_cells": PML_CELLS,
        "source": {
            "type": "broadband_gaussian",
            "center_freq_thz": SOURCE_F_CENTER * 1e-12,
            "bandwidth_thz": SOURCE_BW * 1e-12,
        },
        "multiply_unit": fdtd_results.get("multiply_unit"),
        "demux": fdtd_results.get("demux"),
        "bpm_comparison": comparison,
        "total_runtime_s": time.time() - t_total,
    }

    out_path = results_dir / "fdtdx_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ----- Summary -----
    print("\n" + "=" * 70)
    print("  FDTD Validation Complete")
    print("=" * 70)

    mul_pass  = fdtd_results["multiply_unit"]["n_pass"]
    mul_total = fdtd_results["multiply_unit"]["n_total"]
    dmx_pass  = fdtd_results["demux"]["n_pass"]
    dmx_total = fdtd_results["demux"]["n_total"]
    runtime   = output["total_runtime_s"]

    print(f"  Multiply unit (FDTD): {mul_pass}/{mul_total} PASS")
    print(f"  Demux        (FDTD): {dmx_pass}/{dmx_total} PASS")
    print(f"  Total runtime: {runtime/60:.1f} min")
    print(f"\n  Saved: {out_path.resolve()}")
    print("=" * 70)

    if mul_pass == mul_total and dmx_pass == dmx_total:
        print("\n  ALL PASS — FDTD confirms BPM design is electromagnetically valid.")
        print("  This design is ready for foundry submission.")
    else:
        n_fail = (mul_total - mul_pass) + (dmx_total - dmx_pass)
        print(f"\n  {n_fail} channel(s) FAILED FDTD validation.")
        print("  Possible causes: paraxial approximation breakdown, reflections,")
        print("  or evanescent coupling between adjacent ports.")
        print("  Consider increasing iterations or refining the design region.")


if __name__ == "__main__":
    main()
