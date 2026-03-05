"""
fdtd_inverse_design.py — FDTD adjoint inverse design for NRadix photonics.
Refines BPM-optimized density using full Maxwell FDTD gradients via JAX autodiff.
"""

import math
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import functools

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_LIGHT = 299_792_458.0
MU0     = 1.2566370614e-6
EPS0    = 8.8541878128e-12
ETA0    = math.sqrt(MU0 / EPS0)

# ---------------------------------------------------------------------------
# Grid / material constants
# ---------------------------------------------------------------------------
DX = DY = 80e-9
N_CORE  = 2.2
N_CLAD  = 1.44
WG_WIDTH = 0.5e-6

# ---------------------------------------------------------------------------
# CPML constants
# ---------------------------------------------------------------------------
PML_CELLS = 20
PML_ORDER = 3
PML_R0    = 1e-8
PML_KAPPA = 1.0
PML_ALPHA = 0.05   # S/m, CFS parameter (NOT 2e10 — that was a bug)

# ---------------------------------------------------------------------------
# FDTD time step
# ---------------------------------------------------------------------------
CFL = 0.98
DT  = CFL * DX / (C_LIGHT * math.sqrt(2.0))

# ---------------------------------------------------------------------------
# Source parameters
# ---------------------------------------------------------------------------
SOURCE_F_CENTER = 197.0e12
SOURCE_BW       = 22.0e12
SOURCE_TAU      = 1.0 / (math.pi * SOURCE_BW)
SOURCE_T0       = 5.0 * SOURCE_TAU


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def density_to_index(density: jnp.ndarray, beta: float = 8.0) -> jnp.ndarray:
    """Map density [0,1] to refractive index via sigmoid projection."""
    return N_CLAD + (N_CORE - N_CLAD) * jax.nn.sigmoid(beta * (density - 0.5))


def build_cpml_sigma(n_cells: int, pml_cells: int = PML_CELLS) -> tuple:
    """Build CPML sigma/kappa/alpha profiles (Taflove formulation).

    Returns (b_e, c_e, kappa_e, b_h, c_h, kappa_h) as float32 numpy arrays
    of shape (n_cells,).
    """
    sigma_max = (
        -(PML_ORDER + 1) * math.log(PML_R0)
        / (2.0 * ETA0 * N_CLAD * pml_cells * DX)
    )

    sigma_e  = np.zeros(n_cells, dtype=np.float64)
    sigma_h  = np.zeros(n_cells, dtype=np.float64)
    kappa_e  = np.ones(n_cells,  dtype=np.float64)
    kappa_h  = np.ones(n_cells,  dtype=np.float64)
    alpha_e  = np.zeros(n_cells, dtype=np.float64)
    alpha_h  = np.zeros(n_cells, dtype=np.float64)

    for i in range(pml_cells):
        rho_e = (pml_cells - i) / pml_cells
        rho_h = max((pml_cells - i - 0.5) / pml_cells, 0.0)

        sigma_e[i]  = sigma_max * rho_e ** PML_ORDER
        sigma_h[i]  = sigma_max * rho_h ** PML_ORDER
        alpha_e[i]  = PML_ALPHA * (1.0 - rho_e) ** PML_ORDER
        alpha_h[i]  = PML_ALPHA * (1.0 - rho_h) ** PML_ORDER

        # Mirror to right side
        sigma_e[n_cells - 1 - i] = sigma_e[i]
        sigma_h[n_cells - 1 - i] = sigma_h[i]
        alpha_e[n_cells - 1 - i] = alpha_e[i]
        alpha_h[n_cells - 1 - i] = alpha_h[i]

    def _bc(sigma, kappa, alpha):
        b = np.exp(-(sigma / kappa + alpha) * DT / EPS0)
        denom = kappa * (sigma + kappa * alpha)
        c = np.where(np.abs(denom) > 1e-30, sigma / denom * (b - 1.0), 0.0)
        return b.astype(np.float32), c.astype(np.float32), kappa.astype(np.float32)

    b_e, c_e, kappa_e = _bc(sigma_e, kappa_e, alpha_e)
    b_h, c_h, kappa_h = _bc(sigma_h, kappa_h, alpha_h)

    return b_e, c_e, kappa_e, b_h, c_h, kappa_h


def make_n_background(grid_shape, input_waveguides, output_waveguides) -> jnp.ndarray:
    """Build background index profile with input/output waveguide cores."""
    n_profile = np.full(grid_shape, N_CLAD, dtype=np.float32)

    for wg in input_waveguides:
        y_c  = wg["y_center"]
        w    = wg["width"]
        x_hi = int(wg["x_end"] / DX)
        y_lo = int((y_c - w / 2.0) / DY)
        y_hi = int((y_c + w / 2.0) / DY)
        n_profile[:x_hi, y_lo:y_hi] = N_CORE

    for wg in output_waveguides:
        y_c  = wg["y_center"]
        w    = wg["width"]
        x_lo = int(wg["x_start"] / DX)
        y_lo = int((y_c - w / 2.0) / DY)
        y_hi = int((y_c + w / 2.0) / DY)
        n_profile[x_lo:, y_lo:y_hi] = N_CORE

    return jnp.array(n_profile, dtype=jnp.float32)


def make_n_profile(
    density: jnp.ndarray,
    n_bg: jnp.ndarray,
    design_region: dict,
    beta: float = 8.0,
) -> jnp.ndarray:
    """Differentiable: embed design density into background index grid.

    design_region keys: x0, y0, nx, ny
    """
    x0, y0 = design_region["x0"], design_region["y0"]
    nx, ny  = design_region["nx"], design_region["ny"]
    return n_bg.at[x0 : x0 + nx, y0 : y0 + ny].set(
        density_to_index(density, beta)
    )
# Chunk 2: Differentiable FDTD forward pass (requires chunk 1 constants)

import functools
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple


def run_fdtd_differentiable(
    n_profile: jnp.ndarray,       # (nx, ny) JAX float32, DIFFERENTIABLE input
    input_wg: dict,
    output_monitors: list,
    target_freqs_hz: np.ndarray,
    n_steps: int,
    source_x_cell: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Differentiable FDTD forward pass using JAX.

    Parameters
    ----------
    n_profile       : (nx, ny) JAX float32 refractive index distribution.
                      Gradients flow through this array via autodiff.
    input_wg        : dict with keys "y_center" (float, meters) and
                      "width" (float, meters) describing the source waveguide.
    output_monitors : list of dicts, each with key "grid_y" (int cell index)
                      at which output power is monitored.
    target_freqs_hz : 1-D numpy array of frequencies [Hz] for DFT.
    n_steps         : total number of FDTD time steps.
    source_x_cell   : x-cell index where the soft Ey source is injected.

    Returns
    -------
    P_out : jnp.ndarray, shape (n_freqs, n_monitors) — output power spectrum.
    P_src : jnp.ndarray, shape (n_freqs,)             — source power spectrum.
    """

    nx, ny = n_profile.shape
    n_freqs = len(target_freqs_hz)
    n_monitors = len(output_monitors)

    # ------------------------------------------------------------------
    # 1. Precompute material coefficients (JAX — autodiff flows through)
    # ------------------------------------------------------------------
    coeff_E = jnp.float32(DT) / (jnp.float32(EPS0) * n_profile ** 2)  # (nx, ny)
    dt_mu   = jnp.float32(DT / MU0)
    inv_DX  = jnp.float32(1.0 / DX)
    inv_DY  = jnp.float32(1.0 / DY)

    # ------------------------------------------------------------------
    # 2. CPML coefficients (numpy — not differentiable, that is fine)
    # ------------------------------------------------------------------
    # X-direction CPML (applied to Ey, Hz rows)
    sig_ex, b_ex, c_ex, k_ex = build_cpml_sigma(nx)   # each length nx
    sig_hx, b_hx, c_hx, k_hx = build_cpml_sigma(nx)

    # Y-direction CPML (applied to Ex, Hz columns)
    sig_ey, b_ey, c_ey, k_ey = build_cpml_sigma(ny)
    sig_hy, b_hy, c_hy, k_hy = build_cpml_sigma(ny)

    # Reshape for broadcasting: x-direction -> (nx, 1), y-direction -> (1, ny)
    bex_j = jnp.array(b_ex, dtype=jnp.float32).reshape(nx, 1)
    cex_j = jnp.array(c_ex, dtype=jnp.float32).reshape(nx, 1)
    kex_j = jnp.array(k_ex, dtype=jnp.float32).reshape(nx, 1)

    bhx_j = jnp.array(b_hx, dtype=jnp.float32).reshape(nx, 1)
    chx_j = jnp.array(c_hx, dtype=jnp.float32).reshape(nx, 1)
    khx_j = jnp.array(k_hx, dtype=jnp.float32).reshape(nx, 1)

    bey_j = jnp.array(b_ey, dtype=jnp.float32).reshape(1, ny)
    cey_j = jnp.array(c_ey, dtype=jnp.float32).reshape(1, ny)
    key_j = jnp.array(k_ey, dtype=jnp.float32).reshape(1, ny)

    bhy_j = jnp.array(b_hy, dtype=jnp.float32).reshape(1, ny)
    chy_j = jnp.array(c_hy, dtype=jnp.float32).reshape(1, ny)
    khy_j = jnp.array(k_hy, dtype=jnp.float32).reshape(1, ny)

    # ------------------------------------------------------------------
    # 3. Source parameters and Gaussian spatial profile
    # ------------------------------------------------------------------
    y_c = input_wg["y_center"]   # meters
    bw  = input_wg["width"]      # meters

    # Convert waveguide center / half-width to grid indices
    y_c_idx  = int(round(y_c / DY))
    half_bw  = int(round((bw / 2.0) / DY))

    src_y_lo = int(np.clip(y_c_idx - half_bw, PML_CELLS + 1, ny - PML_CELLS - 1))
    src_y_hi = int(np.clip(y_c_idx + half_bw, PML_CELLS + 1, ny - PML_CELLS - 1))

    # Gaussian transverse profile across the source slice (float32 JAX array)
    y_indices   = np.arange(src_y_lo, src_y_hi, dtype=np.float32)
    sigma_gauss = (bw / 2.0) / DY / 2.0          # ~2 sigma spans the waveguide
    src_prof_np = np.exp(-0.5 * ((y_indices - y_c_idx) / sigma_gauss) ** 2)
    src_prof_np = src_prof_np / (src_prof_np.max() + 1e-30)  # normalise
    src_prof_slice = jnp.array(src_prof_np, dtype=jnp.float32)  # (slice_len,)

    # Time-domain pulse parameters (float32 scalars for JIT)
    fc_f  = jnp.float32(SOURCE_F_CENTER)
    tau_f = jnp.float32(SOURCE_TAU)
    t0_f  = jnp.float32(SOURCE_T0)
    dt_f  = jnp.float32(DT)

    # ------------------------------------------------------------------
    # 4. DFT setup
    # ------------------------------------------------------------------
    omega_j = jnp.array(2.0 * np.pi * target_freqs_hz, dtype=jnp.float32)  # (n_freqs,)

    # Monitor y-indices (integer cell positions)
    mon_y_idx = jnp.array(
        [int(m["grid_y"]) for m in output_monitors], dtype=jnp.int32
    )  # (n_monitors,)

    # Output DFT x-position (fixed column just inside the right PML)
    out_x = int(nx - PML_CELLS - 5)

    # Source DFT x-position (same column as source injection)
    src_x = source_x_cell

    # ------------------------------------------------------------------
    # 5. Step function (TE: Hz, Ex, Ey)
    # ------------------------------------------------------------------
    @functools.partial(jax.checkpoint)
    def step_fn(carry, t_idx):
        (Hz, Ex, Ey,
         psi_Hz_x, psi_Hz_y,
         psi_Ex_y, psi_Ey_x,
         dft_ro, dft_io,    # (n_freqs, n_monitors) — output DFT real/imag
         dft_rs, dft_is,    # (n_freqs,)            — source DFT real/imag
         ) = carry

        t = jnp.float32(t_idx) * dt_f

        # ---- a) Hz update ------------------------------------------------
        dEy_dx = (Ey[1:, :] - Ey[:-1, :]) * inv_DX   # (nx-1, ny)
        dEx_dy = (Ex[:, 1:] - Ex[:, :-1]) * inv_DY    # (nx, ny-1)

        # CPML auxiliary field updates
        psi_Hz_x = psi_Hz_x.at[:-1, :].set(
            bhx_j[:-1] * psi_Hz_x[:-1, :] + chx_j[:-1] * dEy_dx
        )
        psi_Hz_y = psi_Hz_y.at[:, :-1].set(
            bhy_j[:, :-1] * psi_Hz_y[:, :-1] + chy_j[:, :-1] * dEx_dy
        )

        Hz = Hz.at[:-1, :-1].add(
            dt_mu * (
                  dEx_dy[:-1, :] / khy_j[:, :-1] + psi_Hz_y[:-1, :-1]
                - dEy_dx[:, :-1] / khx_j[:-1]    - psi_Hz_x[:-1, :-1]
            )
        )

        # ---- b) Ex update ------------------------------------------------
        dHz_dy = (Hz[:, 1:] - Hz[:, :-1]) * inv_DY    # (nx, ny-1)

        psi_Ex_y = psi_Ex_y.at[:, 1:].set(
            bey_j[:, 1:] * psi_Ex_y[:, 1:] + cey_j[:, 1:] * dHz_dy
        )

        Ex = Ex.at[:, 1:].add(
            coeff_E[:, 1:] * (dHz_dy / key_j[:, 1:] + psi_Ex_y[:, 1:])
        )

        # ---- c) Ey update ------------------------------------------------
        dHz_dx = (Hz[1:, :] - Hz[:-1, :]) * inv_DX    # (nx-1, ny)

        psi_Ey_x = psi_Ey_x.at[1:, :].set(
            bex_j[1:] * psi_Ey_x[1:, :] + cex_j[1:] * dHz_dx
        )

        Ey = Ey.at[1:, :].add(
            -coeff_E[1:, :] * (dHz_dx / kex_j[1:] + psi_Ey_x[1:, :])
        )

        # ---- d) Soft Ey source injection ---------------------------------
        src_amp = (
            jnp.exp(-((t - t0_f) / tau_f) ** 2)
            * jnp.sin(2.0 * jnp.float32(np.pi) * fc_f * t)
        )
        Ey = Ey.at[source_x_cell, src_y_lo:src_y_hi].add(
            src_amp * src_prof_slice
        )

        # ---- e) DFT accumulation on Ey -----------------------------------
        # Phase factor for each frequency at this time step
        phase = omega_j * t                            # (n_freqs,)
        cos_p = jnp.cos(phase)                         # (n_freqs,)
        sin_p = jnp.sin(phase)                         # (n_freqs,)

        # Output monitors: sample Ey[out_x, mon_y_idx] for each monitor
        ey_out = Ey[out_x, mon_y_idx]                  # (n_monitors,)

        # outer-product accumulation: (n_freqs, n_monitors)
        dft_ro = dft_ro + cos_p[:, None] * ey_out[None, :]
        dft_io = dft_io + sin_p[:, None] * ey_out[None, :]

        # Source monitor: sample Ey at source plane, centre of waveguide
        ey_src = Ey[src_x, y_c_idx]                   # scalar
        dft_rs = dft_rs + cos_p * ey_src
        dft_is = dft_is + sin_p * ey_src

        new_carry = (Hz, Ex, Ey,
                     psi_Hz_x, psi_Hz_y,
                     psi_Ex_y, psi_Ey_x,
                     dft_ro, dft_io,
                     dft_rs, dft_is)
        return new_carry, None

    # ------------------------------------------------------------------
    # 6. Initial carry — all zeros
    # ------------------------------------------------------------------
    Hz      = jnp.zeros((nx,     ny    ), dtype=jnp.float32)
    Ex      = jnp.zeros((nx,     ny    ), dtype=jnp.float32)
    Ey      = jnp.zeros((nx,     ny    ), dtype=jnp.float32)

    psi_Hz_x = jnp.zeros((nx,     ny    ), dtype=jnp.float32)
    psi_Hz_y = jnp.zeros((nx,     ny    ), dtype=jnp.float32)
    psi_Ex_y = jnp.zeros((nx,     ny    ), dtype=jnp.float32)
    psi_Ey_x = jnp.zeros((nx,     ny    ), dtype=jnp.float32)

    dft_ro  = jnp.zeros((n_freqs, n_monitors), dtype=jnp.float32)
    dft_io  = jnp.zeros((n_freqs, n_monitors), dtype=jnp.float32)
    dft_rs  = jnp.zeros((n_freqs,),            dtype=jnp.float32)
    dft_is  = jnp.zeros((n_freqs,),            dtype=jnp.float32)

    init_carry = (Hz, Ex, Ey,
                  psi_Hz_x, psi_Hz_y,
                  psi_Ex_y, psi_Ey_x,
                  dft_ro, dft_io,
                  dft_rs, dft_is)

    # ------------------------------------------------------------------
    # 6 (cont). Run full scan — JAX handles the entire computation graph
    # NO chunking loop: this is the differentiable version
    # ------------------------------------------------------------------
    final_carry, _ = jax.lax.scan(
        step_fn,
        init_carry,
        jnp.arange(n_steps, dtype=jnp.int32),
    )

    # ------------------------------------------------------------------
    # 7. Extract DFT accumulators and compute power spectra
    # ------------------------------------------------------------------
    (_, _, _,
     _, _,
     _, _,
     dft_ro_f, dft_io_f,
     dft_rs_f, dft_is_f) = final_carry

    # Power = |DFT|^2  (proportional; normalise against source externally)
    P_out = dft_ro_f ** 2 + dft_io_f ** 2   # (n_freqs, n_monitors)
    P_src = dft_rs_f ** 2 + dft_is_f ** 2   # (n_freqs,)

    return P_out, P_src
# Chunk 3: Loss function (requires chunks 1-2)

import numpy as np
import jax.numpy as jnp

# Assumes from chunks 1-2:
#   make_n_profile(density, n_bg, design_region, beta) -> jnp.ndarray
#   run_fdtd_differentiable(n_profile, input_wg, output_monitors,
#                           target_freqs_hz, n_steps, source_x_cell) -> (P_out, P_src)
#   Constants: DX, DY, N_CORE, N_CLAD, WG_WIDTH, PML_CELLS, DT


def compute_loss(
    density: jnp.ndarray,        # (design_nx, design_ny) in [0,1]
    n_bg: jnp.ndarray,           # (nx, ny) background n_profile (waveguides only)
    design_region: dict,         # {"x0","y0","nx","ny"}
    input_wg: dict,
    output_monitors: list,       # list of dicts with "grid_y", "port_index"
    target_freqs_hz: np.ndarray, # (n_freqs,)
    n_steps: int,
    source_x_cell: int,
    beta: float = 8.0,
) -> jnp.ndarray:                # scalar loss
    # Step 1: build refractive index profile with binarization
    n_profile = make_n_profile(density, n_bg, design_region, beta)

    # Step 2: run FDTD, get power spectra
    # P_out: (n_freqs, n_monitors), P_src: (n_freqs,)
    P_out, P_src = run_fdtd_differentiable(
        n_profile, input_wg, output_monitors, target_freqs_hz, n_steps, source_x_cell
    )

    # Step 3: normalize to transmittance
    # T[freq_idx, monitor_idx] = fraction of source power reaching that monitor
    T = P_out / (P_src[:, None] + 1e-30)   # (n_freqs, n_monitors)

    # Step 4: vectorized efficiency computation
    # Diagonal: T[i, i] = power at correct port for freq i
    correct_power = jnp.diag(T)             # (n_freqs,)
    total_power   = jnp.sum(T, axis=1)      # (n_freqs,)
    efficiency    = correct_power / (total_power + 1e-30)  # (n_freqs,)

    # Step 5: loss = negative mean efficiency (maximize efficiency -> minimize loss)
    loss = -jnp.mean(efficiency)
    return loss


def compute_loss_and_metrics(
    density,
    n_bg,
    design_region,
    input_wg,
    output_monitors,
    target_freqs_hz,
    n_steps,
    source_x_cell,
    beta: float = 8.0,
):
    """
    Same computation as compute_loss, but also returns a metrics dict.
    Intended for logging only — do NOT use inside gradient computation.

    Returns
    -------
    loss_scalar : float
    metrics     : dict
        "efficiency_per_freq" : list[float]   fraction at correct port per freq
        "mean_efficiency"     : float
        "n_pass"              : int            freqs with efficiency > 0.5
        "n_total"             : int
    """
    # Rebuild profile and run simulation (same as compute_loss)
    n_profile = make_n_profile(density, n_bg, design_region, beta)
    P_out, P_src = run_fdtd_differentiable(
        n_profile, input_wg, output_monitors, target_freqs_hz, n_steps, source_x_cell
    )

    T             = P_out / (P_src[:, None] + 1e-30)
    correct_power = jnp.diag(T)
    total_power   = jnp.sum(T, axis=1)
    efficiency    = correct_power / (total_power + 1e-30)
    loss          = -jnp.mean(efficiency)

    # Convert to numpy for safe logging / Python logic
    eff_np        = np.array(efficiency)
    loss_scalar   = float(loss)

    metrics = {
        "efficiency_per_freq": eff_np.tolist(),
        "mean_efficiency":     float(np.mean(eff_np)),
        "n_pass":              int(np.sum(eff_np > 0.5)),
        "n_total":             int(len(eff_np)),
    }

    return loss_scalar, metrics
# Chunk 4: Adam optimizer loop (requires chunks 1-3)

import time
import functools
import numpy as np
import jax
import jax.numpy as jnp


def optimize_density(
    initial_density: np.ndarray,
    n_bg: jnp.ndarray,
    design_region: dict,
    input_wg: dict,
    output_monitors: list,
    target_freqs_hz: np.ndarray,
    n_steps: int,
    source_x_cell: int,
    n_iterations: int = 300,
    learning_rate: float = 0.02,
    beta_schedule: str = "fixed",  # "fixed" uses beta=8.0 throughout; "anneal" goes 4->12
    log_every: int = 10,
    save_every: int = 50,
    save_path: str = "results/density_fdtd_checkpoint.npy",
    beta_projection: float = 8.0,
) -> np.ndarray:
    density = jnp.array(initial_density, dtype=jnp.float32)
    m = jnp.zeros_like(density)
    v = jnp.zeros_like(density)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # TODO: "anneal" schedule requires re-jitting each time beta_projection changes,
    # which is expensive. Full implementation deferred; falls back to fixed for now.
    if beta_schedule == "anneal":
        print("WARNING: beta_schedule='anneal' not fully implemented; using fixed beta.")

    def make_grad_fn(beta):
        return jax.jit(jax.value_and_grad(
            lambda d: compute_loss(
                d, n_bg, design_region, input_wg, output_monitors,
                target_freqs_hz, n_steps, source_x_cell, beta
            )
        ))

    grad_fn = make_grad_fn(beta_projection)

    iter_times = []

    for i in range(1, n_iterations + 1):
        t_start = time.perf_counter()

        loss_val, grad = grad_fn(density)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** i)
        v_hat = v / (1 - beta2 ** i)
        density = density - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        density = jnp.clip(density, 0.0, 1.0)

        t_end = time.perf_counter()
        if i > 1:
            iter_times.append(t_end - t_start)

        if i % log_every == 0:
            avg_t = np.mean(iter_times[-log_every:]) if iter_times else float("nan")
            print(
                f"  iter {i:4d}/{n_iterations}  loss={float(loss_val):.4f}  "
                f"avg_iter_time={avg_t:.3f}s"
            )

        if i % save_every == 0:
            np.save(save_path, np.array(density))
            print(f"  Saved checkpoint: {save_path}")

    np.save(save_path, np.array(density))
    print(f"  Final save: {save_path}")

    if iter_times:
        print(f"  Avg iter time (excluding JIT warmup): {np.mean(iter_times):.3f}s")

    return np.array(density)
# Chunk 5: main() — entry point (requires chunks 1-4)

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict


def load_frequency_assignment(results_dir: Path) -> dict:
    with open(results_dir / "frequency_assignment.json") as f:
        return json.load(f)


def get_multiply_unit_freqs(fa: dict) -> Dict[int, float]:
    mf = fa["multiply_product_freqs"]
    return {int(k): float(v) * 1e12 for k, v in mf.items()}


def get_demux_freqs(fa: dict) -> Dict[int, float]:
    vtc = fa["value_to_channel"]
    ctf = fa["channel_to_freq"]
    result = {}
    for val_str, (port, sub) in vtc.items():
        key = f"{port},{sub}"
        if key in ctf and float(ctf[key]) > 0:
            result[int(val_str)] = float(ctf[key]) * 1e12
    return result


def make_multiply_unit_waveguides():
    domain_x = 20e-6
    domain_y = 30e-6
    design_x = 15e-6
    design_y = 25e-6
    design_x_off = 2.5e-6
    design_y_off = 2.5e-6

    input_wgs = [
        {
            "label": "sfg_product_input",
            "x_end": design_x_off,
            "y_center": domain_y / 2,
            "width": WG_WIDTH,
        }
    ]

    logical_labels = [-1, 0, +1]
    y_margin = 3e-6
    usable_y = domain_y - 2 * y_margin
    x_start = design_x_off + design_x

    output_wgs = []
    for i, lval in enumerate(logical_labels):
        y_center = y_margin + usable_y * (i + 0.5) / 3
        output_wgs.append(
            {
                "label": f"port_{i}_val{lval:+d}",
                "logical_value": lval,
                "port_index": i,
                "x_start": x_start,
                "x_end": domain_x,
                "y_center": y_center,
                "width": WG_WIDTH,
            }
        )

    design_region = {
        "x0": int(design_x_off / DX),
        "y0": int(design_y_off / DY),
        "nx": int(design_x / DX),
        "ny": int(design_y / DY),
    }

    monitors = [{"grid_y": int(wg["y_center"] / DY), **wg} for wg in output_wgs]

    return input_wgs, output_wgs, monitors, design_region, (int(domain_x / DX), int(domain_y / DY))


def make_demux_waveguides(fa: dict):
    domain_x = 40e-6
    domain_y = 96e-6
    design_x = 35e-6
    design_y = 90e-6
    design_x_off = 2.5e-6
    design_y_off = 3.0e-6
    port_spacing = 5.0e-6

    input_wgs = [
        {
            "label": "cascade_output_input",
            "x_end": design_x_off,
            "y_center": domain_y / 2,
            "width": WG_WIDTH,
        }
    ]

    vtc = fa["value_to_channel"]
    ctf = fa["channel_to_freq"]
    values_sorted = sorted(
        int(v) for v in vtc.keys()
        if float(ctf.get(f"{vtc[v][0]},{vtc[v][1]}", 0.0)) > 0
    )

    n_ports = len(values_sorted)
    y_total = (n_ports - 1) * port_spacing
    y_start = domain_y / 2 - y_total / 2
    x_start = design_x_off + design_x

    output_wgs = []
    for i, val in enumerate(values_sorted):
        y_center = y_start + i * port_spacing
        output_wgs.append(
            {
                "label": f"answer_{val:+d}",
                "mac_value": val,
                "port_index": i,
                "x_start": x_start,
                "x_end": domain_x,
                "y_center": y_center,
                "width": WG_WIDTH,
            }
        )

    design_region = {
        "x0": int(design_x_off / DX),
        "y0": int(design_y_off / DY),
        "nx": int(design_x / DX),
        "ny": int(design_y / DY),
    }

    monitors = [{"grid_y": int(wg["y_center"] / DY), **wg} for wg in output_wgs]

    return input_wgs, output_wgs, monitors, design_region, (int(domain_x / DX), int(domain_y / DY))


def main():
    print("NRadix FDTD Adjoint Inverse Design")

    component = sys.argv[1] if len(sys.argv) > 1 else "multiply_unit"
    if component not in ("multiply_unit", "demux"):
        print(f"Error: unknown component '{component}'. Choose 'multiply_unit' or 'demux'.")
        sys.exit(1)

    results_dir = Path("results")
    if not results_dir.exists():
        print(f"Error: results dir '{results_dir}' does not exist.")
        sys.exit(1)

    fa = load_frequency_assignment(results_dir)

    if component == "multiply_unit":
        initial_density = np.load(results_dir / "multiply_unit_density.npy")
        mul_freqs = get_multiply_unit_freqs(fa)
        sorted_vals = sorted(mul_freqs.keys())
        target_freqs_hz = np.array([mul_freqs[v] for v in sorted_vals])
        input_wgs, output_wgs, monitors, design_region, grid_shape = make_multiply_unit_waveguides()
        n_steps = 40_000
        save_path = str(results_dir / "multiply_unit_density_fdtd.npy")

    else:  # demux
        demux_freqs = get_demux_freqs(fa)
        input_wgs, output_wgs, monitors, design_region, grid_shape = make_demux_waveguides(fa)
        sorted_vals = sorted(demux_freqs.keys())
        target_freqs_hz = np.array([demux_freqs[v] for v in sorted_vals])
        initial_density = np.load(results_dir / "demux_density.npy")
        n_steps = 60_000
        save_path = str(results_dir / "demux_density_fdtd.npy")

    nx, ny = grid_shape
    print(f"Grid: {nx} x {ny} cells  (dx={DX*1e9:.1f} nm, dy={DY*1e9:.1f} nm)")
    print(f"Target frequencies ({len(target_freqs_hz)}):")
    for val, freq in zip(sorted_vals, target_freqs_hz):
        print(f"  val={val:+d}  {freq/1e12:.4f} THz")

    source_x_cell = max(int(input_wgs[0]["x_end"] / DX) - 3, PML_CELLS + 2)
    n_bg = make_n_background(grid_shape, input_wgs, output_wgs)

    print("Starting FDTD adjoint optimization...")
    print("  Initial density shape:", initial_density.shape)
    print("  Grid:", grid_shape)
    print("  n_steps per forward pass:", n_steps)
    print("  Gradient computed via JAX autodiff through lax.scan + jax.checkpoint")

    optimized_density = optimize_density(
        initial_density=initial_density,
        n_bg=n_bg,
        design_region=design_region,
        input_wg=input_wgs[0],
        output_monitors=monitors,
        target_freqs_hz=target_freqs_hz,
        n_steps=n_steps,
        source_x_cell=source_x_cell,
        n_iterations=300,
        learning_rate=0.02,
        save_path=save_path,
    )

    np.save(save_path, optimized_density)
    print("Optimization complete. Saved to", save_path)
    print("Run fdtdx_validation.py to validate.")


if __name__ == "__main__":
    main()
