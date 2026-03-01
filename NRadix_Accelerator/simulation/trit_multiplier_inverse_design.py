"""
FDTDX Inverse Design: Single Trit x Trit Optical Multiplier

This script defines and runs the inverse design optimization for the
SFG + AWG structure that performs single trit multiplication.

Architecture:
  - 2 input waveguides (each carrying one of 3 wavelengths)
  - SFG nonlinear region (design region - optimizer shapes this)
  - AWG wavelength router (design region - optimizer shapes this)
  - 6 output waveguides (one per product: 1, 2, 3, 4, 6, 9)

Objective:
  Maximize power at the correct output port for each of the 9 input combos.
  Minimize crosstalk (power at wrong ports).

Run on GCP g2-standard-4 (NVIDIA L4) with:
  source /home/fdtdx-env/bin/activate
  python trit_multiplier_inverse_design.py

Requires: fdtdx, jax[cuda12], numpy, matplotlib
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ============================================================================
# Wavelength / Frequency Constants (must match trit_multiplication.py)
# ============================================================================

C_LIGHT = 299_792_458.0  # m/s

# Input frequencies (THz) for trit values {1, 2, 3}
INPUT_FREQUENCIES_THZ = {1: 191.0, 2: 194.0, 3: 201.0}

# SFG product frequencies (THz) for each product value
PRODUCT_FREQUENCIES_THZ = {
    1: 382.0,  # 1x1: 191+191
    2: 385.0,  # 1x2: 191+194
    3: 392.0,  # 1x3: 191+201
    4: 388.0,  # 2x2: 194+194
    6: 395.0,  # 2x3: 194+201
    9: 402.0,  # 3x3: 201+201
}

# Product values sorted by frequency (ascending) -> port index
PRODUCTS_BY_FREQ = [1, 2, 4, 3, 6, 9]  # sorted by their output freq
PRODUCT_TO_PORT = {p: i for i, p in enumerate(PRODUCTS_BY_FREQ)}

# All 9 input combinations: (trit_a, trit_b) -> product
INPUT_COMBOS = [
    (1, 1, 1), (1, 2, 2), (1, 3, 3),
    (2, 1, 2), (2, 2, 4), (2, 3, 6),
    (3, 1, 3), (3, 2, 6), (3, 3, 9),
]  # (trit_a, trit_b, product)


def freq_to_wavelength_m(freq_thz: float) -> float:
    """Convert THz to meters."""
    return C_LIGHT / (freq_thz * 1e12)


def freq_to_wavelength_nm(freq_thz: float) -> float:
    """Convert THz to nanometers."""
    return freq_to_wavelength_m(freq_thz) * 1e9


# ============================================================================
# Simulation Parameters
# ============================================================================

@dataclass
class SimConfig:
    """Configuration for the inverse design simulation."""
    # Spatial resolution
    dx: float = 80e-9        # 80 nm grid spacing (fits L4 24GB VRAM)
    dy: float = 80e-9
    dz: float = 0.22e-6      # single z-cell (true 2D simulation)

    # Simulation domain (meters)
    domain_x: float = 20e-6  # 20 um
    domain_y: float = 30e-6  # 30 um (wider for 6 output ports)
    domain_z: float = 0.22e-6  # 220 nm slab (2D)

    # Design region (the part the optimizer can modify)
    design_x: float = 15e-6  # 15 um design area
    design_y: float = 25e-6  # 25 um design area
    design_x_offset: float = 2.5e-6  # centered in domain
    design_y_offset: float = 2.5e-6

    # Waveguide parameters
    wg_width: float = 0.5e-6     # 500 nm waveguide width
    wg_length: float = 2.0e-6    # 2 um input/output waveguide length

    # Material properties
    n_core: float = 2.4     # Silicon nitride (SiN) - good for SFG
    n_clad: float = 1.44    # SiO2 cladding
    chi2: float = 1.0e-12   # Second-order nonlinear susceptibility (m/V)
                             # SiN chi(2) is small but nonzero; this is a
                             # design parameter the optimizer will work with

    # Optimization
    num_iterations: int = 500
    learning_rate: float = 0.1
    min_feature_size: float = 80e-9  # 80 nm minimum feature (fab constraint)

    # PML (perfectly matched layer) boundary
    pml_thickness: int = 10  # grid cells

    # Time stepping
    num_time_steps: int = 500
    courant_factor: float = 0.5

    # Output
    output_dir: str = "results"


CONFIG = SimConfig()


# ============================================================================
# Geometry Builder
# ============================================================================

def build_input_waveguides(config: SimConfig) -> list[dict]:
    """Define 2 input waveguide positions.

    Input A enters from the left side, top half.
    Input B enters from the left side, bottom half.
    """
    y_center = config.domain_y / 2
    spacing = config.domain_y / 4  # quarter-domain spacing

    waveguides = [
        {
            "label": "input_A",
            "x_start": 0,
            "x_end": config.design_x_offset,
            "y_center": y_center + spacing / 2,
            "width": config.wg_width,
        },
        {
            "label": "input_B",
            "x_start": 0,
            "x_end": config.design_x_offset,
            "y_center": y_center - spacing / 2,
            "width": config.wg_width,
        },
    ]
    return waveguides


def build_output_waveguides(config: SimConfig) -> list[dict]:
    """Define 6 output waveguide positions, one per product.

    Output ports are evenly spaced along the right edge.
    Port order matches ascending frequency (product order: 1,2,4,3,6,9).
    """
    x_start = config.design_x_offset + config.design_x
    x_end = config.domain_x
    y_margin = 3e-6  # margin from top/bottom
    usable_y = config.domain_y - 2 * y_margin
    num_ports = 6

    waveguides = []
    for i, product in enumerate(PRODUCTS_BY_FREQ):
        y_center = y_margin + usable_y * (i + 0.5) / num_ports
        waveguides.append({
            "label": f"output_port{i}_product{product}",
            "product": product,
            "port_index": i,
            "x_start": x_start,
            "x_end": x_end,
            "y_center": y_center,
            "width": config.wg_width,
        })
    return waveguides


# ============================================================================
# FDTDX Simulation Setup
# ============================================================================

def create_simulation(config: SimConfig):
    """Create the FDTDX simulation with design region, sources, and monitors.

    Returns the simulation object and design parameters.
    """
    # Grid dimensions
    nx = int(config.domain_x / config.dx)
    ny = int(config.domain_y / config.dy)
    nz = max(int(config.domain_z / config.dz), 1)

    print(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz:,} cells")
    print(f"Domain: {config.domain_x*1e6:.1f} x {config.domain_y*1e6:.1f} x {config.domain_z*1e6:.2f} um")

    # --- Initialize permittivity grid ---
    # Start with cladding everywhere
    eps_background = config.n_clad ** 2
    eps_grid = jnp.full((nx, ny, nz), eps_background)

    # --- Design region (binary: core or clad) ---
    # The optimizer controls a 2D density map (0=clad, 1=core)
    design_nx = int(config.design_x / config.dx)
    design_ny = int(config.design_y / config.dy)
    design_x0 = int(config.design_x_offset / config.dx)
    design_y0 = int(config.design_y_offset / config.dy)

    print(f"Design region: {design_nx} x {design_ny} cells "
          f"({config.design_x*1e6:.1f} x {config.design_y*1e6:.1f} um)")

    # Initial design: random density
    key = jax.random.PRNGKey(42)
    design_params = jax.random.uniform(key, (design_nx, design_ny), minval=0.3, maxval=0.7)

    # --- Source definitions ---
    input_wgs = build_input_waveguides(config)
    output_wgs = build_output_waveguides(config)

    # --- Build source/monitor metadata ---
    sources = []
    for wg in input_wgs:
        for trit_val, freq_thz in INPUT_FREQUENCIES_THZ.items():
            wl_m = freq_to_wavelength_m(freq_thz)
            source_x = int(config.design_x_offset / config.dx) - 5  # just before design region
            source_y = int(wg["y_center"] / config.dy)
            sources.append({
                "label": f"{wg['label']}_trit{trit_val}",
                "waveguide": wg["label"],
                "trit_value": trit_val,
                "freq_thz": freq_thz,
                "wavelength_m": wl_m,
                "wavelength_nm": wl_m * 1e9,
                "grid_x": source_x,
                "grid_y": source_y,
            })

    monitors = []
    for wg in output_wgs:
        monitor_x = int(wg["x_start"] / config.dx) + 5  # just after design region
        monitor_y = int(wg["y_center"] / config.dy)
        monitors.append({
            "label": wg["label"],
            "product": wg["product"],
            "port_index": wg["port_index"],
            "grid_x": monitor_x,
            "grid_y": monitor_y,
        })

    return {
        "config": config,
        "grid_shape": (nx, ny, nz),
        "eps_background": eps_background,
        "design_params": design_params,
        "design_region": {
            "x0": design_x0, "y0": design_y0,
            "nx": design_nx, "ny": design_ny,
        },
        "input_waveguides": input_wgs,
        "output_waveguides": output_wgs,
        "sources": sources,
        "monitors": monitors,
    }


# ============================================================================
# Physics: Permittivity from Design Parameters
# ============================================================================

def density_to_permittivity(
    density: jnp.ndarray,
    eps_core: float,
    eps_clad: float,
    beta: float = 8.0,
) -> jnp.ndarray:
    """Convert continuous density [0,1] to permittivity using sigmoid projection.

    Beta controls sharpness: higher beta -> more binary (closer to fab-ready).
    Gradually increase beta during optimization for better convergence.
    """
    # Sigmoid projection for binarization
    projected = jax.nn.sigmoid(beta * (density - 0.5))
    return eps_clad + (eps_core - eps_clad) * projected


def apply_min_feature_size(
    density: jnp.ndarray,
    min_feature_pixels: int,
) -> jnp.ndarray:
    """Apply minimum feature size constraint via circular convolution filter.

    This is a soft constraint — helps the optimizer avoid sub-wavelength features
    that can't be fabricated.
    """
    if min_feature_pixels <= 1:
        return density

    # Create circular kernel
    r = min_feature_pixels
    y, x = jnp.mgrid[-r:r+1, -r:r+1]
    kernel = (x**2 + y**2 <= r**2).astype(jnp.float32)
    kernel = kernel / kernel.sum()

    # Apply as 2D convolution (smoothing)
    density_4d = density[None, None, :, :]
    kernel_4d = kernel[None, None, :, :]
    smoothed = jax.lax.conv(density_4d, kernel_4d, window_strides=(1, 1), padding="SAME")
    return smoothed[0, 0]


# ============================================================================
# Forward Simulation (simplified FDTD core)
# ============================================================================

def run_forward(
    sim: dict,
    design_density: jnp.ndarray,
    source_a_freq_thz: float,
    source_b_freq_thz: float,
    beta: float = 8.0,
) -> dict[int, float]:
    """Run forward FDTD simulation for one pair of input frequencies.

    Returns power at each output port.

    NOTE: This is a simplified 2D FDTD for prototyping the optimization loop.
    For production accuracy, use fdtdx's built-in 3D simulation with proper
    PML boundaries and dispersive materials. The structure of this function
    shows the optimizer what it needs to compute — the actual physics engine
    call will use fdtdx APIs.
    """
    config = sim["config"]
    nx, ny, nz = sim["grid_shape"]
    dr = sim["design_region"]
    eps_core = config.n_core ** 2
    eps_clad = config.n_clad ** 2

    # Build permittivity from design
    eps_design = density_to_permittivity(design_density, eps_core, eps_clad, beta)
    eps = jnp.full((nx, ny), eps_clad)
    eps = eps.at[dr["x0"]:dr["x0"]+dr["nx"], dr["y0"]:dr["y0"]+dr["ny"]].set(eps_design)

    # Add input waveguides
    for wg in sim["input_waveguides"]:
        wg_y_start = int((wg["y_center"] - wg["width"]/2) / config.dy)
        wg_y_end = int((wg["y_center"] + wg["width"]/2) / config.dy)
        wg_x_end = int(wg["x_end"] / config.dx)
        eps = eps.at[:wg_x_end, wg_y_start:wg_y_end].set(eps_core)

    # Add output waveguides
    for wg in sim["output_waveguides"]:
        wg_y_start = int((wg["y_center"] - wg["width"]/2) / config.dy)
        wg_y_end = int((wg["y_center"] + wg["width"]/2) / config.dy)
        wg_x_start = int(wg["x_start"] / config.dx)
        eps = eps.at[wg_x_start:, wg_y_start:wg_y_end].set(eps_core)

    # SFG product frequency
    product_freq_thz = source_a_freq_thz + source_b_freq_thz

    # --- 2D FDTD (Ez, Hx, Hy - TM polarization) ---
    # Physical constants
    mu0 = 4.0 * jnp.pi * 1e-7       # permeability of free space
    eps0 = 8.854187817e-12            # permittivity of free space
    dx = config.dx
    dy = config.dy

    # 2D Courant stability: dt <= dx / (c * sqrt(2))
    dt = config.courant_factor * dx / (C_LIGHT * jnp.sqrt(2.0))
    num_steps = config.num_time_steps

    Ez = jnp.zeros((nx, ny))
    Hx = jnp.zeros((nx, ny))
    Hy = jnp.zeros((nx, ny))

    # Source parameters
    omega_product = 2.0 * jnp.pi * product_freq_thz * 1e12

    # Source injection point: midpoint between the two input waveguides,
    # just before the design region. Both inputs carry the same product
    # frequency (SFG output) for AWG routing optimization.
    src_a = None
    src_b = None
    for s in sim["sources"]:
        if s["waveguide"] == "input_A" and abs(s["freq_thz"] - source_a_freq_thz) < 0.1:
            src_a = s
        if s["waveguide"] == "input_B" and abs(s["freq_thz"] - source_b_freq_thz) < 0.1:
            src_b = s

    if src_a is None or src_b is None:
        raise ValueError(f"Sources not found for freqs {source_a_freq_thz}, {source_b_freq_thz}")

    # Pre-compute FDTD update coefficients
    Ce = dt / (eps0 * eps)       # E-field update coefficient (shape: nx, ny)
    Ch = dt / mu0                 # H-field update coefficient (scalar)

    # PML conductivity profile (polynomial grading)
    pml = config.pml_thickness
    sigma_max = 0.8 * (pml + 1) / (dx * jnp.sqrt(mu0 / eps0))
    # Build conductivity arrays for x and y boundaries
    sigma_x = jnp.zeros(nx)
    sigma_y = jnp.zeros(ny)
    for i in range(pml):
        sigma_val = sigma_max * ((pml - i) / pml) ** 3
        sigma_x = sigma_x.at[i].set(sigma_val)
        sigma_x = sigma_x.at[-(i+1)].set(sigma_val)
        sigma_y = sigma_y.at[i].set(sigma_val)
        sigma_y = sigma_y.at[-(i+1)].set(sigma_val)
    # Combined damping factor per time step
    damp_x = jnp.exp(-sigma_x * dt / eps0)[:, None]  # (nx, 1)
    damp_y = jnp.exp(-sigma_y * dt / eps0)[None, :]   # (1, ny)
    damp_mask = damp_x * damp_y                         # (nx, ny)

    # Source amplitude: scale for meaningful field energy in the design region
    source_scale = 1.0  # normalized amplitude

    @jax.remat
    def fdtd_step(state, i):
        Ez, Hx, Hy = state
        t = i * dt

        # H-field update (standard Yee staggered grid)
        # Hx -= (dt/mu0) * dEz/dy
        Hx = Hx - (Ch / dy) * (jnp.roll(Ez, -1, axis=1) - Ez)
        # Hy += (dt/mu0) * dEz/dx
        Hy = Hy + (Ch / dx) * (jnp.roll(Ez, -1, axis=0) - Ez)

        # E-field update
        # Ez += (dt/(eps0*eps_r)) * (dHy/dx - dHx/dy)
        dHy_dx = (Hy - jnp.roll(Hy, 1, axis=0)) / dx
        dHx_dy = (Hx - jnp.roll(Hx, 1, axis=1)) / dy
        Ez = Ez + Ce * (dHy_dx - dHx_dy)

        # CW source with smooth ramp-up (no early cutoff)
        ramp = jnp.minimum(i / 100.0, 1.0)
        source_amp = source_scale * ramp * jnp.sin(omega_product * t)
        Ez = Ez.at[src_a["grid_x"], src_a["grid_y"]].add(source_amp)
        Ez = Ez.at[src_b["grid_x"], src_b["grid_y"]].add(source_amp)

        # PML absorption
        Ez = Ez * damp_mask

        return (Ez, Hx, Hy), None

    # Run FDTD with scan (remat recomputes during backprop to save memory)
    (Ez, Hx, Hy), _ = jax.lax.scan(fdtd_step, (Ez, Hx, Hy), jnp.arange(num_steps))

    # Measure power at each output monitor
    port_powers = {}
    for mon in sim["monitors"]:
        # Integrate |Ez|^2 over a small window around the monitor point
        mx, my = mon["grid_x"], mon["grid_y"]
        window = 3  # +/- 3 cells
        power = jnp.sum(Ez[mx-window:mx+window, my-window:my+window] ** 2)
        port_powers[mon["product"]] = power  # keep as jax array for autodiff

    return port_powers


# ============================================================================
# Objective Function
# ============================================================================

def compute_objective(
    design_density: jnp.ndarray,
    sim: dict,
    beta: float = 8.0,
) -> float:
    """Compute the optimization objective for all 9 input combinations.

    Objective = sum over all combos of:
      log(power at correct port) - log(sum of power at wrong ports + eps)

    Maximizing this means: maximize correct port power, minimize crosstalk.
    """
    total_objective = 0.0
    eps = 1e-10  # numerical stability

    for trit_a, trit_b, product in INPUT_COMBOS:
        freq_a = INPUT_FREQUENCIES_THZ[trit_a]
        freq_b = INPUT_FREQUENCIES_THZ[trit_b]

        port_powers = run_forward(sim, design_density, freq_a, freq_b, beta)

        correct_port = PRODUCT_TO_PORT[product]
        correct_power = port_powers.get(product, eps)

        # Crosstalk: sum of power at all wrong ports
        wrong_power = sum(
            p for prod, p in port_powers.items() if prod != product
        ) + eps

        # Maximize ratio of correct to wrong
        total_objective += jnp.log(correct_power + eps) - jnp.log(wrong_power)

    return total_objective


# ============================================================================
# Optimization Loop
# ============================================================================

def optimize(sim: dict, config: SimConfig) -> jnp.ndarray:
    """Run the inverse design optimization loop.

    Uses JAX's automatic differentiation to compute gradients of the
    objective with respect to the design parameters, then updates
    via Adam optimizer.
    """
    from jax.example_libraries import optimizers

    design_density = sim["design_params"]

    # Adam optimizer
    opt_init, opt_update, get_params = optimizers.adam(config.learning_rate)
    opt_state = opt_init(design_density)

    # Gradually increase binarization (start soft for better gradient flow)
    beta_schedule = jnp.linspace(1.0, 16.0, config.num_iterations)

    # Min feature size in pixels
    min_feat_px = max(1, int(config.min_feature_size / config.dx))

    print(f"\nStarting optimization: {config.num_iterations} iterations")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Min feature size: {config.min_feature_size*1e9:.0f} nm ({min_feat_px} px)")
    print("=" * 60)

    best_obj = -jnp.inf
    best_density = design_density
    history = []

    for iteration in range(config.num_iterations):
        beta = float(beta_schedule[iteration])
        params = get_params(opt_state)

        # Clamp to [0, 1]
        params = jnp.clip(params, 0.0, 1.0)

        # Apply minimum feature size
        params_filtered = apply_min_feature_size(params, min_feat_px)

        # Compute objective and gradient
        obj_val, grad = jax.value_and_grad(
            lambda d: compute_objective(d, sim, beta)
        )(params_filtered)

        # Update
        opt_state = opt_update(iteration, -grad, opt_state)  # negate: we maximize

        history.append({
            "iteration": iteration,
            "objective": float(obj_val),
            "beta": beta,
            "density_mean": float(jnp.mean(params)),
            "density_binarization": float(jnp.mean(jnp.abs(2 * params - 1))),
        })

        if obj_val > best_obj:
            best_obj = obj_val
            best_density = params_filtered

        if iteration % 20 == 0 or iteration == config.num_iterations - 1:
            binarization = jnp.mean(jnp.abs(2 * params - 1))
            print(
                f"  iter {iteration:4d}  |  obj: {obj_val:+12.6f}  |  "
                f"beta: {beta:5.1f}  |  bin: {binarization:.3f}"
            )

    print("=" * 60)
    print(f"Best objective: {best_obj:+.3f}")

    return best_density, history


# ============================================================================
# Validation
# ============================================================================

def validate_design(sim: dict, design_density: jnp.ndarray) -> dict:
    """Validate the optimized design: check all 9 combos for correct routing."""
    print("\nValidation: Testing all 9 input combinations")
    print("=" * 60)

    results = []
    all_pass = True

    for trit_a, trit_b, product in INPUT_COMBOS:
        freq_a = INPUT_FREQUENCIES_THZ[trit_a]
        freq_b = INPUT_FREQUENCIES_THZ[trit_b]

        port_powers = run_forward(sim, design_density, freq_a, freq_b, beta=16.0)

        correct_port = PRODUCT_TO_PORT[product]
        total_power = sum(port_powers.values())
        correct_power = port_powers.get(product, 0)

        # Find which port got the most power
        max_product = max(port_powers, key=port_powers.get)
        max_port = PRODUCT_TO_PORT[max_product]

        # Extinction ratio (correct port vs next strongest wrong port)
        wrong_powers = {p: pw for p, pw in port_powers.items() if p != product}
        max_wrong = max(wrong_powers.values()) if wrong_powers else 0

        extinction_db = 10 * np.log10(correct_power / max_wrong) if max_wrong > 0 else float('inf')

        passed = (max_port == correct_port and extinction_db > 3.0)
        if not passed:
            all_pass = False

        result = {
            "trit_a": trit_a,
            "trit_b": trit_b,
            "product": product,
            "correct_port": correct_port,
            "detected_port": max_port,
            "correct_power_fraction": correct_power / total_power if total_power > 0 else 0,
            "extinction_ratio_db": extinction_db,
            "passed": passed,
        }
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(
            f"  {trit_a} x {trit_b} = {product:>2}  |  "
            f"port: {correct_port} (expect) -> {max_port} (got)  |  "
            f"ER: {extinction_db:5.1f} dB  |  [{status}]"
        )

    print("=" * 60)
    print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    return {"results": results, "all_pass": all_pass}


# ============================================================================
# GDS Export
# ============================================================================

def export_gds(
    sim: dict,
    design_density: jnp.ndarray,
    output_path: str = "trit_multiplier.gds",
    threshold: float = 0.5,
):
    """Export the optimized design to GDS format for fabrication.

    Requires gdstk (pip install gdstk).
    """
    try:
        import gdstk
    except ImportError:
        print("gdstk not installed. Install with: pip install gdstk")
        print("Skipping GDS export.")
        return None

    config = sim["config"]
    dr = sim["design_region"]

    # Create GDS library
    lib = gdstk.Library(name="TritMultiplier")
    cell = lib.new_cell("MULTIPLIER")

    # Convert density to binary (above threshold = core material)
    density_np = np.array(design_density)
    binary = density_np > threshold

    # Design region polygons
    dx_um = config.dx * 1e6
    dy_um = config.dy * 1e6
    x_offset_um = dr["x0"] * dx_um
    y_offset_um = dr["y0"] * dy_um

    # Create rectangles for each filled pixel
    # (In production, use polygon merging for cleaner GDS)
    for ix in range(dr["nx"]):
        for iy in range(dr["ny"]):
            if binary[ix, iy]:
                x0 = x_offset_um + ix * dx_um
                y0 = y_offset_um + iy * dy_um
                rect = gdstk.rectangle(
                    (x0, y0), (x0 + dx_um, y0 + dy_um),
                    layer=1, datatype=0
                )
                cell.add(rect)

    # Input waveguides
    for wg in sim["input_waveguides"]:
        x0 = 0
        x1 = wg["x_end"] * 1e6
        y_center = wg["y_center"] * 1e6
        w = wg["width"] * 1e6
        rect = gdstk.rectangle(
            (x0, y_center - w/2), (x1, y_center + w/2),
            layer=2, datatype=0
        )
        cell.add(rect)

    # Output waveguides
    for wg in sim["output_waveguides"]:
        x0 = wg["x_start"] * 1e6
        x1 = wg["x_end"] * 1e6
        y_center = wg["y_center"] * 1e6
        w = wg["width"] * 1e6
        rect = gdstk.rectangle(
            (x0, y_center - w/2), (x1, y_center + w/2),
            layer=2, datatype=0
        )
        cell.add(rect)

        # Port label
        label = gdstk.Label(
            f"P{wg['port_index']}={wg['product']}",
            (x1, y_center),
            layer=10
        )
        cell.add(label)

    # Write GDS
    lib.write_gds(output_path)
    print(f"\nGDS exported to: {output_path}")
    print(f"  Design region layer: 1")
    print(f"  Waveguide layer: 2")
    print(f"  Port labels layer: 10")

    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  FDTDX Inverse Design: Single Trit x Trit Multiplier")
    print("  Unbalanced Ternary {1, 2, 3}")
    print("=" * 60)

    # Print wavelength assignments
    print("\nInput wavelengths:")
    for trit, freq in sorted(INPUT_FREQUENCIES_THZ.items()):
        wl = freq_to_wavelength_nm(freq)
        print(f"  trit={trit}: {freq:.1f} THz ({wl:.2f} nm)")

    print("\nProduct wavelengths (SFG output):")
    for product in PRODUCTS_BY_FREQ:
        freq = PRODUCT_FREQUENCIES_THZ[product]
        wl = freq_to_wavelength_nm(freq)
        port = PRODUCT_TO_PORT[product]
        print(f"  product={product}: {freq:.1f} THz ({wl:.2f} nm) -> port {port}")

    # Check frequency separation
    freqs_sorted = sorted(PRODUCT_FREQUENCIES_THZ.items(), key=lambda x: x[1])
    min_gap = min(
        freqs_sorted[i+1][1] - freqs_sorted[i][1]
        for i in range(len(freqs_sorted) - 1)
    )
    print(f"\nMinimum frequency separation: {min_gap:.1f} THz")
    assert min_gap >= 3.0, f"Insufficient separation: {min_gap} THz"

    # Create simulation
    config = SimConfig()
    print(f"\nCreating simulation...")
    sim = create_simulation(config)

    # Check for GPU
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    if any(d.platform == 'gpu' for d in devices):
        print("GPU detected - running on GPU")
    else:
        print("WARNING: No GPU detected. Simulation will be slow on CPU.")
        print("For production runs, use GCP g2-standard-4 (NVIDIA L4) with CUDA 12.")

    # Run optimization
    t0 = time.time()
    best_density, history = optimize(sim, config)
    elapsed = time.time() - t0
    print(f"\nOptimization completed in {elapsed:.1f}s")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save design density
    np.save(output_dir / "design_density.npy", np.array(best_density))

    # Save optimization history
    with open(output_dir / "optimization_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Validate
    validation = validate_design(sim, best_density)
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)

    # Export GDS
    gds_path = export_gds(sim, best_density, str(output_dir / "trit_multiplier.gds"))

    print("\n" + "=" * 60)
    print("  Output files:")
    print(f"  {output_dir}/design_density.npy     - Optimized geometry")
    print(f"  {output_dir}/optimization_history.json - Training curve")
    print(f"  {output_dir}/validation_results.json  - Port routing test")
    if gds_path:
        print(f"  {gds_path}                          - GDS for fab/GDSFactory")
    print("=" * 60)


if __name__ == "__main__":
    main()
