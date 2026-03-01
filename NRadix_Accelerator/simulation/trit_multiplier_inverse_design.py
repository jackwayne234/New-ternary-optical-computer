"""
Inverse Design: Single Trit x Trit Optical Multiplier — AWG Wavelength Router

Optimizes the refractive index distribution of a photonic structure to route
6 different SFG product wavelengths to 6 output ports.

Method: Beam Propagation Method (BPM) with split-step Fourier.
  - Propagates fields along x through the design region
  - One propagation per product frequency (6 total, not 9 — symmetric combos share freqs)
  - ~250 propagation steps (one per x-slice), each fully differentiable
  - Much better gradient flow than FDTD for this routing optimization

Architecture:
  - Input: SFG product beam enters from left (one frequency at a time)
  - Design region: optimizer shapes refractive index to act as wavelength router
  - Output: 6 ports on the right, one per product value {1, 2, 3, 4, 6, 9}

Run on RunPod (NVIDIA L4) with:
  python trit_multiplier_inverse_design.py

Requires: jax[cuda12], numpy, matplotlib
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

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

# Unique products (avoid duplicate simulations for symmetric combos)
UNIQUE_PRODUCTS = [1, 2, 3, 4, 6, 9]


def freq_to_wavelength_m(freq_thz: float) -> float:
    return C_LIGHT / (freq_thz * 1e12)


def freq_to_wavelength_nm(freq_thz: float) -> float:
    return freq_to_wavelength_m(freq_thz) * 1e9


# ============================================================================
# Simulation Parameters
# ============================================================================

@dataclass
class SimConfig:
    # Spatial resolution
    dx: float = 80e-9        # 80 nm grid spacing
    dy: float = 80e-9

    # Simulation domain (meters)
    domain_x: float = 20e-6  # 20 um propagation length
    domain_y: float = 30e-6  # 30 um width (6 output ports)

    # Design region (the part the optimizer can modify)
    design_x: float = 15e-6  # 15 um design area
    design_y: float = 25e-6  # 25 um design area
    design_x_offset: float = 2.5e-6
    design_y_offset: float = 2.5e-6

    # Waveguide parameters
    wg_width: float = 0.5e-6     # 500 nm waveguide width

    # Material properties
    n_core: float = 2.2     # SiN core
    n_clad: float = 1.44    # SiO2 cladding

    # Optimization
    num_iterations: int = 300
    learning_rate: float = 0.01

    # Output
    output_dir: str = "results"


CONFIG = SimConfig()


# ============================================================================
# Geometry Builder
# ============================================================================

def build_input_waveguides(config: SimConfig) -> list[dict]:
    y_center = config.domain_y / 2
    spacing = config.domain_y / 4
    return [
        {"label": "input_A", "x_end": config.design_x_offset,
         "y_center": y_center + spacing / 2, "width": config.wg_width},
        {"label": "input_B", "x_end": config.design_x_offset,
         "y_center": y_center - spacing / 2, "width": config.wg_width},
    ]


def build_output_waveguides(config: SimConfig) -> list[dict]:
    x_start = config.design_x_offset + config.design_x
    y_margin = 3e-6
    usable_y = config.domain_y - 2 * y_margin
    num_ports = 6

    waveguides = []
    for i, product in enumerate(PRODUCTS_BY_FREQ):
        y_center = y_margin + usable_y * (i + 0.5) / num_ports
        waveguides.append({
            "label": f"output_port{i}_product{product}",
            "product": product, "port_index": i,
            "x_start": x_start, "x_end": config.domain_x,
            "y_center": y_center, "width": config.wg_width,
        })
    return waveguides


# ============================================================================
# Simulation Setup
# ============================================================================

def create_simulation(config: SimConfig):
    nx = int(config.domain_x / config.dx)
    ny = int(config.domain_y / config.dy)

    print(f"Grid: {nx} x {ny} = {nx*ny:,} cells")
    print(f"Domain: {config.domain_x*1e6:.1f} x {config.domain_y*1e6:.1f} um")

    design_nx = int(config.design_x / config.dx)
    design_ny = int(config.design_y / config.dy)
    design_x0 = int(config.design_x_offset / config.dx)
    design_y0 = int(config.design_y_offset / config.dy)

    print(f"Design region: {design_nx} x {design_ny} cells "
          f"({config.design_x*1e6:.1f} x {config.design_y*1e6:.1f} um)")
    print(f"BPM propagation steps: {nx}")

    key = jax.random.PRNGKey(42)
    design_params = jax.random.uniform(key, (design_nx, design_ny), minval=0.4, maxval=0.6)

    input_wgs = build_input_waveguides(config)
    output_wgs = build_output_waveguides(config)

    monitors = []
    for wg in output_wgs:
        monitor_y = int(wg["y_center"] / config.dy)
        monitors.append({
            "label": wg["label"], "product": wg["product"],
            "port_index": wg["port_index"], "grid_y": monitor_y,
        })

    return {
        "config": config,
        "grid_shape": (nx, ny),
        "design_params": design_params,
        "design_region": {
            "x0": design_x0, "y0": design_y0,
            "nx": design_nx, "ny": design_ny,
        },
        "input_waveguides": input_wgs,
        "output_waveguides": output_wgs,
        "monitors": monitors,
    }


# ============================================================================
# Physics: Density to Refractive Index
# ============================================================================

def density_to_index(density, n_core, n_clad, beta=8.0):
    projected = jax.nn.sigmoid(beta * (density - 0.5))
    return n_clad + (n_core - n_clad) * projected


# ============================================================================
# BPM Forward Simulation
# ============================================================================

def run_forward_bpm(sim, design_density, product_freq_thz, beta=8.0):
    """Propagate a beam at the given product frequency through the device.

    Uses split-step Fourier BPM:
      1. Apply phase screen (refractive index at current x-slice)
      2. Propagate in free space (FFT method)

    Returns power at each output port.
    """
    config = sim["config"]
    nx, ny = sim["grid_shape"]
    dr = sim["design_region"]
    dx = config.dx
    dy = config.dy

    # Build refractive index profile from design
    n_design = density_to_index(design_density, config.n_core, config.n_clad, beta)
    n_profile = jnp.full((nx, ny), config.n_clad)
    n_profile = n_profile.at[dr["x0"]:dr["x0"]+dr["nx"],
                             dr["y0"]:dr["y0"]+dr["ny"]].set(n_design)

    # Add input waveguides to index profile
    for wg in sim["input_waveguides"]:
        wg_y_start = int((wg["y_center"] - wg["width"]/2) / dy)
        wg_y_end = int((wg["y_center"] + wg["width"]/2) / dy)
        wg_x_end = int(wg["x_end"] / dx)
        n_profile = n_profile.at[:wg_x_end, wg_y_start:wg_y_end].set(config.n_core)

    # Add output waveguides
    for wg in sim["output_waveguides"]:
        wg_y_start = int((wg["y_center"] - wg["width"]/2) / dy)
        wg_y_end = int((wg["y_center"] + wg["width"]/2) / dy)
        wg_x_start = int(wg["x_start"] / dx)
        n_profile = n_profile.at[wg_x_start:, wg_y_start:wg_y_end].set(config.n_core)

    # BPM parameters
    wavelength = C_LIGHT / (product_freq_thz * 1e12)
    k0 = 2.0 * jnp.pi / wavelength
    n_ref = (config.n_core + config.n_clad) / 2.0  # reference index

    # Spatial frequency axis for y-direction
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, dy)

    # Free-space propagation kernel (paraxial approximation)
    # kx = k0*n_ref - ky^2 / (2*k0*n_ref)  (Fresnel approximation)
    prop_phase = jnp.exp(-1j * ky**2 * dx / (2.0 * k0 * n_ref))

    # Absorbing boundary in y (prevent wraparound from FFT)
    absorber = jnp.ones(ny)
    abs_width = 20  # cells
    for i in range(abs_width):
        a = 0.5 * (1.0 - jnp.cos(jnp.pi * i / abs_width))  # Hann window
        absorber = absorber.at[i].set(a)
        absorber = absorber.at[-(i+1)].set(a)

    # Initial field: launch from both input waveguides
    y_coords = jnp.arange(ny) * dy
    input_a_y = sim["input_waveguides"][0]["y_center"]
    input_b_y = sim["input_waveguides"][1]["y_center"]
    beam_w = config.wg_width  # beam width matches waveguide

    E = (jnp.exp(-((y_coords - input_a_y) / beam_w)**2) +
         jnp.exp(-((y_coords - input_b_y) / beam_w)**2))
    E = E.astype(jnp.complex64)

    # Propagate through all x-slices
    def bpm_step(E, x_idx):
        # Phase screen: effect of refractive index at this slice
        dn = n_profile[x_idx, :] - n_ref
        phase_screen = jnp.exp(1j * k0 * dn * dx).astype(jnp.complex64)
        E = E * phase_screen

        # Free-space propagation (split-step Fourier)
        E_k = jnp.fft.fft(E)
        E_k = E_k * prop_phase.astype(jnp.complex64)
        E = jnp.fft.ifft(E_k)

        # Absorbing boundary
        E = E * absorber

        return E, None

    E_final, _ = jax.lax.scan(bpm_step, E, jnp.arange(nx))

    # Measure power at each output port
    port_powers = {}
    for mon in sim["monitors"]:
        my = mon["grid_y"]
        # Integrate |E|^2 over waveguide width at output
        hw = max(int(config.wg_width / dy / 2), 2)  # half-width in pixels
        y_lo = jnp.maximum(my - hw, 0)
        y_hi = jnp.minimum(my + hw, ny)
        power = jnp.sum(jnp.abs(E_final[y_lo:y_hi])**2)
        port_powers[mon["product"]] = power

    return port_powers


# ============================================================================
# Objective Function
# ============================================================================

def compute_objective(design_density, sim, beta=8.0):
    """Maximize power at correct port, minimize crosstalk, for all 6 products."""
    total_objective = 0.0
    eps = 1e-8

    for product in UNIQUE_PRODUCTS:
        freq = PRODUCT_FREQUENCIES_THZ[product]
        port_powers = run_forward_bpm(sim, design_density, freq, beta)

        correct_power = port_powers[product]

        # Crosstalk: power at all wrong ports
        wrong_power = sum(
            p for prod, p in port_powers.items() if prod != product
        ) + eps

        total_objective += jnp.log(correct_power + eps) - jnp.log(wrong_power)

    return total_objective


# ============================================================================
# Optimization Loop
# ============================================================================

def optimize(sim, config):
    from jax.example_libraries import optimizers

    design_density = sim["design_params"]

    opt_init, opt_update, get_params = optimizers.adam(config.learning_rate)
    opt_state = opt_init(design_density)

    # Beta schedule: start soft, end sharp
    beta_schedule = jnp.linspace(1.0, 12.0, config.num_iterations)

    print(f"\nStarting optimization: {config.num_iterations} iterations")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Method: BPM (split-step Fourier)")
    print(f"Frequencies: 6 unique products per iteration")
    print("=" * 70)

    best_obj = -jnp.inf
    best_density = design_density
    history = []

    for iteration in range(config.num_iterations):
        beta = float(beta_schedule[iteration])
        params = get_params(opt_state)
        params = jnp.clip(params, 0.0, 1.0)

        obj_val, grad = jax.value_and_grad(
            lambda d: compute_objective(d, sim, beta)
        )(params)

        # Negate gradient because we maximize
        opt_state = opt_update(iteration, -grad, opt_state)

        grad_norm = float(jnp.linalg.norm(grad))
        obj_float = float(obj_val)

        history.append({
            "iteration": iteration,
            "objective": obj_float,
            "beta": beta,
            "grad_norm": grad_norm,
            "density_mean": float(jnp.mean(params)),
            "density_binarization": float(jnp.mean(jnp.abs(2 * params - 1))),
        })

        if obj_val > best_obj:
            best_obj = obj_val
            best_density = params

        if iteration % 10 == 0 or iteration == config.num_iterations - 1:
            binarization = jnp.mean(jnp.abs(2 * params - 1))
            print(
                f"  iter {iteration:4d}  |  obj: {obj_float:+10.4f}  |  "
                f"|grad|: {grad_norm:.2e}  |  beta: {beta:4.1f}  |  "
                f"bin: {binarization:.3f}"
            )

    print("=" * 70)
    print(f"Best objective: {float(best_obj):+.4f}")

    return best_density, history


# ============================================================================
# Validation
# ============================================================================

def validate_design(sim, design_density):
    print("\nValidation: Testing all 9 input combinations")
    print("=" * 70)

    results = []
    all_pass = True

    for trit_a, trit_b, product in INPUT_COMBOS:
        freq_a = INPUT_FREQUENCIES_THZ[trit_a]
        freq_b = INPUT_FREQUENCIES_THZ[trit_b]
        product_freq = freq_a + freq_b

        port_powers = run_forward_bpm(sim, design_density, product_freq, beta=12.0)

        correct_port = PRODUCT_TO_PORT[product]
        total_power = sum(float(p) for p in port_powers.values())
        correct_power = float(port_powers.get(product, 0))

        max_product = max(port_powers, key=lambda k: float(port_powers[k]))
        max_port = PRODUCT_TO_PORT[max_product]

        wrong_powers = {p: float(pw) for p, pw in port_powers.items() if p != product}
        max_wrong = max(wrong_powers.values()) if wrong_powers else 0

        if max_wrong > 0 and correct_power > 0:
            extinction_db = 10 * np.log10(correct_power / max_wrong)
        else:
            extinction_db = float('inf') if correct_power > 0 else float('-inf')

        passed = (max_port == correct_port and extinction_db > 3.0)
        if not passed:
            all_pass = False

        result = {
            "trit_a": trit_a, "trit_b": trit_b, "product": product,
            "correct_port": correct_port, "detected_port": max_port,
            "correct_power_fraction": correct_power / total_power if total_power > 0 else 0,
            "extinction_ratio_db": extinction_db, "passed": passed,
        }
        results.append(result)

        status = "PASS" if passed else "FAIL"
        pct = 100 * correct_power / total_power if total_power > 0 else 0
        print(
            f"  {trit_a} x {trit_b} = {product:>2}  |  "
            f"port: {correct_port} -> {max_port}  |  "
            f"ER: {extinction_db:+6.1f} dB  |  pwr: {pct:5.1f}%  |  [{status}]"
        )

    print("=" * 70)
    n_pass = sum(1 for r in results if r["passed"])
    print(f"Result: {n_pass}/9 PASS {'— ALL PASS!' if all_pass else ''}")

    return {"results": results, "all_pass": all_pass}


# ============================================================================
# GDS Export
# ============================================================================

def export_gds(sim, design_density, output_path="trit_multiplier.gds", threshold=0.5):
    try:
        import gdstk
    except ImportError:
        print("gdstk not installed — skipping GDS export")
        return None

    config = sim["config"]
    dr = sim["design_region"]

    lib = gdstk.Library(name="TritMultiplier")
    cell = lib.new_cell("MULTIPLIER")

    density_np = np.array(design_density)
    binary = density_np > threshold

    dx_um = config.dx * 1e6
    dy_um = config.dy * 1e6
    x_offset_um = dr["x0"] * dx_um
    y_offset_um = dr["y0"] * dy_um

    for ix in range(dr["nx"]):
        for iy in range(dr["ny"]):
            if binary[ix, iy]:
                x0 = x_offset_um + ix * dx_um
                y0 = y_offset_um + iy * dy_um
                cell.add(gdstk.rectangle(
                    (x0, y0), (x0 + dx_um, y0 + dy_um), layer=1))

    for wg in sim["input_waveguides"]:
        y_c = wg["y_center"] * 1e6
        w = wg["width"] * 1e6
        cell.add(gdstk.rectangle((0, y_c - w/2), (wg["x_end"]*1e6, y_c + w/2), layer=2))

    for wg in sim["output_waveguides"]:
        y_c = wg["y_center"] * 1e6
        w = wg["width"] * 1e6
        cell.add(gdstk.rectangle(
            (wg["x_start"]*1e6, y_c - w/2), (wg["x_end"]*1e6, y_c + w/2), layer=2))
        cell.add(gdstk.Label(f"P{wg['port_index']}={wg['product']}", (wg["x_end"]*1e6, y_c), layer=10))

    lib.write_gds(output_path)
    print(f"\nGDS exported: {output_path} (design=L1, waveguides=L2, labels=L10)")
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Inverse Design: Trit Multiplier Wavelength Router (BPM)")
    print("  Unbalanced Ternary {1, 2, 3}")
    print("=" * 70)

    print("\nInput wavelengths:")
    for trit, freq in sorted(INPUT_FREQUENCIES_THZ.items()):
        print(f"  trit={trit}: {freq:.1f} THz ({freq_to_wavelength_nm(freq):.2f} nm)")

    print("\nProduct wavelengths (SFG output):")
    for product in PRODUCTS_BY_FREQ:
        freq = PRODUCT_FREQUENCIES_THZ[product]
        print(f"  product={product}: {freq:.1f} THz ({freq_to_wavelength_nm(freq):.2f} nm) -> port {PRODUCT_TO_PORT[product]}")

    freqs_sorted = sorted(PRODUCT_FREQUENCIES_THZ.items(), key=lambda x: x[1])
    min_gap = min(freqs_sorted[i+1][1] - freqs_sorted[i][1] for i in range(len(freqs_sorted) - 1))
    print(f"\nMinimum frequency separation: {min_gap:.1f} THz")
    assert min_gap >= 3.0, f"Insufficient separation: {min_gap} THz"

    config = SimConfig()
    print(f"\nCreating simulation...")
    sim = create_simulation(config)

    devices = jax.devices()
    print(f"JAX devices: {devices}")
    if any(d.platform == 'gpu' for d in devices):
        print("GPU detected — running on GPU")
    else:
        print("WARNING: No GPU detected. Will be slow on CPU.")

    t0 = time.time()
    best_density, history = optimize(sim, config)
    elapsed = time.time() - t0
    print(f"\nOptimization completed in {elapsed:.1f}s ({elapsed/config.num_iterations:.1f}s/iter)")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "design_density.npy", np.array(best_density))
    with open(output_dir / "optimization_history.json", "w") as f:
        json.dump(history, f, indent=2)

    validation = validate_design(sim, best_density)
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)

    gds_path = export_gds(sim, best_density, str(output_dir / "trit_multiplier.gds"))

    print("\n" + "=" * 70)
    print("  Output files:")
    print(f"  {output_dir}/design_density.npy       — Optimized geometry")
    print(f"  {output_dir}/optimization_history.json — Training curve")
    print(f"  {output_dir}/validation_results.json   — Port routing test")
    if gds_path:
        print(f"  {gds_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
