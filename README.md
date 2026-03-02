# NRadix Accelerator — Ternary Optical Matrix Multiplier

**Single trit × trit optical multiplier using Sum Frequency Generation (SFG) and inverse-designed AWG routing on SiN/SiO₂ — scaled to a 9×9 photonic systolic array.**

> Multiplication is performed entirely by physics. No logic gates. No lookup tables. The output *color* is the answer.

---

## How It Works

Standard ternary computing failed commercially because encoding three states costs ~40× more transistors per trit than binary. This project sidesteps that entirely using wavelength-division encoding: each trit value is a laser frequency, and optical physics does the rest.

### The Core Trick — SFG as a Multiplier

Sum Frequency Generation (SFG) is a nonlinear optical process where two input photons combine to produce one output photon at their combined frequency:

```
f_out = f_A + f_B
```

If trit values are encoded as distinct input frequencies, then the output frequency *is* the product — determined entirely by the physics of the interaction, not by any circuit logic.

**Why unbalanced ternary {1, 2, 3}?**
With balanced {0, 1, 2}, any input involving 0 produces a degenerate SFG output — you can't distinguish which input was zero from the output frequency alone. {1, 2, 3} guarantees every pairwise sum is non-zero and uniquely identifiable. It also maps cleanly onto matrix multiply, where values are added (accumulated) after the optical multiply step.

---

## Wavelength Assignments

Three telecom C/L-band laser sources encode trit values:

| Trit | Frequency (THz) | Wavelength (nm) |
|:----:|:---------------:|:---------------:|
| 1    | 191.0           | 1569.59         |
| 2    | 194.0           | 1545.32         |
| 3    | 201.0           | 1491.54         |

SFG produces six unique product frequencies, one per possible output value:

| Product | SFG Freq (THz) | Wavelength (nm) | AWG Port |
|:-------:|:--------------:|:---------------:|:--------:|
| 1       | 382.0          | 784.97          | 0        |
| 2       | 385.0          | 778.83          | 1        |
| 4       | 388.0          | 772.78          | 2        |
| 3       | 392.0          | 764.88          | 3        |
| 6       | 395.0          | 759.06          | 4        |
| 9       | 402.0          | 745.77          | 5        |

**Minimum frequency separation: 3.0 THz** — well within AWG resolving capability.

> Note: The frequency ordering of products 3 and 4 is non-intuitive (product 4 lands at a *lower* frequency than product 3). This is a direct consequence of the non-uniform input spacing chosen to maximize channel separation.

---

## Architecture

### Single Trit × Trit NRIOC Module

```
trit_a, trit_b
  → LaserSource A, B        (encode trit value as telecom wavelength)
  → SFG unit                (f_out = f_A + f_B  →  product encoded as color)
  → AWG router              (route by wavelength to dedicated output port)
  → Photodetector array     (6 ports, one per unique product value)
  → integer product
```

### 9×9 2D Optical Systolic Array

81 NRIOC processing elements (PEs) arranged in a weight-stationary 9×9 grid.

- **Optical**: Each PE multiplies its weight × input via SFG + AWG — instantaneous within the clock cycle
- **Electronic**: Each PE accumulates products in a 7-bit register over systolic cycles

```
Dataflow (weight-stationary):
  Matrix A (9×9) pre-loaded — PE(i,j) holds A[i][j] as a trit
  Input vector b enters at column 0 with systolic skew
  After 17 cycles: output vector C = A × b is ready in accumulators
```

| Operation       | Cycles |
|:----------------|:------:|
| Matrix-vector   | 17     |
| Matrix-matrix   | 153    |

---

## Inverse Design

The AWG routing region is not hand-designed — it is optimized using inverse design to shape a SiN refractive index distribution that routes each of the 6 SFG product wavelengths to its dedicated output port.

**Method:** Beam Propagation Method (BPM), split-step Fourier, fully differentiable via JAX
**Why BPM over FDTD:** BPM has far better gradient flow for this routing problem (~250 propagation steps, each autodiffable vs. FDTD stiffness)

### Design Parameters

| Parameter          | Value                   |
|:-------------------|:------------------------|
| Grid spacing       | 80 nm                   |
| Domain size        | 20 × 30 µm              |
| Design region      | 15 × 25 µm              |
| Core material      | SiN (n = 2.2)           |
| Cladding material  | SiO₂ (n = 1.44)         |
| Waveguide width    | 500 nm                  |
| Optimizer          | Adam (JAX)              |
| Iterations         | 300                     |
| Beta schedule      | 1.0 → 12.0 (binarization)|
| Objective          | log(correct) − log(crosstalk) summed over 6 products |

---

## Validated Results

Optimized on **RunPod NVIDIA L4 24 GB** — 300 iterations, ~18 minutes.
All 9 input combinations tested post-optimization.

| A × B | Product | Port | Extinction Ratio | Power Fraction | Status |
|:-----:|:-------:|:----:|:----------------:|:--------------:|:------:|
| 1 × 1 | 1       | 0    | 12.2 dB          | 81.7%          | ✅ PASS |
| 1 × 2 | 2       | 1    | 12.7 dB          | 83.3%          | ✅ PASS |
| 1 × 3 | 3       | 3    | 14.8 dB          | 90.9%          | ✅ PASS |
| 2 × 1 | 2       | 1    | 12.7 dB          | 83.3%          | ✅ PASS |
| 2 × 2 | 4       | 2    | 11.5 dB          | 80.5%          | ✅ PASS |
| 2 × 3 | 6       | 4    | 13.3 dB          | 88.9%          | ✅ PASS |
| 3 × 1 | 3       | 3    | 14.8 dB          | 90.9%          | ✅ PASS |
| 3 × 2 | 6       | 4    | 13.3 dB          | 88.9%          | ✅ PASS |
| 3 × 3 | 9       | 5    | 14.0 dB          | 88.1%          | ✅ PASS |

**9/9 PASS** — extinction ratios 11.5–14.8 dB, power fractions 80–91%.

---

## Power Budget

| Component              | Power     |
|:-----------------------|----------:|
| 3 laser sources        | ~100 mW   |
| 486 InGaAs detectors   | ~1 mW     |
| 81 accumulators (7-bit)| ~4 mW     |
| Timing + I/O           | ~50 mW    |
| **Total**              | **~155 mW** |

**Throughput:** 1.6 GMAC/s
**Energy efficiency:** ~96 pJ/MAC

Laser sources dominate (~65% of power) and are shared across all 81 PEs.

---

## Repository Structure

```
NRadix_Accelerator/
├── math/
│   └── trit_multiplication.py        # Wavelength assignments, SFG math, port mapping
├── components/
│   └── nrioc_module.py               # Full NRIOC pipeline: laser → SFG → AWG → detect
├── architecture/
│   ├── systolic_array_2d.py          # 9×9 weight-stationary systolic array
│   ├── optical_systolic_array.py     # 1D prototype (reference)
│   ├── accumulator_bank.py
│   ├── timing_controller.py
│   ├── photodetector_array.py
│   ├── physical_layout.py
│   └── power_budget.py
├── simulation/
│   ├── trit_multiplier_inverse_design.py  # BPM optimizer (run on GPU)
│   └── results/
│       ├── validation_results.json        # 9/9 PASS results
│       └── optimization_log.json
├── tests/
│   └── test_systolic_array_2d.py
└── scripts/
    ├── aws_setup_fdtdx.sh            # AWS GPU setup for FDTDX
    └── gcp_setup_fdtdx.sh            # GCP GPU setup for FDTDX
```

---

## Running the Inverse Design

Requires JAX with CUDA 12. Recommended: RunPod with an L4 or A100.

```bash
pip install jax[cuda12] numpy matplotlib gdstk

cd NRadix_Accelerator/simulation
python trit_multiplier_inverse_design.py
```

Outputs:
- `results/design_density.npy` — optimized geometry (float array, 0–1)
- `results/optimization_history.json` — objective and gradient norms per iteration
- `results/validation_results.json` — per-combination extinction ratios and port routing
- `results/trit_multiplier.gds` — foundry-ready GDS layout (if gdstk installed)

For cloud GPU setup, see `scripts/aws_setup_fdtdx.sh` or `scripts/gcp_setup_fdtdx.sh`.

---

## Status

- [x] Wavelength assignment and SFG multiplication math
- [x] NRIOC module (full pipeline, software model)
- [x] BPM inverse design optimizer (JAX, differentiable)
- [x] Inverse design validated — 9/9 PASS, ER 11.5–14.8 dB
- [x] 9×9 systolic array architecture
- [x] Power budget and timing model
- [x] GDS export for foundry
- [ ] Full FDTDX electromagnetic validation (beyond BPM paraxial approximation)
- [ ] Multi-trit accumulation and carry logic
- [ ] Foundry tapeout

---

## Related Work

- **[optical-computing-workspace](https://github.com/jackwayne234/optical-computing-workspace)** — Broader NRadix accelerator workspace
- **[binary-optical-computer](https://github.com/jackwayne234/binary-optical-computer)** — Binary version for direct comparison
- **[binary-optical-chip](https://github.com/jackwayne234/binary-optical-chip)** — 2-wavelength binary photonic chip on lithium niobate

---

## Author

**Christopher Riner**
Independent Researcher | Active Duty U.S. Navy
ORCID: [0009-0008-9448-9033](https://orcid.org/0009-0008-9448-9033)
GitHub: [@jackwayne234](https://github.com/jackwayne234)
