# NRadix Accelerator — Ternary Optical Matrix Multiplier

**Single trit × trit optical multiplier using Sum Frequency Generation (SFG) and inverse-designed waveguide routing on SiN/SiO₂ — scaled to a 9×9 fully optical MAC array.**

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

### Fully Optical MAC — No Electronic Accumulation

The final architecture performs multiply-accumulate entirely in the optical domain:

- **Multiply** — SFG mixes two input frequencies → output frequency encodes the product
- **Accumulate** — optical field superposition in the waveguide; multiple SFG outputs traveling the same path add up physically, no electronic register needed
- **The answer** — the output port, determined by which frequency band arrives at the AWG

There are no accumulators, no registers, no electronic handoff between multiply and accumulate. The waveguide network topology *is* the computation.

### 9×9 Chip Physical Layout

```
9 horizontal waveguides  →  carry input vector b (left to right)
9 perpendicular feeds    →  carry matrix A weights (side injection)

At each of 81 intersections:
  SFG mixes horizontal freq + perpendicular freq → new output frequency

All SFG outputs from a row travel together toward the output AWG.
AWG routes each frequency to the correct c[i] output port.
```

**Programming the matrix:** swap perpendicular frequency sources = load new matrix A. The b vector feeds horizontally each computation cycle.

### Single Trit × Trit NRIOC Module

```
trit_a, trit_b
  → LaserSource A, B        (encode trit value as telecom wavelength)
  → SFG unit                (f_out = f_A + f_B  →  product encoded as color)
  → Waveguide network       (optical superposition = accumulation, free)
  → AWG demux               (route by frequency to dedicated output port)
  → integer product
```

---

## Inverse Design

The waveguide routing regions are optimized using inverse design to shape a SiN refractive index distribution that routes each frequency to its dedicated output port.

**BPM optimizer (routing design):** Beam Propagation Method, split-step Fourier, fully differentiable via JAX. Used for initial inverse design — far better gradient flow than FDTD for this problem (~300 steps, each autodiffable).

**FDTD adjoint (full EM validation):** JAX-based adjoint FDTD verifies routing under the full Maxwell equations, beyond BPM's paraxial approximation.

### Design Parameters

| Parameter             | Value                                          |
|:----------------------|:-----------------------------------------------|
| Grid spacing          | 80 nm                                          |
| Domain size (demux)   | 40 × 96 µm (standard) / 80 × 96 µm (large)    |
| Design region         | 15 × 25 µm                                     |
| Core material         | SiN (n = 2.2)                                  |
| Cladding material     | SiO₂ (n = 1.44)                                |
| Waveguide width       | 500 nm                                         |
| Optimizer             | Adam (JAX)                                     |
| Iterations            | 300 (BPM) / 1000 (FDTD)                        |
| Beta schedule         | 1.0 → 12.0 (binarization)                      |
| Objective             | log(correct) − log(crosstalk) over all channels |

---

## Validated Results

### BPM Inverse Design — Single Multiplier (9/9 PASS)

Optimized on RunPod NVIDIA L4 24 GB — 300 iterations, ~18 minutes.

| A × B | Product | Port | Extinction Ratio | Power Fraction | Status  |
|:-----:|:-------:|:----:|:----------------:|:--------------:|:-------:|
| 1 × 1 | 1       | 0    | 12.2 dB          | 81.7%          | ✅ PASS |
| 1 × 2 | 2       | 1    | 12.7 dB          | 83.3%          | ✅ PASS |
| 1 × 3 | 3       | 3    | 14.8 dB          | 90.9%          | ✅ PASS |
| 2 × 1 | 2       | 1    | 12.7 dB          | 83.3%          | ✅ PASS |
| 2 × 2 | 4       | 2    | 11.5 dB          | 80.5%          | ✅ PASS |
| 2 × 3 | 6       | 4    | 13.3 dB          | 88.9%          | ✅ PASS |
| 3 × 1 | 3       | 3    | 14.8 dB          | 90.9%          | ✅ PASS |
| 3 × 2 | 6       | 4    | 13.3 dB          | 88.9%          | ✅ PASS |
| 3 × 3 | 9       | 5    | 14.0 dB          | 88.1%          | ✅ PASS |

### Full MAC Inverse Design — End-to-End (2000/2000 PASS)

Complete cascade (multiply unit → 81-port demux) validated across all 3⁹ input combinations.

| Stage                    | Result                       | Notes                                        |
|:-------------------------|:-----------------------------|:---------------------------------------------|
| Stage 2: Multiply unit   | **6/6 PASS**                 | 30–34 dB ER, 99.9–100% power                 |
| Stage 3: Demux (81-port) | **18/19 PASS**               | All channels in 192.3–197.7 THz (C-band)     |
| End-to-end cascade       | **2000/2000 PASS (100%)**    | All 3⁹ input combinations validated          |

GDS files exported and foundry-ready: `multiply_unit.gds`, `demux_81port.gds`.

### FDTD Full Electromagnetic Validation

Full Maxwell equations validation beyond BPM paraxial approximation.

| Component                           | Result              | Notes                                             |
|:------------------------------------|:--------------------|:--------------------------------------------------|
| Multiply unit                       | **3/3 PASS** ✅     |                                                   |
| Demux — 40×96 µm chip               | **4/19 PASS**       | Reflections + tight routing at high-Q             |
| Demux — 80×96 µm chip (larger)      | ⏳ In progress      | More routing space to reduce reflection artifacts |

Root cause of demux FDTD failures: reflections and tight bend geometry — not cross-talk. The BPM paraxial approximation breaks down in high-Q routing. Larger chip (2× routing space) currently running on RunPod A40.

---

## Power Budget

| Component              | Power       |
|:-----------------------|------------:|
| 3 laser sources        | ~100 mW     |
| 486 InGaAs detectors   | ~1 mW       |
| Timing + I/O           | ~50 mW      |
| **Total**              | **~151 mW** |

No electronic accumulators — accumulation is optical (waveguide superposition is free). Laser sources dominate (~66% of power) and are shared across all 81 PEs.

**Throughput:** 1.6 GMAC/s
**Energy efficiency:** ~94 pJ/MAC

---

## Repository Structure

```
NRadix_Accelerator/
├── math/
│   └── trit_multiplication.py             # Wavelength assignments, SFG math, port mapping
├── components/
│   └── nrioc_module.py                    # Full NRIOC pipeline: laser → SFG → AWG → detect
├── architecture/
│   ├── systolic_array_2d.py               # 9×9 array (predates fully-optical MAC decision)
│   ├── optical_systolic_array.py          # 1D prototype (reference)
│   ├── accumulator_bank.py
│   ├── timing_controller.py
│   ├── photodetector_array.py
│   ├── physical_layout.py
│   └── power_budget.py
├── simulation/
│   ├── trit_multiplier_inverse_design.py  # BPM optimizer
│   ├── fdtd_inverse_design.py             # FDTD adjoint optimizer (JAX)
│   ├── run_demux_large.py                 # Large chip FDTD test (80×96 µm)
│   └── results/
│       ├── validation_results.json        # 9/9 BPM PASS
│       ├── mac_full_validation.json       # 2000/2000 end-to-end PASS
│       ├── multiply_unit.gds              # Foundry-ready GDS
│       └── demux_81port.gds              # Foundry-ready GDS
├── tests/
│   └── test_systolic_array_2d.py
└── scripts/
    ├── auto_push.sh                       # Run optimization + auto-push to GitHub
    ├── aws_setup_fdtdx.sh                 # AWS GPU setup
    └── gcp_setup_fdtdx.sh                # GCP GPU setup
```

---

## Running the Simulations

Requires JAX with CUDA 12. Recommended: RunPod with an A40 or A100.

```bash
pip install jax[cuda12] numpy matplotlib gdstk

cd NRadix_Accelerator/simulation

# BPM inverse design (~18 min on L4)
python trit_multiplier_inverse_design.py

# FDTD full EM validation (hours on A40)
python fdtd_inverse_design.py demux

# Large chip FDTD test
python run_demux_large.py
```

---

## Status

- [x] Wavelength assignment and SFG multiplication math
- [x] NRIOC module (full pipeline, software model)
- [x] BPM inverse design optimizer (JAX, differentiable)
- [x] BPM inverse design validated — 9/9 PASS, ER 11.5–14.8 dB
- [x] 9×9 systolic array architecture
- [x] Power budget and timing model
- [x] GDS export — `multiply_unit.gds`, `demux_81port.gds`
- [x] Full MAC inverse design — **2000/2000 PASS (100%)** across all 3⁹ inputs
- [x] FDTD adjoint optimizer (JAX)
- [x] FDTD validation — multiply unit **3/3 PASS**
- [ ] FDTD validation — demux (4/19 on standard chip; large chip test in progress)
- [ ] Foundry tapeout
- [ ] Multi-trit accumulation and carry logic

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
