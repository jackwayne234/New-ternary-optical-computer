# Ternary Optical Computer — NRadix Accelerator

Role: Project knowledge base for CJ's ternary optical computing work.

---

## Project Overview

**N-Radix**: Open-source wavelength-division ternary optical AI accelerator on lithium niobate.
The core insight: multiplication is performed *by physics* (SFG), not by logic gates.

**GitHub repos:**
- `jackwayne234/New-ternary-optical-computer` — Main: single trit×trit optical multiplier (BPM inverse design)
- `jackwayne234/optical-computing-workspace` — NRadix workspace (broader accelerator architecture)
- `jackwayne234/binary-optical-computer` — Binary version for comparison
- `jackwayne234/binary-optical-chip` — Binary chip (2-wavelength photonic, lithium niobate)

---

## Physics

**Unbalanced ternary: {1, 2, 3}**

**Sum Frequency Generation (SFG):**
- Two input photons (freq f_A, f_B) → one output photon at f_A + f_B
- Encoding trit values as optical frequencies makes frequency addition = multiplication

**Input frequency assignments (final/FINAL mapping):**
| Trit | Freq (THz) | Wavelength (nm) |
|------|-----------|-----------------|
| 1    | 191.0     | 1569.59         |
| 2    | 194.0     | 1545.32         |
| 3    | 201.0     | 1491.54         |

**SFG Product frequencies:**
| Product | Freq (THz) | AWG Port |
|---------|-----------|----------|
| 1       | 382.0     | 0        |
| 2       | 385.0     | 1        |
| 4       | 388.0     | 2        |
| 3       | 392.0     | 3        |
| 6       | 395.0     | 4        |
| 9       | 402.0     | 5        |

Min freq separation: **3.0 THz** (excellent for AWG routing).
Note: product 3 (391+201) and product 4 (194+194) are the tightest pair — careful frequency selection was required.

---

## Architecture

**Single trit×trit NRIOC module:**
```
trit_a, trit_b
  → LaserSource A, B (encode as wavelengths)
  → SFG unit (f_out = f_a + f_b)
  → AWG router (routes color to port)
  → Photodetector array (6 ports → electronic)
  → product integer
```

**9×9 2D Optical Systolic Array:**
- 81 PEs, weight-stationary dataflow
- Matrix-vector: C = A×b in **17 clock cycles**
- Matrix-matrix: C = A×B in **153 clock cycles** (9 columns)
- All PEs operate in parallel; optical multiply is instantaneous within cycle

---

## Inverse Design (BPM Simulation)

**Method:** Beam Propagation Method (split-step Fourier), differentiable via JAX

**Design region:** 15×25 µm, 80 nm grid spacing (250×375 cells)
- Optimizer shapes refractive index distribution to route 6 SFG product wavelengths to 6 ports

**Materials:**
- Core: SiN (n=2.2)
- Cladding: SiO₂ (n=1.44)

**Run on:** RunPod NVIDIA L4 24GB

**Validated results (2026-03-01):**
- 300 iterations, 18 min total
- All 9/9 combinations PASS
- Extinction ratios: 11.5–14.8 dB
- Power fractions: 80.5–90.9%

---

## Repo Structure (New-ternary-optical-computer)

```
NRadix_Accelerator/
  math/trit_multiplication.py       — Wavelength assignments, SFG math, validation
  components/nrioc_module.py        — Full NRIOC pipeline (laser→SFG→AWG→detect)
  architecture/
    systolic_array_2d.py            — 9×9 systolic array
    optical_systolic_array.py       — Original 1D array
    accumulator_bank.py
    timing_controller.py
    photodetector_array.py
    physical_layout.py
    power_budget.py
  simulation/
    trit_multiplier_inverse_design.py — BPM inverse design optimizer
    results/
      validation_results.json        — 9/9 PASS results
      optimization_log.json
  tests/
    test_systolic_array_2d.py
  scripts/
    gcp_setup_fdtdx.sh
    aws_setup_fdtdx.sh
```

---

## Key Decisions & Notes

- Frequency spacing was non-trivially designed: uniform spacing causes product 3 and 4 to collide. The final mapping (191, 194, 201 THz) gives 3 THz minimum gap.
- BPM chosen over FDTD: much better gradient flow for this routing optimization (~250 steps vs FDTD's stiffness)
- Beta schedule (1→12 over 300 iters): starts soft for gradient flow, ends sharp for binarization
- GDS export via gdstk: design_density → binary mask → foundry-ready GDS (L1=design, L2=waveguides, L10=labels)
- JAX used for auto-differentiation of the full BPM propagation chain

---

## Advanced Architecture (from GitHub issues, Feb 28)

### 9-Trit Tensor Element (Issue #2 — binary-optical-chip repo)
- Fundamental compute unit = **9 physical PPLN waveguides** = 1 tensor element
- Each waveguide: single-period PPLN (catalog fab, no exotic poling)
- IOC firmware interprets positions as 3⁰ through 3⁸
- **14.26 bits precision** per tensor element (exceeds INT8)
- Target: 6,400 PEs (80×80) to match H100 at ~150–200W vs H100's 700W
- Chiplet + interposer strategy, yield-based SKUs:
  - NR-81 (9×9), NR-64 (8×8), NR-36 (6×6), NR-9 (single eval)

### Optical Accumulator Inverse Design (Issue #1)
- Accumulator waveguide geometry determined by inverse design, not hand-designed
- Tools evaluated: Lumerical, SPINS, angler, Tidy3D
- Must handle LiNbO₃ + SFG + multi-input degenerate geometry

### Key Final Architectural Decision
**No log domain. No electronic accumulation. Fully optical MAC.**

- SFG handles **multiply** (frequency encoding: f_A + f_B = product)
- Waveguide superposition handles **accumulate** (optical field addition is free — it's just physics)
- Output port = the answer
- Unbalanced ternary {1,2,3} chosen specifically because all values are positive — no sign ambiguity, no zero-collapse, clean superposition
- The waveguide network topology encodes the matrix
- Light flows through, doing multiply-accumulate at the speed of light with no electronic handoff

The current code (electronic accumulator in systolic_array_2d.py) predates this decision and does NOT reflect the final architecture.

## Status

- [x] Single trit×trit multiplier: inverse design complete, all 9 combos validated
- [x] 9×9 systolic array: architecture implemented
- [ ] FDTDX run (full EM validation beyond BPM paraxial approx)
- [ ] Foundry tapeout / GDS finalization
- [ ] Multi-trit accumulation / carry logic

---

## Published Paper

**"Wavelength-Division Ternary Logic: Bypassing the Radix Economy Penalty in Optical Computing"**
- Author: Christopher Riner | Independent Researcher, Chesapeake, VA
- File: `~/Library/CloudStorage/GoogleDrive-.../Physics/Wavelength_Ternary_Computing.pdf`
- One of papers 1-4 already published on Zenodo

**Core argument:**
- Ternary failed commercially due to 40× transistor overhead per trit (not math)
- Wavelength-selection encoding has *constant* cost independent of radix → bypasses radix economy penalty
- Unlocks the full 1.58× information density advantage of base-3

**Paper's encoding scheme vs. current code:**
- Paper (v1): standard ternary {0,1,2} with visible wavelengths — λ1=650nm, λ2=532nm, λ3=473nm
- Code (v2): unbalanced ternary {1,2,3} in telecom C-band (191/194/201 THz, ~1491–1570 nm)
- **Why the switch:** SFG requires non-zero inputs to produce a uniquely identifiable output frequency. {0,1,2} breaks down — anything × 0 collapses and you can't distinguish inputs. {1,2,3} ensures every pairwise sum f_A + f_B is non-zero and maps to one of 6 unique product frequencies. Also, matrix multiply is addition-based (accumulate products), so unbalanced positive integers are the natural fit.

**3-layer architecture (from paper):**
1. Wavelength Source Layer (External): 3 CW lasers at distinct wavelengths
2. Logic Processing Layer (Internal): Wavelength-selective switches (ring resonators, MZI, AWG)
3. Detection Layer: Wavelength demux + photodetectors

## Google Drive

Located at: `~/Library/CloudStorage/GoogleDrive-chrisriner45@gmail.com/My Drive/Physics/`
Contents mirror the GitHub repo (same files, last synced Mar 1). Also contains:
- `Wavelength_Ternary_Computing.pdf` — published paper
- `CST_Paper_Session_Handoff.md` — handoff notes for Paper 5 (Prime Numbers as CST)
- `Prime_Numbers_as_Causal_Set_Theory.docx` — Paper 5 draft

## Research Paper Sequence

- Papers 1–4: Published on Zenodo (modified Schwarzschild, Bose-Einstein, optical computing, AI accelerators)
- Paper 5: "Prime Numbers as Causal Set Theory" — drafted, needs review + Zenodo upload
- Paper 6: Entropy rate in GR — detailed simulation results (GPS, Hawking radiation, 11× time acceleration)
- Paper 7: Full 5D/6D framework synthesis
