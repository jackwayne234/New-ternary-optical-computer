"""
Timing Controller for 9x9 Optical Systolic Array

Manages the 3-phase clock cycle for the optical multiply-accumulate pipeline:
  Phase 1 — Laser modulation:     1.50 ns  (switch wavelength for trit encoding)
  Phase 2 — Optical propagation:  0.15 ns  (SFG + AWG waveguide transit)
  Phase 3 — Detection + accumulate: 0.80 ns (photodetector + register add)

Total clock cycle: 2.45 ns (~408 MHz) - updated from plan based on phase timings
Conservative estimate with margins: ~2.95 ns (~339 MHz)

Weight-stationary dataflow timing:
  - Matrix-vector (9×9 × 9×1): 9 + 8 = 17 cycles for systolic fill + drain
  - Matrix-matrix (9×9 × 9×9): 9 columns × 17 cycles = 153 cycles
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

ARRAY_SIZE = 9


class ClockPhase(IntEnum):
    """Three phases of each clock cycle."""
    LASER_MODULATION = 1       # Phase 1: switch laser wavelength
    OPTICAL_PROPAGATION = 2    # Phase 2: SFG + AWG
    DETECTION_ACCUMULATE = 3   # Phase 3: photodetect + accumulate


@dataclass
class TimingSpec:
    """Timing specification for each clock phase."""
    # Phase durations in nanoseconds
    laser_modulation_ns: float = 1.50
    optical_propagation_ns: float = 0.15
    detection_accumulate_ns: float = 0.80

    # Guard band between phases (accounts for skew, settling)
    guard_band_ns: float = 0.25

    @property
    def cycle_time_ns(self) -> float:
        """Total clock cycle time including guard bands."""
        phases = (
            self.laser_modulation_ns
            + self.optical_propagation_ns
            + self.detection_accumulate_ns
        )
        guards = 2 * self.guard_band_ns  # guard between phase 1-2 and 2-3
        return phases + guards

    @property
    def clock_freq_mhz(self) -> float:
        """Clock frequency in MHz."""
        return 1000.0 / self.cycle_time_ns

    @property
    def clock_freq_ghz(self) -> float:
        """Clock frequency in GHz."""
        return 1.0 / self.cycle_time_ns


@dataclass
class TimingController:
    """Controls clock phases and tracks cycle count for the systolic array.

    Weight-stationary dataflow:
    - Weights A[i][j] are pre-loaded into PE(i,j) before computation starts
    - Input vector b[k] enters at column 0 and propagates right
    - Systolic skew: b[k] enters column c at cycle (k + c)
    - After N+N-1 cycles, all partial products are accumulated
    """
    spec: TimingSpec = None
    current_cycle: int = 0
    current_phase: ClockPhase = ClockPhase.LASER_MODULATION
    array_size: int = ARRAY_SIZE

    def __post_init__(self):
        if self.spec is None:
            self.spec = TimingSpec()

    def reset(self):
        """Reset to cycle 0."""
        self.current_cycle = 0
        self.current_phase = ClockPhase.LASER_MODULATION

    def advance_phase(self):
        """Advance to the next phase within the current cycle."""
        if self.current_phase == ClockPhase.DETECTION_ACCUMULATE:
            self.current_cycle += 1
            self.current_phase = ClockPhase.LASER_MODULATION
        else:
            self.current_phase = ClockPhase(self.current_phase + 1)

    def advance_cycle(self):
        """Advance one full clock cycle."""
        self.current_cycle += 1
        self.current_phase = ClockPhase.LASER_MODULATION

    # --- Latency calculations ---

    @property
    def mat_vec_cycles(self) -> int:
        """Clock cycles for one matrix-vector multiply (N + N - 1 for systolic fill+drain)."""
        return self.array_size + (self.array_size - 1)

    @property
    def mat_mat_cycles(self) -> int:
        """Clock cycles for full matrix-matrix multiply (N columns × mat_vec_cycles)."""
        return self.array_size * self.mat_vec_cycles

    @property
    def mat_vec_latency_ns(self) -> float:
        """Matrix-vector latency in nanoseconds."""
        return self.mat_vec_cycles * self.spec.cycle_time_ns

    @property
    def mat_mat_latency_ns(self) -> float:
        """Matrix-matrix latency in nanoseconds."""
        return self.mat_mat_cycles * self.spec.cycle_time_ns

    @property
    def macs_per_mat_vec(self) -> int:
        """Number of multiply-accumulate operations per mat-vec."""
        return self.array_size * self.array_size  # N² MACs

    @property
    def macs_per_mat_mat(self) -> int:
        """Number of MACs per mat-mat."""
        return self.array_size ** 3  # N³ MACs

    @property
    def throughput_gmacs(self) -> float:
        """Sustained throughput in GMAC/s (mat-mat pipelined)."""
        # N³ MACs every mat_mat_cycles cycles
        macs = self.macs_per_mat_mat
        time_s = self.mat_mat_latency_ns * 1e-9
        return macs / time_s / 1e9

    def systolic_input_cycle(self, k: int, col: int) -> int:
        """Cycle at which input b[k] arrives at column col.

        Systolic skew: element k enters column c at cycle k + c.
        """
        return k + col

    def print_summary(self):
        """Print timing and performance summary."""
        print("=" * 60)
        print("Timing Controller — 9x9 Optical Systolic Array")
        print("=" * 60)

        print(f"\n  Clock Phases:")
        print(f"    Phase 1 (Laser modulation):     {self.spec.laser_modulation_ns:.2f} ns")
        print(f"    Phase 2 (Optical propagation):   {self.spec.optical_propagation_ns:.2f} ns")
        print(f"    Phase 3 (Detection + accumulate): {self.spec.detection_accumulate_ns:.2f} ns")
        print(f"    Guard bands (×2):                {2 * self.spec.guard_band_ns:.2f} ns")
        print(f"    Total cycle time:                {self.spec.cycle_time_ns:.2f} ns")
        print(f"    Clock frequency:                 {self.spec.clock_freq_mhz:.0f} MHz")

        print(f"\n  Latency:")
        print(f"    Matrix-vector (9×9 × 9×1):  {self.mat_vec_cycles} cycles = "
              f"{self.mat_vec_latency_ns:.1f} ns")
        print(f"    Matrix-matrix (9×9 × 9×9):  {self.mat_mat_cycles} cycles = "
              f"{self.mat_mat_latency_ns:.1f} ns")

        print(f"\n  Throughput:")
        print(f"    MACs per mat-vec:  {self.macs_per_mat_vec}")
        print(f"    MACs per mat-mat:  {self.macs_per_mat_mat}")
        print(f"    Throughput:        {self.throughput_gmacs:.2f} GMAC/s")

        print(f"\n  Systolic Skew (first 5 cycles, b[k] arrival at column c):")
        print(f"    {'Cycle':>6}", end="")
        for c in range(min(5, self.array_size)):
            print(f"  col{c}", end="")
        print()
        for cycle in range(min(8, self.mat_vec_cycles)):
            print(f"    {cycle:>6}", end="")
            for c in range(min(5, self.array_size)):
                k = cycle - c  # b[k] arrives at col c on cycle k+c
                if 0 <= k < self.array_size:
                    print(f"  b[{k}] ", end="")
                else:
                    print(f"   —  ", end="")
            print()
        print()


if __name__ == "__main__":
    controller = TimingController()
    controller.print_summary()
