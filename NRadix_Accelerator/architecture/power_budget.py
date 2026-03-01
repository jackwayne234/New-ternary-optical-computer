"""
Power Budget for 9x9 Optical Systolic Array

Estimates total power consumption for the full chip.

Components:
  - 3 shared tunable laser sources:  ~100 mW  (dominant)
  - 486 InGaAs photodetectors:        ~1 mW   (negligible)
  - 81 accumulator registers (7-bit):  ~4 mW
  - Timing controller + I/O:          ~50 mW
  - Total:                            ~155 mW

Energy efficiency: ~96 pJ per MAC
  (155 mW / 1.6 GMAC/s ≈ 96 pJ/MAC)
"""

from __future__ import annotations

from dataclasses import dataclass

ARRAY_SIZE = 9
PORTS_PER_PE = 6


@dataclass
class LaserPowerSpec:
    """Power for the 3 shared laser sources.

    Three tunable/switchable telecom lasers (191, 194, 201 THz).
    These are shared across all PEs — light is split/fanned out.
    Laser power dominates the total budget.
    """
    num_lasers: int = 3
    power_per_laser_mw: float = 30.0   # conservative telecom DFB laser
    driver_overhead_mw: float = 10.0   # laser driver electronics

    @property
    def total_mw(self) -> float:
        return self.num_lasers * self.power_per_laser_mw + self.driver_overhead_mw


@dataclass
class DetectorPowerSpec:
    """Power for 486 InGaAs photodetectors.

    Each detector has a TIA (trans-impedance amplifier).
    Very low power — photodetectors are passive, TIAs are small.
    """
    num_detectors: int = ARRAY_SIZE * ARRAY_SIZE * PORTS_PER_PE  # 486
    power_per_detector_uw: float = 2.0  # TIA power per detector

    @property
    def total_mw(self) -> float:
        return self.num_detectors * self.power_per_detector_uw / 1000.0


@dataclass
class AccumulatorPowerSpec:
    """Power for 81 accumulator registers."""
    num_registers: int = ARRAY_SIZE * ARRAY_SIZE  # 81
    power_per_register_uw: float = 50.0  # 7-bit adder + flip-flops

    @property
    def total_mw(self) -> float:
        return self.num_registers * self.power_per_register_uw / 1000.0


@dataclass
class ControllerPowerSpec:
    """Power for timing controller, I/O pads, and clock distribution."""
    clock_tree_mw: float = 15.0
    io_pads_mw: float = 20.0
    control_logic_mw: float = 10.0
    misc_mw: float = 5.0

    @property
    def total_mw(self) -> float:
        return self.clock_tree_mw + self.io_pads_mw + self.control_logic_mw + self.misc_mw


@dataclass
class PowerBudget:
    """Complete power budget for the 9x9 optical systolic array chip."""
    lasers: LaserPowerSpec = None
    detectors: DetectorPowerSpec = None
    accumulators: AccumulatorPowerSpec = None
    controller: ControllerPowerSpec = None

    # Performance numbers for efficiency calculation
    throughput_gmacs: float = 1.6  # from timing controller

    def __post_init__(self):
        if self.lasers is None:
            self.lasers = LaserPowerSpec()
        if self.detectors is None:
            self.detectors = DetectorPowerSpec()
        if self.accumulators is None:
            self.accumulators = AccumulatorPowerSpec()
        if self.controller is None:
            self.controller = ControllerPowerSpec()

    @property
    def total_power_mw(self) -> float:
        return (
            self.lasers.total_mw
            + self.detectors.total_mw
            + self.accumulators.total_mw
            + self.controller.total_mw
        )

    @property
    def energy_per_mac_pj(self) -> float:
        """Energy per MAC in picojoules."""
        power_w = self.total_power_mw / 1000.0
        macs_per_s = self.throughput_gmacs * 1e9
        energy_j = power_w / macs_per_s
        return energy_j * 1e12  # convert to pJ

    def print_summary(self):
        """Print detailed power budget."""
        print("=" * 60)
        print("Power Budget — 9x9 Optical Systolic Array")
        print("=" * 60)

        print(f"\n  Laser Sources:")
        print(f"    {self.lasers.num_lasers} lasers × {self.lasers.power_per_laser_mw:.0f} mW "
              f"= {self.lasers.num_lasers * self.lasers.power_per_laser_mw:.0f} mW")
        print(f"    Driver overhead: {self.lasers.driver_overhead_mw:.0f} mW")
        print(f"    Subtotal: {self.lasers.total_mw:.0f} mW")

        print(f"\n  Photodetectors:")
        print(f"    {self.detectors.num_detectors} detectors × "
              f"{self.detectors.power_per_detector_uw:.0f} μW "
              f"= {self.detectors.total_mw:.1f} mW")

        print(f"\n  Accumulators:")
        print(f"    {self.accumulators.num_registers} registers × "
              f"{self.accumulators.power_per_register_uw:.0f} μW "
              f"= {self.accumulators.total_mw:.1f} mW")

        print(f"\n  Controller + I/O:")
        print(f"    Clock tree:     {self.controller.clock_tree_mw:.0f} mW")
        print(f"    I/O pads:       {self.controller.io_pads_mw:.0f} mW")
        print(f"    Control logic:  {self.controller.control_logic_mw:.0f} mW")
        print(f"    Misc:           {self.controller.misc_mw:.0f} mW")
        print(f"    Subtotal:       {self.controller.total_mw:.0f} mW")

        print(f"\n  {'─' * 40}")
        print(f"  TOTAL POWER:      {self.total_power_mw:.0f} mW")
        print(f"  Throughput:       {self.throughput_gmacs:.2f} GMAC/s")
        print(f"  Energy/MAC:       {self.energy_per_mac_pj:.0f} pJ")

        print(f"\n  Power Breakdown:")
        total = self.total_power_mw
        components = [
            ("Lasers", self.lasers.total_mw),
            ("Detectors", self.detectors.total_mw),
            ("Accumulators", self.accumulators.total_mw),
            ("Controller", self.controller.total_mw),
        ]
        for name, power in components:
            pct = power / total * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"    {name:<14} {bar} {pct:5.1f}%  ({power:.1f} mW)")

        print()


if __name__ == "__main__":
    budget = PowerBudget()
    budget.print_summary()
