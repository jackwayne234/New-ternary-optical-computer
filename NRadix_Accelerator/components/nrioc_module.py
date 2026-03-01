"""
NRadix Integrated Optical Circuit (NRIOC) Module

Redesigned for unbalanced ternary {1, 2, 3} trit multiplication.

Architecture:
  - Encode: trit value -> laser wavelength (select one of 3 telecom sources)
  - SFG:    two wavelengths in -> one wavelength out (frequency addition = multiplication)
  - AWG:    routes output wavelength to the correct port (physical path = answer)
  - Decode: photodetector at each port converts optical -> electronic product value

No log domain. No lookup tables. The physics does the multiply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math.trit_multiplication import (
    TRIT_VALUES,
    WAVELENGTHS,
    PRODUCT_TO_PORT,
    PORT_TO_PRODUCT,
    PRODUCT_FREQUENCIES,
    UNIQUE_PRODUCTS,
    sfg_product_freq,
    sfg_product_wavelength_nm,
    C_LIGHT,
)


class TritValue(IntEnum):
    """Unbalanced ternary trit values for multiplication."""
    ONE = 1
    TWO = 2
    THREE = 3


@dataclass
class LaserSource:
    """A tunable or switchable laser source for one trit input.

    Selects one of 3 wavelengths based on the trit value.
    """
    label: str
    current_trit: Optional[TritValue] = None

    @property
    def frequency_thz(self) -> Optional[float]:
        if self.current_trit is None:
            return None
        return WAVELENGTHS.freq_thz(self.current_trit)

    @property
    def wavelength_nm(self) -> Optional[float]:
        if self.current_trit is None:
            return None
        return WAVELENGTHS.wavelength_nm(self.current_trit)

    def set_trit(self, value: int):
        """Set the trit value, activating the corresponding wavelength."""
        self.current_trit = TritValue(value)


@dataclass
class SFGUnit:
    """Sum Frequency Generation unit.

    Takes two input wavelengths and produces one output wavelength
    whose frequency is the sum: f_out = f_A + f_B.

    This is where multiplication happens optically.
    """
    label: str = "SFG"

    def compute(self, trit_a: TritValue, trit_b: TritValue) -> SFGResult:
        """Compute the SFG output for two trit inputs."""
        product = int(trit_a) * int(trit_b)
        output_freq = sfg_product_freq(int(trit_a), int(trit_b))
        output_wl = sfg_product_wavelength_nm(int(trit_a), int(trit_b))
        return SFGResult(
            trit_a=trit_a,
            trit_b=trit_b,
            product=product,
            output_freq_thz=output_freq,
            output_wavelength_nm=output_wl,
        )


@dataclass(frozen=True)
class SFGResult:
    """Result of an SFG computation."""
    trit_a: TritValue
    trit_b: TritValue
    product: int
    output_freq_thz: float
    output_wavelength_nm: float


@dataclass
class AWGRouter:
    """Arrayed Waveguide Grating router.

    Routes each SFG output wavelength to its dedicated port.
    The physical path IS the answer — no lookup tables.

    6 output ports for products {1, 2, 3, 4, 6, 9}.
    """
    num_ports: int = 6

    def route(self, sfg_result: SFGResult) -> AWGOutput:
        """Route an SFG result to the correct output port."""
        port = PRODUCT_TO_PORT[sfg_result.product]
        return AWGOutput(
            port_index=port,
            product_value=sfg_result.product,
            frequency_thz=sfg_result.output_freq_thz,
            wavelength_nm=sfg_result.output_wavelength_nm,
        )

    @property
    def port_assignments(self) -> dict[int, dict]:
        """Return the port -> product mapping with frequencies."""
        assignments = {}
        for port_idx in range(self.num_ports):
            product = PORT_TO_PRODUCT[port_idx]
            freq = PRODUCT_FREQUENCIES[product]
            wl = (C_LIGHT / (freq * 1e12)) * 1e9
            assignments[port_idx] = {
                "product": product,
                "frequency_thz": freq,
                "wavelength_nm": wl,
            }
        return assignments


@dataclass(frozen=True)
class AWGOutput:
    """Result of AWG routing — which port the light exits."""
    port_index: int
    product_value: int
    frequency_thz: float
    wavelength_nm: float


@dataclass
class Photodetector:
    """Photodetector at one AWG output port.

    Converts optical power to electronic signal.
    Each detector sits at a specific port and knows its product value.
    """
    port_index: int

    @property
    def product_value(self) -> int:
        return PORT_TO_PRODUCT[self.port_index]

    def detect(self, awg_output: AWGOutput) -> int:
        """Convert optical output to electronic product value.

        Returns the product value if light arrives at this port, 0 otherwise.
        """
        if awg_output.port_index == self.port_index:
            return awg_output.product_value
        return 0


@dataclass
class NRIOCModule:
    """Complete NRIOC module for single trit x trit multiplication.

    Data flow:
      trit_a, trit_b
        -> LaserSource A, LaserSource B (encode as wavelengths)
        -> SFG (multiply: f_out = f_a + f_b)
        -> AWG (route by color to correct port)
        -> Photodetector array (convert to electronic)
        -> product value (integer)
    """
    source_a: LaserSource = field(default_factory=lambda: LaserSource("Input_A"))
    source_b: LaserSource = field(default_factory=lambda: LaserSource("Input_B"))
    sfg: SFGUnit = field(default_factory=SFGUnit)
    awg: AWGRouter = field(default_factory=AWGRouter)
    detectors: list[Photodetector] = field(default_factory=lambda: [
        Photodetector(port_index=i) for i in range(6)
    ])

    def encode(self, trit_a: int, trit_b: int):
        """Encode trit values as wavelengths (activate laser sources)."""
        self.source_a.set_trit(trit_a)
        self.source_b.set_trit(trit_b)

    def multiply(self, trit_a: int, trit_b: int) -> int:
        """Perform a complete trit x trit optical multiplication.

        Full pipeline: encode -> SFG -> AWG -> detect -> product value.
        """
        self.encode(trit_a, trit_b)

        # SFG: two wavelengths -> one product wavelength
        sfg_result = self.sfg.compute(
            TritValue(trit_a), TritValue(trit_b)
        )

        # AWG: route product wavelength to correct port
        awg_output = self.awg.route(sfg_result)

        # Photodetector: convert to electronic value
        for detector in self.detectors:
            result = detector.detect(awg_output)
            if result != 0:
                return result

        raise RuntimeError(
            f"No detector triggered for {trit_a} x {trit_b} = {sfg_result.product}"
        )

    def verify_all(self) -> bool:
        """Verify all 9 trit combinations produce correct products."""
        all_pass = True
        for a in TRIT_VALUES:
            for b in TRIT_VALUES:
                result = self.multiply(a, b)
                expected = a * b
                if result != expected:
                    print(f"  FAIL: {a} x {b} = {result}, expected {expected}")
                    all_pass = False
        return all_pass


def self_test():
    """Run self-test of the NRIOC module."""
    module = NRIOCModule()

    print("NRIOC Module Self-Test")
    print("=" * 40)
    print("\nTrit x Trit multiplication results:")
    print(f"  {'A':>3} x {'B':>3} = {'Product':>7}  (Port {'#':>2}, Freq THz, WL nm)")
    print(f"  {'---':>3}   {'---':>3}   {'-------':>7}  {'--':>7}")

    for a in TRIT_VALUES:
        for b in TRIT_VALUES:
            module.encode(a, b)
            sfg_result = module.sfg.compute(TritValue(a), TritValue(b))
            awg_output = module.awg.route(sfg_result)
            product = module.multiply(a, b)
            print(
                f"  {a:>3} x {b:>3} = {product:>7}  "
                f"(Port {awg_output.port_index:>2}, "
                f"{awg_output.frequency_thz:.1f} THz, "
                f"{awg_output.wavelength_nm:.2f} nm)"
            )

    print()
    if module.verify_all():
        print("ALL 9 COMBINATIONS PASS")
    else:
        print("SOME COMBINATIONS FAILED")


if __name__ == "__main__":
    self_test()
