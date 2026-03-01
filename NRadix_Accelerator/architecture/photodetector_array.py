"""
Photodetector Array for 9x9 Optical Systolic Array

486 InGaAs photodetectors (81 PEs × 6 AWG output ports).
Each detector converts optical power at a specific SFG product frequency
into an electronic signal representing the product value.

Addressing: (pe_row, pe_col, port_index) -> detector_id
Each detector maps to one of the 6 product values {1, 2, 3, 4, 6, 9}.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math.trit_multiplication import (
    PORT_TO_PRODUCT,
    PRODUCT_TO_PORT,
    PRODUCT_FREQUENCIES,
    UNIQUE_PRODUCTS,
    C_LIGHT,
)

ARRAY_SIZE = 9
PORTS_PER_PE = 6
TOTAL_DETECTORS = ARRAY_SIZE * ARRAY_SIZE * PORTS_PER_PE  # 486


@dataclass(frozen=True)
class DetectorAddress:
    """Unique address for one photodetector in the array."""
    pe_row: int
    pe_col: int
    port_index: int

    @property
    def detector_id(self) -> int:
        """Flat detector index (0-485)."""
        return (self.pe_row * ARRAY_SIZE + self.pe_col) * PORTS_PER_PE + self.port_index

    @property
    def product_value(self) -> int:
        """The trit product this detector responds to."""
        return PORT_TO_PRODUCT[self.port_index]

    @property
    def frequency_thz(self) -> float:
        """SFG product frequency this detector is tuned to."""
        return PRODUCT_FREQUENCIES[self.product_value]

    @property
    def wavelength_nm(self) -> float:
        """Wavelength this detector is tuned to."""
        return (C_LIGHT / (self.frequency_thz * 1e12)) * 1e9

    def __repr__(self) -> str:
        return (
            f"Detector(row={self.pe_row}, col={self.pe_col}, "
            f"port={self.port_index}, product={self.product_value})"
        )


@dataclass
class PhotodetectorArray:
    """Complete photodetector array for the 9x9 systolic array.

    Manages 486 InGaAs photodetectors with addressing and readout.

    Material: InGaAs (Indium Gallium Arsenide)
    Responsivity: ~0.9 A/W at 780 nm (SFG output band)
    Dark current: ~1 nA (typical for 20 μm diameter)
    Bandwidth: >1 GHz (sufficient for 339 MHz clock)
    """
    array_size: int = ARRAY_SIZE
    detectors: list[DetectorAddress] = field(init=False)

    # Physical parameters
    responsivity_a_per_w: float = 0.9
    dark_current_na: float = 1.0
    bandwidth_ghz: float = 1.0
    diameter_um: float = 20.0

    def __post_init__(self):
        self.detectors = []
        for row in range(self.array_size):
            for col in range(self.array_size):
                for port in range(PORTS_PER_PE):
                    self.detectors.append(DetectorAddress(row, col, port))

    @property
    def total_detectors(self) -> int:
        return len(self.detectors)

    def get_detector(self, pe_row: int, pe_col: int, port_index: int) -> DetectorAddress:
        """Look up a specific detector by PE coordinates and port."""
        idx = (pe_row * self.array_size + pe_col) * PORTS_PER_PE + port_index
        return self.detectors[idx]

    def get_pe_detectors(self, pe_row: int, pe_col: int) -> list[DetectorAddress]:
        """Get all 6 detectors for a specific PE."""
        return [self.get_detector(pe_row, pe_col, p) for p in range(PORTS_PER_PE)]

    def get_row_detectors(self, row: int) -> list[DetectorAddress]:
        """Get all detectors in a row of PEs (9 PEs × 6 ports = 54 detectors)."""
        result = []
        for col in range(self.array_size):
            result.extend(self.get_pe_detectors(row, col))
        return result

    def detector_for_product(self, pe_row: int, pe_col: int, product: int) -> DetectorAddress:
        """Find the detector at a specific PE that responds to a given product value."""
        port = PRODUCT_TO_PORT[product]
        return self.get_detector(pe_row, pe_col, port)

    def print_summary(self):
        """Print detector array summary."""
        print("=" * 60)
        print("Photodetector Array — 9x9 Optical Systolic Array")
        print("=" * 60)
        print(f"\n  Array size:        {self.array_size}x{self.array_size} PEs")
        print(f"  Ports per PE:      {PORTS_PER_PE}")
        print(f"  Total detectors:   {self.total_detectors}")
        print(f"  Material:          InGaAs")
        print(f"  Responsivity:      {self.responsivity_a_per_w} A/W")
        print(f"  Dark current:      {self.dark_current_na} nA")
        print(f"  Bandwidth:         {self.bandwidth_ghz} GHz")
        print(f"  Diameter:          {self.diameter_um} μm")

        print(f"\n  Port -> Product Mapping:")
        for port in range(PORTS_PER_PE):
            product = PORT_TO_PRODUCT[port]
            freq = PRODUCT_FREQUENCIES[product]
            wl = (C_LIGHT / (freq * 1e12)) * 1e9
            print(f"    Port {port}: product={product}, "
                  f"freq={freq:.1f} THz, λ={wl:.2f} nm")

        print(f"\n  Addressing example (PE at row=4, col=7):")
        for port in range(PORTS_PER_PE):
            det = self.get_detector(4, 7, port)
            print(f"    {det} -> detector_id={det.detector_id}")

        print()


if __name__ == "__main__":
    array = PhotodetectorArray()
    array.print_summary()
