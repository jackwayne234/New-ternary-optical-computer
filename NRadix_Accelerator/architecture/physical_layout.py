"""
Physical Layout for 9x9 Optical Systolic Array

Chip floor plan with realistic component dimensions:

  PE footprint: 90 × 30 μm
    - SFG crystal:     20 × 30 μm
    - AWG router:      50 × 30 μm
    - Detector array:  20 × 30 μm (6 photodetectors)

  PE pitch: 140 × 70 μm (includes routing channels)
    - Horizontal gap: 50 μm (waveguides + electrical interconnect)
    - Vertical gap: 40 μm (waveguides + electrical interconnect)

  Array core (9×9 PEs): 1.26 × 0.63 mm
  Periphery (lasers, I/O pads, controller): 0.2 mm border on each side

  Full chip: 1.66 × 0.83 mm = 1.38 mm²
"""

from __future__ import annotations

from dataclasses import dataclass

ARRAY_SIZE = 9


@dataclass
class ComponentDimensions:
    """Dimensions of sub-components within a PE (in μm)."""
    # SFG nonlinear crystal (periodically-poled lithium niobate)
    sfg_width_um: float = 20.0
    sfg_height_um: float = 30.0

    # AWG demultiplexer (6 output ports)
    awg_width_um: float = 50.0
    awg_height_um: float = 30.0

    # Photodetector array (6 InGaAs detectors in a row)
    detector_width_um: float = 20.0
    detector_height_um: float = 30.0


@dataclass
class PELayout:
    """Physical layout of a single Processing Element."""
    components: ComponentDimensions = None

    # Routing channel widths between PEs
    horizontal_gap_um: float = 50.0  # waveguides + metal
    vertical_gap_um: float = 40.0

    def __post_init__(self):
        if self.components is None:
            self.components = ComponentDimensions()

    @property
    def footprint_width_um(self) -> float:
        """PE width (horizontal): SFG + AWG + detectors side by side."""
        return (
            self.components.sfg_width_um
            + self.components.awg_width_um
            + self.components.detector_width_um
        )

    @property
    def footprint_height_um(self) -> float:
        """PE height (vertical): all components same height."""
        return self.components.sfg_height_um  # 30 μm

    @property
    def pitch_x_um(self) -> float:
        """Center-to-center horizontal spacing."""
        return self.footprint_width_um + self.horizontal_gap_um

    @property
    def pitch_y_um(self) -> float:
        """Center-to-center vertical spacing."""
        return self.footprint_height_um + self.vertical_gap_um


@dataclass
class ChipLayout:
    """Full chip layout for the 9×9 optical systolic array."""
    array_size: int = ARRAY_SIZE
    pe: PELayout = None

    # Periphery: laser sources, I/O pads, controller
    periphery_um: float = 200.0  # border on each side

    def __post_init__(self):
        if self.pe is None:
            self.pe = PELayout()

    @property
    def array_width_um(self) -> float:
        """Width of the PE array core."""
        return self.array_size * self.pe.pitch_x_um - self.pe.horizontal_gap_um

    @property
    def array_height_um(self) -> float:
        """Height of the PE array core."""
        return self.array_size * self.pe.pitch_y_um - self.pe.vertical_gap_um

    @property
    def array_width_mm(self) -> float:
        return self.array_width_um / 1000.0

    @property
    def array_height_mm(self) -> float:
        return self.array_height_um / 1000.0

    @property
    def chip_width_um(self) -> float:
        return self.array_width_um + 2 * self.periphery_um

    @property
    def chip_height_um(self) -> float:
        return self.array_height_um + 2 * self.periphery_um

    @property
    def chip_width_mm(self) -> float:
        return self.chip_width_um / 1000.0

    @property
    def chip_height_mm(self) -> float:
        return self.chip_height_um / 1000.0

    @property
    def chip_area_mm2(self) -> float:
        return self.chip_width_mm * self.chip_height_mm

    def pe_position_um(self, row: int, col: int) -> tuple[float, float]:
        """Get the (x, y) position of PE(row, col) bottom-left corner.

        Origin is at the bottom-left of the array core (excludes periphery).
        """
        x = col * self.pe.pitch_x_um
        y = row * self.pe.pitch_y_um
        return (x, y)

    def print_summary(self):
        """Print chip layout summary."""
        print("=" * 60)
        print("Physical Layout — 9x9 Optical Systolic Array")
        print("=" * 60)

        print(f"\n  PE Components:")
        c = self.pe.components
        print(f"    SFG crystal:    {c.sfg_width_um:.0f} × {c.sfg_height_um:.0f} μm")
        print(f"    AWG router:     {c.awg_width_um:.0f} × {c.awg_height_um:.0f} μm")
        print(f"    Detector array: {c.detector_width_um:.0f} × {c.detector_height_um:.0f} μm")

        print(f"\n  PE Layout:")
        print(f"    Footprint:      {self.pe.footprint_width_um:.0f} × "
              f"{self.pe.footprint_height_um:.0f} μm")
        print(f"    Pitch:          {self.pe.pitch_x_um:.0f} × "
              f"{self.pe.pitch_y_um:.0f} μm")
        print(f"    Routing gaps:   {self.pe.horizontal_gap_um:.0f} μm (H) × "
              f"{self.pe.vertical_gap_um:.0f} μm (V)")

        print(f"\n  Array Core ({self.array_size}×{self.array_size} PEs):")
        print(f"    Dimensions:     {self.array_width_um:.0f} × "
              f"{self.array_height_um:.0f} μm")
        print(f"                    {self.array_width_mm:.2f} × "
              f"{self.array_height_mm:.2f} mm")

        print(f"\n  Full Chip:")
        print(f"    Periphery:      {self.periphery_um:.0f} μm border")
        print(f"    Dimensions:     {self.chip_width_um:.0f} × "
              f"{self.chip_height_um:.0f} μm")
        print(f"                    {self.chip_width_mm:.2f} × "
              f"{self.chip_height_mm:.2f} mm")
        print(f"    Area:           {self.chip_area_mm2:.2f} mm²")

        print(f"\n  PE Grid (corner positions):")
        corners = [(0, 0), (0, 8), (8, 0), (8, 8)]
        for r, c in corners:
            x, y = self.pe_position_um(r, c)
            print(f"    PE({r},{c}): ({x:.0f}, {y:.0f}) μm")

        print(f"\n  Floor Plan Sketch:")
        print(f"    ┌{'─' * 42}┐")
        print(f"    │{'Periphery (lasers, I/O, controller)':^42}│")
        print(f"    │  ┌{'─' * 36}┐  │")
        print(f"    │  │{'':^36}│  │")
        print(f"    │  │{'9 × 9  PE  Array':^36}│  │")
        print(f"    │  │{f'{self.array_width_mm:.2f} × {self.array_height_mm:.2f} mm':^36}│  │")
        print(f"    │  │{'':^36}│  │")
        print(f"    │  └{'─' * 36}┘  │")
        print(f"    │{f'Total: {self.chip_width_mm:.2f} × {self.chip_height_mm:.2f} mm = {self.chip_area_mm2:.2f} mm²':^42}│")
        print(f"    └{'─' * 42}┘")
        print()


if __name__ == "__main__":
    layout = ChipLayout()
    layout.print_summary()
