"""
Accumulator Bank for 9x9 Optical Systolic Array

81 electronic accumulator registers (one per PE).
Each register accumulates trit products over multiple clock cycles
to compute dot products and matrix elements.

7-bit registers: max single product = 9 (trit 3×3),
max accumulated value for 9-element dot product = 9×9 = 81,
which fits in 7 bits (2^7 = 128 > 81).

Row-sum readout: C[i] = sum of 9 PE accumulators across row i.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ARRAY_SIZE = 9
TOTAL_REGISTERS = ARRAY_SIZE * ARRAY_SIZE  # 81
REGISTER_BITS = 7
MAX_ACCUMULATOR_VALUE = 2**REGISTER_BITS - 1  # 127
MAX_DOT_PRODUCT = 9 * 9  # 81 (9 terms, each max 9)


@dataclass
class AccumulatorRegister7Bit:
    """7-bit electronic accumulator register for one PE.

    Accumulates product values from the photodetector.
    Max value: 81 (9 terms of max product 9), fits in 7 bits.
    """
    value: int = 0
    cycle_count: int = 0

    def accumulate(self, product: int):
        """Add a product value to the running sum."""
        self.value += product
        self.cycle_count += 1

    def reset(self):
        """Clear for a new computation."""
        self.value = 0
        self.cycle_count = 0

    @property
    def saturated(self) -> bool:
        """Check if the register has exceeded 7-bit capacity."""
        return self.value > MAX_ACCUMULATOR_VALUE


@dataclass
class AccumulatorBank:
    """Bank of 81 accumulator registers for the 9x9 systolic array.

    Layout: 9 rows × 9 columns, matching PE grid.
    Row-sum readout produces one element of the output vector/matrix.

    Power: ~50 μW per register (7-bit adder + flip-flops at 339 MHz)
    Total: 81 × 50 μW ≈ 4 mW
    """
    array_size: int = ARRAY_SIZE
    registers: list[list[AccumulatorRegister7Bit]] = field(init=False)

    # Physical parameters
    power_per_register_uw: float = 50.0  # μW
    technology_node_nm: int = 45

    def __post_init__(self):
        self.registers = [
            [AccumulatorRegister7Bit() for _ in range(self.array_size)]
            for _ in range(self.array_size)
        ]

    def get(self, row: int, col: int) -> AccumulatorRegister7Bit:
        """Get the accumulator at (row, col)."""
        return self.registers[row][col]

    def accumulate(self, row: int, col: int, product: int):
        """Add a product to the register at (row, col)."""
        self.registers[row][col].accumulate(product)

    def read_row(self, row: int) -> list[int]:
        """Read all accumulator values across one row (9 values)."""
        return [self.registers[row][col].value for col in range(self.array_size)]

    def read_row_sum(self, row: int) -> int:
        """Sum all accumulators in a row — one element of the output vector."""
        return sum(self.read_row(row))

    def read_all(self) -> list[list[int]]:
        """Read the full 9×9 accumulator state."""
        return [self.read_row(row) for row in range(self.array_size)]

    def reset_all(self):
        """Clear all registers for a new computation."""
        for row in self.registers:
            for reg in row:
                reg.reset()

    def reset_row(self, row: int):
        """Clear one row of registers."""
        for reg in self.registers[row]:
            reg.reset()

    @property
    def total_registers(self) -> int:
        return self.array_size * self.array_size

    @property
    def total_power_mw(self) -> float:
        """Total power consumption in mW."""
        return self.total_registers * self.power_per_register_uw / 1000.0

    def print_summary(self):
        """Print accumulator bank summary."""
        print("=" * 60)
        print("Accumulator Bank — 9x9 Optical Systolic Array")
        print("=" * 60)
        print(f"\n  Array size:          {self.array_size}x{self.array_size}")
        print(f"  Total registers:     {self.total_registers}")
        print(f"  Register width:      {REGISTER_BITS} bits")
        print(f"  Max register value:  {MAX_ACCUMULATOR_VALUE}")
        print(f"  Max dot product:     {MAX_DOT_PRODUCT} (fits in {REGISTER_BITS} bits)")
        print(f"  Technology node:     {self.technology_node_nm} nm")
        print(f"  Power per register:  {self.power_per_register_uw} μW")
        print(f"  Total power:         {self.total_power_mw:.1f} mW")
        print()

    def print_state(self):
        """Print current accumulator values as a 9×9 grid."""
        print("  Accumulator State:")
        header = "     " + "".join(f"  c{c}" for c in range(self.array_size))
        print(header)
        for row in range(self.array_size):
            vals = self.read_row(row)
            line = f"  r{row} " + "".join(f"{v:>4}" for v in vals)
            print(line)
        print()


if __name__ == "__main__":
    bank = AccumulatorBank()
    bank.print_summary()
