"""
9x9 2D Optical Systolic Array for Ternary Matrix Multiplication

Weight-stationary dataflow:
  - Weight matrix A (9×9) is pre-loaded: PE(i,j) holds A[i][j]
  - Input vector b (or column of B) enters at column 0 with systolic skew
  - Each PE multiplies its weight × input optically (SFG + AWG)
  - Products accumulate electronically in per-PE registers
  - After 17 cycles (N + N-1), row sums give the output vector

Operations:
  - Matrix-vector: C = A × b  in 17 clock cycles
  - Matrix-matrix: C = A × B  in 153 clock cycles (9 columns of B)

All 81 PEs operate in parallel. The optical multiply is instantaneous
within each clock cycle; the systolic skew handles the data flow timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math.trit_multiplication import TRIT_VALUES
from components.nrioc_module import NRIOCModule


ARRAY_SIZE = 9


@dataclass
class SystolicPE:
    """Processing Element for the 2D systolic array.

    Each PE holds:
    - A weight register (one trit value from matrix A)
    - An NRIOC module (optical multiplier)
    - An accumulator (electronic sum of products)
    - Input/output registers for systolic data flow

    Data flow per cycle:
    1. Receive input from left neighbor (or external input for col 0)
    2. Multiply weight × input optically
    3. Accumulate product electronically
    4. Pass input to right neighbor
    """
    row: int = 0
    col: int = 0
    weight: int = 0           # loaded trit value from A[row][col]
    accumulator: int = 0      # running sum of products
    input_reg: int = 0        # current input (from left or external)
    output_reg: int = 0       # passes input to right neighbor
    nrioc: NRIOCModule = field(default_factory=NRIOCModule)
    active: bool = False      # True when a valid input is present

    def load_weight(self, trit: int):
        """Load a weight value from matrix A."""
        self.weight = trit

    def reset_accumulator(self):
        """Clear accumulator for a new computation."""
        self.accumulator = 0
        self.active = False
        self.input_reg = 0
        self.output_reg = 0

    def feed_input(self, trit: int):
        """Receive an input trit from the left (or external for col 0)."""
        self.input_reg = trit
        self.active = True

    def feed_idle(self):
        """No valid input this cycle (before/after systolic window)."""
        self.active = False
        self.input_reg = 0

    def execute(self):
        """Execute one clock cycle: multiply + accumulate + pass right.

        The optical multiply (SFG + AWG + detect) happens here.
        """
        if self.active and self.weight != 0 and self.input_reg != 0:
            product = self.nrioc.multiply(self.weight, self.input_reg)
            self.accumulator += product
        # Pass input to right neighbor
        self.output_reg = self.input_reg if self.active else 0


@dataclass
class SystolicArray2D:
    """9x9 2D Optical Systolic Array.

    Weight-stationary dataflow for ternary matrix multiplication.
    Uses unbalanced ternary {1, 2, 3}.

    Architecture:
    - 81 PEs arranged in a 9×9 grid
    - Weight matrix A pre-loaded: PE(i,j) holds A[i][j]
    - Input vector b enters column 0 with systolic skew:
        b[k] arrives at col c on cycle (k + c)
    - After N + N - 1 = 17 cycles, each PE(i,j) has accumulated A[i][j] * b[j]
    - Row sums give C[i] = sum_j A[i][j] * b[j]
    """
    size: int = ARRAY_SIZE
    pes: list[list[SystolicPE]] = field(init=False)

    def __post_init__(self):
        self.pes = [
            [SystolicPE(row=r, col=c) for c in range(self.size)]
            for r in range(self.size)
        ]

    def load_weights(self, matrix: list[list[int]]):
        """Load weight matrix A into the PE array.

        A[i][j] goes into PE(i, j).
        """
        for i in range(self.size):
            for j in range(self.size):
                self.pes[i][j].load_weight(matrix[i][j])

    def _reset_accumulators(self):
        """Clear all PE accumulators for a new computation."""
        for row in self.pes:
            for pe in row:
                pe.reset_accumulator()

    def _read_results(self) -> list[int]:
        """Read output vector: C[i] = sum of accumulators across row i."""
        results = []
        for i in range(self.size):
            row_sum = sum(self.pes[i][j].accumulator for j in range(self.size))
            results.append(row_sum)
        return results

    def _read_accumulator_grid(self) -> list[list[int]]:
        """Read the full 9×9 accumulator state (before row-summing)."""
        return [
            [self.pes[i][j].accumulator for j in range(self.size)]
            for i in range(self.size)
        ]

    def matrix_vector_multiply(self, vector: list[int]) -> list[int]:
        """Compute C = A × b where A is pre-loaded as weights.

        Weight-stationary systolic dataflow:
        - b[k] enters column 0 on cycle k
        - b[k] arrives at column c on cycle k + c (systolic skew)
        - Total: N + (N-1) = 17 cycles

        Args:
            vector: Input vector b of length 9, values in {1, 2, 3}

        Returns:
            Output vector C of length 9
        """
        self._reset_accumulators()
        total_cycles = self.size + (self.size - 1)  # 17

        for cycle in range(total_cycles):
            # Phase 1: Determine inputs for each column
            # Column c receives b[k] where k = cycle - c
            for c in range(self.size):
                k = cycle - c
                if 0 <= k < self.size:
                    # Valid input: b[k] arrives at column c
                    for r in range(self.size):
                        self.pes[r][c].feed_input(vector[k])
                else:
                    # No valid input (before first or after last)
                    for r in range(self.size):
                        self.pes[r][c].feed_idle()

            # Phase 2+3: All PEs execute in parallel (multiply + accumulate)
            for r in range(self.size):
                for c in range(self.size):
                    self.pes[r][c].execute()

        return self._read_results()

    def matrix_matrix_multiply(self, matrix_b: list[list[int]]) -> list[list[int]]:
        """Compute C = A × B where A is pre-loaded as weights.

        Processes B column by column. Each column is a matrix-vector multiply.
        Total: 9 × 17 = 153 clock cycles.

        Args:
            matrix_b: Input matrix B (9×9), values in {1, 2, 3}

        Returns:
            Output matrix C (9×9)
        """
        result = []
        # Extract columns of B
        for j in range(self.size):
            col_b = [matrix_b[k][j] for k in range(self.size)]
            col_c = self.matrix_vector_multiply(col_b)
            result.append(col_c)

        # result[j] is column j of C, transpose to row-major
        return [[result[j][i] for j in range(self.size)] for i in range(self.size)]


def self_test():
    """Run self-test of the 2D systolic array."""
    print("9x9 2D Optical Systolic Array — Self-Test")
    print("=" * 55)

    array = SystolicArray2D()

    # Test 1: Identity-like weight test (all weights = 1)
    print("\nTest 1: All weights = 1, vector = [1,2,3,1,2,3,1,2,3]")
    ones = [[1]*9 for _ in range(9)]
    array.load_weights(ones)
    vec = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    result = array.matrix_vector_multiply(vec)
    expected_sum = sum(vec)  # each row sums all elements
    expected = [expected_sum] * 9
    status = "PASS" if result == expected else "FAIL"
    print(f"  Result:   {result}")
    print(f"  Expected: {expected}")
    print(f"  [{status}]")

    # Test 2: Simple 9x9 matrix-vector
    print("\nTest 2: Diagonal weight matrix × vector")
    diag = [[0]*9 for _ in range(9)]
    # Can't use 0 in ternary {1,2,3}, so use identity-like: A[i][i] = 2, rest = 1
    weights = [[1]*9 for _ in range(9)]
    for i in range(9):
        weights[i][i] = 3  # diagonal = 3, off-diagonal = 1
    array.load_weights(weights)
    vec = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    result = array.matrix_vector_multiply(vec)
    # Each row: 8 × (1×1) + 1 × (3×1) = 8 + 3 = 11
    expected = [11] * 9
    status = "PASS" if result == expected else "FAIL"
    print(f"  Result:   {result}")
    print(f"  Expected: {expected}")
    print(f"  [{status}]")

    # Test 3: Known 3x3 subproblem (embed in 9x9)
    print("\nTest 3: Manual 3x3 in top-left corner")
    # Use a minimal 3x3 case, padding rest with 1s
    weights = [[1]*9 for _ in range(9)]
    # Top-left 3x3:
    # [2 3 1]   [1]   [2*1 + 3*2 + 1*3] = [11]
    # [1 2 3] × [2] = [1*1 + 2*2 + 3*3] = [14]
    # [3 1 2]   [3]   [3*1 + 1*2 + 2*3] = [11]
    sub = [[2, 3, 1], [1, 2, 3], [3, 1, 2]]
    for i in range(3):
        for j in range(3):
            weights[i][j] = sub[i][j]
    array.load_weights(weights)
    # Vector: first 3 are [1,2,3], rest are 1
    vec = [1, 2, 3, 1, 1, 1, 1, 1, 1]
    result = array.matrix_vector_multiply(vec)
    # Row 0: 2*1 + 3*2 + 1*3 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1 = 11 + 6 = 17
    # Row 1: 1*1 + 2*2 + 3*3 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1 = 14 + 6 = 20
    # Row 2: 3*1 + 1*2 + 2*3 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1 = 11 + 6 = 17
    # Rows 3-8: all 1s × vec = 1+2+3+1+1+1+1+1+1 = 12
    expected = [17, 20, 17] + [12]*6
    status = "PASS" if result == expected else "FAIL"
    print(f"  Result:   {result}")
    print(f"  Expected: {expected}")
    print(f"  [{status}]")

    # Test 4: Matrix-matrix multiply (small verification)
    print("\nTest 4: Matrix-matrix multiply (all 2s × all 1s)")
    weights = [[2]*9 for _ in range(9)]
    array.load_weights(weights)
    mat_b = [[1]*9 for _ in range(9)]
    result = array.matrix_matrix_multiply(mat_b)
    # Each element: sum of 9 terms of (2×1) = 18
    expected = [[18]*9 for _ in range(9)]
    status = "PASS" if result == expected else "FAIL"
    print(f"  Result[0]: {result[0]}")
    print(f"  Expected:  {expected[0]}")
    all_match = all(result[i] == expected[i] for i in range(9))
    print(f"  All rows match: {all_match} [{status}]")

    print()


if __name__ == "__main__":
    self_test()
