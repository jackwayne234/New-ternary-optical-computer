"""
Optical Systolic Array — Math-First Redesign

Architecture: multiply-only optical + electronic accumulate.

Each Processing Element (PE):
  OPTICAL: SFG multiplies (two wavelengths in, product wavelength out)
           AWG routes product to correct photodetector port
  ELECTRONIC: Photodetectors convert to current
              Accumulator register adds products over time

The optical side does the expensive part (parallel multiplication).
The electronic side does the cheap part (sequential accumulation).

This module defines the PE and a 1D systolic array for matrix-vector multiply.
For the initial prototype, we focus on a single PE (1 trit x 1 trit)
but structure the code to scale to arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math.trit_multiplication import TRIT_VALUES, UNIQUE_PRODUCTS
from components.nrioc_module import NRIOCModule


@dataclass
class AccumulatorRegister:
    """Electronic accumulator register.

    Receives integer product values from the photodetector array
    and accumulates them over multiple clock cycles.
    """
    value: int = 0
    history: list[int] = field(default_factory=list)

    def accumulate(self, product: int):
        """Add a product value to the running sum."""
        self.history.append(product)
        self.value += product

    def reset(self):
        """Clear the accumulator for a new computation."""
        self.value = 0
        self.history.clear()


@dataclass
class ProcessingElement:
    """Single Processing Element in the optical systolic array.

    Combines:
    - NRIOC module (optical multiply: SFG + AWG + photodetectors)
    - Accumulator register (electronic add)

    One PE computes one element of a dot product:
      result = sum over k of (A[k] * B[k])
    where each A[k]*B[k] is done optically and the sum is electronic.
    """
    pe_id: int = 0
    nrioc: NRIOCModule = field(default_factory=NRIOCModule)
    accumulator: AccumulatorRegister = field(default_factory=AccumulatorRegister)

    def multiply_and_accumulate(self, trit_a: int, trit_b: int):
        """Perform one optical multiply and accumulate the result electronically.

        This is one clock cycle of the PE:
        1. Optical: SFG produces product wavelength, AWG routes it
        2. Electronic: Photodetector converts, accumulator adds
        """
        product = self.nrioc.multiply(trit_a, trit_b)
        self.accumulator.accumulate(product)

    @property
    def result(self) -> int:
        """Current accumulated value."""
        return self.accumulator.value

    def reset(self):
        """Reset for a new dot product computation."""
        self.accumulator.reset()


@dataclass
class OpticalSystolicArray:
    """1D Optical Systolic Array for trit-vector dot products.

    Contains N processing elements, each computing one output element.
    For matrix-vector multiply C = A * b:
      - Each PE[i] computes C[i] = sum_k A[i][k] * b[k]
      - A values are stationary in each PE
      - b values flow through the array

    For the initial prototype (single trit x trit), this is just 1 PE.
    The structure is here to scale up.
    """
    num_pes: int = 1
    pes: list[ProcessingElement] = field(default_factory=list)

    def __post_init__(self):
        if not self.pes:
            self.pes = [ProcessingElement(pe_id=i) for i in range(self.num_pes)]

    def dot_product(self, vec_a: list[int], vec_b: list[int], pe_index: int = 0) -> int:
        """Compute dot product of two trit vectors using one PE.

        Each element pair is multiplied optically and accumulated electronically.
        """
        if len(vec_a) != len(vec_b):
            raise ValueError(
                f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
            )

        pe = self.pes[pe_index]
        pe.reset()

        for a, b in zip(vec_a, vec_b):
            pe.multiply_and_accumulate(a, b)

        return pe.result

    def matrix_vector_multiply(
        self, matrix: list[list[int]], vector: list[int]
    ) -> list[int]:
        """Compute matrix-vector product C = A * b.

        Each row of A is dot-producted with b using a separate PE.
        All PEs operate in parallel (in hardware); here we simulate sequentially.
        """
        num_rows = len(matrix)
        if num_rows > self.num_pes:
            raise ValueError(
                f"Matrix has {num_rows} rows but array only has {self.num_pes} PEs"
            )

        results = []
        for i, row in enumerate(matrix):
            result = self.dot_product(row, vector, pe_index=i)
            results.append(result)

        return results

    def reset_all(self):
        """Reset all PEs for a new computation."""
        for pe in self.pes:
            pe.reset()


def self_test():
    """Run self-test of the optical systolic array."""
    print("Optical Systolic Array Self-Test")
    print("=" * 50)

    # Test 1: Single trit x trit (1 PE)
    print("\nTest 1: Single trit x trit multiplication")
    array = OpticalSystolicArray(num_pes=1)
    for a in TRIT_VALUES:
        for b in TRIT_VALUES:
            result = array.dot_product([a], [b])
            expected = a * b
            status = "PASS" if result == expected else "FAIL"
            print(f"  {a} x {b} = {result} (expected {expected}) [{status}]")
            array.reset_all()

    # Test 2: Dot product of trit vectors
    print("\nTest 2: Dot product [2, 3, 1] . [1, 2, 3]")
    array = OpticalSystolicArray(num_pes=1)
    vec_a = [2, 3, 1]
    vec_b = [1, 2, 3]
    result = array.dot_product(vec_a, vec_b)
    expected = 2*1 + 3*2 + 1*3  # = 2 + 6 + 3 = 11
    print(f"  Result: {result}, Expected: {expected}, {'PASS' if result == expected else 'FAIL'}")

    # Test 3: Matrix-vector multiply
    print("\nTest 3: Matrix-vector multiply")
    matrix = [
        [1, 2],
        [3, 1],
        [2, 3],
    ]
    vector = [2, 3]
    expected_result = [
        1*2 + 2*3,  # = 8
        3*2 + 1*3,  # = 9
        2*2 + 3*3,  # = 13
    ]
    array = OpticalSystolicArray(num_pes=3)
    result = array.matrix_vector_multiply(matrix, vector)
    print(f"  Matrix: {matrix}")
    print(f"  Vector: {vector}")
    print(f"  Result:   {result}")
    print(f"  Expected: {expected_result}")
    print(f"  {'PASS' if result == expected_result else 'FAIL'}")

    # Test 4: Accumulator history
    print("\nTest 4: Accumulator history for dot product [2,3,1].[1,2,3]")
    array = OpticalSystolicArray(num_pes=1)
    pe = array.pes[0]
    for a, b in zip([2, 3, 1], [1, 2, 3]):
        pe.multiply_and_accumulate(a, b)
        print(f"  {a} x {b} = {a*b}, accumulator = {pe.result}, history = {pe.accumulator.history}")

    print()


if __name__ == "__main__":
    self_test()
