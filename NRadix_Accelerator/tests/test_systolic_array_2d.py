"""
Tests for 9x9 2D Optical Systolic Array

Verifies matrix-vector and matrix-matrix multiply against numpy reference.
All matrices use unbalanced ternary values {1, 2, 3}.
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from architecture.systolic_array_2d import SystolicArray2D, ARRAY_SIZE


def random_trit_vector(n: int = ARRAY_SIZE) -> list[int]:
    """Generate a random vector with values in {1, 2, 3}."""
    return [random.choice([1, 2, 3]) for _ in range(n)]


def random_trit_matrix(n: int = ARRAY_SIZE) -> list[list[int]]:
    """Generate a random n×n matrix with values in {1, 2, 3}."""
    return [random_trit_vector(n) for _ in range(n)]


def numpy_mat_vec(matrix: list[list[int]], vector: list[int]) -> list[int]:
    """Reference matrix-vector multiply using numpy."""
    a = np.array(matrix, dtype=np.int64)
    b = np.array(vector, dtype=np.int64)
    return (a @ b).tolist()


def numpy_mat_mat(matrix_a: list[list[int]], matrix_b: list[list[int]]) -> list[list[int]]:
    """Reference matrix-matrix multiply using numpy."""
    a = np.array(matrix_a, dtype=np.int64)
    b = np.array(matrix_b, dtype=np.int64)
    return (a @ b).tolist()


def test_mat_vec_all_ones():
    """A = all 1s, b = all 1s => C[i] = 9 for all i."""
    array = SystolicArray2D()
    weights = [[1] * 9 for _ in range(9)]
    vector = [1] * 9
    array.load_weights(weights)
    result = array.matrix_vector_multiply(vector)
    expected = numpy_mat_vec(weights, vector)
    assert result == expected, f"all_ones: {result} != {expected}"
    print("  PASS: mat_vec_all_ones")


def test_mat_vec_all_threes():
    """A = all 3s, b = all 3s => C[i] = 81 for all i."""
    array = SystolicArray2D()
    weights = [[3] * 9 for _ in range(9)]
    vector = [3] * 9
    array.load_weights(weights)
    result = array.matrix_vector_multiply(vector)
    expected = numpy_mat_vec(weights, vector)
    assert result == expected, f"all_threes: {result} != {expected}"
    print("  PASS: mat_vec_all_threes")


def test_mat_vec_diagonal():
    """Diagonal 2s, off-diagonal 1s."""
    array = SystolicArray2D()
    weights = [[1] * 9 for _ in range(9)]
    for i in range(9):
        weights[i][i] = 2
    vector = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    array.load_weights(weights)
    result = array.matrix_vector_multiply(vector)
    expected = numpy_mat_vec(weights, vector)
    assert result == expected, f"diagonal: {result} != {expected}"
    print("  PASS: mat_vec_diagonal")


def test_mat_vec_random(num_trials: int = 10):
    """Random matrices and vectors, compare against numpy."""
    array = SystolicArray2D()
    for trial in range(num_trials):
        weights = random_trit_matrix()
        vector = random_trit_vector()
        array.load_weights(weights)
        result = array.matrix_vector_multiply(vector)
        expected = numpy_mat_vec(weights, vector)
        assert result == expected, (
            f"random trial {trial}: {result} != {expected}\n"
            f"  weights={weights}\n  vector={vector}"
        )
    print(f"  PASS: mat_vec_random ({num_trials} trials)")


def test_mat_mat_all_ones():
    """A = all 1s, B = all 1s => C[i][j] = 9 for all i, j."""
    array = SystolicArray2D()
    weights = [[1] * 9 for _ in range(9)]
    mat_b = [[1] * 9 for _ in range(9)]
    array.load_weights(weights)
    result = array.matrix_matrix_multiply(mat_b)
    expected = numpy_mat_mat(weights, mat_b)
    assert result == expected, f"mat_mat_all_ones: {result} != {expected}"
    print("  PASS: mat_mat_all_ones")


def test_mat_mat_mixed():
    """A = all 2s, B = all 3s => C[i][j] = 54 for all i, j."""
    array = SystolicArray2D()
    weights = [[2] * 9 for _ in range(9)]
    mat_b = [[3] * 9 for _ in range(9)]
    array.load_weights(weights)
    result = array.matrix_matrix_multiply(mat_b)
    expected = numpy_mat_mat(weights, mat_b)
    assert result == expected, f"mat_mat_mixed: {result} != {expected}"
    print("  PASS: mat_mat_mixed")


def test_mat_mat_random(num_trials: int = 5):
    """Random matrix-matrix multiply, compare against numpy."""
    array = SystolicArray2D()
    for trial in range(num_trials):
        weights = random_trit_matrix()
        mat_b = random_trit_matrix()
        array.load_weights(weights)
        result = array.matrix_matrix_multiply(mat_b)
        expected = numpy_mat_mat(weights, mat_b)
        assert result == expected, (
            f"random mat-mat trial {trial}: result != expected\n"
            f"  result[0]={result[0]}\n  expected[0]={expected[0]}"
        )
    print(f"  PASS: mat_mat_random ({num_trials} trials)")


def test_accumulator_isolation():
    """Verify that consecutive multiplies don't leak state."""
    array = SystolicArray2D()
    weights = [[2] * 9 for _ in range(9)]
    array.load_weights(weights)

    vec1 = [1] * 9
    result1 = array.matrix_vector_multiply(vec1)
    expected1 = numpy_mat_vec(weights, vec1)

    vec2 = [3] * 9
    result2 = array.matrix_vector_multiply(vec2)
    expected2 = numpy_mat_vec(weights, vec2)

    assert result1 == expected1, f"isolation vec1: {result1} != {expected1}"
    assert result2 == expected2, f"isolation vec2: {result2} != {expected2}"
    print("  PASS: accumulator_isolation")


def run_all_tests():
    """Run all tests."""
    random.seed(42)
    print("=" * 55)
    print("Test Suite: 9x9 2D Optical Systolic Array")
    print("=" * 55)

    print("\nMatrix-Vector Tests:")
    test_mat_vec_all_ones()
    test_mat_vec_all_threes()
    test_mat_vec_diagonal()
    test_mat_vec_random()

    print("\nMatrix-Matrix Tests:")
    test_mat_mat_all_ones()
    test_mat_mat_mixed()
    test_mat_mat_random()

    print("\nRegression Tests:")
    test_accumulator_isolation()

    print("\n" + "=" * 55)
    print("ALL TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    run_all_tests()
