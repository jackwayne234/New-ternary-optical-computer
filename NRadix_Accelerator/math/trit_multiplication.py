"""
Single Trit x Trit Optical Multiplier — Math & Wavelength Assignments

Unbalanced ternary {1, 2, 3} multiplication.
SFG (sum frequency generation): f_out = f_A + f_B
AWG routes each unique product frequency to a dedicated output port.

Wavelength selection rationale:
  We pick 3 input frequencies in the telecom C-band (~1530-1565 nm) where:
  - SFG outputs land in the near-IR (~765-783 nm), well within silicon photodetector range
  - All 6 product frequencies are separated by >= 2 THz, easily resolvable by AWG
  - The input wavelengths are compatible with standard telecom laser sources

  Trit value -> input frequency mapping:
    1 -> f1 = 192.0 THz  (1561.42 nm)
    2 -> f2 = 194.0 THz  (1545.32 nm)
    3 -> f3 = 196.0 THz  (1529.55 nm)

  Spacing: 2 THz between adjacent input values.

  SFG product frequencies (f_A + f_B):
    1x1 = 1 -> 192.0 + 192.0 = 384.0 THz  (780.88 nm)
    1x2 = 2 -> 192.0 + 194.0 = 386.0 THz  (776.81 nm)
    1x3 = 3 -> 192.0 + 196.0 = 388.0 THz  (772.78 nm)
    2x2 = 4 -> 194.0 + 194.0 = 388.0 THz  -- COLLISION with 1x3!

  Problem: with uniform 2 THz spacing, products 3 and 4 collide at 388.0 THz.

  Fix: use non-uniform spacing so all 6 sums are distinct.

  Revised mapping (non-uniform spacing):
    1 -> f1 = 192.0 THz  (1561.42 nm)
    2 -> f2 = 195.0 THz  (1537.89 nm)
    3 -> f3 = 199.0 THz  (1506.28 nm)

  SFG products with revised mapping:
    1x1 = 1 -> 192.0 + 192.0 = 384.0 THz  (780.88 nm)
    1x2 = 2 -> 192.0 + 195.0 = 387.0 THz  (774.78 nm)
    1x3 = 3 -> 192.0 + 199.0 = 391.0 THz  (766.84 nm)
    2x2 = 4 -> 195.0 + 195.0 = 390.0 THz  (768.80 nm)
    2x3 = 6 -> 195.0 + 199.0 = 394.0 THz  (761.04 nm)
    3x3 = 9 -> 199.0 + 199.0 = 398.0 THz  (753.39 nm)

  All 6 output frequencies: {384, 387, 390, 391, 394, 398} THz
  Minimum separation: 1.0 THz (between 390 and 391 THz, products 4 and 3)

  1 THz separation is tight but achievable for a well-designed AWG.
  If more margin is needed, the FDTDX optimizer can explore slightly adjusted inputs.

  BETTER revised mapping (wider minimum separation):
    1 -> f1 = 191.0 THz  (1569.59 nm)
    2 -> f2 = 195.5 THz  (1533.96 nm)
    3 -> f3 = 201.0 THz  (1491.54 nm)

  SFG products:
    1x1 = 1 -> 191.0 + 191.0 = 382.0 THz  (784.97 nm)
    1x2 = 2 -> 191.0 + 195.5 = 386.5 THz  (775.81 nm)
    1x3 = 3 -> 191.0 + 201.0 = 392.0 THz  (764.88 nm)
    2x2 = 4 -> 195.5 + 195.5 = 391.0 THz  (766.84 nm)
    2x3 = 6 -> 195.5 + 201.0 = 396.5 THz  (756.19 nm)
    3x3 = 9 -> 201.0 + 201.0 = 402.0 THz  (745.77 nm)

  Output frequencies: {382.0, 386.5, 391.0, 392.0, 396.5, 402.0} THz
  Minimum separation: 1.0 THz (between 391.0 and 392.0, products 4 and 3)

  The 4-vs-3 collision is inherent: product 4 = 2*f2, product 3 = f1+f3.
  For these to differ: 2*f2 != f1+f3, i.e., f2 != (f1+f3)/2.
  With our choice: (191+201)/2 = 196.0 != 195.5. Gap = 1.0 THz.

  To widen: push f2 further from the midpoint.

  FINAL mapping (maximized minimum separation):
    1 -> f1 = 191.0 THz  (1569.59 nm)
    2 -> f2 = 194.0 THz  (1545.32 nm)
    3 -> f3 = 201.0 THz  (1491.54 nm)

  SFG products:
    1x1 = 1 -> 382.0 THz  (784.97 nm)
    1x2 = 2 -> 385.0 THz  (778.83 nm)
    1x3 = 3 -> 392.0 THz  (764.88 nm)
    2x2 = 4 -> 388.0 THz  (772.78 nm)
    2x3 = 6 -> 395.0 THz  (759.06 nm)
    3x3 = 9 -> 402.0 THz  (745.77 nm)

  Sorted output frequencies: {382.0, 385.0, 388.0, 392.0, 395.0, 402.0} THz
  Gaps: 3.0, 3.0, 4.0, 3.0, 7.0 THz
  Minimum separation: 3.0 THz -- excellent for AWG routing!

  Verification: 2*f2 = 388.0, f1+f3 = 392.0, gap = 4.0 THz. Clean.
"""

from dataclasses import dataclass

# Speed of light in vacuum (m/s)
C_LIGHT = 299_792_458.0

# --- Unbalanced ternary trit values ---
TRIT_VALUES = (1, 2, 3)


@dataclass(frozen=True)
class WavelengthAssignment:
    """Maps trit values to optical frequencies for SFG multiplication."""
    # Input frequencies in THz (telecom C/L band)
    f1: float = 191.0   # trit=1 -> 1569.59 nm
    f2: float = 194.0   # trit=2 -> 1545.32 nm
    f3: float = 201.0   # trit=3 -> 1491.54 nm

    @property
    def trit_to_freq(self) -> dict[int, float]:
        return {1: self.f1, 2: self.f2, 3: self.f3}

    @property
    def freq_to_trit(self) -> dict[float, int]:
        return {self.f1: 1, self.f2: 2, self.f3: 3}

    def freq_thz(self, trit: int) -> float:
        """Get input frequency in THz for a trit value."""
        return self.trit_to_freq[trit]

    def wavelength_nm(self, trit: int) -> float:
        """Get input wavelength in nm for a trit value."""
        freq_hz = self.trit_to_freq[trit] * 1e12
        return (C_LIGHT / freq_hz) * 1e9


# Singleton for the chosen assignment
WAVELENGTHS = WavelengthAssignment()


def sfg_product_freq(trit_a: int, trit_b: int) -> float:
    """Compute SFG output frequency (THz) for two input trits.

    SFG: f_out = f_A + f_B (frequency-domain addition).
    This is the physics that performs multiplication:
    the output color IS the product.
    """
    return WAVELENGTHS.freq_thz(trit_a) + WAVELENGTHS.freq_thz(trit_b)


def sfg_product_wavelength_nm(trit_a: int, trit_b: int) -> float:
    """Compute SFG output wavelength (nm) for two input trits."""
    freq_hz = sfg_product_freq(trit_a, trit_b) * 1e12
    return (C_LIGHT / freq_hz) * 1e9


# --- Multiplication table ---

def build_multiplication_table() -> dict[tuple[int, int], int]:
    """Build the full 3x3 trit multiplication table."""
    table = {}
    for a in TRIT_VALUES:
        for b in TRIT_VALUES:
            table[(a, b)] = a * b
    return table


MULTIPLICATION_TABLE = build_multiplication_table()

# Unique products sorted
UNIQUE_PRODUCTS = sorted(set(MULTIPLICATION_TABLE.values()))
# -> [1, 2, 3, 4, 6, 9]

assert UNIQUE_PRODUCTS == [1, 2, 3, 4, 6, 9], (
    f"Expected 6 unique products [1,2,3,4,6,9], got {UNIQUE_PRODUCTS}"
)


# --- AWG output port mapping ---
# Each unique product maps to a dedicated AWG output port.
# Port index 0-5 corresponds to products sorted by output frequency (ascending).

def build_product_to_port() -> dict[int, int]:
    """Map each product value to its AWG output port index.

    Ports are ordered by ascending SFG output frequency.
    """
    # For each unique product, find the SFG frequency that produces it.
    # Use the canonical (min trit_a, max trit_b) combo.
    product_freq_pairs = []
    seen = set()
    for (a, b), product in MULTIPLICATION_TABLE.items():
        if product not in seen:
            seen.add(product)
            product_freq_pairs.append((product, sfg_product_freq(a, b)))

    # Sort by frequency (ascending) -> port 0 is lowest freq
    product_freq_pairs.sort(key=lambda pf: pf[1])

    return {product: port_idx for port_idx, (product, _freq) in enumerate(product_freq_pairs)}


PRODUCT_TO_PORT = build_product_to_port()
PORT_TO_PRODUCT = {v: k for k, v in PRODUCT_TO_PORT.items()}


def build_product_frequency_table() -> dict[int, float]:
    """Map each product value to its SFG output frequency in THz."""
    table = {}
    seen = set()
    for (a, b), product in MULTIPLICATION_TABLE.items():
        if product not in seen:
            seen.add(product)
            table[product] = sfg_product_freq(a, b)
    return table


PRODUCT_FREQUENCIES = build_product_frequency_table()


# --- Validation ---

def validate_frequency_separation(min_sep_thz: float = 1.0) -> list[str]:
    """Check that all product frequencies are separated by at least min_sep_thz.

    Returns list of issues (empty = pass).
    """
    issues = []
    freqs = sorted(PRODUCT_FREQUENCIES.items(), key=lambda pf: pf[1])

    for i in range(len(freqs) - 1):
        prod_a, freq_a = freqs[i]
        prod_b, freq_b = freqs[i + 1]
        gap = freq_b - freq_a
        if gap < min_sep_thz:
            issues.append(
                f"Products {prod_a} ({freq_a:.1f} THz) and {prod_b} ({freq_b:.1f} THz) "
                f"are only {gap:.1f} THz apart (need >= {min_sep_thz:.1f})"
            )

    return issues


def print_summary():
    """Print a human-readable summary of the wavelength assignments."""
    print("=" * 60)
    print("Single Trit x Trit Optical Multiplier")
    print("Unbalanced Ternary {1, 2, 3}")
    print("=" * 60)

    print("\nInput Wavelength Assignments:")
    print(f"  {'Trit':>4}  {'Freq (THz)':>10}  {'Wavelength (nm)':>15}")
    print(f"  {'----':>4}  {'----------':>10}  {'---------------':>15}")
    for t in TRIT_VALUES:
        print(f"  {t:>4}  {WAVELENGTHS.freq_thz(t):>10.1f}  {WAVELENGTHS.wavelength_nm(t):>15.2f}")

    print("\nMultiplication Table (product values):")
    print(f"  {'x':>4}", end="")
    for b in TRIT_VALUES:
        print(f"  {b:>4}", end="")
    print()
    print(f"  {'----':>4}  {'----':>4}  {'----':>4}  {'----':>4}")
    for a in TRIT_VALUES:
        print(f"  {a:>4}", end="")
        for b in TRIT_VALUES:
            print(f"  {a*b:>4}", end="")
        print()

    print("\nSFG Output Frequencies (sorted by frequency):")
    print(f"  {'Product':>7}  {'Freq (THz)':>10}  {'Wavelength (nm)':>15}  {'AWG Port':>8}")
    print(f"  {'-------':>7}  {'----------':>10}  {'---------------':>15}  {'--------':>8}")
    for product in UNIQUE_PRODUCTS:
        freq = PRODUCT_FREQUENCIES[product]
        wl = (C_LIGHT / (freq * 1e12)) * 1e9
        port = PRODUCT_TO_PORT[product]
        print(f"  {product:>7}  {freq:>10.1f}  {wl:>15.2f}  {port:>8}")

    print("\nFrequency Gaps Between Adjacent Products:")
    freqs_sorted = sorted(PRODUCT_FREQUENCIES.items(), key=lambda pf: pf[1])
    for i in range(len(freqs_sorted) - 1):
        pa, fa = freqs_sorted[i]
        pb, fb = freqs_sorted[i + 1]
        print(f"  Product {pa} -> {pb}: {fb - fa:.1f} THz")

    issues = validate_frequency_separation(min_sep_thz=1.0)
    if issues:
        print("\nWARNINGS:")
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print("\n  All product frequencies separated by >= 1.0 THz")

    print()


if __name__ == "__main__":
    print_summary()
