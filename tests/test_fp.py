"""The arithmetic unit against exact arithmetic.

The theme: every result must equal the true mathematical result rounded once,
ties to even, and every flag must describe that one rounding honestly.
"""
import math
import random
import struct
import unittest
from fractions import Fraction

from mantissa import fp
from mantissa.fp import (
    D, DIVZERO, F, H, INF, INEXACT, INVALID, NAN, NINF, NNORMAL, NSUBNORMAL, NZERO,
    OVERFLOW, PINF, PNORMAL, PSUBNORMAL, PZERO, QNAN, UNDERFLOW,
)
from . import support

FORMATS = (H, F, D)  # the formats a double holds exactly; quad runs its own class

MAX_FINITE = {H: 65504.0, F: 3.4028234663852886e+38, D: 1.7976931348623157e+308}
LAST_ULP = {H: 32.0, F: 2.0 ** 104, D: 2.0 ** 971}
MIN_SUBNORMAL = {H: 2.0 ** -24, F: 2.0 ** -149, D: 2.0 ** -1074}


def struct_double(bits):
    return struct.unpack("<d", bits.to_bytes(8, "little"))[0]


def bits32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


class CellsAndBitPatterns(unittest.TestCase):
    def test_every_half_round_trips_through_its_cells(self):
        for bits in range(1 << 16):
            value = fp.from_bits(H, bits)
            self.assertEqual(fp.to_bits(H, value), bits, f"half {bits:#06x}")
            self.assertEqual(fp.pack(H, value), (bits,))
            self.assertTrue(support.same(fp.unpack(H, (bits,)), value))

    def test_agrees_with_the_struct_module_on_half_values(self):
        for bits in range(1 << 16):
            if bits & 0x7C00 != 0x7C00:  # not NaN: the two must see eye to eye
                self.assertEqual(fp.from_bits(H, bits), support.from_pattern(H, bits), f"half {bits:#06x}")

    def test_random_patterns_round_trip_in_every_format(self):
        rng = random.Random(16)
        for fmt in FORMATS:
            for _ in range(20000):
                bits = rng.getrandbits(fmt.bits)
                value = fp.from_bits(fmt, bits)
                self.assertEqual(fp.to_bits(fmt, value), bits, f"{fmt.name} {bits:#x}")
                cells = fp.pack(fmt, value)
                self.assertEqual(len(cells), fmt.cells)
                self.assertTrue(support.same(fp.unpack(fmt, cells), value))

    def test_the_most_significant_cell_comes_first(self):
        self.assertEqual(fp.pack(F, 1.0), (0x3F80, 0x0000))
        self.assertEqual(fp.pack(D, 1.0), (0x3FF0, 0x0000, 0x0000, 0x0000))
        for fmt, value in ((H, 3.5), (F, 0.1), (D, -2.75)):
            pattern = int.from_bytes(struct.pack("<" + support.STRUCT_CODE[fmt.name], value), "little")
            expected = tuple(pattern >> 16 * (fmt.cells - 1 - i) & 0xFFFF for i in range(fmt.cells))
            self.assertEqual(fp.pack(fmt, value), expected)

    def test_nan_payloads_survive_a_widening_and_a_narrowing(self):
        for bits in (0x7C01, 0x7E00, 0x7FFF, 0xFC01, 0xFE00, 0x7D2A):
            wide = fp.from_bits(H, bits)
            self.assertNotEqual(wide, wide)
            self.assertEqual(fp.to_bits(H, wide), bits, f"half {bits:#06x}")
        for bits in (0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0xFF800001, 0xFFC01234):
            wide = fp.from_bits(F, bits)
            self.assertNotEqual(wide, wide)
            self.assertEqual(fp.to_bits(F, wide), bits, f"float {bits:#010x}")

    def test_the_canonical_nan_is_the_one_with_no_payload(self):
        self.assertEqual(fp.to_bits(H, NAN), 0x7E00)
        self.assertEqual(fp.to_bits(F, NAN), 0x7FC00000)
        self.assertEqual(fp.to_bits(D, NAN), 0x7FF8000000000000)
        self.assertEqual(fp.to_bits(F, -NAN), 0xFFC00000)

    def test_flag_names(self):
        self.assertEqual(fp.flag_names(0), [])
        self.assertEqual(fp.flag_names(INVALID | INEXACT), ["invalid", "inexact"])
        self.assertEqual(len(fp.flag_names(31)), 5)


class Rounding(unittest.TestCase):
    """mantissa.fp.round_literal against the exact reference."""

    def check(self, fmt, x):
        got = fp.round_literal(fmt, x)
        expected, _ = (x, 0) if x == 0 else support.reference(fmt, Fraction(x))
        self.assertTrue(support.same(got, expected), f"{fmt.name} {x!r}: got {got!r}, want {expected!r}")

    def test_neighbours_and_what_lies_between_them(self):
        rng = random.Random(754)
        for fmt in FORMATS:
            for _ in range(3000):
                while True:  # two adjacent positive format values
                    bits = rng.getrandbits(fmt.bits - 1)
                    lo, hi = support.from_pattern(fmt, bits), support.from_pattern(fmt, bits + 1)
                    if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
                        break
                mid = lo + (hi - lo) / 2
                for x in (lo, hi, mid, math.nextafter(mid, lo), math.nextafter(mid, hi),
                          math.nextafter(mid, -INF), math.nextafter(mid, INF),
                          lo + (hi - lo) * rng.random()):
                    self.check(fmt, x)

    def test_random_doubles_and_random_decimals(self):
        rng = random.Random(755)
        for fmt in FORMATS:
            for _ in range(3000):
                x = struct_double(rng.getrandbits(64))
                if math.isfinite(x):
                    self.check(fmt, x)
                self.check(fmt, rng.uniform(-1e6, 1e6))
                self.check(fmt, rng.uniform(-1, 1) * 10.0 ** rng.randrange(-307, 308))

    def test_signed_zeros_are_kept(self):
        self.assertEqual(fp.to_bits(H, fp.round_literal(H, -1e-300)), 0x8000)
        self.assertEqual(fp.to_bits(H, fp.round_literal(H, 1e-300)), 0x0000)
        self.assertEqual(fp.to_bits(F, fp.round_literal(F, -0.0)), 0x80000000)
        self.assertEqual(fp.to_bits(F, fp.round_literal(F, 0.0)), 0)

    def test_special_values_pass_through(self):
        for fmt in FORMATS:
            self.assertEqual(fp.round_literal(fmt, INF), INF)
            self.assertEqual(fp.round_literal(fmt, -INF), -INF)
            self.assertTrue(support.same(fp.round_literal(fmt, NAN), NAN))

    def test_half_ties_go_to_even(self):
        self.assertEqual(fp.round_literal(H, 1 + 2.0 ** -11), 1.0)          # tie: 1.0 has the even significand
        self.assertEqual(fp.round_literal(H, 1 + 3 * 2.0 ** -11), 1 + 2.0 ** -9)  # tie: the upper neighbour is even
        self.assertEqual(fp.round_literal(H, 1 + 2.0 ** -11 + 2.0 ** -20), 1 + 2.0 ** -10)  # just past the tie
        self.assertEqual(fp.round_literal(H, 2051), 2052.0)
        self.assertEqual(fp.round_literal(H, 2.0 ** -25), 0.0)

    def test_a_half_literally_cannot_hold_2049(self):
        self.assertEqual(fp.round_literal(H, 2049), 2048.0)
        self.assertEqual(fp.round_literal(H, 3001), 3000.0)
        self.assertEqual(fp.round_literal(H, 4097), 4096.0)

    def test_overflow_starts_at_the_midpoint_above_the_largest_finite_value(self):
        for fmt in FORMATS:
            top, ulp = MAX_FINITE[fmt], LAST_ULP[fmt]
            for x in (top, math.nextafter(top + ulp / 2, 0), top + ulp / 2, top + ulp):
                if math.isfinite(x):  # in double precision adding the ulp already overflows
                    self.check(fmt, x)

    def test_narrowing_from_a_double_rounds_once_not_twice(self):
        x = 1 + 2.0 ** -11 + 2.0 ** -40
        via_float = fp.round_literal(H, fp.round_literal(F, x))
        self.assertEqual(via_float, 1.0, "the detour through float loses the tie-breaker")
        self.assertEqual(fp.round_literal(H, x), 1 + 2.0 ** -10, "a direct narrowing keeps it")


class ExactArithmetic(unittest.TestCase):
    def check(self, fmt, got, exact, what):
        value, flags = got
        expected, expected_flags = support.reference(fmt, exact)
        self.assertTrue(support.same(value, expected), f"{what} {fmt.name}: got {value!r}, want {expected!r}")
        self.assertEqual(flags, expected_flags, f"{what} {fmt.name} {value!r}")

    def test_add_sub_mul_div_fma_against_exact_arithmetic(self):
        rng = random.Random(2024)
        plan = (
            ("add", fp.add, lambda a, b, c: Fraction(a) + Fraction(b)),
            ("sub", fp.sub, lambda a, b, c: Fraction(a) - Fraction(b)),
            ("mul", fp.mul, lambda a, b, c: Fraction(a) * Fraction(b)),
            ("div", fp.div, lambda a, b, c: Fraction(a) / Fraction(b)),
            ("fma", fp.fma, lambda a, b, c: Fraction(a) * Fraction(b) + Fraction(c)),
        )
        for fmt in FORMATS:
            for _ in range(1500):
                a, b, c = (support.sample(fmt, rng) for _ in range(3))
                for name, fn, exact_of in plan:
                    operands = (a, b, c) if name == "fma" else (a, b)
                    if any(x != x for x in operands):
                        got = fn(fmt, *operands)
                        self.assertTrue(support.same(got[0], NAN) and support.canonical(got[0])
                                        and got[1] == 0, f"{name} with NaN")
                        continue
                    if any(math.isinf(x) for x in operands) or (name == "div" and b == 0.0):
                        continue  # special values have their own test
                    exact = exact_of(a, b, c)
                    if exact == 0 and any(x == 0.0 for x in operands):
                        continue  # the sign of an exact zero follows IEEE rules, tested below
                    self.check(fmt, fn(fmt, *operands), exact, name)

    def test_sqrt_against_exact_arithmetic(self):
        rng = random.Random(2025)
        for fmt in FORMATS:
            for _ in range(1500):
                a = support.sample(fmt, rng)
                if a != a:
                    got = fp.sqrt(fmt, a)
                    self.assertTrue(support.same(got[0], NAN) and support.canonical(got[0]) and got[1] == 0)
                    continue
                if a < 0 or math.isinf(a):
                    continue
                value, flags = fp.sqrt(fmt, a)
                if a == 0.0:
                    self.assertTrue(support.same(value, a) and flags == 0)
                    continue
                # the value: let the struct module round the correctly rounded double root
                root = math.sqrt(a)
                packed = struct.pack("<" + support.STRUCT_CODE[fmt.name], root)
                self.assertTrue(support.same(value, support.from_pattern(fmt, int.from_bytes(packed, "little"))))
                # the flags: exact only when a is the square of a format value
                num, den = Fraction(a).numerator, Fraction(a).denominator
                exact_root = math.isqrt(num) ** 2 == num and math.isqrt(den) ** 2 == den
                self.assertEqual(flags, 0 if exact_root else INEXACT, f"sqrt {fmt.name} {a!r}")
                if flags and abs(value) < fmt.min_normal:
                    self.assertEqual(flags, INEXACT | UNDERFLOW)

    def test_narrowing_conversions_round_once(self):
        rng = random.Random(7)
        for _ in range(4000):
            x = struct_double(rng.getrandbits(64))
            if not math.isfinite(x) or x == 0.0:
                continue
            for dst in (F, H):
                value, flags = fp.convert(dst, x)
                expected, expected_flags = support.reference(dst, Fraction(x))
                self.assertTrue(support.same(value, expected), f"{dst.name} {x!r}")
                self.assertEqual(flags, expected_flags)

    def test_widening_is_exact(self):
        rng = random.Random(8)
        for narrow, wide in ((H, F), (H, D), (F, D)):
            for _ in range(4000):
                x = support.sample(narrow, rng)
                if x != x:
                    self.assertTrue(support.same(fp.convert(wide, x)[0], NAN))
                    continue
                self.assertTrue(support.same(fp.convert(wide, x)[0], x))
                self.assertEqual(fp.convert(wide, x)[1], 0)

    def test_scaling_by_a_power_of_two(self):
        rng = random.Random(9)
        for _ in range(3000):
            fmt = rng.choice(FORMATS)
            x = support.sample(fmt, rng)
            n = float(rng.randrange(-1300, 1301))
            if x != x:
                self.assertTrue(support.same(fp.scaleb(fmt, x, n)[0], NAN))
            elif x == 0.0 or math.isinf(x):
                self.assertEqual(fp.scaleb(fmt, x, n), (x, 0))
            else:
                self.check(fmt, fp.scaleb(fmt, x, n), Fraction(x) * Fraction(2) ** int(n), "scaleb")

    def test_special_values_follow_ieee_754(self):
        cases = [
            (fp.add, (INF, 1.0), INF, 0),
            (fp.add, (INF, -INF), NAN, INVALID),
            (fp.add, (NAN, 1.0), NAN, 0),
            (fp.sub, (INF, INF), NAN, INVALID),
            (fp.mul, (INF, 0.0), NAN, INVALID),
            (fp.mul, (INF, -2.0), -INF, 0),
            (fp.mul, (NAN, 0.0), NAN, 0),
            (fp.div, (1.0, 0.0), INF, DIVZERO),
            (fp.div, (-1.0, 0.0), -INF, DIVZERO),
            (fp.div, (1.0, -0.0), -INF, DIVZERO),
            (fp.div, (-1.0, -0.0), INF, DIVZERO),
            (fp.div, (0.0, 0.0), NAN, INVALID),
            (fp.div, (INF, 0.0), NAN, INVALID),
            (fp.div, (INF, INF), NAN, INVALID),
            (fp.div, (INF, -2.0), -INF, 0),
            (fp.sqrt, (-1.0,), NAN, INVALID),
            (fp.sqrt, (NAN,), NAN, 0),
            (fp.fma, (INF, 0.0, 1.0), NAN, INVALID),
            (fp.fma, (0.0, INF, 1.0), NAN, INVALID),
            (fp.fma, (INF, 2.0, -INF), NAN, INVALID),
            (fp.fma, (INF, -2.0, INF), NAN, INVALID),
            (fp.fma, (INF, 2.0, INF), INF, 0),
            (fp.fma, (2.0, 3.0, -INF), -INF, 0),
            (fp.fma, (1.0, 2.0, NAN), NAN, 0),
            (fp.remainder, (INF, 2.0), NAN, INVALID),
            (fp.remainder, (1.0, 0.0), NAN, INVALID),
            (fp.remainder, (-0.0, 2.0), -0.0, 0),
            (fp.remainder, (3.0, INF), 3.0, 0),
        ]
        for fn, args, value, flags in cases:
            for fmt in FORMATS:
                got = fn(fmt, *args)
                self.assertTrue(support.same(got[0], value), f"{fn.__name__}{args} {fmt.name}: {got[0]!r}")
                self.assertEqual(got[1], flags, f"{fn.__name__}{args} {fmt.name}")

    def test_the_sign_of_an_exact_zero_follows_ieee_754(self):
        for fmt in FORMATS:
            zeros = [
                (fp.add, (0.0, 0.0), 0.0), (fp.add, (-0.0, -0.0), -0.0),
                (fp.add, (0.0, -0.0), 0.0), (fp.add, (-0.0, 0.0), 0.0), (fp.add, (1.0, -1.0), 0.0),
                (fp.sub, (0.0, 0.0), 0.0), (fp.sub, (-0.0, 0.0), -0.0),
                (fp.sub, (0.0, -0.0), 0.0), (fp.sub, (-0.0, -0.0), 0.0), (fp.sub, (-1.0, -1.0), 0.0),
                (fp.mul, (0.0, 5.0), 0.0), (fp.mul, (-0.0, 5.0), -0.0),
                (fp.mul, (3.0, -0.0), -0.0), (fp.mul, (-0.0, -0.0), 0.0), (fp.mul, (-3.0, -0.0), 0.0),
                (fp.div, (0.0, 5.0), 0.0), (fp.div, (-0.0, 5.0), -0.0),
                (fp.div, (0.0, -5.0), -0.0), (fp.div, (-0.0, -5.0), 0.0),
                (fp.fma, (0.0, 5.0, -0.0), 0.0), (fp.fma, (-0.0, 5.0, -0.0), -0.0),
                (fp.fma, (0.0, -5.0, -0.0), -0.0), (fp.fma, (0.0, -5.0, 0.0), 0.0),
                (fp.fma, (2.0, -4.0, 8.0), 0.0),
                (fp.fma, (1.0, 1.0, -1.0), 0.0),
            ]
            for fn, args, expected in zeros:
                got = fn(fmt, *args)
                self.assertTrue(support.same(got[0], expected), f"{fn.__name__}{args} {fmt.name}: {got[0]!r}")
                self.assertEqual(got[1], 0)

    def test_overflow_is_flagged_at_the_midpoint(self):
        for fmt in FORMATS:
            top, ulp = MAX_FINITE[fmt], LAST_ULP[fmt]
            self.assertEqual(fp.add(fmt, top, ulp / 4), (top, INEXACT))
            self.assertEqual(fp.add(fmt, top, ulp / 2), (INF, OVERFLOW | INEXACT))
            self.assertEqual(fp.add(fmt, top, ulp), (INF, OVERFLOW | INEXACT))
            self.assertEqual(fp.mul(fmt, top, 2.0), (INF, OVERFLOW | INEXACT))

    def test_underflow_is_gradual_and_flagged_only_when_something_was_lost(self):
        tiny = MIN_SUBNORMAL
        self.assertEqual(fp.mul(H, 2.0 ** -23, 0.5), (tiny[H], 0))
        self.assertEqual(fp.mul(H, tiny[H], 0.5), (0.0, UNDERFLOW | INEXACT))
        self.assertEqual(fp.mul(H, 3 * tiny[H], 0.5), (2 * tiny[H], UNDERFLOW | INEXACT))
        self.assertEqual(fp.mul(F, tiny[F], 0.5), (0.0, UNDERFLOW | INEXACT))
        self.assertEqual(fp.mul(F, 3 * tiny[F], 0.5), (2 * tiny[F], UNDERFLOW | INEXACT))
        self.assertEqual(fp.mul(D, tiny[D], 0.5), (0.0, UNDERFLOW | INEXACT))
        self.assertEqual(fp.mul(D, 3 * tiny[D], 0.5), (2 * tiny[D], UNDERFLOW | INEXACT))
        for fmt in FORMATS:
            self.assertEqual(math.copysign(1.0, fp.mul(fmt, -tiny[fmt], 0.5)[0]), -1.0)
            self.assertEqual(fp.mul(fmt, tiny[fmt], 1.0), (tiny[fmt], 0))

    def test_fma_rounds_once(self):
        for fmt in FORMATS:
            a = 1.0 + 2.0 ** -(fmt.p - 1)  # a * a needs one bit more than the format has
            self.assertEqual(fp.mul(fmt, a, a), (1.0 + 2.0 ** -(fmt.p - 2), INEXACT))
            self.assertEqual(fp.fma(fmt, a, a, -(1.0 + 2.0 ** -(fmt.p - 2))), (2.0 ** -(2 * fmt.p - 2), 0))
            self.assertEqual(fp.fma(fmt, a, a, 0.0), (1.0 + 2.0 ** -(fmt.p - 2), INEXACT))

    def test_flags_describe_the_true_result_not_an_intermediate_rounding(self):
        # A result can look exact in double precision and still be inexact in the format.
        self.assertEqual(fp.add(F, 2.0 ** 60, 2.0 ** -10), (2.0 ** 60, INEXACT))
        self.assertEqual(fp.add(H, 2.0 ** 14, 2.0 ** -20), (2.0 ** 14, INEXACT))
        # ...and the other way round: tiny errors far below the format's resolution.
        self.assertEqual(fp.add(H, 1.0, 2.0 ** -60), (1.0, INEXACT))
        # In double precision the host arithmetic is the rounding, and it is still reported.
        self.assertEqual(fp.add(D, 0.1, 0.2), (0.30000000000000004, INEXACT))
        self.assertEqual(fp.add(D, 0.5, 0.25), (0.75, 0))
        self.assertEqual(fp.mul(D, 0.1, 3.0), (0.30000000000000004, INEXACT))
        self.assertEqual(fp.mul(D, 0.5, 0.25), (0.125, 0))
        self.assertEqual(fp.div(D, 1.0, 3.0), (1.0 / 3.0, INEXACT))
        self.assertEqual(fp.div(D, 1.0, 4.0), (0.25, 0))
        self.assertEqual(fp.sqrt(D, 2.0), (math.sqrt(2.0), INEXACT))
        self.assertEqual(fp.sqrt(D, 4.0), (2.0, 0))
        # Subnormal products, where a double addendum to the error would itself underflow.
        self.assertEqual(fp.mul(D, 3 * 2.0 ** -1070, 2.0 ** -10), (0.0, UNDERFLOW | INEXACT))
        self.assertEqual(fp.mul(D, 2.0 ** -1073, 0.5), (2.0 ** -1074, 0))
        self.assertEqual(fp.div(D, 2.0 ** -1074, 3.0), (0.0, INEXACT | UNDERFLOW))


class TheCountingWalls(unittest.TestCase):
    """The limitation the whole machine hangs on: numbers are the only counters."""

    def test_a_half_cannot_count_past_2048(self):
        self.assertEqual(fp.add(H, 2047.0, 1.0), (2048.0, 0))
        self.assertEqual(fp.add(H, 2048.0, 1.0), (2048.0, INEXACT))
        self.assertEqual(fp.add(H, 2048.0, 2.0), (2050.0, 0))

    def test_a_float_cannot_count_past_16777216(self):
        self.assertEqual(fp.add(F, 16777215.0, 1.0), (16777216.0, 0))
        self.assertEqual(fp.add(F, 16777216.0, 1.0), (16777216.0, INEXACT))
        self.assertEqual(fp.add(F, 16777216.0, 2.0), (16777218.0, 0))

    def test_a_double_counts_on(self):
        self.assertEqual(fp.add(D, 16777216.0, 1.0), (16777217.0, 0))

    def test_subtracting_one_freezes_at_the_same_wall(self):
        # Above 2**p the spacing is 2, so x - 1 is a tie: it rounds back to x
        # (or, from an odd slot, two down to an even one, where it then sticks).
        self.assertEqual(fp.sub(H, 2048.0, 1.0), (2047.0, 0))  # lands below the wall: exact
        self.assertEqual(fp.sub(H, 4094.0, 1.0), (4092.0, INEXACT))
        self.assertEqual(fp.sub(H, 4092.0, 1.0), (4092.0, INEXACT))
        self.assertEqual(fp.sub(F, 16777216.0, 1.0), (16777215.0, 0))
        self.assertEqual(fp.sub(F, 16777220.0, 1.0), (16777220.0, INEXACT))


class TheRestOfTheInstructionSet(unittest.TestCase):
    def test_remainder_is_exact_and_within_half_the_divisor(self):
        rng = random.Random(11)
        for _ in range(2000):
            fmt = rng.choice(FORMATS)
            a, b = support.sample(fmt, rng), support.sample(fmt, rng)
            if any(x != x or math.isinf(x) or x == 0.0 for x in (a, b)):
                continue
            value, flags = fp.remainder(fmt, a, b)
            self.assertEqual(flags, 0)
            self.assertLessEqual(abs(value), abs(b) / 2)
            self.assertEqual(value, math.remainder(a, b))
            multiple = (Fraction(a) - Fraction(value)) / Fraction(b)  # a whole number, or a is returned as is
            self.assertTrue(multiple.denominator == 1 or value == a)

    def test_integral_rounding(self):
        # x, floor, ceil, trunc, round-to-even
        table = [(2.5, 2.0, 3.0, 2.0, 2.0), (-2.5, -3.0, -2.0, -2.0, -2.0), (3.5, 3.0, 4.0, 3.0, 4.0),
                 (-3.5, -4.0, -3.0, -3.0, -4.0), (0.25, 0.0, 1.0, 0.0, 0.0), (-0.25, -1.0, -0.0, -0.0, -0.0)]
        functions = (fp.floor, fp.ceil, fp.trunc, fp.round_even)
        for fmt in FORMATS:
            for row in table:
                if fp.round_literal(fmt, row[0]) != row[0]:  # the format cannot hold x
                    continue
                for fn, want in zip(functions, row[1:]):
                    got = fn(fmt, row[0])
                    self.assertTrue(support.same(got[0], want), f"{fn.__name__} {fmt.name} {row[0]!r}: {got[0]!r}")
                    self.assertEqual(got[1], 0 if support.same(row[0], want) else INEXACT)
            for fn in functions:
                self.assertEqual(fn(fmt, 5.0), (5.0, 0))
                self.assertEqual(fn(fmt, MAX_FINITE[fmt]), (MAX_FINITE[fmt], 0))
                self.assertEqual(fn(fmt, INF), (INF, 0))
                self.assertEqual(fn(fmt, -INF), (-INF, 0))
                self.assertTrue(support.same(fn(fmt, -0.0)[0], -0.0))
                self.assertTrue(support.same(fn(fmt, NAN)[0], NAN))

    def test_min_max_and_comparisons(self):
        for fmt in FORMATS:
            self.assertEqual(fp.fmin(fmt, 1.0, 2.0), (1.0, 0))
            self.assertEqual(fp.fmax(fmt, 1.0, 2.0), (2.0, 0))
            self.assertEqual(fp.fmin(fmt, -1.0, -2.0), (-2.0, 0))
            self.assertEqual(fp.fmin(fmt, NAN, 2.0)[0], 2.0)
            self.assertEqual(fp.fmin(fmt, 1.0, NAN)[0], 1.0)
            self.assertEqual(fp.fmax(fmt, NAN, 2.0)[0], 2.0)
            self.assertTrue(support.same(fp.fmin(fmt, NAN, NAN)[0], NAN))
            self.assertEqual(math.copysign(1.0, fp.fmin(fmt, 0.0, -0.0)[0]), -1.0)
            self.assertEqual(math.copysign(1.0, fp.fmin(fmt, -0.0, 0.0)[0]), -1.0)
            self.assertEqual(math.copysign(1.0, fp.fmax(fmt, 0.0, -0.0)[0]), 1.0)
            self.assertEqual(math.copysign(1.0, fp.fmax(fmt, -0.0, 0.0)[0]), 1.0)
            for a, b in ((1.0, 2.0), (2.0, 2.0), (2.0, 1.0), (NAN, 1.0), (1.0, NAN), (-0.0, 0.0)):
                self.assertEqual(fp.lt(fmt, a, b), (1.0 if a < b else 0.0, 0))
                self.assertEqual(fp.le(fmt, a, b), (1.0 if a <= b else 0.0, 0))
                self.assertEqual(fp.eq(fmt, a, b), (1.0 if a == b else 0.0, 0))
            self.assertEqual(fp.eq(fmt, NAN, NAN), (0.0, 0))

    def test_sign_operations_keep_nan_payloads_and_arithmetic_does_not(self):
        payload = fp.from_bits(F, 0x7FC12345)
        self.assertEqual(fp.to_bits(F, fp.copy(F, payload)[0]), 0x7FC12345)
        self.assertEqual(fp.to_bits(F, fp.fabs(F, payload)[0]), 0x7FC12345)
        self.assertEqual(fp.to_bits(F, fp.neg(F, payload)[0]), 0xFFC12345)
        self.assertEqual(fp.to_bits(F, fp.copysign(F, payload, -1.0)[0]), 0xFFC12345)
        self.assertEqual(fp.to_bits(F, fp.add(F, payload, 0.0)[0]), 0x7FC00000)
        self.assertEqual(fp.to_bits(F, fp.mul(F, payload, 1.0)[0]), 0x7FC00000)
        self.assertEqual(bits32(fp.fabs(F, -0.0)[0]), 0)
        self.assertEqual(bits32(fp.neg(F, 0.0)[0]), 0x80000000)

    def test_logb(self):
        for fmt in FORMATS:
            self.assertEqual(fp.logb(fmt, 1.0), (0.0, 0))
            self.assertEqual(fp.logb(fmt, 8.0), (3.0, 0))
            self.assertEqual(fp.logb(fmt, 0.5), (-1.0, 0))
            self.assertEqual(fp.logb(fmt, -5.0), (2.0, 0))
            self.assertEqual(fp.logb(fmt, fmt.min_normal), (float(fmt.emin), 0))
            self.assertEqual(fp.logb(fmt, MIN_SUBNORMAL[fmt]), (float(fmt.emin), 0))
            self.assertEqual(fp.logb(fmt, 0.0), (-INF, DIVZERO))
            self.assertEqual(fp.logb(fmt, -0.0), (-INF, DIVZERO))
            self.assertEqual(fp.logb(fmt, -INF), (INF, 0))
            self.assertTrue(support.same(fp.logb(fmt, NAN)[0], NAN))

    def test_nextafter_walks_the_grid_of_the_format(self):
        self.assertEqual(fp.nextafter(H, 2048.0, INF), (2050.0, 0))
        self.assertEqual(fp.nextafter(F, 1.0, 2.0), (support.from_pattern(F, 0x3F800001), 0))
        self.assertEqual(fp.nextafter(F, 1.0, 0.0), (support.from_pattern(F, 0x3F7FFFFF), 0))
        self.assertEqual(fp.nextafter(F, 0.0, 1.0), (MIN_SUBNORMAL[F], 0))
        self.assertEqual(fp.nextafter(F, 0.0, -1.0), (-MIN_SUBNORMAL[F], 0))
        self.assertEqual(fp.nextafter(F, -0.0, -1.0), (-MIN_SUBNORMAL[F], 0))
        self.assertEqual(fp.nextafter(F, MIN_SUBNORMAL[F], 0.0)[0], 0.0)
        self.assertEqual(fp.nextafter(F, MAX_FINITE[F], INF), (INF, OVERFLOW | INEXACT))
        self.assertEqual(fp.nextafter(F, 1.0, 1.0), (1.0, 0))
        self.assertTrue(support.same(fp.nextafter(F, NAN, 1.0)[0], NAN))
        rng = random.Random(12)
        for _ in range(20000):
            x, y = struct_double(rng.getrandbits(64)), struct_double(rng.getrandbits(64))
            if x != x or y != y:
                continue
            want = math.nextafter(x, y)
            flags = OVERFLOW | INEXACT if math.isinf(want) and math.isfinite(x) else 0
            self.assertTrue(support.same(fp.nextafter(D, x, y)[0], want))
            self.assertEqual(fp.nextafter(D, x, y)[1], flags)

    def test_classify(self):
        codes = [(NAN, QNAN), (-INF, NINF), (INF, PINF), (-0.0, NZERO), (0.0, PZERO),
                 (-1.0, NNORMAL), (1.0, PNORMAL)]
        for fmt in FORMATS:
            table = codes + [(-MIN_SUBNORMAL[fmt], NSUBNORMAL), (MIN_SUBNORMAL[fmt], PSUBNORMAL)]
            for x, code in table:
                self.assertEqual(fp.classify(fmt, x)[0], float(code), f"{fmt.name} {x!r}")
                self.assertEqual(fp.classify(fmt, x)[1], 0)


class GridsAndCasts(unittest.TestCase):
    """The instructions added for whole-number work, against exact arithmetic."""

    def grid(self, fmt, fn, choose, x, step):
        got = fn(fmt, x, step)
        if x != x or step != step:
            self.assertTrue(support.same(got[0], NAN) and got[1] == 0)
            return
        if step == 0.0 or abs(step) == INF or abs(x) == INF:
            self.assertTrue(support.same(got[0], NAN) and got[1] == INVALID)
            return
        if x == 0.0:
            self.assertTrue(support.same(got[0], x) and got[1] == 0)
            return
        quotient = Fraction(x) / Fraction(abs(step))  # exact
        multiple = choose(quotient)
        exact = multiple * Fraction(abs(step))
        value, flags = support.reference(fmt, exact)
        if value == 0.0 and x < 0:  # toward zero from below leaves a negative zero
            value = -0.0
        self.assertTrue(support.same(got[0], value), f"{fn.__name__} {fmt.name} {x!r} to {step!r}: {got[0]!r}, want {value!r}")
        self.assertEqual(got[1], flags | (INEXACT if exact != Fraction(x) else 0),
                         f"{fn.__name__} {fmt.name} {x!r} to {step!r}")

    def test_chop_floor_ceil_and_round_to_a_step_against_exact_arithmetic(self):
        rng = random.Random(61)
        floors = lambda f: f.numerator // f.denominator  # noqa: E731
        ceils = lambda f: -((-f.numerator) // f.denominator)  # noqa: E731
        plan = ((fp.floor_to, floors), (fp.ceil_to, ceils), (fp.trunc_to, int), (fp.round_to, round))
        steps = [1.0, 2.0, 5.0, 10.0, 0.5, 0.25, 0.01, 0.1, 3.0, 7.0, 360.0, 0.001, 12345.678, 4096.0, 1e-7]
        for fmt in FORMATS:
            for _ in range(600):
                x = support.sample(fmt, rng)
                step = rng.choice(steps) if rng.random() < 0.7 else support.sample(fmt, rng)
                for fn, choose in plan:
                    self.grid(fmt, fn, choose, x, step)

    def test_step_edge_cases(self):
        for fmt in FORMATS:
            for fn in (fp.floor_to, fp.ceil_to, fp.trunc_to, fp.round_to):
                self.assertEqual(fn(fmt, 7.0, 0.0)[1], INVALID)
                self.assertEqual(fn(fmt, INF, 1.0)[1], INVALID)
                self.assertEqual(fn(fmt, 7.0, INF)[1], INVALID)
                self.assertEqual(fn(fmt, 0.0, 5.0), (0.0, 0))
                value, flags = fn(fmt, -0.0, 5.0)
                self.assertTrue(support.same(value, -0.0) and flags == 0)
            self.assertEqual(fp.round_to(fmt, 47.5, 5.0)[0], 50.0)  # a tie goes to the even multiple
            self.assertEqual(fp.round_to(fmt, 42.5, 5.0)[0], 40.0)
            self.assertEqual(fp.round_to(fmt, 0.125, 0.25)[0], 0.0)
            self.assertEqual(fp.round_to(fmt, 0.375, 0.25)[0], 0.5)

    def test_mod_against_exact_arithmetic(self):
        rng = random.Random(62)
        for fmt in FORMATS:
            for _ in range(1500):
                a, b = support.sample(fmt, rng), support.sample(fmt, rng)
                if any(x != x or math.isinf(x) or x == 0.0 for x in (a, b)):
                    continue
                value, flags = fp.fmod(fmt, a, b)
                exact = Fraction(a) - Fraction(b) * math.trunc(Fraction(a) / Fraction(b))
                self.assertEqual(Fraction(value), exact, f"mod {fmt.name} {a!r} {b!r}")
                self.assertEqual(flags, 0)
                self.assertTrue(value == 0.0 or math.copysign(1.0, value) == math.copysign(1.0, a))

    def test_split_then_join_loses_almost_nothing(self):
        rng = random.Random(63)
        for _ in range(4000):
            x = struct_double(rng.getrandbits(64))
            if not math.isfinite(x):
                continue
            high, low, flags = fp.split(F, x)
            if math.isinf(high):  # too big for any float: the pair is an infinity and its opposite
                self.assertEqual((low, flags), (-high, INEXACT))
                continue
            self.assertEqual(flags, 0 if high + low == x else INEXACT)
            self.assertEqual(fp.join(D, high, low)[0], high + low)
            # A pair has float's range, not double's.  Well inside that range it keeps
            # close to 48 bits; near the bottom the remainder is subnormal and coarser,
            # and below 2**-149 both halves vanish.
            self.assertLessEqual(abs(x - (high + low)), max(abs(x) * 2.0 ** -48, 2.0 ** -150))
        self.assertEqual(fp.split(H, 0.0999755859375), (0.0999755859375, 0.0, 0))  # already a half: nothing left over
        self.assertEqual(fp.split(H, 0.1), (0.0999755859375, 2.4437904357910156e-05, INEXACT))  # a double that is not
        self.assertEqual(fp.split(F, 0.0), (0.0, 0.0, 0))
        self.assertEqual(fp.split(F, 1e-114), (0.0, 0.0, INEXACT))  # too small for any float: both halves vanish
        self.assertEqual(fp.mantissa(1e-114)[1], -378.0)  # ... which is what mantissa and exponent are for
        self.assertTrue(support.same(fp.split(F, NAN)[0], NAN))

    def test_mantissa_scales_into_the_first_binade(self):
        rng = random.Random(64)
        for _ in range(4000):
            x = struct_double(rng.getrandbits(64))
            m, e = fp.mantissa(x)
            if x != x:
                self.assertTrue(m != m and e != e)
                continue
            if x == 0.0 or math.isinf(x):
                self.assertTrue(support.same(m, x) and e == 0.0)
                continue
            self.assertTrue(0.5 <= abs(m) < 1.0)
            self.assertEqual(m * 2.0 ** int(e), x)
            self.assertEqual(fp.scaleb(D, m, e)[0], x)


if __name__ == "__main__":
    unittest.main()


class QuadPrecision(unittest.TestCase):
    """binary128: values are Quad numbers, arithmetic is exact and rounded
    once, and the format a double can never hold is still held honestly."""

    def test_a_number_no_double_can_hold(self):
        x = fp.Quad(0, 10 ** 34)
        self.assertEqual(fp.q_fraction(x), 10 ** 34)
        self.assertEqual(fp.to_bits(fp.Q, x) >> 112, 16383 + 112)  # 10^34 sits in binade 2^112
        back = fp.from_bits(fp.Q, fp.to_bits(fp.Q, x))
        self.assertEqual(back, x)

    def test_arithmetic_is_exact_then_rounded_once(self):
        v, flags = fp.mul(fp.Q, fp.Quad(0, 6), fp.Quad(0, 1, -1))
        self.assertEqual((fp.q_fraction(v), flags), (Fraction(3), 0))
        v, flags = fp.div(fp.Q, fp.Quad(0, 1), fp.Quad(0, 3))
        self.assertEqual(flags, fp.INEXACT)               # 1/3 is not a dyadic rational
        self.assertTrue(0.333333333333333 < float(fp.q_fraction(v)) < 0.3333333333333334)

    def test_signed_zeros_follow_ieee(self):
        z, _ = fp.add(fp.Q, fp.Quad(1), fp.Quad(1))
        self.assertEqual(z, fp.Quad(1))                   # -0 + -0 is -0
        z, _ = fp.add(fp.Q, fp.Quad(0), fp.Quad(1))
        self.assertEqual(z, fp.Quad(0))                   # +0 + -0 is +0
        z, _ = fp.mul(fp.Q, fp.Quad(0, 5), fp.Quad(1))
        self.assertEqual(z, fp.Quad(1))                   # (+5) * (-0) is -0, the sign is xor
        self.assertEqual(fp.fmin(fp.Q, fp.Quad(0), fp.Quad(1))[0], fp.Quad(1))
        self.assertEqual(fp.fmax(fp.Q, fp.Quad(0), fp.Quad(1))[0], fp.Quad(0))

    def test_overflow_underflow_and_subnormals(self):
        v, flags = fp.mul(fp.Q, fp.Quad(0, (1 << 113) - 1, 16384), fp.Quad(0, 1 << 113, 0))
        self.assertTrue(v.inf)
        self.assertEqual(fp.flag_names(flags), ["overflow", "inexact"])
        v, flags = fp.mul(fp.Q, fp.Quad(0, 1, -16494), fp.Quad(0, 1))   # exact subnormal
        self.assertEqual(v, fp.Quad(0, 1, -16494))
        self.assertEqual(flags, 0)

    def test_sqrt_division_and_remainder(self):
        v, flags = fp.sqrt(fp.Q, fp.Quad(0, 4))
        self.assertEqual((fp.q_fraction(v), flags), (Fraction(2), 0))
        v, flags = fp.sqrt(fp.Q, fp.Quad(0, 2))
        self.assertEqual(flags, fp.INEXACT)
        v, flags = fp.remainder(fp.Q, fp.Quad(0, 7), fp.Quad(0, 3))
        self.assertEqual((fp.q_fraction(v), flags), (Fraction(1), 0))
        v, flags = fp.div(fp.Q, fp.Quad(0, 1), fp.Quad(0))
        self.assertTrue(v.inf)
        self.assertEqual(fp.flag_names(flags), ["divzero"])

    def test_conversion_widens_doubles_exactly(self):
        v, flags = fp.convert(fp.Q, 0.1)
        self.assertEqual(flags, 0)
        self.assertEqual(fp.q_fraction(v), Fraction(0.1))
        v, flags = fp.convert(fp.D, fp.Quad(0, 3, -1075))  # 1.5 double ulps: ties to even, up to 2
        self.assertEqual(flags, fp.UNDERFLOW | fp.INEXACT)
        self.assertEqual(v, 2.0 ** -1073)

    def test_the_counting_wall_is_2_to_the_113(self):
        wall = fp.Quad(0, 1, 113)
        step, flags = fp.add(fp.Q, wall, fp.Quad(0, 1))
        self.assertEqual(fp.q_fraction(step), 2 ** 113)    # the spacing is 2 here, as the wall demands
        below = fp.sub(fp.Q, wall, fp.Quad(0, 1))[0]
        self.assertEqual(fp.q_fraction(below), 2 ** 113 - 1)  # exact below the wall, in a double never
