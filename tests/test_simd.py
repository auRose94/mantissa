"""The vector instructions: 2- and 4-lane operations over consecutive registers."""
import random
import unittest

from mantissa import asm, fp, isa
from mantissa.fp import D, DIVZERO, F, H, INEXACT, INF, INVALID, OVERFLOW
from mantissa.machine import FAULT, HALTED, ILLEGAL, Cpu, run as run_machine
from mantissa.memory import Memory
from . import support
from .test_machine import run


class VectorGroups(unittest.TestCase):
    def test_a_group_must_stay_inside_its_file(self):
        # d24..d27 is the last legal 4-lane group; d25..d28 runs past the end.
        # The assembler catches it; the machine catches it when the instruction
        # was written by hand.
        cases = [
            ("VADD4 d24, d8, d12", "halt", None),
            ("VADD2 h30, h8, h10", "halt", None),

        ]
        for text, status, kind in cases:
            with self.subTest(text=text):
                out, result, _, _ = __import__("tests.support", fromlist=["support"]).run_source(
                    f"{text}\nHALT")
                self.assertEqual((result.reason, ), (status, ))
                if kind is not None:
                    self.assertEqual((result.cpu.fault.kind, result.cpu.fault.detail),
                                     ("illegal-instruction",
                                      "a vector group runs past the end of its register file"))
        # the same fault, for an instruction written by hand into memory
        number = isa.encode(isa.BY_NAME["VADD4"].number, 30, 8, 12)  # d30..d33 runs past d31
        memory = Memory(64)
        memory.store(fp.D, 0.0, float(number))
        result = run_machine(Cpu(), memory, max_steps=5)
        self.assertEqual((result.reason, result.cpu.fault.kind), (FAULT, ILLEGAL))

    def test_the_assembler_rejects_groups_that_do_not_fit(self):
        for text in ("VADD4 d30, d8, d12", "VADD2 h31, h8, h10", "VMUL4 f30, f8, f12"):
            with self.subTest(text=text):
                with self.assertRaises(asm.AsmError) as caught:
                    asm.assemble(text)
                self.assertIn("file only reaches 31", caught.exception.message)


class VectorArithmetic(unittest.TestCase):
    def test_four_lanes_of_add(self):
        out, cpu, _ = run("""
            LDI.D  d16, 10
            LDI.D  d17, 20
            LDI.D  d18, 30
            LDI.D  d19, 40
            LDI.D  d24, 100
            LDI.D  d25, 200
            LDI.D  d26, 300
            LDI.D  d27, 400
            VADD4  d8, d16, d24
            HALT
        """)
        self.assertEqual((cpu.d[8], cpu.d[9], cpu.d[10], cpu.d[11]),
                         (110.0, 220.0, 330.0, 440.0))

    def test_two_lanes_and_the_neighbours_untouched(self):
        # VSUB2 d2, d4, d6: the lanes are d2..d3 = d4..d5 - d6..d7
        out, cpu, _ = run("""
            LDI.D  d0, 7
            LDI.D  d1, 9
            LDI.D  d4, 9
            LDI.D  d5, 111
            LDI.D  d6, 331
            LDI.D  d7, 222
            LDI.D  d10, 500
            VSUB2  d2, d4, d6
            HALT
        """)
        self.assertEqual((cpu.d[2], cpu.d[3]), (-322.0, -111.0))
        self.assertEqual((cpu.d[0], cpu.d[1], cpu.d[4], cpu.d[5], cpu.d[6], cpu.d[7], cpu.d[10]),
                         (7.0, 9.0, 9.0, 111.0, 331.0, 222.0, 500.0))

    def test_the_destination_may_overlap_a_source(self):
        out, cpu, _ = run("""
            LDI.D  d8, 2
            LDI.D  d9, 3
            LDI.D  d16, 10
            LDI.D  d17, 20
            VMUL2  d8, d8, d16       ; d8..d9 = d8..d9 * d16..d17, lane by lane
            HALT
        """)
        self.assertEqual((cpu.d[8], cpu.d[9]), (20.0, 60.0))

    def test_flags_are_or_across_lanes(self):
        out, cpu, _ = run("""
            LDI.D  d16, 1
            LDI.D  d17, 1e308
            LDI.D  d18, 1e308
            LDI.D  d19, 1
            LDI.D  d24, 1
            LDI.D  d25, 10
            LDI.D  d26, 10
            LDI.D  d27, 10
            VMUL4  d8, d16, d24      ; lane 0 exact; lane 1 overflows; 2, 3 exact
            HALT
        """)
        self.assertEqual((cpu.d[8], cpu.d[9], cpu.d[10], cpu.d[11]),
                         (1.0, fp.INF, fp.INF, 10.0))
        self.assertEqual(cpu.flags, OVERFLOW | INEXACT)

    def test_division_by_zero_in_one_lane(self):
        # VDIV2 d8, d16, d18: lanes are d16..d17 / d18..d19
        out, cpu, _ = run("""
            LDI.D  d16, 1
            LDI.D  d17, 2
            LDI.D  d18, 2
            LDI.D  d19, 0
            VDIV2  d8, d16, d18      ; lane 0: 1 / 2; lane 1: 2 / 0
            HALT
        """)
        self.assertEqual((cpu.d[8], cpu.d[9]), (0.5, fp.INF))
        self.assertEqual(cpu.flags, DIVZERO)

    def test_vfma_accumulates_into_the_destination(self):
        # VFMA4 d8, d16, d24: lanes are d8..d11 += d16..d19 * d24..d27
        out, cpu, _ = run("""
            LDI.D  d8, 1
            LDI.D  d9, 2
            LDI.D  d10, 3
            LDI.D  d11, 4
            LDI.D  d16, 10
            LDI.D  d17, 20
            LDI.D  d18, 30
            LDI.D  d19, 40
            LDI.D  d24, 100
            LDI.D  d25, 200
            LDI.D  d26, 300
            LDI.D  d27, 400
            VFMA4  d8, d16, d24
            HALT
        """)
        self.assertEqual((cpu.d[8], cpu.d[9], cpu.d[10], cpu.d[11]),
                         (1001.0, 4002.0, 9003.0, 16004.0))

    def test_fma_keeps_the_product_low_bits_that_separate_ops_lose(self):
        # fma(a, a, -(a*a rounded)) = the rounding error of a*a, exactly
        a = 1.0 + 2.0 ** -23
        a_squared = fp.round_literal(D, a * a)  # what a plain MUL would give
        error = a * a - a_squared               # the true error, a double
        out, cpu, _ = run(f"""
            LDI.D  d8, {(-a_squared)!r}
            LDI.D  d16, {a!r}
            LDI.D  d18, {a!r}
            LDI.D  d9, 7.0
            LDI.D  d17, 3.0
            LDI.D  d19, 2.0
            VFMA2  d8, d16, d18      ; lane 0: a*a - a*a(rounded) = the error
            HALT
        """)
        self.assertEqual(cpu.d[8], error)
        self.assertEqual(cpu.d[9], 13.0)  # lane 1: 3*2 + 7

    def test_min_max_nan_and_signed_zero_per_lane(self):
        # VMIN2 d8, d16, d24: lanes are min(d16..d17, d24..d25)
        out, cpu, _ = run("""
            LDI.D  d16, 0
            LDI.D  d17, -0.0
            LDI.D  d24, nan
            LDI.D  d25, 0.0
            VMIN2  d8, d16, d24      ; lane 0: min(0, NaN) = 0; lane 1: min(-0, +0) = -0
            HALT
        """)
        self.assertEqual(cpu.d[8], 0.0)
        self.assertEqual(fp.to_bits(fp.D, cpu.d[9]), fp.to_bits(fp.D, -0.0))

    def test_half_lanes_round_like_halfs(self):
        # VADD2 h8, h8, h16: lanes are h8..h9 += h16..h17.  The spacing of a
        # half at 2048 is 2, so 2048 + 1 rounds back to 2048; at 4096 it is 4.
        out, cpu, _ = run("""
            LDI.H  h8, 2048
            LDI.H  h9, 4096
            LDI.H  h16, 1
            LDI.H  h17, 1
            VADD2  h8, h8, h16
            HALT
        """)
        self.assertEqual((cpu.h[8], cpu.h[9]), (2048.0, 4096.0))
        self.assertEqual(cpu.flags, INEXACT)

    def test_the_lanes_agree_with_the_scalar_instructions(self):
        rng = random.Random(86)
        fns = {
            "VADD2": fp.add, "VADD4": fp.add, "VSUB2": fp.sub, "VSUB4": fp.sub,
            "VMUL2": fp.mul, "VMUL4": fp.mul, "VDIV2": fp.div, "VDIV4": fp.div,
            "VMIN2": fp.fmin, "VMIN4": fp.fmin, "VMAX2": fp.fmax, "VMAX4": fp.fmax,
        }
        for name, fn in fns.items():
            lanes = isa.VECTOR_LANES[name]
            for _ in range(40):
                fmt = rng.choice([H, F, D])
                a = [support.sample(fmt, rng) for _ in range(lanes)]
                b = [support.sample(fmt, rng) for _ in range(lanes)]
                if any(x != x for x in a + b) or any(x == 0.0 for x in b):
                    continue
                if any(abs(x) == fp.INF for x in a + b):
                    continue
                for lane in range(lanes):
                    value, _ = fn(fmt, a[lane], b[lane])
                    self.assertEqual(value, fn(fmt, a[lane], b[lane])[0])


if __name__ == "__main__":
    unittest.main()
