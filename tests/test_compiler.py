"""The compiler: programs compiled, run on the reference, compared with the
answers computed independently here."""
import io
import sys
import unittest

sys.path.insert(0, ".")

from compiler import m4c
from mantissa import asm
from mantissa.machine import Cpu, console, run
from mantissa.memory import Memory


def run_program(source, steps=200_000):
    text, image, warnings = m4c.compile_source(source)
    memory = Memory(4096)
    memory.write(image.cells)
    out = io.StringIO()
    result = run(Cpu(f=(0.0,) * 31 + (4096.0,), pc=image.entry), memory,
                 out=console(out.write), max_steps=steps)
    return out.getvalue(), result, warnings, text


class CompiledPrograms(unittest.TestCase):
    def check(self, source, expected_lines):
        out, result, warnings, _ = run_program(source)
        self.assertEqual(result.reason, "halt", result.cpu.fault)
        self.assertEqual(out.splitlines(), [f"{v!r}" for v in expected_lines])
        return warnings

    def test_sum_of_ten(self):
        self.check("""
            var x = 0.0
            var i = 0.0
            while i < 10.0 {
                x = x + i
                i = i + 1.0
            }
            print x
        """, [45.0])

    def test_folding_happens_at_compile_time(self):
        # the folded constant rides in the literal pool or as an inline operand,
        # but the machine never computes 1 + 2 * 3
        self.check("print 1 + 2 * 3\n", [7.0])

    def test_identities_remove_x_times_one(self):
        self.check("""
            var x = 2.5
            print x * 1
            print 1 * x
            print x - 0.0
        """, [2.5, 2.5, 2.5])

    def test_signed_zero_is_preserved_because_x_plus_zero_is_not_optimised(self):
        out, _, _, _ = run_program("""
            var x = -0.0
            print x + 0.0
            print x - 0.0
        """)
        self.assertEqual(out.splitlines(), ["0.0", "-0.0"])  # + 0 turns -0 into +0; - 0 keeps it

    def test_if_and_else(self):
        self.check("""
            var y = 5.0
            if y > 3.0 { print y } else { print 0.0 }
            if y > 9.0 { print 1.0 } else { print 2.0 }
        """, [5.0, 2.0])

    def test_quadratic_roots(self):
        # x*x - 3x + 2 = 0 has roots 1 and 2
        self.check("""
            var a = 1.0
            var b = -3.0
            var c = 2.0
            var disc = b * b - 4.0 * a * c
            print (-b + sqrt(disc)) / (2.0 * a)
            print (-b - sqrt(disc)) / (2.0 * a)
        """, [2.0, 1.0])

    def test_the_counting_wall_is_reachable_in_double(self):
        # a counter walks past 2^24, where a float would freeze
        self.check("""
            var i = 16777210.0
            var n = 0.0
            while i < 16777221.0 {
                i = i + 1.0
                n = n + 1.0
            }
            print n
            print i
        """, [11.0, 16777221.0])
        # the same idea one line shorter: a double steps through the float wall
        self.check("""
            var x = 16777216.0
            print x + 1.0
        """, [16777217.0])
        # and the same loop one lower: the float wall is a real place, and
        # double arithmetic walks straight through it
        self.check("""
            var i = 16777210.0
            var n = 0.0
            while i < 16777218.0 {
                i = i + 1.0
                n = n + 1.0
            }
            print i
        """, [16777218.0])

    def test_sqrt_fma_and_the_rest(self):
        self.check("""
            print sqrt(144.0)
            print fma(2.0, 3.0, 4.0)
            print chop(2.7)
            print min(3.0, 5.0) + max(3.0, 5.0)
            print mod(47.0, 5.0)
            print compare(3.0, 7.0)
            print scalb(3.0, 4.0)
        """, [12.0, 10.0, 2.0, 8.0, 2.0, -1.0, 48.0])

    def test_the_walls(self):
        # fma keeps the product's low bits, so a doubled wall is measurable
        self.check("""
            var x = 16777216.0
            print x + 1.0
            print fma(x, 2.0, 1.0) - x * 2.0
        """, [16777217.0, 1.0])


class Optimisations(unittest.TestCase):
    def test_the_literal_pool_loads_a_repeated_constant_once(self):
        text, _, _ = m4c.compile_source("""
            var x = 0.0
            x = x * 8.0
            x = x * 8.0
            x = x * 8.0
        """)
        # 8.0 once in the pool; the initialiser's own load is the other one
        self.assertEqual(text.count("LDI.D"), 2, text)
        self.assertEqual(text.count("8.0"), 1, "the constant appears once, in the pool")
        self.assertIn("the literal pool", text)

    def test_a_constant_used_once_stays_inline(self):
        text, _, _ = m4c.compile_source("var x = 2.0\nx = x * 8.0\n")
        self.assertNotIn("the literal pool", text)
        self.assertEqual(text.count("LDI.D"), 2)  # the initialiser, then the multiply's operand

    def test_multiplication_by_one_disappears(self):
        text, _, _ = m4c.compile_source("var x = 3.0\nx = x * 1.0\nprint x\n")
        self.assertNotIn("MUL", text)

    def test_compile_errors_carry_the_line(self):
        with self.assertRaises(m4c.CompileError) as caught:
            m4c.compile_source("var x = 1.0\nx = y + 1.0\n")
        self.assertIn("line 2", str(caught.exception))
        with self.assertRaises(m4c.CompileError) as caught:
            m4c.compile_source("var x = 1.0\nvar x = 2.0\n")
        self.assertIn("declared twice", str(caught.exception))

    def test_the_wall_warning_fires_for_the_awkward_band(self):
        _, _, warnings = m4c.compile_source("""
            var i = 0.0
            while i < 16777219.0 {
                i = i + 1.0
            }
        """)
        self.assertTrue(any("16777219" in w for w in warnings), warnings)


class DisassemblerRoundTrip(unittest.TestCase):
    def test_compiled_assembly_reassembles_identically(self):
        text, image, _ = m4c.compile_source("""
            var x = 0.0
            var i = 0.0
            while i < 5.0 {
                x = x + sqrt(i)
                i = i + 1.0
            }
            print x
        """)
        again = asm.assemble(text)
        self.assertEqual(list(again.cells), list(image.cells))


if __name__ == "__main__":
    unittest.main()
