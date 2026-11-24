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
        self.assertEqual(text.count("LDI.D"), 2, text)  # 8.0 once in the pool, + the SIZE register
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


class Arrays(unittest.TestCase):
    def check(self, source, expected_lines):
        out, result, warnings, _ = run_program(source)
        self.assertEqual(result.reason, "halt", result.cpu.fault)
        self.assertEqual(out.splitlines(), [f"{v!r}" for v in expected_lines])
        return warnings

    def test_fill_sum_and_read(self):
        self.check("""
            array a[10]
            var i = 0.0
            var sum = 0.0
            while i < 10.0 {
                a[i] = i * i
                i = i + 1.0
            }
            i = 0.0
            while i < 10.0 {
                sum = sum + a[i]
                i = i + 1.0
            }
            print sum
            print a[3]
        """, [285.0, 9.0])

    def test_reverse_in_place(self):
        self.check("""
            array a[6]
            var i = 0.0
            while i < 6.0 {
                a[i] = i + 1.0
                i = i + 1.0
            }
            var lo = 0.0
            var hi = 5.0
            var t = 1.0
            while lo < hi {
                var tmp = a[lo]
                a[lo] = a[hi]
                a[hi] = tmp
                lo = lo + t
                hi = hi - t
            }
            print a[0]
            print a[5]
        """, [6.0, 1.0])

    def test_arrays_are_zero_filled(self):
        self.check("array a[4]\nprint a[0]\nprint a[3]\n", [0.0, 0.0])

    def test_index_expressions_and_reads_in_expressions(self):
        self.check("""
            array a[4]
            a[0] = 10.0
            a[1] = 20.0
            a[2] = a[0] + a[1]
            print a[2] * 2.0 - a[0] - a[1]
        """, [30.0])

    def test_two_arrays(self):
        self.check("""
            array a[4]
            array b[4]
            var i = 0.0
            while i < 4.0 {
                a[i] = i + 1.0
                b[i] = a[i] * 10.0
                i = i + 1.0
            }
            print b[2]
        """, [30.0])

    def test_a_fractional_index_faults_the_machine(self):
        # run through the reference: the fractional address must fault
        text, image, _ = m4c.compile_source("""
            array a[4]
            var i = 1.5
            a[i] = 7.0
            print a[1]
        """)
        memory = Memory(4096)
        memory.write(image.cells)
        result = run(Cpu(f=(0.0,) * 31 + (4096.0,), pc=image.entry), memory, max_steps=1000)
        self.assertEqual(result.reason, "fault")
        self.assertEqual(result.cpu.fault.kind, "address")
        self.assertIn("multiple of 4", result.cpu.fault.detail)

    def test_array_errors(self):
        with self.assertRaises(m4c.CompileError) as caught:
            m4c.compile_source("array a[4]\nprint b[0]\n")
        self.assertIn("never declared", str(caught.exception))
        with self.assertRaises(m4c.CompileError) as caught:
            m4c.compile_source("var x = 1.0\nprint x[0]\n")
        self.assertIn("cannot be indexed", str(caught.exception))
        with self.assertRaises(m4c.CompileError) as caught:
            m4c.compile_source("array a[4]\nx = 1.0\n")
        self.assertIn("never declared", str(caught.exception))
        with self.assertRaises(m4c.CompileError) as caught:
            m4c.compile_source("array a[4]\narray a[8]\n")
        self.assertIn("declared twice", str(caught.exception))
        with self.assertRaises(m4c.CompileError) as caught:
            m4c.compile_source("array a[0]\n")
        self.assertIn("positive whole number", str(caught.exception))

    def test_the_space_is_reserved_aligned_and_before_the_code(self):
        text, image, _ = m4c.compile_source("array a[4]\narray b[2]\nprint b[1]\n")
        self.assertIn("a:     SPACE 16", text)
        self.assertIn("b:     SPACE 8", text)
        self.assertLess(text.index("b:     SPACE"), text.index("LDI.D  d6"))
        self.assertEqual(list(image.cells[:16]), [0] * 16)  # zero-filled, before the code


class Functions(unittest.TestCase):
    def check(self, source, expected_lines):
        out, result, warnings, _ = run_program(source)
        self.assertEqual(result.reason, "halt", result.cpu.fault)
        self.assertEqual(out.splitlines(), [f"{v!r}" for v in expected_lines])
        return warnings

    def test_a_call_and_its_value(self):
        self.check("""
            fn area(r) {
                return 3.0 * r * r
            }
            print area(2.0)
            print area(3.0)
        """, [12.0, 27.0])

    def test_two_arguments_in_order(self):
        self.check("""
            fn sub(a, b) {
                return a - b
            }
            print sub(10.0, 4.0)
            print sub(4.0, 10.0)
        """, [6.0, -6.0])

    def test_callers_variables_survive_the_call(self):
        # the function has its own x; the main's x must come back untouched
        self.check("""
            fn bump(r) {
                var x = r + 1.0
                return x
            }
            var x = 5.0
            print bump(1.0)
            print bump(2.0)
            print x
        """, [2.0, 3.0, 5.0])

    def test_early_return_from_inside_a_loop(self):
        self.check("""
            fn first_below(x) {
                var i = 10.0
                while i > 0.0 {
                    if i < x {
                        return i
                    }
                    i = i - 1.0
                }
                return 0.0
            }
            print first_below(7.5)
            print first_below(0.5)
        """, [7.0, 0.0])

    def test_recursion(self):
        self.check("""
            fn fact(n) {
                if n <= 1.0 {
                    return 1.0
                }
                return n * fact(n - 1.0)
            }
            print fact(6.0)
        """, [720.0])

    def test_a_call_before_the_definition(self):
        self.check("""
            print twice(21.0)
            fn twice(x) {
                return x + x
            }
        """, [42.0])

    def test_calls_inside_expressions_with_live_temps(self):
        self.check("""
            fn triple(x) {
                return x * 3.0
            }
            var a = 2.0
            print triple(a) * triple(a + 1.0) + triple(a + 2.0)
            print a
        """, [6.0 * 9.0 + 12.0, 2.0])

    def test_zero_argument_function(self):
        self.check("""
            fn seven() {
                return 7.0
            }
            print seven() + 1.0
        """, [8.0])

    def test_bare_return_is_zero(self):
        self.check("""
            fn f(x) {
                if x < 0.0 {
                    return
                }
                return x
            }
            print f(4.0)
            print f(-4.0)
        """, [4.0, 0.0])

    def test_falling_off_the_end_is_zero(self):
        self.check("""
            fn f(x) {
                var y = x + 1.0
            }
            print f(9.0)
        """, [0.0])

    def test_a_function_reads_an_array(self):
        self.check("""
            array a[3]
            fn sum3() {
                return a[0] + a[1] + a[2]
            }
            a[0] = 2.0
            a[1] = 4.0
            a[2] = 6.0
            print sum3()
        """, [12.0])

    def test_deep_recursion(self):
        # fib(10): 177 calls; every frame's variables and link ride the stack
        self.check("""
            fn fib(n) {
                if n < 2.0 {
                    return n
                }
                return fib(n - 1.0) + fib(n - 2.0)
            }
            print fib(10.0)
        """, [55.0])

    def test_a_function_writes_an_array(self):
        self.check("""
            array a[4]
            fn fill(step) {
                var i = 0.0
                while i < 4.0 {
                    a[i] = i * step
                    i = i + 1.0
                }
                return a[3]
            }
            print fill(5.0)
            print a[1]
        """, [15.0, 5.0])

    def test_function_errors(self):
        cases = [
            ("print nope(1.0)\n", "not a fn"),
            ("fn f(x) { return x }\nprint f(1.0, 2.0)\n", "takes 1 argument(s)"),
            ("fn f(x) { return x }\nfn f(y) { return y }\n", "declared twice"),
            ("fn main() { return 1.0 }\n", "cannot be named 'main'"),
            ("fn sqrt(x) { return x }\n", "is a builtin"),
            ("return 1.0\n", "return outside a function"),
            ("while 1.0 < 2.0 { fn f() { return 1.0 } }\n", "top level only"),
            ("fn f(a, b, c, d, e, g, h) { return a }\n", "too many arguments"),
            ("fn f(a, a) { return a }\n", "declared twice"),
            ("fn f() { array a[4] return 1.0 }\n", "top level only"),
            ("array f[4]\nfn f() { return 1.0 }\n", "declared twice"),
        ]
        for source, expected in cases:
            with self.assertRaises(m4c.CompileError) as caught:
                m4c.compile_source(source)
            self.assertIn(expected, str(caught.exception), source)

    def test_params_count_toward_the_variable_registers(self):
        self.check("""
            fn f(a, b, c, d, e, g) {
                return a + b + c + d + e + g
            }
            print f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        """, [21.0])
        with self.assertRaises(m4c.CompileError) as caught:
            m4c.compile_source("fn f(a, b, c, d, e, g) { var h = 1.0 return h }\n")
        self.assertIn("no register left", str(caught.exception))
