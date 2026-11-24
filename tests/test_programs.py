"""The programs in programs/, each a small proof of what the machine is like,
and the command line around them."""
import importlib.util
import json
import math
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

from mantissa import asm, fp, trace
from mantissa.machine import Cpu, run
from mantissa.memory import Memory
from . import support

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROGRAMS = ROOT / "programs"


def source(name):
    return (PROGRAMS / name).read_text(encoding="utf-8")


def execute(text, steps=3_000_000, cells=65536):
    out, result, _, image = support.run_source(text, steps=steps, cells=cells)
    return out, result


def single(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def zoom_module():
    """programs/mandel_zoom.py, loaded as a module."""
    spec = importlib.util.spec_from_file_location("mandel_zoom", PROGRAMS / "mandel_zoom.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Programs(unittest.TestCase):
    def test_hello(self):
        out, result = execute(source("hello.mant"))
        self.assertEqual((result.reason, out), ("halt", "Hello from a computer with no integers.\n"))

    def test_stall(self):
        # three setup instructions, then three per loop: eleven numbers in 36 steps
        out, result = execute(source("stall.mant"), steps=36)
        self.assertEqual(out, "2046.0\n2047.0\n" + "2048.0\n" * 9)
        self.assertEqual(result.reason, "step limit")
        self.assertEqual(result.cpu.h[0], 2048.0)

    def test_wall24(self):
        out, result = execute(source("wall24.mant"))
        self.assertEqual(result.reason, "halt")
        self.assertEqual(out, "".join(
            f"float {float(f)!r}\ndouble {float(d)!r}\n" for f, d in (
                (16777215, 16777215), (16777216, 16777216), (16777216, 16777217),
                (16777216, 16777218), (16777216, 16777219))))

    def test_tenths_against_an_independent_simulation(self):
        term_h, term_f = half(0.1), single(0.1)
        h_sum = f_sum = d_sum = c = 0.0
        k_sum = 0.0
        for _ in range(1000):
            h_sum = half(h_sum + term_h)
            f_sum = single(f_sum + term_f)
            d_sum += 0.1
            y = single(term_f - c)
            t = single(k_sum + y)
            c = single(single(t - k_sum) - y)
            k_sum = t
        expected = (f"half, plain:       {h_sum!r}\n"
                    f"float, plain:      {f_sum!r}\n"
                    f"double, plain:     {d_sum!r}\n"
                    f"float, compensated: {k_sum!r}\n")
        out, result = execute(source("tenths.mant"))
        self.assertEqual((result.reason, out), ("halt", expected))
        self.assertGreater(abs(h_sum - 100), 5, "a half is off by several percent after a thousand steps")
        self.assertGreater(abs(f_sum - 100), 0.0005)
        self.assertLess(abs(d_sum - 100), 1e-10)
        self.assertEqual(k_sum, 100.0, "compensation keeps the float sum exact to the last bit")

    def test_a_counter_of_four_floats_crosses_the_wall(self):
        out, result = execute(source("bignum.mant"))
        # where a single float freezes, four 12-bit limbs keep counting; printed with chop and mod
        self.assertEqual((result.reason, out), ("halt", "".join(f"{n}\n" for n in range(16777214, 16777221))))
        self.assertNotEqual(single(16777217), 16777217.0)  # what the plain float could not have done

    def test_ternary_search(self):
        out, result = execute(source("ternary.mant"))
        guesses = [float(line) for line in out.split()]
        self.assertEqual(result.reason, "halt")
        self.assertEqual(guesses[-1], 7734.0)
        low, high = 0.0, 10000.0
        for guess in guesses:  # every guess is honest: halfway, and the right half is kept
            self.assertEqual(guess, math.floor((low + high) / 2))
            if guess < 7734:
                low = guess + 1
            elif guess > 7734:
                high = guess - 1
        self.assertEqual(out, "5000.0\n7500.0\n8750.0\n8125.0\n7812.0\n7656.0\n7734.0\n")

    def test_grid(self):
        out, result = execute(source("grid.mant"))
        rows = [((2.675, 0.01), (2.67, 2.67, 2.68, 2.67)), ((47, 5), (45, 45, 50, 45)),
                ((47.5, 5), (45, 45, 50, 50)), ((42.5, 5), (40, 40, 45, 40)), ((-7.3, 2), (-6, -8, -6, -8))]
        expected = "".join("".join(f"{float(v)!r}\n" for v in (x, step) + results) + "\n"
                           for (x, step), results in rows)
        self.assertEqual((result.reason, out), ("halt", expected))

    def test_pairs(self):
        out, result = execute(source("pairs.mant"))
        kept = float(562949970198528)
        self.assertEqual(out, "a plain cast to float:  10000000000.0\n"      # the nearest float, by struct
                         "split into two floats:  10000000000.0\n1.0\n"
                         f"joined back:            {10000000001.0!r}\n"
                         f"the last one kept:      {kept!r}\n"
                         "6.02e23 as a half:      inf\n"
                         "its mantissa, exponent: 0.99609375\n79.0\n"
                         f"scaled back:            {0.99609375 * 2.0 ** 79!r}\n")
        self.assertEqual(single(10000000001.0), 10000000000.0)


class Mandelbrot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zoom = zoom_module()
        cls.source = source("mandel.mant")

    def reference(self, r):
        """The picture mandel.mant draws, operation for operation, every result
        rounded by r.  Addresses and counters are floats throughout, as in
        the program."""
        left, right, bottom, top = r(-2.0), r(0.6), r(-1.1), r(1.1)
        two, one, four = r(2.0), r(1.0), r(4.0)
        factor = single(0.225)
        ramp = " .:-=+*#%@"
        out = []
        for row in range(22):
            for col in range(64):
                cx = r(r(r(r(right - left) * r(float(col))) / r(64.0)) + left)
                cy = r(r(r(r(bottom - top) * r(float(row))) / r(22.0)) + top)
                zx = zy = 0.0
                n = 0.0
                while True:
                    zx2, zy2 = r(zx * zx), r(zy * zy)
                    if four <= r(zx2 + zy2):
                        out.append(ramp[min(int(math.trunc(single(float(n) * factor))), 9)])
                        break
                    if not n < 40.0:
                        out.append("@")
                        break
                    folded = r(r(zx * zy) * two)
                    zx, zy = r(r(zx2 - zy2) + cx), r(folded + cy)
                    n = r(n + one)
            out.append("\n")
        return "".join(out)

    def render(self, text):
        image = asm.assemble(text)
        memory = Memory(65536)
        memory.write(image.cells)
        out, result = execute(text)
        self.assertEqual(result.reason, "halt")
        return out

    def test_double_matches_plain_python_arithmetic(self):
        self.assertEqual(self.render(self.source), self.reference(lambda v: v))

    def test_float_matches_arithmetic_rounded_to_float_after_every_operation(self):
        text = self.zoom.retyped(self.source, "F", lambda n: f"f{n + 15}", True)
        self.assertEqual(self.render(text), self.reference(single))

    def test_half_matches_arithmetic_rounded_to_half_after_every_operation(self):
        text = self.zoom.retyped(self.source, "H", lambda n: f"h{n}", False)
        self.assertEqual(self.render(text), self.reference(half))

    def test_the_zoom_tour_rewrites_a_program_that_still_assembles(self):
        for _, suffix, register, counters_too in self.zoom.VARIANTS:
            text = self.zoom.windowed(
                self.zoom.retyped(self.source, suffix, register, counters_too), -1e-7, 1e-7, 0.999999915, 1.00000008)
            self.assertGreater(len(asm.assemble(text).cells), 0)


class GoldenTrace(unittest.TestCase):
    def record(self, name, steps=1_000_000):
        text = source(name)
        image = asm.assemble(text)
        memory = Memory(4096)
        floats = [0.0] * 32
        floats[31] = 4096.0
        memory.write(image.cells)
        return image, memory, trace.lines(image, memory, Cpu(f=tuple(floats), pc=image.entry), steps)

    def test_header_and_image(self):
        image, memory, lines = self.record("hello.mant")
        self.assertEqual(lines[:6], ["# mantissa-1 trace", "cells 4096", "org 0", "entry 0", "sp 4096",
                                     f"image {len(image.cells)}"])
        image_lines = []
        for line in lines[6:]:
            if line == "run":
                break
            image_lines.append(line)
        cells = [int(cell, 16) for line in image_lines for cell in line.split()]
        self.assertEqual(cells, list(image.cells))
        rebuilt = Memory(4096)
        rebuilt.write(cells)
        self.assertEqual(rebuilt.fetch(48.0), 2000000.0)  # the HALT, read back as a number

    def test_one_line_per_step_then_the_ending(self):
        image, memory, lines = self.record("wall24.mant")
        run_lines = lines[lines.index("run") + 1:]
        steps = [line for line in run_lines if line.startswith("S ")]
        effects = [line for line in run_lines if not line.startswith("S ")]
        self.assertEqual(lines[-1], "H")
        self.assertEqual(len(steps) + len(effects), len(run_lines))
        self.assertTrue(all(int(line.split()[1]) < 4096 for line in steps))
        characters = "".join(chr(int(line.split()[3])) for line in effects if line.startswith("C "))
        values = [struct.unpack("<d", bytes.fromhex(line.split()[3]))[0] for line in effects if line.startswith("P ")]
        self.assertEqual((characters.count("float "), characters.count("double "), len(values)), (5, 5, 10))
        self.assertEqual(values[:6], [16777215.0, 16777215.0, 16777216.0, 16777216.0, 16777216.0, 16777217.0])
        self.assertEqual(characters[:6], "float ")  # the label is spelled one character at a time
    def test_replaying_the_stores_rebuilds_the_final_memory(self):
        image, memory, lines = self.record("smc_emit.mant")
        final = list(memory.cells)
        rebuilt = Memory(4096)
        rebuilt.write(image.cells)
        for line in lines:
            if line.startswith("M "):
                _, step, pc, addr, cells, bits = line.split()
                value = struct.unpack("<d", bytes.fromhex(bits))[0]
                rebuilt.store({1: fp.H, 2: fp.F, 4: fp.D}[int(cells)], float(addr), value)
        self.assertEqual(list(rebuilt.cells), final)

    def test_a_fault_and_a_step_limit_are_recorded_as_such(self):
        image = asm.assemble("LDI.F f0, table\nJMP f0\nHALT\ntable: DATA.D 1000")
        memory = Memory(256)
        memory.write(image.cells)
        lines = trace.lines(image, memory, Cpu(pc=image.entry), 100)
        self.assertEqual(lines[-1], "F illegal-instruction")
        image = asm.assemble("loop: BR loop")
        memory = Memory(256)
        memory.write(image.cells)
        lines = trace.lines(image, memory, Cpu(pc=image.entry), 7)
        self.assertEqual(lines[-1], "L")
        self.assertEqual(sum(line == "X 3 0" for line in lines), 1)  # a branch changes no register


class CommandLine(unittest.TestCase):
    def mantissa(self, *args):
        return subprocess.run([sys.executable, "-m", "mantissa", *args], cwd=ROOT,
                              capture_output=True, text=True)

    def test_program_output_and_simulator_notes_go_to_different_streams(self):
        done = self.mantissa("run", "programs/hello.mant")
        self.assertEqual((done.returncode, done.stdout), (0, "Hello from a computer with no integers.\n"))
        self.assertRegex(done.stderr, r"^; halted: HALT at 48 after 245 steps\n$")

    def test_exit_codes_tell_halt_fault_and_step_limit_apart(self):
        stalled = self.mantissa("run", "programs/stall.mant", "--steps", "36")
        self.assertEqual((stalled.returncode, stalled.stdout.count("\n")), (2, 11))
        self.assertIn("; stopped: step limit 36 reached", stalled.stderr)
        with tempfile.TemporaryDirectory() as folder:
            off = pathlib.Path(folder) / "off.mant"
            off.write_text("LDI.F f0, 2\nLD.D d0, [f0]\nHALT\n", encoding="utf-8")
            faulted = self.mantissa("run", str(off))
        self.assertEqual(faulted.returncode, 1)
        self.assertIn("; fault: address at 8 (LD d0, [f0]): 2: misaligned, needs a multiple of 4", faulted.stderr)

    def test_panel_and_trace(self):
        panel = self.mantissa("run", "programs/stall.mant", "--steps", "36", "--panel")
        self.assertIn("; h: h0=2048.0 h1=1.0 h2=2050.0", panel.stderr)
        self.assertIn("; pc: 24  flags: inexact", panel.stderr)
        trace = self.mantissa("run", "programs/stall.mant", "--steps", "3", "--trace")
        self.assertEqual(trace.stderr.splitlines()[:3],
                         [";      0  LDI h1, 1.0", ";      8  LDI h2, 2050.0", ";     16  LDI h0, 2045.0"])

    def test_a_fault_mid_line_starts_its_report_on_its_own_line(self):
        faulted = self.mantissa("run", "programs/pairs.mant", "--steps", "90")
        self.assertEqual(faulted.returncode, 2)
        self.assertRegex(faulted.stderr, r"(?s).*\n; stopped: step limit 90 reached")

    def test_listing_shows_instructions_as_numbers(self):
        listing = self.mantissa("disasm", "programs/smc_edit.mant")
        self.assertEqual(listing.returncode, 0)
        lines = listing.stdout.splitlines()
        self.assertEqual(lines[0], "     0  11 37 00 00  LDI f5, 88.0")
        self.assertTrue(any(line.endswith("tick PUT h0") for line in lines), lines)
        self.assertTrue(any(line.endswith("loop LD d0, [f5]") for line in lines), lines)

    def test_trace_command(self):
        with tempfile.TemporaryDirectory() as folder:
            out = str(pathlib.Path(folder) / "hello")
            done = self.mantissa("trace", "programs/hello.mant", "--out", out, "--cells", "1024")
            self.assertEqual((done.returncode, done.stdout), (0, ""))
            self.assertIn("hello: 245 steps", done.stderr)
            text = pathlib.Path(out).read_text(encoding="utf-8")
        self.assertEqual(text.splitlines()[1], "cells 1024")
        self.assertEqual(text.splitlines()[-1], "H")

    def test_an_assemble_error_names_the_line(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = pathlib.Path(folder) / "bad.mant"
            bad.write_text("NOP\nADD.H h0, f1, h0\n", encoding="utf-8")
            failed = self.mantissa("run", str(bad))
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(failed.stdout, "")
        self.assertIn("bad.mant:2: ADD.H is written for halfs here, but f1 holds a float", failed.stderr)


if __name__ == "__main__":
    unittest.main()
