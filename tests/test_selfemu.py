"""The self-interpreter: programs/selfemu.mant running targets of its own kind.

Every target runs twice on the same console contract: directly under the
reference machine, and spliced into the interpreter's image between the
`target` and `end` lines of programs/selfemu.mant.  The console output must
match value for value and both runs must end in HALT -- which is also the
proof that the interpreter is itself a legal program: the reference
assembler accepted it, and the machine ran it to a halt without a fault.
The interpreted steps divided by the direct steps are the slowdown recorded
in EXTENSIONS.md.
"""
import io
import pathlib
import unittest

from mantissa import asm
from mantissa.machine import Cpu, console, run
from mantissa.memory import Memory

ROOT = pathlib.Path(__file__).resolve().parent.parent
SELFEMU = (ROOT / "programs" / "selfemu.mant").read_text(encoding="utf-8")
HEAD, _, REST = SELFEMU.partition("\ntarget:\n")  # the interpreter, then its demo target
DEMO, _, _ = REST.partition("\nend:")

HELLO = """
t_start:
        LDI.F  f0, t_msg
        LDI.F  f1, 1
t_again:
        LD.H   h0, [f0]
        BZ.H   h0, t_done
        PUTC.H h0
        ADD.F  f0, f0, f1
        BR     t_again
t_done:
        HALT
t_msg:  TEXT "interpreting is the sincerest form of flattery\\n"
        DATA.H 0
"""

COUNT = """
t0:     LDI.H  h1, 1
        LDI.H  h4, 1
        LDI.H  h9, 10
        LDI.H  h2, 48
t_again:
        ADD.H  h0, h1, h2      ; the digit for the count so far
        PUTC.H h0
        ADD.H  h1, h1, h4
        SUB.H  h5, h1, h9
        BNZ.H  h5, t_again
        LDI.H  h6, 10
        PUTC.H h6
        LDI.H  h7, 7
        LDI.H  h8, 6
        MUL.H  h3, h7, h8
        PUT.H  h3
        HALT
"""

BRANCHES = """
t0:     LDI.H  h0, 5
        LDI.H  h1, 5
        LDI.H  h2, 61         ; '='
        BEQ.H  h0, h1, t_eq
        LDI.H  h2, 33         ; '!', skipped
t_eq:   PUTC.H h2
        LDI.H  h3, 2
        LDI.H  h4, 7
        SUB.H  h5, h3, h4
        BZ.H   h5, t_zero     ; not taken
        LDI.H  h6, 110        ; 'n'
        PUTC.H h6
        BR     t_fin
t_zero: LDI.H  h6, 122        ; 'z'
        PUTC.H h6
t_fin:  LDI.F  f0, 2.5
        LDI.F  f1, 3.5
        BLT.F  f0, f1, t_lt
        LDI.H  h7, 78         ; 'N', skipped
        PUTC.H h7
        BR     t_dif
t_lt:   LDI.H  h7, 121        ; 'y'
        PUTC.H h7
        LDI.H  h8, 3
        LDI.H  h9, 5
        BNE.H  h8, h9, t_dif
        LDI.H  h0, 63         ; '?', skipped
        PUTC.H h0
t_dif:  LDI.H  h0, 100        ; 'd'
        PUTC.H h0
        LDI.H  h3, 5
        LDI.H  h4, 5
        BLE.H  h3, h4, t_le
        LDI.H  h2, 33         ; '!', skipped
t_le:   LDI.H  h2, 101        ; 'e'
        PUTC.H h2
        LDI.H  h1, 10
        PUTC.H h1
        HALT
"""

ROUNDTRIP = """
t0:     LDI.D  d0, 1234
        LDI.F  f0, tbuf
        ST.D   [f0], d0
        LD.D   d1, [f0]
        MOV.D  d2, d1
        PUT.D  d2
        LDI.D  d3, 40
        LDI.D  d4, 2
        ADD.D  d5, d3, d4
        PUT.D  d5
        LDI.H  h0, 2000
        ST.H   [f0], h0
        LD.H   h1, [f0]
        PUT.H  h1
        LDI.F  f1, 2.5
        ST.F   [f0], f1
        LD.F   f2, [f0]
        PUT.F  f2
        LDI.F  f3, 7
        LDI.F  f4, 2
        DIV.F  f5, f3, f4
        PUT.F  f5
        HALT
tbuf:   SPACE 4
"""

CALLOUT = """
t0:     LDI.H  h9, 112        ; 'p'
        LDI.F  f0, 5
        LDI.F  f7, t_show
        JAL    f6, t_show
        LDI.H  h9, 110        ; 'n'
        LDI.F  f0, -3
        JAL    f6, t_show
        LDI.H  h9, 122        ; 'z'
        LDI.F  f0, 0
        JAL    f6, t_show
        LDI.H  h8, 10
        PUTC.H h8
        HALT
t_show: PUTC.H h9
        BSGN.F f0, t_neg, t_pos
t_neg:  LDI.H  h1, 45         ; '-'
        PUTC.H h1
        JMP    f6
t_pos:  LDI.H  h1, 43         ; '+'
        PUTC.H h1
        JMP    f6
"""

NARROWING = """
t0:     LDI.H  h0, 4094
        LDI.H  h1, 3
        ADD.H  h2, h0, h1     ; 4097, which half rounds down to 4096
        PUT.H  h2
        LDI.H  h3, 2049
        MUL.H  h4, h3, h3     ; 4198401 overflows a half: inf
        PUT.H  h4
        LDI.F  f0, 0.1
        LDI.F  f1, 300
        MUL.F  f2, f0, f1     ; rounded in float: 30.0, like the double
        PUT.F  f2
        LDI.D  d0, 0.1
        LDI.D  d1, 300
        MUL.D  d2, d0, d1     ; and in double, also 30.0
        PUT.D  d2
        LDI.F  f3, 65504
        LDI.F  f4, 2
        MUL.F  f5, f3, f4     ; overflows the half file's reach, not a float's
        PUT.F  f5
        HALT
"""

TARGETS = {"demo": DEMO, "hello": HELLO, "count": COUNT, "branches": BRANCHES,
           "roundtrip": ROUNDTRIP, "callout": CALLOUT, "narrowing": NARROWING}


def run_image(source, steps=400_000, cells=8192):
    image = asm.assemble(source)
    memory = Memory(cells)
    memory.write(image.cells)
    out = io.StringIO()
    result = run(Cpu(pc=image.entry), memory, out=console(out.write), max_steps=steps)
    return out.getvalue(), result


def spliced(target):
    """The interpreter with the target placed between `target` and `end`."""
    return HEAD + "\ntarget:\n" + target + "\nend:\n"


class SelfEmulator(unittest.TestCase):
    def run_both(self, target):
        direct_out, direct = run_image(target)
        interp_out, interp = run_image(spliced(target))
        self.assertEqual(direct.reason, "halt")
        self.assertEqual(interp.reason, "halt", "the interpreter faulted or stalled")
        self.assertEqual(direct_out, interp_out)
        return direct_out, direct.steps, interp.steps

    def test_each_target_matches_the_reference(self):
        self.assertEqual(run_image(spliced(""))[1].reason, "halt")  # an empty target just stops
        for name, target in TARGETS.items():
            with self.subTest(target=name):
                out, direct_steps, interp_steps = self.run_both(target)
                self.assertTrue(out)
                print(f"selfemu {name:>9}: {direct_steps:4d} steps directly, "
                      f"{interp_steps:5d} interpreted, slowdown {interp_steps / direct_steps:.1f}x")

    def test_the_expected_output_of_three_targets(self):
        out, _, _ = self.run_both(HELLO)
        self.assertEqual(out, "interpreting is the sincerest form of flattery\n")
        out, _, _ = self.run_both(COUNT)
        self.assertEqual(out, "123456789\n42.0\n")
        out, _, _ = self.run_both(CALLOUT)
        self.assertEqual(out, "p+n-z-\n")

    def test_the_slowdown_is_in_the_usual_band(self):
        for name, target in TARGETS.items():
            with self.subTest(target=name):
                _, direct_steps, interp_steps = self.run_both(target)
                ratio = interp_steps / direct_steps
                self.assertGreater(ratio, 5, "the interpreter is suspiciously cheap")
                self.assertLess(ratio, 200, "the interpreter has gone hopelessly slow")

    def test_the_whole_file_is_legal_and_agrees_with_its_own_demo(self):
        out, result = run_image(SELFEMU)
        self.assertEqual(result.reason, "halt")
        demo_out, demo = run_image(DEMO)
        self.assertEqual(out, demo_out)
        self.assertEqual(demo.reason, "halt")

    def test_a_target_that_falls_off_its_end_stops_the_interpreter(self):
        runaway = "LDI.H h0, 65\nPUTC.H h0\n"
        out, result = run_image(spliced(runaway))
        self.assertEqual((out, result.reason), ("A", "halt"))  # zeroed memory decodes to opcode 00
        out, result = run_image(runaway)
        self.assertEqual((out, result.reason), ("A", "fault"))  # the reference faults instead

    def test_an_illegal_target_instruction_stops_the_interpreter(self):
        junk = "junk: DATA.D 99000000"
        out, result = run_image(spliced(junk))
        self.assertEqual((out, result.reason), ("", "halt"))
        out, result = run_image(junk)
        self.assertEqual((out, result.reason), ("", "fault"))


if __name__ == "__main__":
    unittest.main()
