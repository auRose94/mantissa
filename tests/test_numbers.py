"""Whole numbers, the casts that keep what a plain cast drops, the ternary
branch, and self-modifying code."""
import pathlib
import random
import unittest

from mantissa import asm, fp, isa, machine
from mantissa.machine import Cpu
from mantissa.memory import Memory
from . import support
from .test_machine import run

ROOT = pathlib.Path(__file__).resolve().parent.parent
EDGE = 562949970198528  # the largest whole number a pair of floats keeps


class WholeNumbers(unittest.TestCase):
    def test_chop_shaves_off_what_is_right_of_the_point(self):
        for letter, loader in (("h", "LDI.H"), ("f", "LDI.F"), ("d", "LDI.D")):
            with self.subTest(register=letter):
                out, cpu, _ = run(f"""
                    {loader} {letter}0, 2.7
                    {loader} {letter}1, -2.7
                    {loader} {letter}2, -0.5
                    CHOP {letter}0, {letter}0
                    CHOP {letter}1, {letter}1
                    CHOP {letter}2, {letter}2
                    PUT {letter}0
                    PUT {letter}1
                    PUT {letter}2
                    HALT
                """)
                self.assertEqual(out, "2.0\n-2.0\n-0.0\n")
                self.assertEqual(cpu.flags, fp.INEXACT)

    def test_chop_floor_ceil_and_round_to_a_step(self):
        table = [((2.675, 0.01), (2.67, 2.67, 2.68, 2.67)), ((47, 5), (45, 45, 50, 45)),
                 ((47.5, 5), (45, 45, 50, 50)), ((42.5, 5), (40, 40, 45, 40)), ((-7.3, 2), (-6, -8, -6, -8))]
        for (x, step), (chopped, floored, ceiling, rounded) in table:
            with self.subTest(x=x, step=step):
                out, cpu, _ = run(f"""
                    LDI.D d0, {x!r}
                    LDI.D d1, {step!r}
                    CHOP d2, d0, d1
                    FLOOR d3, d0, d1
                    CEIL d4, d0, d1
                    ROUND d5, d0, d1
                    HALT
                """)
                self.assertEqual(cpu.d[2:6], (chopped, floored, ceiling, rounded))

    def test_rounding_to_a_cent_is_honest_about_what_the_number_really_is(self):
        out, cpu, _ = run("LDI.D d0, 2.675\nLDI.D d1, 0.01\nROUND d2, d0, d1\nHALT")
        self.assertEqual(cpu.d[2], 2.67)  # the double nearest 2.675 is a little below 2.675

    def test_a_step_of_zero_is_not_a_number(self):
        out, cpu, _ = run("LDI.F f0, 7\nLDI.F f1, 0\nCHOP f2, f0, f1\nHALT")
        self.assertNotEqual(cpu.f[2], cpu.f[2])
        self.assertEqual(cpu.flags, fp.INVALID)

    def test_mod_keeps_the_sign_of_its_left_operand(self):
        for a, b, expected in ((47, 5, 2), (-7, 5, -2), (7, -5, 2), (7.5, 2, 1.5), (0, 5, 0)):
            with self.subTest(a=a, b=b):
                out, cpu, _ = run(f"LDI.D d0, {a!r}\nLDI.D d1, {b!r}\nMOD d2, d0, d1\nHALT")
                self.assertEqual(cpu.d[2], expected)

    def test_chop_and_mod_split_a_number_into_quotient_and_remainder(self):
        for a, b in ((1000000007.0, 97.0), (123456789.0, 1000.0), (16777217.0, 10.0)):
            with self.subTest(a=a, b=b):
                out, cpu, _ = run(f"""
                    LDI.D d0, {a!r}
                    LDI.D d1, {b!r}
                    DIV d2, d0, d1
                    CHOP d2, d2
                    MOD d3, d0, d1
                    MUL d4, d2, d1
                    ADD d5, d4, d3
                    HALT
                """)
                self.assertEqual(cpu.d[5], a)


class CastsThatKeepWhatAPlainCastDrops(unittest.TestCase):
    def test_split_and_join_round_trip(self):
        out, cpu, _ = run("""
            LDI.D d0, 10000000001
            SPLIT f1, f2, d0
            JOIN d1, f1, f2
            HALT
        """)
        self.assertEqual((cpu.f[1], cpu.f[2]), (10000000000.0, 1.0))
        self.assertEqual(cpu.d[1], 10000000001.0)
        self.assertEqual(cpu.flags, 0)

    def test_where_the_float_pair_runs_out(self):
        for n, kept in ((EDGE, True), (EDGE + 1, False), (EDGE + 2, True)):
            with self.subTest(n=n):
                out, cpu, _ = run(f"""
                    LDI.D d0, {n}.0
                    SPLIT f1, f2, d0
                    JOIN d1, f1, f2
                    HALT
                """)
                self.assertEqual(cpu.d[1] == float(n), kept)
                self.assertEqual(bool(cpu.flags & fp.INEXACT), not kept)

    def test_every_whole_number_up_to_the_edge_survives(self):
        rng = random.Random(49)
        for _ in range(500):
            n = rng.randrange(0, EDGE + 1)
            high, low, flags = fp.split(F := fp.F, float(n))
            self.assertEqual(fp.join(fp.D, high, low)[0], float(n), f"{n} was lost")
            self.assertEqual(flags, 0)

    def test_a_half_pair_is_limited_by_range_before_precision(self):
        high, low, flags = fp.split(fp.H, 65520.0)
        self.assertEqual(high, fp.INF)  # the nearest half to 65520 overflows

    def test_mant_scales_a_number_into_the_first_binade(self):
        out, cpu, _ = run("""
            LDI.D d0, 6.02e23
            MANT f0, f1, d0
            MANT h2, h3, d0
            CVT.D d2, h2
            CVT.D d3, h3
            SCALB d4, d2, d3
            HALT
        """)
        self.assertTrue(0.5 <= cpu.f[0] < 1.0)
        self.assertEqual((cpu.f[1], cpu.h[3]), (79.0, 79.0))
        self.assertEqual(cpu.d[4], cpu.h[2] * 2.0 ** 79)
        self.assertTrue(0.99 * 6.02e23 < cpu.d[4] < 1.01 * 6.02e23)

    def test_mant_of_special_values(self):
        self.assertEqual(fp.mantissa(0.0), (0.0, 0.0))
        self.assertEqual(fp.mantissa(fp.INF), (fp.INF, 0.0))
        m, e = fp.mantissa(fp.NAN)
        self.assertNotEqual(m, m)

    def test_an_operand_word_written_by_hand_is_rounded_and_flagged(self):
        out, cpu, symbols = run("""
            LDI.F f0, slot + 4
            LDI.D d0, 0.1
            ST.D [f0], d0
            BR slot
        slot: LDI.H h0, 0
            PUT.H h0
            HALT
        """)
        self.assertEqual((cpu.status, cpu.h[0]), ("halted", 0.0999755859375))
        self.assertEqual(cpu.flags, fp.INEXACT)

    def test_type_rules_are_faults_for_an_instruction_written_by_hand(self):
        o = isa.OP
        cases = [
            (isa.encode(o["SPLIT"], 32, 1, 64), "SPLIT writes its two halves to registers of one type"),
            (isa.encode(o["JOIN"], 64, 32, 1), "JOIN reads its two parts from registers of one type"),
            (isa.encode(o["ADD"], 0, 33, 2), "ADD needs operands of one type"),
            (isa.encode(o["CVT"], 32, 33, 0), "CVT changes type"),
            (isa.encode(o["LD"], 0, 1, 0), "LD reads its address from a float register"),
            (isa.encode(o["JMP"], 0, 0, 0), "JMP reads its target from a float register"),
        ]
        for number, fragment in cases:
            with self.subTest(number=number):
                memory = Memory(64)
                memory.store(fp.D, 0.0, float(number))
                result = machine.run(Cpu(), memory, max_steps=5)
                self.assertEqual(result.reason, "fault")
                self.assertEqual((result.cpu.fault.kind, fragment in result.cpu.fault.detail),
                                 ("illegal-instruction", True))


class Ternary(unittest.TestCase):
    def where(self, value):
        out, cpu, _ = run(f"""
            LDI.D d0, {value}
            BSGN d0, negative, positive
            LDI.H h0, 0
            BR put
            negative: LDI.H h0, 1
            BR put
            positive: LDI.H h0, 2
            put: PUT h0
            HALT
        """)
        return out

    def test_negative_jumps_positive_jumps_and_zero_falls_through(self):
        self.assertEqual(self.where("-3"), "1.0\n")
        self.assertEqual(self.where("3"), "2.0\n")
        self.assertEqual(self.where("0"), "0.0\n")
        self.assertEqual(self.where("-0"), "0.0\n")
        self.assertEqual(self.where("nan"), "0.0\n")  # a NaN is neither negative nor positive

    def test_a_three_way_branch_needs_no_operand_word_of_its_own_to_fall_through(self):
        out, cpu, symbols = run("back: LDI.D d0, 0\nBSGN d0, 4, back\nstop: HALT\n")
        self.assertEqual((cpu.status, cpu.pc), ("halted", symbols["stop"]))  # it fell through twelve cells

    def test_cmp_is_three_valued(self):
        for a, b, expected in ((7, 3, 1), (3, 7, -1), (5, 5, 0), (7.5, 7.25, 1)):
            with self.subTest(a=a, b=b):
                out, cpu, _ = run(f"LDI.D d0, {a!r}\nLDI.D d1, {b!r}\nCMP d2, d0, d1\nHALT")
                self.assertEqual(cpu.d[2], expected)
        out, cpu, _ = run("LDI.D d0, nan\nLDI.D d1, 1\nCMP d2, d0, d1\nHALT")
        self.assertNotEqual(cpu.d[2], cpu.d[2])


class SelfModifyingCode(unittest.TestCase):
    def test_a_loop_whose_instruction_is_its_own_counter(self):
        out, result, _, image = support.run_source((ROOT / "programs" / "smc_edit.mant").read_text(encoding="utf-8"))
        self.assertEqual(result.reason, "halt")
        self.assertEqual(out, "70010000.0\n2.0\n70020000.0\n4.0\n70030000.0\n8.0\n70040000.0\n16.0\n")

    def test_a_store_is_seen_by_the_very_next_fetch(self):
        out, cpu, symbols = run("""
            LDI.F f0, slot
            LDI.D d0, 2000000       ; the number of the HALT instruction
            ST.D [f0], d0
            BR slot
            HALT
        slot: NOP
        """)
        self.assertEqual((cpu.status, cpu.pc), ("halted", symbols["slot"]))

    def test_an_instruction_is_a_number_a_program_can_compute(self):
        out, cpu, symbols = run("""
            LDI.H h1, 2
            LDI.H h2, 3
            LDI.F f0, slot
            LDI.D d0, 20000000      ; opcode 20: ADD
            LDI.D d1, 10102         ; the operand digits: h1, h1, h2
            ADD.D d0, d0, d1
            ST.D [f0], d0           ; write ADD h1, h1, h2
            BR slot
        slot: NOP
            PUT.H h1
            HALT
        """)
        self.assertEqual(out, "5.0\n")

    def test_a_program_that_writes_a_program(self):
        out, result, memory, image = support.run_source((ROOT / "programs" / "smc_emit.mant").read_text(encoding="utf-8"))
        self.assertEqual(result.reason, "halt")
        self.assertTrue(out.endswith("25.0\n"), out)
        o = isa.OP
        code = int(image.symbols["code"])
        words = [fp.unpack(fp.D, memory.cells[at:at + 4]) for at in range(code, code + 64, 4)]
        self.assertEqual(words, [11000000.0, 1.0, 11010000.0, 3.0, 20000001.0, 11010000.0, 5.0, 20000001.0,
                                 11010000.0, 7.0, 20000001.0, 11010000.0, 9.0, 20000001.0, 70000000.0, 2000000.0])

    def test_a_program_that_takes_its_own_instructions_apart(self):
        out, result, memory, image = support.run_source((ROOT / "programs" / "selflist.mant").read_text(encoding="utf-8"))
        start, end = int(image.symbols["main"]), int(image.symbols["end"])
        expected = []
        for at in range(start, end, 4):
            word = fp.unpack(fp.D, memory.cells[at:at + 4])
            ins = isa.decode(word)
            expected.append(f"{ins[0]:02d} {ins[1]:02d} {ins[2]:02d} {ins[3]:02d}" if ins else repr(word))
        self.assertEqual(result.reason, "halt")
        self.assertEqual(out, "\n".join(expected) + "\n")

    def test_arithmetic_that_misses_is_an_illegal_instruction(self):
        out, cpu, symbols = run("""
            LDI.F f0, slot
            LDI.D d0, 70000100      ; a PUT with a stray second operand field
            ST.D [f0], d0
            BR slot
            HALT
        slot: NOP
        """)
        self.assertEqual(cpu.fault.detail, "70000100.0 is not an instruction")

    def test_zero_and_data_are_not_instructions(self):
        out, cpu, _ = run("LDI.F f0, table\nJMP f0\nHALT\ntable: DATA.D 1000")
        self.assertEqual(cpu.fault.detail, "1000.0 is not an instruction")
        out, cpu, _ = run("LDI.F f0, 1000\nJMP f0\nHALT")
        self.assertEqual(cpu.fault.detail, "0.0 is not an instruction")

    def test_a_pc_between_instructions_is_an_address_fault(self):
        out, cpu, _ = run("LDI.F f0, 2\nJMP f0\nHALT")
        self.assertEqual((cpu.fault.kind, cpu.fault.detail), ("address", "2: misaligned, needs a multiple of 4"))


if __name__ == "__main__":
    unittest.main()
