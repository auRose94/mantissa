"""Programs, as the machine runs them."""
import struct
import unittest

from mantissa import asm
from mantissa.fp import DIVZERO, INEXACT
from mantissa.machine import FAULT, HALTED, RUNNING, Cpu, step
from mantissa.memory import Memory
from . import support

TWO_24 = 16777216.0


def run(source, steps=200_000):
    """(output, final Cpu, symbols) for a snippet."""
    out, result, _, image = support.run_source(source, steps=steps)
    return out, result.cpu, image.symbols


def halves_and_floats_of(value):
    """What a double looks like when read back as two floats and four halfs."""
    raw = struct.pack("<d", value)  # little endian; the cells hold it most significant first
    floats = struct.unpack("<ff", raw[4:8] + raw[0:4])
    halfs = struct.unpack("<eeee", raw[6:8] + raw[4:6] + raw[2:4] + raw[0:2])
    return floats, halfs


class Registers(unittest.TestCase):
    def test_each_type_has_its_own_registers(self):
        out, cpu, _ = run("""
            LDI.H  h1, 1.5
            LDI.F  f1, 1.5
            LDI.D  d1, 1.5
            ADD.H  h0, h1, h1
            ADD.F  f0, f1, f1
            ADD.D  d0, d1, d1
            LDI.H  h2, 0.1
            ADD.H  h3, h2, h2
            LDI.F  f2, 0.1
            ADD.F  f3, f2, f2
            HALT
        """)
        self.assertEqual(cpu.status, HALTED)
        self.assertEqual((cpu.h[0], cpu.f[0], cpu.d[0]), (3.0, 3.0, 3.0))
        self.assertEqual(cpu.h[3], 0.199951171875)  # the half nearest 0.1, doubled exactly
        self.assertEqual(cpu.f[3], 0.20000000298023224)
        self.assertEqual(out, "")

    def test_there_are_32_registers_of_each_type(self):
        out, cpu, _ = run("""
            LDI.H  h29, 9
            LDI.F  f30, 8
            LDI.D  d31, 7
            HALT
        """)
        self.assertEqual((cpu.h[29], cpu.f[30], cpu.d[31]), (9.0, 8.0, 7.0))

    def test_mov_copies_within_one_type(self):
        out, cpu, _ = run("""
            LDI.F  f1, 7.5
            MOV.F  f0, f1
            LDI.H  h0, 9
            MOV.H  h31, h0
            LDI.D  d0, nan
            MOV.D  d1, d0
            HALT
        """)
        self.assertEqual((cpu.f[0], cpu.h[31]), (7.5, 9.0))
        self.assertTrue(support.canonical(cpu.d[1]) and support.canonical(cpu.d[0]))

    def test_a_step_returns_a_new_cpu_and_leaves_the_old_one_alone(self):
        image = asm.assemble("LDI.H h0, 1\nADD.H h0, h0, h0\nHALT")
        memory = Memory(64)
        memory.write(image.cells)
        before = Cpu(pc=image.entry)
        middle = step(before, memory)
        after = step(middle, memory)
        self.assertEqual((before.h[0], before.pc), (0.0, 0.0))
        self.assertEqual((middle.h[0], middle.pc), (1.0, 8.0))
        self.assertEqual((after.h[0], after.pc, after.status), (2.0, 12.0, RUNNING))
        self.assertEqual(step(after, memory).status, HALTED)


class TheCountingWalls(unittest.TestCase):
    def test_a_half_counter_freezes_at_2048(self):
        out, cpu, _ = run("""
            LDI.H  h1, 1
            LDI.H  h2, 2050
            LDI.H  h0, 2045
            loop:   ADD.H  h0, h0, h1
                    BLT.H  h0, h2, loop
            HALT
        """, steps=400)
        self.assertEqual(cpu.status, RUNNING)
        self.assertEqual(cpu.h[0], 2048.0, "the counter is stuck short of the limit")
        self.assertEqual(cpu.flags, INEXACT, "every stalled increment is flagged inexact")

    def test_a_float_counter_freezes_at_16777216(self):
        out, cpu, _ = run("""
            LDI.F  f1, 1
            LDI.F  f2, 16777218
            LDI.F  f0, 16777215
            loop:   ADD.F  f0, f0, f1
                    BLT.F  f0, f2, loop
            HALT
        """, steps=400)
        self.assertEqual(cpu.status, RUNNING)
        self.assertEqual(cpu.f[0], TWO_24)

    def test_the_same_loop_in_double_ends(self):
        out, cpu, _ = run("""
            LDI.D  d1, 1
            LDI.D  d2, 16777218
            LDI.D  d0, 16777215
            loop:   ADD.D  d0, d0, d1
                    BLT.D  d0, d2, loop
            PUT.D  d0
            HALT
        """, steps=400)
        self.assertEqual((cpu.status, out), (HALTED, f"{TWO_24 + 2.0!r}\n"))

    def test_a_down_counter_stalls_at_the_same_wall(self):
        out, cpu, _ = run("""
            LDI.H  h1, 1
            LDI.H  h0, 4094       ; spacing is 2 here: an odd slot steps two down to an even one
            SUB.H  h0, h0, h1
            PUT.H  h0
            SUB.H  h0, h0, h1     ; 4092 - 1 is a tie, and it rounds back to 4092
            PUT.H  h0
            HALT
        """)
        self.assertEqual((cpu.status, out), (HALTED, "4092.0\n4092.0\n"))

    def test_a_down_counter_ends_below_the_wall(self):
        out, cpu, _ = run("""
            LDI.H  h1, 1
            LDI.H  h0, 2047
            loop:   SUB.H  h0, h0, h1
                    BNZ.H  h0, loop
            PUT.H  h0
            HALT
        """, steps=10_000)
        self.assertEqual((cpu.status, out), (HALTED, "0.0\n"))


class MemoryAccess(unittest.TestCase):
    def test_a_double_reads_as_two_floats_and_four_halfs(self):
        floats, halfs = halves_and_floats_of(1.0)
        out, cpu, _ = run("""
            LDI.F  f1, cell
            LDI.D  d0, 1.0
            ST.D   [f1], d0
            LD.F   f0, [f1]        ; the leading half of the double, read as a float
            LD.H   h0, [f1]
            LDI.F  f2, 1
            ADD.F  f1, f1, f2
            LD.H   h1, [f1]
            ADD.F  f1, f1, f2
            LD.F   f3, [f1]        ; the trailing half, read as a float
            LD.H   h2, [f1]
            ADD.F  f1, f1, f2
            LD.H   h3, [f1]
            HALT
        cell:   DATA.D 0
        """)
        self.assertEqual(cpu.status, HALTED)
        self.assertEqual((cpu.f[0], cpu.f[3]), floats)
        self.assertEqual(cpu.h[0:4], halfs)
        self.assertEqual((floats, halfs), ((1.875, 0.0), (1.984375, 0.0, 0.0, 0.0)))

    def test_floats_need_an_even_address(self):
        out, cpu, _ = run("LDI.F f0, 1\nLDI.F f1, 5\nST.F [f0], f1\nHALT")
        self.assertEqual((cpu.status, cpu.fault.kind, cpu.fault.detail),
                         (FAULT, "address", "1: misaligned, needs a multiple of 2"))

    def test_doubles_need_a_multiple_of_four(self):
        out, cpu, _ = run("LDI.F f0, 2\nLDI.D d1, 5\nLD.D d0, [f0]\nHALT")
        self.assertEqual((cpu.status, cpu.fault.detail), (FAULT, "2: misaligned, needs a multiple of 4"))

    def test_a_half_lives_in_any_cell(self):
        out, cpu, symbols = run("""
            LDI.F  f0, odd
            LD.H   h0, [f0]
            LDI.H  h1, 2.5
            ST.H   [f0], h1
            LD.H   h2, [f0]
            HALT
        even:   DATA.H 1
        odd:    DATA.H 7
        """)
        self.assertEqual(symbols["odd"] % 2, 1.0)
        self.assertEqual((cpu.h[0], cpu.h[2]), (7.0, 2.5))

    def test_the_bits_of_a_double_are_not_the_number_you_stored(self):
        low_float = halves_and_floats_of(0.1)[0][1]
        out, cpu, _ = run("""
            LDI.F  f0, cell
            LDI.D  d0, 0.1
            ST.D   [f0], d0
            LDI.F  f1, 2
            ADD.F  f0, f0, f1
            LD.F   f1, [f0]        ; the trailing half of the double 0.1, read as a float
            HALT
        cell:   DATA.D 0
        """)
        self.assertEqual(cpu.f[1], low_float)
        self.assertLess(cpu.f[1], 0.0)


class Addresses(unittest.TestCase):
    def test_an_address_must_be_a_whole_cell_number_inside_memory(self):
        cases = [("nan", "NaN address"), ("-1", "negative address"), ("2.5", "not a whole number of cells"),
                 ("4096", "past the end of memory"), ("inf", "past the end of memory"),
                 ("1e30", "past the end of memory")]
        for literal, reason in cases:
            with self.subTest(literal=literal):
                out, cpu, _ = run(f"LDI.F f0, {literal}\nLD.H h0, [f0]\nHALT")
                self.assertEqual(cpu.status, FAULT)
                self.assertEqual(cpu.fault.kind, "address")
                self.assertIn(reason, cpu.fault.detail)

    def test_a_fault_leaves_the_registers_as_they_were(self):
        out, cpu, symbols = run("""
            LDI.H  h0, 5
            LDI.F  f0, -1
            bad:    LD.H  h0, [f0]        ; faults before it writes h0
            HALT
        """)
        self.assertEqual((cpu.h[0], cpu.pc), (5.0, symbols["bad"]))

    def test_a_branch_target_is_checked_before_the_jump(self):
        out, cpu, _ = run("LDI.F f0, 2.5\nJMP f0\nHALT")
        self.assertEqual(cpu.fault.detail, "2.5: not a whole number of cells")

    def test_jumping_past_the_program(self):
        out, cpu, _ = run("LDI.F f0, 4096\nJMP f0\nHALT")
        self.assertEqual((cpu.status, cpu.fault.detail), (FAULT, "4096: past the end of memory"))

    def test_a_half_register_is_not_an_address(self):
        # Addresses live in float registers; a half would only reach the first 2048 cells.
        with self.assertRaises(asm.AsmError) as caught:
            asm.assemble("LDI.H h0, 4\nLD.H h1, [h0]\nHALT")
        self.assertIn("float register", caught.exception.message)


class ControlFlow(unittest.TestCase):
    def branch(self, mnemonic, a, b):
        out, cpu, _ = run(f"""
            LDI.F  f0, {a}
            LDI.F  f1, {b}
            {mnemonic}.F f0, f1, yes
            LDI.H  h0, 0
            BR     put
            yes:    LDI.H  h0, 1
            put:    PUT.H  h0
            HALT
        """)
        return out

    def test_compare_and_branch(self):
        cases = [("BEQ", "1", "1", "1"), ("BEQ", "1", "2", "0"), ("BEQ", "nan", "nan", "0"),
                 ("BEQ", "-0", "0", "1"), ("BNE", "nan", "nan", "1"), ("BNE", "1", "1", "0"),
                 ("BLT", "1", "2", "1"), ("BLT", "2", "1", "0"), ("BLT", "1", "1", "0"),
                 ("BLT", "nan", "1", "0"), ("BLE", "1", "1", "1"), ("BLE", "nan", "1", "0")]
        for mnemonic, a, b, expected in cases:
            with self.subTest(mnemonic=mnemonic, a=a, b=b):
                self.assertEqual(self.branch(mnemonic, a, b), f"{expected}.0\n")

    def test_branch_on_zero_and_on_not_zero(self):
        cases = [("BZ", "0", "1"), ("BZ", "-0", "1"), ("BZ", "nan", "0"), ("BZ", "1e-300", "0"),
                 ("BNZ", "0", "0"), ("BNZ", "nan", "1"), ("BNZ", "1e-300", "1")]
        for mnemonic, value, expected in cases:
            with self.subTest(mnemonic=mnemonic, value=value):
                out, cpu, _ = run(f"""
                    LDI.D  d0, {value}
                    {mnemonic}.D d0, yes
                    LDI.H  h0, 0
                    BR     put
                    yes:    LDI.H  h0, 1
                    put:    PUT.H  h0
                    HALT
                """)
                self.assertEqual(out, f"{expected}.0\n")

    def test_a_branch_skips_its_own_operand(self):
        out, cpu, _ = run("""
            LDI.H  h0, 1
            BZ.H   h0, elsewhere    ; not taken: the operand is stepped over, not executed
            LDI.H  h1, 4
            PUT.H  h1
            HALT
            elsewhere:  LDI.H h1, 9
            PUT.H  h1
            HALT
        """)
        self.assertEqual(out, "4.0\n")

    def test_call_and_return(self):
        out, cpu, _ = run("""
            LDI.H  h0, 1
            PUT.H  h0
            JAL    f4, twice
            PUT.H  h0
            JAL    f4, thrice
            PUT.H  h0
            HALT
            twice:  LDI.H h1, 2
            ADD.H  h0, h0, h1
            JMP    f4
            thrice: LDI.H h1, 3
            sub:    ADD.H  h0, h0, h1
            JMP    f4
        """)
        self.assertEqual((cpu.status, out), (HALTED, "1.0\n3.0\n6.0\n"))

    def test_the_link_register_holds_the_address_after_the_call(self):
        out, cpu, symbols = run("JAL f4, sub\nback: HALT\nsub: JMP f4")
        self.assertEqual((cpu.f[4], cpu.status), (symbols["back"], HALTED))

    def test_halt_stops_on_the_halt_instruction(self):
        out, cpu, symbols = run("LDI.H h0, 1\nstop: HALT\nLDI.H h0, 2")
        self.assertEqual((cpu.status, cpu.pc, cpu.h[0]), (HALTED, symbols["stop"], 1.0))

    def test_nop(self):
        out, cpu, _ = run("NOP\nNOP\nBR next\nNOP\nnext: HALT")
        self.assertEqual(cpu.status, HALTED)


class Arithmetic(unittest.TestCase):
    def test_fma_rounds_once(self):
        out, cpu, _ = run("""
            LDI.F  f1, 1.000030517578125       ; 1 + 2**-15
            LDI.F  f2, -1.00006103515625       ; -(1 + 2**-14)
            MUL.F  f3, f1, f1                  ; 1 + 2**-14 + 2**-30, rounded to 1 + 2**-14
            ADD.F  f4, f3, f2                  ; separately: nothing is left
            MOV.F  f3, f2
            FMA.F  f3, f1, f1                  ; fused: the low bit of the product survives
            HALT
        """)
        self.assertEqual(cpu.f[4], 0.0)
        self.assertEqual(cpu.f[3], 2.0 ** -30)

    def test_flags_are_sticky(self):
        out, cpu, _ = run("""
            LDI.F  f1, 1
            LDI.F  f2, 0
            DIV.F  f0, f1, f2        ; 1 / 0: infinity, and division by zero is flagged
            ADD.F  f2, f1, f1        ; exact afterwards: the flag stays
            SQRT.H  h0, h0
            HALT
        """)
        self.assertEqual(cpu.f[0], float("inf"))
        self.assertEqual(cpu.flags, DIVZERO)

    def test_conversions(self):
        x = 1 + 2.0 ** -11 + 2.0 ** -40  # just above the midpoint between two halfs
        out, cpu, _ = run(f"""
            LDI.H  h0, 0.1
            CVT.F  f0, h0           ; widening: nothing lost
            LDI.F  f1, 0.1
            CVT.H  h1, f1           ; narrowing: rounded
            LDI.D  d0, {x!r}
            CVT.H  h2, d0           ; narrowed straight from the double
            CVT.F  f2, d0
            CVT.H  h3, f2           ; narrowed by way of a float: one rounding too many
            CVT.D  d1, h0
            HALT
        """)
        self.assertEqual(cpu.f[0], 0.0999755859375)
        self.assertEqual(cpu.h[1], 0.0999755859375)
        self.assertEqual(cpu.h[2], 1 + 2.0 ** -10)
        self.assertEqual(cpu.h[3], 1.0)
        self.assertEqual(cpu.d[1], 0.0999755859375)

    def test_exponent_surgery(self):
        out, cpu, _ = run("""
            LDI.F  f1, 3
            LDI.F  f2, 48
            LOGB.F f3, f2
            SCALB.F f4, f1, f3     ; 3 * 2**5
            LDI.F  f5, 1
            LDI.F  f6, inf
            NEXT.F f7, f5, f6      ; the next float after 1
            CLASS.F f8, f6
            CLASS.F f9, f1
            HALT
        """)
        self.assertEqual((cpu.f[3], cpu.f[4], cpu.f[7], cpu.f[8], cpu.f[9]),
                         (5.0, 96.0, 1.0000001192092896, 9.0, 8.0))

    def test_the_rest_of_the_typed_instructions(self):
        out, cpu, _ = run("""
            LDI.H  h0, 0.1           ; the half nearest 0.1
            LDI.H  h1, 0.2           ; exactly twice h0
            LDI.H  h14, -0.5
            LDI.H  h15, 2.5
            EQ.H   h2, h0, h1
            LT.H   h3, h0, h1
            REM.H  h4, h1, h0        ; an exact quotient of 2: remainder 0
            MIN.H  h5, h0, h1
            MAX.H  h6, h0, h1
            FLOOR.H h7, h0
            CEIL.H  h8, h0
            CHOP.H h9, h14
            ROUND.H h10, h15
            CPSGN.H h11, h0, h14
            ABS.H  h12, h14
            NEG.H  h13, h0
            HALT
        """)
        self.assertEqual(cpu.h[2:14], (
            0.0, 1.0, 0.0, cpu.h[0], cpu.h[1], 0.0, 1.0, -0.0, 2.0,
            -cpu.h[0], 0.5, -cpu.h[0]))


class Output(unittest.TestCase):
    def test_put_prints_the_value_as_it_is_stored(self):
        out, cpu, _ = run("""
            LDI.H  h0, 0.1
            LDI.F  f0, 0.1
            LDI.D  d0, 0.1
            PUT.H  h0
            PUT.F  f0
            PUT.D  d0
            LDI.H  h1, -0
            PUT.H  h1
            LDI.D  d1, inf
            PUT.D  d1
            NEG.D  d1, d1
            PUT.D  d1
            LDI.H  h2, nan
            PUT.H  h2
            HALT
        """)
        self.assertEqual(out, "0.0999755859375\n0.10000000149011612\n0.1\n"
                              "-0.0\ninf\n-inf\nnan\n")

    def test_putc_prints_one_code_point_per_value(self):
        out, cpu, _ = run("""
            LDI.H  h0, 72
            PUTC.H h0
            LDI.F  f0, 105
            PUTC.F f0
            LDI.F  f0, 10084
            PUTC.F f0            ; a heavy black heart, U+2764: more than a half can hold
            LDI.H  h0, 33
            PUTC.H h0
            LDI.H  h0, 10
            PUTC.H h0
            HALT
        """)
        self.assertEqual(out, "Hi❤!\n")

    def test_putc_refuses_what_is_not_a_code_point(self):
        for value in ("2.5", "-1", "1114112", "55296", "nan", "inf"):
            with self.subTest(value=value):
                out, cpu, _ = run(f"LDI.D d0, {value}\nPUTC.D d0\nHALT")
                self.assertEqual(cpu.status, FAULT)
                self.assertEqual(cpu.fault.kind, "not-a-character")

    def test_a_string_is_half_code_points_terminated_by_zero(self):
        out, result, _, _ = support.run_source("""
            LDI.F  f0, msg
            LDI.H  h1, 1
            CVT.F  f1, h1
            loop:   LD.H   h0, [f0]
            BZ.H   h0, done
            PUTC.H h0
            ADD.F  f0, f0, f1
            BR     loop
            done:  HALT
        msg:    TEXT "Hi\\tthere!\\n\\x41\\"quote\\"\\\\"
                DATA.H 0
        """)
        self.assertEqual(result.reason, "halt")
        self.assertEqual(out, "Hi\tthere!\nA\"quote\"\\")


if __name__ == "__main__":
    unittest.main()
