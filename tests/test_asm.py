"""The assembler and the disassembler."""
import math
import unittest

from mantissa import asm, fp, isa
from mantissa.disasm import disassemble
from mantissa.memory import Memory


def number(image, address=0):
    """The instruction number at a four-cell boundary of the image."""
    return int(fp.unpack(fp.D, image.cells[address:address + 4]))


def word_at(image, address):
    """The double stored at a four-cell boundary, inline operands included."""
    return fp.unpack(fp.D, image.cells[address:address + 4])


def float_of(image, address):
    return fp.unpack(fp.F, image.cells[address:address + 2])


def double_of(image, address):
    return fp.unpack(fp.D, image.cells[address:address + 4])


def halfs_of(image, address, count):
    return [fp.unpack(fp.H, image.cells[a:a + 1]) for a in range(address, address + count)]


def text_of(image, address, count):
    return "".join(chr(int(value)) for value in halfs_of(image, address, count))


class Encodings(unittest.TestCase):
    def test_hand_computed_numbers(self):
        for text, expected in (("HALT", 2000000), ("ADD h1, h2, h3", 20010203), ("JMP f5", 69370000),
                               ("MOV f0, f1", 10323300), ("PUT h0", 70000000), ("SPLIT f1, f2, d0", 15333464)):
            with self.subTest(text=text):
                self.assertEqual(number(asm.assemble(text)), expected)

    def test_every_form_puts_its_registers_in_the_digit_fields(self):
        o = isa.OP
        cases = [
            ("ADD.H h1, h2, h3", o["ADD"], 1, 2, 3), ("FMA.D d4, d5, d6", o["FMA"], 68, 69, 70),
            ("CPSGN.F f1, f2, f3", o["CPSGN"], 33, 34, 35), ("MOD h1, h2, h3", o["MOD"], 1, 2, 3),
            ("SQRT.F f7, f8", o["SQRT"], 39, 40, 0), ("CLASS.D d9, d10", o["CLASS"], 73, 74, 0),
            ("MOV.F f0, f1", o["MOV"], 32, 33, 0), ("PUT.H h9", o["PUT"], 9, 0, 0), ("PUTC.F f3", o["PUTC"], 35, 0, 0),
            ("LD.H h1, [f2]", o["LD"], 1, 34, 0), ("ST.D [f4], d5", o["ST"], 36, 69, 0),
            ("CVT.H h1, d2", o["CVT"], 1, 66, 0), ("CVT.F f1, h2", o["CVT"], 33, 2, 0), ("CVT.D d1, f2", o["CVT"], 65, 34, 0),
            ("SPLIT f1, f2, d2", o["SPLIT"], 33, 34, 66), ("JOIN d1, f2, f3", o["JOIN"], 65, 34, 35),
            ("MANT h1, f2, d3", o["MANT"], 1, 34, 67), ("MANT f1, f2, h3", o["MANT"], 33, 34, 3),
            ("CHOP h1, h2", o["CHOP"], 1, 2, 0), ("CHOP h1, h2, h3", o["CHOP_TO"], 1, 2, 3),
            ("ROUND d1, d2", o["ROUND"], 65, 66, 0), ("ROUND d1, d2, d3", o["ROUND_TO"], 65, 66, 67),
            ("FLOOR h1, h2, h3", o["FLOOR_TO"], 1, 2, 3), ("CEIL h1, h2, h3", o["CEIL_TO"], 1, 2, 3),
            ("CMP f1, f2, f3", o["CMP"], 33, 34, 35), ("LT h1, h2, h3", o["LT"], 1, 2, 3),
            ("BZ.F f1, 0", o["BZ"], 33, 0, 0), ("BNZ.D d2, 0", o["BNZ"], 66, 0, 0),
            ("HALT", o["HALT"], 0, 0, 0), ("NOP", o["NOP"], 0, 0, 0), ("JMP f5", o["JMP"], 37, 0, 0),
            ("JAL f6, 0", o["JAL"], 38, 0, 0), ("BR 0", o["BR"], 0, 0, 0),
            ("BEQ.H h1, h2, 0", o["BEQ"], 1, 2, 0), ("BNE.F f1, f2, 0", o["BNE"], 33, 34, 0),
            ("BLT.D d1, d2, 0", o["BLT"], 65, 66, 0), ("BLE.H h3, h4, 0", o["BLE"], 3, 4, 0),
            ("BSGN f1, 0, 0", o["BSGN"], 33, 0, 0),
        ]
        for text, op, a, b, c in cases:
            with self.subTest(text=text):
                self.assertEqual(number(asm.assemble(text)), isa.encode(op, a, b, c))

    def test_inline_operands_are_doubles_that_follow_the_instruction(self):
        image = asm.assemble("LDI h0, 3.5\nHALT")
        self.assertEqual(number(image), 11000000)
        self.assertEqual(word_at(image, 4), 3.5)
        self.assertEqual(number(image, 8), 2000000)  # the next instruction is eight cells along
        image = asm.assemble("LDI.F f0, 0.1")
        self.assertEqual(word_at(image, 4), 0.10000000149011612)  # the literal, rounded to a float
        image = asm.assemble("BR 12\nHALT")
        self.assertEqual((word_at(image, 4), number(image, 8), len(image.cells)), (12.0, 2000000, 12))
        image = asm.assemble("BSGN f0, 16, 24\nNOP\nHALT")
        self.assertEqual((word_at(image, 4), word_at(image, 8), number(image, 12), number(image, 16)),
                         (16.0, 24.0, 1000000, 2000000))

    def test_the_suffix_is_optional_but_must_match(self):
        plain = asm.assemble("add h0, h1, h2\nput f3\ncmp d1, d2, d3\nhalt")
        loud = asm.assemble("ADD.H h0, h1, h2\nPUT.F f3\nCMP.D d1, d2, d3\nHALT")
        self.assertEqual(list(plain.cells), list(loud.cells))

    def test_a_suffix_names_the_value_not_the_address(self):
        load = asm.assemble("LD.H h0, [f5]\nST.D [f5], d0\nCVT.F f0, h1\nJAL.F f4, 0")
        self.assertEqual([number(load, at) for at in (0, 4, 8, 12)],
                         [isa.encode(isa.OP["LD"], 0, 37, 0), isa.encode(isa.OP["ST"], 37, 64, 0),
                          isa.encode(isa.OP["CVT"], 32, 1, 0), isa.encode(isa.OP["JAL"], 36, 0, 0)])

    def test_case_and_sp_and_lr(self):
        loud = asm.assemble("LDI.H H0, 1\nMOV.F SP, LR\nJAL LR, 4\nHALT")
        quiet = asm.assemble("ldi.h h0, 1\nmov f31, f30\njal f30, 4\nhalt")
        self.assertEqual(list(loud.cells), list(quiet.cells))

    def test_whitespace_is_free(self):
        image = asm.assemble("  add.h\th0 ,h1,   h2  ; a comment\n\n\t; another one\n")
        self.assertEqual(number(image), isa.encode(isa.OP["ADD"], 0, 1, 2))


class Layout(unittest.TestCase):
    def test_labels_resolve_to_cell_addresses_forward_and_back(self):
        image = asm.assemble("""
            BR   start
            LDI.H h0, 1
        start:  LDI.H h0, 2
            BR   start
        end:    HALT
        """)
        self.assertEqual((image.symbols["start"], image.symbols["end"]), (16.0, 32.0))
        self.assertEqual((word_at(image, 4), word_at(image, 28)), (16.0, 16.0))
        self.assertEqual(image.entry, 0.0)

    def test_a_label_lands_on_the_thing_not_on_its_padding(self):
        image = asm.assemble("""
            LDI.H h0, 1          ; cells 0 to 7
        text:   TEXT "abc"       ; cells 8, 9, 10
        number: DATA.D 1.0       ; a double needs a multiple of four: cell 11 is padding
        half:   DATA.H 7
        room:   SPACE 5
        after:  HALT             ; an instruction needs a multiple of four: cells 22 and 23 pad
        """)
        self.assertEqual([image.symbols[name] for name in ("text", "number", "half", "room", "after")],
                         [8.0, 12.0, 16.0, 17.0, 24.0])
        self.assertEqual(list(image.cells[11:12]), [0])
        self.assertEqual(double_of(image, 12), 1.0)
        self.assertEqual(halfs_of(image, 16, 1), [7.0])
        self.assertEqual(list(image.cells[17:24]), [0] * 7)

    def test_entry_point(self):
        self.assertEqual(asm.assemble("NOP\nHALT").entry, 0.0)
        self.assertEqual(asm.assemble("DATA.H 1\nmain: HALT").entry, 4.0)
        self.assertEqual(asm.assemble("NOP\nHALT", org=100).entry, 100.0)
        self.assertEqual(asm.assemble("NOP\nHALT", org=101).entry, 104.0)

    def test_directives_place_typed_values(self):
        image = asm.assemble('TEXT "Hi!"\nDATA.H 1, 2\nDATA.F 2.5\nDATA.D -0.5\n')
        self.assertEqual(text_of(image, 0, 3), "Hi!")
        self.assertEqual(halfs_of(image, 3, 2), [1.0, 2.0])  # halfs need no alignment: cells 3 and 4
        self.assertEqual(list(image.cells[5:6]), [0])        # the float at 6 needs one pad cell
        self.assertEqual(float_of(image, 6), 2.5)
        self.assertEqual(double_of(image, 8), -0.5)

    def test_string_escapes_and_a_semicolon_inside_a_string(self):
        quote = '"'
        image = asm.assemble("TEXT " + quote + r'a;b\t\"q\"\x41\\' + quote)
        self.assertEqual(text_of(image, 0, 9), 'a;b\t"q"A\\')

    def test_expressions(self):
        image = asm.assemble("""
            LDI.F f0, buf + 4 - 1
            LDI.F f1, -buf
            LDI.F f2, - 0
            LDI.F f3, +2.5e1
            LDI.D d0, inf
            LDI.D d1, -inf
            LDI.D d2, nan
            HALT
        buf:    DATA.H 10
        """)
        buf = image.symbols["buf"]
        self.assertEqual([word_at(image, a) for a in (4, 12, 20, 28)], [buf + 3, -buf, -0.0, 25.0])
        self.assertEqual(math.copysign(1.0, word_at(image, 20)), -1.0)
        self.assertEqual((word_at(image, 36), word_at(image, 44)), (math.inf, -math.inf))
        self.assertNotEqual(word_at(image, 52), word_at(image, 52))

    def test_literals_are_rounded_to_their_register_like_in_c(self):
        image = asm.assemble("LDI.H h0, 0.1\nLDI.H h1, 2049\nLDI.H h2, 1e10\n"
                             "LDI.F f0, 0.1\nLDI.D d0, 0.1\nHALT")
        # the operand word is always a double, holding the literal already rounded to its register
        self.assertEqual([word_at(image, a) for a in (4, 12, 20, 28, 36)],
                         [0.0999755859375, 2048.0, math.inf, 0.10000000149011612, 0.1])


ROUND_TRIP = """
        MOV.D  d3, d4
        ADD.H  h1, h2, h3
        FMA.F  f1, f2, f3
        SQRT.D d0, d1
        CPSGN.H h0, h1, h2
        CHOP.F f0, f1, f2
        ROUND.H h0, h1, h2
        LDI.H  h0, 3.5
        LDI.F  f0, 0.1
        LD.H   h0, [f1]
        ST.D   [f2], d0
        CVT.F  f0, h1
        SPLIT  f0, f1, d2
        JOIN   d0, f1, f2
        MANT   h0, f1, d2
        MOD.H  h0, h1, h2
        CMP.F  f0, f1, f2
        PUT.D  d0
        PUTC.H h0
        BEQ.F  f0, f1, end
        BNE.H  h0, h1, end
        BLT.D  d0, d1, end
        BLE.F  f0, f1, end
        BZ.H   h0, end
        BNZ.F  f0, end
        BSGN.D d0, start, end
        BR     start
        JAL    f5, end
        JMP    f5
        NOP
start:  HALT
end:    HALT
"""


class Disassembly(unittest.TestCase):
    def test_disassembly_assembles_back_to_the_same_cells(self):
        image = asm.assemble(ROUND_TRIP)
        memory = Memory(len(image.cells) + 16)
        memory.write(image.cells)
        labels = {name: address for name, address in image.symbols.items() if name not in asm.PREDEFINED}
        lines, addr = [], 0
        while addr < len(image.cells):
            for name, address in labels.items():
                if address == addr:
                    lines.append(f"{name}:")
            text, width = disassemble(memory, addr, labels)
            lines.append("    " + text)
            addr += width
        again = asm.assemble("\n".join(lines))
        self.assertEqual(list(again.cells), list(image.cells))
        self.assertEqual(again.symbols, image.symbols)
        for expected in ("    ADD h1, h2, h3", "    CHOP f0, f1, f2", "    ROUND h0, h1, h2",
                         "    LDI h0, 3.5", "    LD h0, [f1]", "    ST [f2], d0", "    CVT f0, h1",
                         "    SPLIT f0, f1, d2", "    JOIN d0, f1, f2", "    MANT h0, f1, d2",
                         "    BEQ f0, f1, end", "    BZ h0, end", "    BSGN d0, start, end",
                         "    BR start", "    JAL f5, end", "    JMP f5", "    HALT"):
            self.assertIn(expected, lines)

    def test_what_is_not_an_instruction_is_shown_as_a_number(self):
        image = asm.assemble("LDI.F f0, 1\nLDI.F f1, 2\nHALT")
        memory = Memory(32)
        memory.write(image.cells)
        self.assertEqual(disassemble(memory, 4), (".double 1.0", 4))  # the operand of the first LDI
        self.assertEqual(disassemble(memory, 16), ("HALT", 4))


class Errors(unittest.TestCase):
    def errors(self, source, fragment):
        with self.assertRaises(asm.AsmError) as caught:
            asm.assemble(source)
        self.assertIn(fragment, caught.exception.message)
        return caught.exception.lineno

    def test_messages(self):
        cases = [
            ("ADD.H h0, f1, h0", "ADD.H is written for halfs here, but f1 holds a float"),
            ("ADD.F f0, h1, f0", "written for floats here, but h1 holds a half"),
            ("ADD.D d0, d1, f2", "written for doubles here, but f2 holds a float"),
            ("ADD f0, h1, f0", "ADD needs operands of one type"),
            ("ADD f0, f1", "ADD takes 3 operands, not 2"),
            ("ADD f0, f1, f2, f3", "ADD takes 3 operands, not 4"),
            ("LDI.H h32, 1", "not a register"),
            ("LDI.H x0, 1", "not a register"),
            ("LD.H h0, f1", "memory operand"),
            ("ST.H h0, h1", "memory operand"),
            ("ST.H [f0], f1", "written for halfs here, but f1 holds a float"),
            ("JMP h0", "float register"),
            ("JAL h0, 4", "float register"),
            ("CVT.F f0, f1", "CVT changes type"),
            ("CVT.F f0, 3", "not a register"),
            ("SPLIT f0, h1, d2", "SPLIT needs two registers of one type"),
            ("JOIN d0, f1, h2", "JOIN needs two registers of one type"),
            ("BR 2.5", "is not an instruction address"),
            ("BR -4", "is not an instruction address"),
            ("BR 6", "four-cell boundary"),
            ("BR nowhere", "'nowhere' is not defined"),
            ("BSGN f0, 4", "BSGN takes 3 operands, not 2"),
            ("LDI.F f0, 1 +", "cannot read"),
            ("LDI.F f0, 1 2", "missing operator"),
            ("LDI.F f0, *2", "cannot read"),
            ("LDI.F f0,", "an operand is missing"),
            ("FROB.F f0, f1, f2", "unknown instruction or directive"),
            ("ADD.X f0, f1, f2", "there is no .X"),
            ("HALT.H", "takes no suffix"),
            ("BR.F 4", "takes no suffix"),
            ("h0: HALT", "cannot be a label"),
            ("sp: HALT", "cannot be a label"),
            ("nan: HALT", "cannot be a label"),
            ("x: HALT\nx: NOP", "is defined twice"),
            ("TEXT 'abc'", "TEXT needs one double-quoted string"),
            ('TEXT "abc', "TEXT needs one double-quoted string"),
            (r'TEXT "\q"', "unknown escape"),
            (r'TEXT "\u2764"', "does not fit in a half"),
            ("SPACE -1", "whole number of cells"),
            ("SPACE 2.5", "whole number of cells"),
            ("SPACE later\nlater: HALT", "'later' is not defined"),
            ("DATA 1", "write DATA.H, DATA.F or DATA.D"),
            ("  ???", "cannot read"),
            ("LDI.F f0, 3 ; fine so far\nLDI.F f0, nowhere\nHALT", "'nowhere' is not defined"),
        ]
        for source, fragment in cases:
            with self.subTest(source=source):
                self.errors(source, fragment)

    def test_the_line_number_is_reported(self):
        self.assertEqual(self.errors("NOP\n\nADD.H h0, f1, h0\nHALT", "holds a float"), 3)
        self.assertEqual(self.errors("NOP\nBR missing", "'missing' is not defined"), 2)
        self.assertEqual(self.errors("NOP\nDATA 1", "write DATA"), 2)
        self.assertEqual(self.errors("LDI.F f0, 3\nLDI.F f0, nowhere\nHALT", "'nowhere' is not defined"), 2)


if __name__ == "__main__":
    unittest.main()
