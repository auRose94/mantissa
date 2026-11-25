"""Instruction numbers."""
import unittest

from mantissa import fp, isa


def decoded(number):
    """What the machine makes of a number used as an instruction."""
    return isa.decode(float(number))


class Encoding(unittest.TestCase):
    def test_an_instruction_is_an_eight_digit_number(self):
        self.assertEqual(isa.encode(isa.OP["ADD"], 0, 1, 2), 20000102)
        self.assertEqual(decoded(20000102), (20, 0, 1, 2))
        self.assertEqual(decoded(20000102.0 + 0.5), None)

    def test_register_codes(self):
        self.assertEqual([isa.code_of(file_number, index)
                          for file_number, index in ((1, 0), (2, 5), (3, 31))], [0, 37, 95])
        self.assertEqual([isa.name_of(code) for code in (0, 31, 32, 37, 64, 95)],
                         ["h0", "h31", "f0", "f5", "d0", "d31"])
        self.assertEqual([isa.file_of(code) for code in (0, 31, 32, 63, 64, 95)], [1, 1, 2, 2, 3, 3])

    def test_every_opcode_decodes_back(self):
        for op in isa.OPS:
            for fill in (0, 1, 31, 32, 64, 95):
                fields = [fill, (fill + 1) % 96, (fill + 2) % 96]
                a, b, c = (fields["abc".index(place)] if place in op.used else 0 for place in "abc")
                with self.subTest(op=op.name, fill=fill):
                    self.assertEqual(decoded(isa.encode(op.number, a, b, c)), (op.number, a, b, c))

    def test_opcode_numbers_are_two_distinct_digits(self):
        numbers = [op.number for op in isa.OPS]
        self.assertEqual(len(numbers), len(set(numbers)))
        self.assertTrue(all(0 <= number < 100 for number in numbers))
        self.assertNotIn(0, numbers, "opcode 00 is kept back so that data does not run")

    def test_numbers_that_are_not_instructions(self):
        strangers = [
            0, 5, 999999,                 # opcode digits 00: data, or nothing
            86000000, 99000000,           # no such opcode
            100000000, -20000102,         # not eight digits
            float("nan"), float("inf"),
            isa.encode(10, 0, 1, 2),      # MOV reads two fields, so the third must be 00
            isa.encode(70, 0, 0, 1),
        ]
        for number in strangers:
            with self.subTest(number=number):
                self.assertIsNone(decoded(number))

    def test_the_guard_band_is_the_quad_file(self):
        # codes 96 to 99 were spare while the operand fields were two digits
        # wide; they are q0 to q3 now, and two digits cannot name anything
        # past them -- the wall that blocked a bigger file enforces this one
        self.assertEqual(decoded(isa.encode(20, 96, 99, 0)), (20, 96, 99, 0))
        self.assertEqual(isa.name_of(96), "q0")
        self.assertEqual(isa.name_of(99), "q3")
        self.assertEqual(isa.file_of(97), 4)

    def test_arithmetic_moves_the_operand_digits(self):
        add = isa.encode(20, 0, 1, 2)  # ADD h0, h1, h2
        self.assertEqual(decoded(add + 10000), (20, 1, 1, 2))    # the first operand, one register on
        self.assertEqual(decoded(add + 320000), (20, 32, 1, 2))  # ... from the half registers to the floats
        self.assertEqual(decoded(add + 640000), (20, 64, 1, 2))  # ... or to the doubles
        self.assertEqual(decoded(add + 100), (20, 0, 2, 2))      # the second operand
        self.assertEqual(decoded(add + 1), (20, 0, 1, 3))        # the third
        self.assertEqual(decoded(add + 5000000), (25, 0, 1, 2))  # ADD became FMA

    def test_chop_and_mod_take_an_instruction_apart(self):
        number = 70524321.0  # PUTC d20 ... read as opcode, then three operand fields
        opcode = fp.trunc(fp.D, number / 1000000.0)[0]
        high = fp.trunc(fp.D, number / 10000.0)[0]
        first = fp.fmod(fp.D, high, 100.0)[0]
        middle = fp.trunc(fp.D, number / 100.0)[0]
        second = fp.fmod(fp.D, middle, 100.0)[0]
        third = fp.fmod(fp.D, number, 100.0)[0]
        self.assertEqual((opcode, first, second, third), (70.0, 52.0, 43.0, 21.0))

    def test_sizes_in_cells(self):
        for name, cells in (("HALT", 4), ("ADD", 4), ("LDI", 8), ("BR", 8), ("BZ", 8), ("JAL", 8),
                            ("BSGN", 12), ("SPLIT", 4), ("BEQ", 8)):
            self.assertEqual(isa.BY_NAME[name].cells, cells, name)


if __name__ == "__main__":
    unittest.main()
