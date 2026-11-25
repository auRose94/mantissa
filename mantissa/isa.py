"""Instruction encoding: an instruction is a whole number.

An instruction is a double whose value is an eight-digit decimal number:

    oo aa bb cc

Read from the left: a two-digit opcode, then three two-digit operand fields.
An operand field names a register:

    00 to 31  h0 to h31      32 to 63  f0 to f31      64 to 95  d0 to d31

96 to 99 are q0 to q3, the quad file: binary128, eight cells a register.
The guard band the two-digit fields left became a file exactly four wide,
and a field an instruction does not use must still be 00.

Because an instruction is just a number, a program can read one, do
arithmetic on it and write it back.  Doubles hold these numbers exactly, so
adding 10000 to an instruction moves its first operand one register along,
and adding 320000 moves that operand from the half registers to the floats.
Any number that does not decode is an illegal instruction.
"""
from __future__ import annotations

LIMIT = 10 ** 8  # an instruction is a whole number below this

# Instructions that carry inline operands, and how many doubles follow them.
WORDS = {"LDI": 1, "BR": 1, "BSGN": 2, "BEQ": 1, "BNE": 1, "BLT": 1, "BLE": 1,
         "BZ": 1, "BNZ": 1, "JAL": 1}


class Op:
    """An opcode: its mnemonic, the operand fields it reads, whether those
    operands must all be registers of one type, and its size in cells."""

    __slots__ = ("name", "number", "used", "same_type", "cells")

    def __init__(self, name, number, used, same_type):
        self.name = name
        self.number = number
        self.used = used
        self.same_type = same_type
        self.cells = 4 * (1 + WORDS.get(name, 0))

    def __repr__(self):
        return f"Op({self.name}, {self.number})"


_TABLE = [
    ("NOP", 1, "", False), ("HALT", 2, "", False),
    ("MOV", 10, "ab", True), ("LDI", 11, "a", False), ("LD", 12, "ab", False),
    ("ST", 13, "ab", False), ("CVT", 14, "ab", False), ("SPLIT", 15, "abc", False),
    ("JOIN", 16, "abc", False),
    ("ADD", 20, "abc", True), ("SUB", 21, "abc", True), ("MUL", 22, "abc", True),
    ("DIV", 23, "abc", True), ("SQRT", 24, "ab", True), ("FMA", 25, "abc", True),
    ("REM", 26, "abc", True), ("MOD", 27, "abc", True), ("MIN", 28, "abc", True),
    ("MAX", 29, "abc", True),
    ("NEG", 30, "ab", True), ("ABS", 31, "ab", True), ("CPSGN", 32, "abc", True),
    ("FLOOR", 33, "ab", True), ("CEIL", 34, "ab", True), ("CHOP", 35, "ab", True),
    ("ROUND", 36, "ab", True),
    ("FLOOR_TO", 37, "abc", True), ("CEIL_TO", 38, "abc", True),
    ("CHOP_TO", 39, "abc", True), ("ROUND_TO", 40, "abc", True),
    ("SCALB", 41, "abc", True), ("LOGB", 42, "ab", True), ("NEXT", 43, "abc", True),
    ("CLASS", 44, "ab", True), ("MANT", 45, "abc", False),
    ("LT", 50, "abc", True), ("LE", 51, "abc", True), ("EQ", 52, "abc", True),
    ("CMP", 53, "abc", True),
    ("BR", 60, "", False), ("BSGN", 61, "a", False), ("BEQ", 62, "ab", True),
    ("BNE", 63, "ab", True), ("BLT", 64, "ab", True), ("BLE", 65, "ab", True),
    ("BZ", 66, "a", False), ("BNZ", 67, "a", False), ("JAL", 68, "a", False),
    ("JMP", 69, "a", False),
    ("PUT", 70, "a", False), ("PUTC", 71, "a", False),
    ("VADD2", 72, "abc", True), ("VSUB2", 73, "abc", True), ("VMUL2", 74, "abc", True),
    ("VDIV2", 75, "abc", True), ("VFMA2", 76, "abc", True), ("VMIN2", 77, "abc", True),
    ("VMAX2", 78, "abc", True),
    ("VADD4", 79, "abc", True), ("VSUB4", 80, "abc", True), ("VMUL4", 81, "abc", True),
    ("VDIV4", 82, "abc", True), ("VFMA4", 83, "abc", True), ("VMIN4", 84, "abc", True),
    ("VMAX4", 85, "abc", True),
]

OPS = [Op(*row) for row in _TABLE]
BY_NUMBER = {op.number: op for op in OPS}
BY_NAME = {op.name: op for op in OPS}
OP = {op.name: op.number for op in OPS}
# The same mnemonic with a step register is a second opcode.
WITH_STEP = {"FLOOR": "FLOOR_TO", "CEIL": "CEIL_TO", "CHOP": "CHOP_TO", "ROUND": "ROUND_TO"}

H_BASE, F_BASE, D_BASE = 0, 32, 64  # register codes are (file - 1) * 32 + index

# The vector instructions work on 2 or 4 consecutive registers of one file.
VECTOR_LANES = {"VADD2": 2, "VSUB2": 2, "VMUL2": 2, "VDIV2": 2, "VFMA2": 2,
                "VMIN2": 2, "VMAX2": 2,
                "VADD4": 4, "VSUB4": 4, "VMUL4": 4, "VDIV4": 4, "VFMA4": 4,
                "VMIN4": 4, "VMAX4": 4}


def group_fits(name, code):
    """Whether a vector group of `lanes` registers starting at `code` stays
    inside registers 0-31 of its file."""
    return (code % 32) + VECTOR_LANES[name] <= 32


def code_of(file_number, index):
    """The operand field for register `index` of file `file_number` (1 half,
    2 float, 3 double, 4 quad -- four registers only)."""
    return (file_number - 1) * 32 + index


def file_of(code):
    return code // 32 + 1


def name_of(code):
    if code >= 96:
        return f"q{code - 96}"        # the guard band became the quad file
    return "hfd"[code // 32] + str(code % 32)


def encode(number, a=0, b=0, c=0):
    """The instruction number with the given opcode number and operand fields."""
    return number * 10 ** 6 + a * 10 ** 4 + b * 100 + c


def decode(value):
    """(op, a, b, c) for the instruction number `value`, or None."""
    if not (0 <= value < LIMIT) or not value.is_integer():
        return None
    n = int(value)
    op = n // 10 ** 6
    a, b, c = n // 10 ** 4 % 100, n // 100 % 100, n % 100
    spec = BY_NUMBER.get(op)
    if spec is None:
        return None
    for position, code in zip("abc", (a, b, c)):
        if position in spec.used:
            if code >= D_BASE + 36:
                return None     # 96 to 99 name q0 to q3; two digits reach no further
        elif code:
            return None  # an unused field must be 00
    return op, a, b, c
