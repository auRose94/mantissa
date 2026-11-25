"""The golden traces: recorded runs that the ports replay.

They are only worth anything while they agree with the reference, and only
useful while they exercise every instruction.
"""
import pathlib
import struct
import unittest

from mantissa import asm, fp, isa, trace
from mantissa.machine import Cpu
from mantissa.memory import Memory

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "golden"
PROGRAMS = {"stall_limit": ("stall", 36)}  # the rest run programs/<name>.mant to the end


def recorded(name):
    """(header, cells, run lines) from a golden trace."""
    lines = (GOLDEN / f"{name}.trace").read_text(encoding="utf-8").splitlines()
    header = dict(line.split(" ", 1) for line in lines[:6])
    count = int(header["image"])
    cells = [int(cell, 16) for line in lines[6:6 + (count + 15) // 16] for cell in line.split()]
    return header, cells, lines[lines.index("run") + 1:]


def fresh(name):
    """The same, recorded now from the reference."""
    program, steps = PROGRAMS.get(name, (name, 1_000_000))
    source = (ROOT / "programs" / f"{program}.mant").read_text(encoding="utf-8")
    image = asm.assemble(source)
    memory = Memory(65536)
    floats = [0.0] * 32
    floats[31] = 65536.0
    memory.write(image.cells)
    lines = trace.lines(image, memory, Cpu(f=tuple(floats), pc=image.entry), steps)
    count = len(image.cells)
    return dict(line.split(" ", 1) for line in lines[:6]), list(image.cells), lines[lines.index("run") + 1:]


def executed(record):
    """The opcodes a trace executes, self-written instructions included."""
    _, cells, lines = record
    memory = dict(enumerate(cells))
    opcodes = set()
    for line in lines:
        if line.startswith("M "):
            _, _, _, address, width, bits = line.split()
            value = int.from_bytes(bytes.fromhex(bits), "little")
            for k in range(int(width)):  # cells are most significant first, whatever the width
                shift = (int(width) - 1 - k) * 16
                memory[int(address) + k] = value >> shift & 0xFFFF
        elif line.startswith("S "):
            pc = int(line.split()[2])
            word = struct.pack(">4H", *(memory.get(pc + k, 0) for k in range(4)))
            instruction = isa.decode(struct.unpack(">d", word)[0])
            if instruction:
                opcodes.add(isa.BY_NUMBER[instruction[0]].name)
    return opcodes


class GoldenTraces(unittest.TestCase):
    NAMES = sorted(path.stem for path in GOLDEN.glob("*.trace"))

    def test_there_are_golden_traces(self):
        self.assertGreaterEqual(len(self.NAMES), 10)

    def test_every_golden_trace_still_matches_the_reference(self):
        for name in self.NAMES:
            with self.subTest(trace=name):
                self.assertEqual(recorded(name), fresh(name))

    def test_the_golden_traces_exercise_every_instruction(self):
        seen = set()
        for name in self.NAMES:
            seen |= executed(recorded(name))
        self.assertEqual(sorted(seen), sorted(op.name for op in isa.OPS))

    def test_the_set_contains_a_halt_a_fault_and_a_step_limit(self):
        endings = {recorded(name)[2][-1][0] for name in self.NAMES}
        self.assertGreaterEqual(endings, {"H", "F", "L"})

    def test_values_in_a_trace_are_the_bits_of_doubles(self):
        header, _, lines = recorded("hello")
        first = next(line for line in lines if line.startswith("R "))
        bits = first.split()[4]
        self.assertEqual(len(bits), 16)
        self.assertEqual(struct.unpack("<d", bytes.fromhex(bits))[0], 52.0)  # f0: the address of the text


if __name__ == "__main__":
    unittest.main()
