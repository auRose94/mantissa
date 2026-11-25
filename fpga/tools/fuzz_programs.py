#!/usr/bin/env python3
"""Co-simulate random programs against the reference.

    python3 tools/fuzz_programs.py [--count N] [--seed S] [--out-dir DIR] [--sim PATH]
                                   [--steps N] [--dirty FRACTION]

Builds random programs as raw cell images, so that nothing has to be legal:
wrong opcodes, unused fields that are not 00, operands that break the type
rules, addresses that are negative, fractional, misaligned or past the end,
branches into data, stores into the code about to run.  The reference
(mantissa/trace.py) records what it does with each; the simulator replays the
image and its trace is compared line for line.  A program is kept only when
the two disagree, so the directory is a list of failures.

With --dirty 0 every instruction is one the assembler could have written, and
the programs run on until HALT or the step limit; with a larger fraction more
of them are built to go wrong.
"""
from __future__ import annotations

import argparse
import math
import random
import struct
import subprocess
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mantissa import fp, isa, trace  # noqa: E402
from mantissa.machine import Cpu  # noqa: E402
from mantissa.memory import Memory  # noqa: E402
from mantissa.asm import Image  # noqa: E402

MAX_CELLS = 65536              # what the simulator is built with
FILES = (1, 2, 3)              # half, float, double


def dbl_bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def bits_dbl(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


class Program:
    """A program under construction."""

    def __init__(self, rng, dirty, org, cells, data_base, data_cells):
        self.rng = rng
        self.dirty = dirty
        self.org = org             # where the image is loaded
        self.cells_count = cells   # how many cells the machine has
        self.data_base = data_base
        self.data_cells = data_cells
        self.image = array("H")
        self.starts = []           # cell addresses where an instruction begins

    @property
    def here(self):
        return self.org + len(self.image)

    def put(self, values):
        """Append values (doubles) as four cells each."""
        for v in values:
            self.image.extend(fp.pack(fp.D, v))

    def value(self):
        """An operand worth trying, as a double."""
        r = self.rng.random()
        if r < 0.30:
            return float(self.rng.choice([0, 1, 2, 3, 4, 5, 7, 8, 10, 16, 100, 255, 256, 1000, 2047, 2048,
                                          4096, 65504, 65505, 65520, 65536, 16777215, 16777216, 2 ** 24]))
        if r < 0.50:
            return self.rng.choice([0.0, -0.0, math.inf, -math.inf, fp.NAN, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5,
                                    0.1, 0.3, 1e-4, 1e10, 1e-30, 1e300, 5e-324, -5e-324, 2.0 ** -1022,
                                    2.0 ** -1074, 1.7976931348623157e308, 6.1e-5, 0.00006103515625 * 1.5])
        if r < 0.62:              # a NaN with a payload, of any format, widened
            # half, float and double: a quad value is not a double, and the
            # core has no quad arithmetic yet (the ISA is legal all the same)
            fmt = self.rng.choice(fp.FORMATS[:3])
            return fp.from_bits(fmt, fmt.inf | (1 << (fmt.p - 2)) | self.rng.randrange(1 << (fmt.p - 2)))
        if r < 0.72:              # a value of a random format
            fmt = self.rng.choice(fp.FORMATS[:3])   # no quads: see the NaN case above
            return fp.from_bits(fmt, self.rng.getrandbits(fmt.bits))
        return bits_dbl(self.rng.getrandbits(64) & 0x7FFFFFFFFFFFFFFF | (self.rng.getrandbits(1) << 63))

    def file(self):
        return self.rng.choice(FILES)

    def reg(self, f):
        return isa.code_of(f, self.rng.randrange(32))

    def address(self):
        """A cell address for a float register to hold: usually good for something."""
        r = self.rng.random()
        if r < 0.80 and self.data_cells >= 4:
            width = self.rng.choice([1, 2, 4, 4])
            return float(self.data_base + width * self.rng.randrange(self.data_cells // width))
        if self.dirty and r < 0.90:
            return float(self.rng.choice([self.data_base + 1, self.data_base + 2, self.data_base + 3,
                                          self.data_base + 5, self.data_base + self.data_cells - 1,
                                          self.data_base + self.data_cells - 2, self.data_base + self.data_cells - 3,
                                          self.here + 4, 4.0, 12.0, 64.0]))
        if self.dirty:
            n = self.cells_count
            return self.rng.choice([-1.0, -0.5, 0.5, 1.5, 2.5, float(n), float(n - 1), float(n - 3), float(n - 5),
                                    1e6, 1e9, 2 ** 53, fp.NAN, math.inf, -math.inf, -float(self.data_base)])
        return float(self.rng.choice([self.org, self.org + 4, self.org + 8, self.org + 12, self.org + 16, 32, 64,
                                      self.here]))


def legal_operand(program, name, spec):
    """The operand fields of a legal `name`, mostly; and sometimes not."""
    rng = program.rng
    a = b = c = 0
    used = spec.used
    if spec.same_type:
        f = program.file()
        a = program.reg(f)
        if "b" in used:
            b = program.reg(f)
        if "c" in used:
            c = program.reg(f)
    elif name == "LDI":
        a = program.reg(program.file())
    elif name == "LD":
        a, b = program.reg(program.file()), isa.code_of(2, rng.randrange(32))
    elif name == "ST":
        a, b = isa.code_of(2, rng.randrange(32)), program.reg(program.file())
    elif name == "CVT":
        fa, fb = rng.sample(FILES, 2)
        a, b = program.reg(fa), program.reg(fb)
    elif name == "SPLIT":
        f = program.file()
        a, b = program.reg(f), program.reg(f)
        c = program.reg(program.file())
    elif name == "JOIN":
        f = program.file()
        b, c = program.reg(f), program.reg(f)
        a = program.reg(program.file())
    elif name == "MANT":
        a, b, c = program.reg(program.file()), program.reg(program.file()), program.reg(program.file())
    elif name in ("JMP", "JAL"):
        a = isa.code_of(2, rng.randrange(32))
    elif name in ("BZ", "BNZ", "BSGN", "PUT", "PUTC"):
        a = program.reg(program.file())
    return a, b, c


def broken_operand(program, name, spec):
    """Operands that break a rule: the decode's, or the type rules."""
    rng = program.rng
    a, b, c = legal_operand(program, name, spec)
    which = rng.randrange(6)
    if which == 0:                        # a field the instruction does not use is not 00
        for pos in "abc":
            if pos not in spec.used:
                if pos == "a":
                    a = rng.randrange(1, 100)
                elif pos == "b":
                    b = rng.randrange(1, 100)
                else:
                    c = rng.randrange(1, 100)
                break
        else:
            return None
    elif which == 1:
        # codes 96 to 99 name q0 to q3 now: legal for the ISA, but the core
        # has no quad arithmetic yet, so a program that uses them is not a
        # fair differential.  Refused along with everything from 96 up.
        return None
    elif which == 2 and spec.same_type:   # operands of mixed types
        fa, fb = rng.sample(FILES, 2)
        a, b = program.reg(fa), program.reg(fb)
        if "c" in spec.used:
            c = program.reg(rng.choice(FILES))
    elif which == 3:
        # LD, ST, JMP, JAL want a float register; CVT wants two types; SPLIT and JOIN one
        nonfloat = isa.code_of(rng.choice([1, 3]), rng.randrange(32))
        if name == "LD":
            b = nonfloat
        elif name in ("ST", "JMP", "JAL"):
            a = nonfloat
        elif name == "CVT":
            b = a
        elif name == "SPLIT":
            b = program.reg(rng.choice([f for f in FILES if f != a // 32 + 1]))
        elif name == "JOIN":
            c = program.reg(rng.choice([f for f in FILES if f != b // 32 + 1]))
        else:
            return None
    else:
        return None
    return a, b, c


def not_an_instruction(program):
    """A double that is not an instruction at all."""
    rng = program.rng
    r = rng.random()
    if r < 0.35:
        n = rng.randrange(100, 100000000)
        if rng.random() < 0.5:
            n = rng.randrange(0, 100) * 1000000 + rng.randrange(1000000)       # a two-digit opcode that may not exist
        return float(n)
    if r < 0.55:
        return float(rng.randrange(0, 100000000)) + rng.choice([0.25, 0.5, 0.75])   # not whole
    if r < 0.75:
        return rng.choice([-1.0, -12345678.0, 100000000.0, 1e9, 1e20, -0.0, 2 ** 70])
    return rng.choice([math.inf, -math.inf, fp.NAN, bits_dbl(rng.getrandbits(64))])


BY_NAME = isa.BY_NAME
# how often each legal instruction comes up, by name
WEIGHTS = {
    "ADD": 10, "SUB": 8, "MUL": 9, "DIV": 6, "SQRT": 4, "FMA": 6, "REM": 4, "MOD": 5, "MIN": 4, "MAX": 4,
    "NEG": 4, "ABS": 4, "CPSGN": 4, "FLOOR": 4, "CEIL": 4, "CHOP": 5, "ROUND": 4,
    "FLOOR_TO": 3, "CEIL_TO": 3, "CHOP_TO": 4, "ROUND_TO": 4, "SCALB": 4, "LOGB": 4, "NEXT": 4, "CLASS": 4,
    "LT": 4, "LE": 4, "EQ": 4, "CMP": 4, "MOV": 6, "CVT": 10, "SPLIT": 7, "JOIN": 7, "MANT": 6,
    "LD": 14, "ST": 12, "LDI": 6, "BR": 3, "BSGN": 4, "BEQ": 3, "BNE": 3, "BLT": 4, "BLE": 4,
    "BZ": 4, "BNZ": 4, "JAL": 3, "JMP": 3, "PUT": 4, "PUTC": 2, "NOP": 2, "HALT": 1,
}
NAMES = sorted(WEIGHTS)
WEIGHT_LIST = [WEIGHTS[n] for n in NAMES]
WITH_TARGET = {"BR": 1, "BSGN": 2, "BEQ": 1, "BNE": 1, "BLT": 1, "BLE": 1, "BZ": 1, "BNZ": 1, "JAL": 1}


def machine(rng):
    """How big a machine, where the image goes, where the stack starts."""
    cells = rng.choice([MAX_CELLS] * 6 + [4096, 2048, 1000, 300, 1500, 3000, 8192, 16384, 100, 60, MAX_CELLS - 4,
                                          MAX_CELLS - 1, 24, 20, 12, 8, 4])
    data_cells = min(1024, max(0, cells // 8)) if cells >= 64 else 0
    data_base = cells - data_cells
    org = 0 if cells < 64 or rng.random() < 0.8 else 4 * rng.randrange(min(8, (data_base - 8) // 4 + 1))
    sp = float(cells) if rng.random() < 0.8 else float(rng.choice([cells - 4, cells - 1, 0, data_base, cells + 4]))
    return cells, org, sp, data_base, data_cells


def generate(seed, dirty, length):
    """A random program: (cells, org, sp, entry, image)."""
    rng = random.Random(seed)
    cells, org, sp, data_base, data_cells = machine(rng)
    program = Program(rng, dirty, org, cells, data_base, data_cells)

    def fits(n):
        """Would n more cells of code still end before the data?"""
        return program.here + n <= data_base - (4 if data_cells else 0)

    # --- data: values of every format written cell by cell at any offset, so
    # that an aligned load sees part of one as often as a whole one
    data = array("H")
    while len(data) < data_cells:
        if rng.random() < 0.55:
            fmt = rng.choice(fp.FORMATS)
            if rng.random() < 0.25:
                bits = fmt.inf | (1 << (fmt.p - 2)) | rng.randrange(1 << (fmt.p - 2))   # a NaN with a payload
            else:
                bits = rng.getrandbits(fmt.bits)
            data.extend(fp.pack(fmt, fp.from_bits(fmt, bits)))     # most significant cell first
        else:
            data.append(rng.getrandbits(16))
    del data[data_cells:]

    # --- the prologue: registers filled with every kind of value, and float
    # registers holding addresses
    for _ in range(rng.randrange(4, 70)):
        if not fits(8):
            break
        program.starts.append(program.here)
        program.put([float(isa.encode(11, program.reg(program.file()), 0, 0)), program.value()])
    for _ in range(rng.randrange(6, 20)):
        if not fits(8):
            break
        program.starts.append(program.here)
        program.put([float(isa.encode(11, isa.code_of(2, rng.randrange(32)), 0, 0)), program.address()])

    # --- the body: what the instructions will be, then where each one starts,
    # so that branches can name them
    plan = []
    planned = 0                     # cells the plan will take
    for _ in range(length):
        name = "BROKEN_NUMBER" if rng.random() < dirty * 0.12 else rng.choices(NAMES, WEIGHT_LIST)[0]
        size = 4 if name == "BROKEN_NUMBER" else BY_NAME[name].cells
        if not fits(planned + size + 4):
            break
        plan.append(name)
        planned += size
    if fits(planned + 4):
        plan.append("HALT")
    at = program.here
    for name in plan:
        program.starts.append(at)
        at += 4 if name == "BROKEN_NUMBER" else BY_NAME[name].cells
    code_end = at

    def target():
        r = rng.random()
        if r < 0.86:
            return float(rng.choice(program.starts))                 # an instruction
        if r < 0.94:
            return float(rng.choice(program.starts) + 4)             # into an inline operand
        if program.dirty and r < 0.98 and data_cells >= 4:
            return float(data_base + 4 * rng.randrange(data_cells // 4))   # into the data
        return float(code_end + 4 * rng.randrange(1, 8))             # past the code: zero cells

    def bad_target():
        r = rng.random()
        if r < 0.25:
            return float(rng.choice(program.starts) + rng.choice([1, 2, 3]))      # misaligned
        if r < 0.45:
            return rng.choice([-4.0, -8.0, float(cells), float(cells - 4), float(cells + 4), 2 ** 40,
                               float(cells - 1), float(cells - 2), float(cells - 3)])
        if r < 0.65:
            return rng.choice([0.5, 2.5, 100.25, fp.NAN, math.inf, -math.inf])   # not a cell number
        if r < 0.85:
            return float(code_end + 4 * rng.randrange(4, 40))
        return float(rng.choice(program.starts))

    for name in plan:
        if name == "BROKEN_NUMBER":
            program.put([not_an_instruction(program)])
            continue
        spec = BY_NAME[name]
        fields = None
        if program.dirty and name != "HALT" and rng.random() < 0.10:
            fields = broken_operand(program, name, spec)
        if fields is None:
            fields = legal_operand(program, name, spec)
        a, b, c = fields
        inline = []
        if name == "LDI":
            inline = [program.value()]
        for _ in range(WITH_TARGET.get(name, 0)):
            inline.append(bad_target() if program.dirty and rng.random() < 0.06 else target())
        program.put([float(isa.encode(spec.number, a, b, c))] + inline)
    assert program.here == code_end, (program.here, code_end)

    # the image is what lies between the code and the data; the machine's zero
    # cells before the image and between it and the data need not be in it
    image = program.image
    if program.dirty and rng.random() < 0.5 and org > 0:
        image.extend(array("H", bytes(2 * rng.randrange(1, 9))))       # a few trailing cells
    entry = float(org)
    if program.dirty and rng.random() < 0.05:
        entry = float(rng.choice(program.starts))
    return cells, org, sp, entry, image


def record(cells, org, sp, entry, image, max_steps):
    """The reference's trace lines for a machine and an image, as trace.lines writes them."""
    picture = Image(org, image, {}, entry)
    memory = Memory(cells)
    floats = [0.0] * 32
    floats[31] = sp
    memory.write(image, org)
    return trace.lines(picture, memory, Cpu(f=tuple(floats), pc=entry), max_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="build/fuzz")
    parser.add_argument("--sim", default="build/obj_model/m1_sim")
    parser.add_argument("--steps", type=int, default=300, help="the step limit each program runs under")
    parser.add_argument("--dirty", type=float, default=0.0, help="fraction of programs built to go wrong")
    parser.add_argument("--length", type=int, default=120, help="instructions in the body")
    parser.add_argument("--batch", type=int, default=400, help="programs per simulator run")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    endings = {}
    refused = 0
    refused_kind = {}
    total_failed = 0
    total = 0
    batch = []
    first = args.seed

    def run_batch():
        nonlocal total_failed, total
        if not batch:
            return
        listing = out / "batch.list"
        listing.write_text("\n".join(str(p) for p in batch) + "\n")
        proc = subprocess.run([args.sim, "--batch", str(listing)], capture_output=True, text=True)
        bad = set()
        for line in proc.stdout.splitlines():
            if line.startswith("FAIL "):
                bad.add(line.split()[1].rstrip(":"))      # "FAIL name: ..."
                total_failed += 1
        for p in batch:
            if p.name not in bad:
                p.unlink()
        total += len(batch)
        batch.clear()
        for line in proc.stdout.splitlines():
            if line.startswith("  run line"):
                print(line)

    rng = random.Random(args.seed * 7919)
    for i in range(args.count):
        seed = first + i
        dirty = args.dirty if rng.random() < 0.75 else 1.0
        try:
            lines = record(*generate(seed, dirty, args.length), args.steps)
        except (OverflowError, ValueError, ZeroDivisionError) as error:
            # the reference raises on some operands (DIV by infinity: NOTES.md); anything
            # else here is a generator bug worth hearing about
            kind = f"{type(error).__name__}: {error}"
            refused_kind[kind] = refused_kind.get(kind, 0) + 1
            refused += 1
            continue
        ending = lines[-1]
        endings[ending] = endings.get(ending, 0) + 1
        path = out / f"fuzz_{seed}.trace"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        batch.append(path)
        if len(batch) >= args.batch:
            run_batch()
    run_batch()
    kinds = ", ".join(f"{k}: {v}" for k, v in sorted(endings.items()))
    print(f"{total} programs, {total_failed} disagree with the reference; endings {kinds}; "
          f"{refused} programs the reference could not run")
    for kind, n in sorted(refused_kind.items(), key=lambda kv: -kv[1]):
        print(f"  refused {n:5d}x: {kind}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
