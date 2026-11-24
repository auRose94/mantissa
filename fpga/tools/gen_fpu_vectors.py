#!/usr/bin/env python3
"""Test vectors for a floating point unit, with the expected answers.

The unit is whatever implements the interface in fpga/rtl/mantissa_pkg.sv
(fpu_op_e): the simulation model rtl/model/fpu_model.sv, or the synthesizable
one.  This script does what mantissa/machine.py does for each opcode, calls
mantissa/fp.py on the operands, and writes one request per line:

    op fmt fmt2 sel a b c y y2 flags

Every value is sixteen hex digits: the 64 bits of the double that holds it,
most significant first.  op and sel are decimal, fmt and fmt2 are 0, 1, 2 for
half, float, double, and flags is two hex digits, the sticky flags in the low
five bits.

Operand order is the interface's: every operation takes x as a and y as b
(FMA is a * b + c, SPLIT and MANT read only a, JOIN reads a as the high part
and b as the low part).

    python3 tools/gen_fpu_vectors.py --out FILE [--seed N] [--count N]

Operands come from every corner: the special values, the smallest and largest
of each kind, the midpoints that decide a rounding, NaNs with payloads, small
whole numbers, and random bit patterns.
"""
from __future__ import annotations

import argparse
import math
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mantissa import fp  # noqa: E402
from mantissa.machine import _BINARY, _UNARY  # noqa: E402  (the reference's own opcode tables)

OP = {
    "ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3, "SQRT": 4, "FMA": 5, "REM": 6, "MOD": 7, "MIN": 8,
    "MAX": 9, "NEG": 10, "ABS": 11, "CPSGN": 12, "FLOOR": 13, "CEIL": 14, "CHOP": 15, "ROUND": 16,
    "FLOOR_TO": 17, "CEIL_TO": 18, "CHOP_TO": 19, "ROUND_TO": 20, "SCALB": 21, "LOGB": 22,
    "NEXT": 23, "CLASS": 24, "LT": 25, "LE": 26, "EQ": 27, "CMP": 28, "CVT": 29, "FROMBITS": 30,
    "TOBITS": 31, "SPLIT": 32, "JOIN": 33, "MANT": 34,
}
BINARY = {fp.add: "ADD", fp.sub: "SUB", fp.mul: "MUL", fp.div: "DIV", fp.remainder: "REM",
          fp.fmod: "MOD", fp.fmin: "MIN", fp.fmax: "MAX", fp.copysign: "CPSGN", fp.scaleb: "SCALB",
          fp.nextafter: "NEXT", fp.lt: "LT", fp.le: "LE", fp.eq: "EQ", fp.compare: "CMP",
          fp.floor_to: "FLOOR_TO", fp.ceil_to: "CEIL_TO", fp.trunc_to: "CHOP_TO",
          fp.round_to: "ROUND_TO"}
UNARY = {fp.sqrt: "SQRT", fp.neg: "NEG", fp.fabs: "ABS", fp.logb: "LOGB", fp.floor: "FLOOR",
         fp.ceil: "CEIL", fp.trunc: "CHOP", fp.round_even: "ROUND", fp.classify: "CLASS"}
# the reference's opcode tables name exactly these functions
assert set(_BINARY.values()) == set(BINARY)
assert set(_UNARY.values()) == set(UNARY)

FORMATS = (fp.H, fp.F, fp.D)
FMT = {"H": 0, "F": 1, "D": 2}


def hex64(value):
    """The 64 bits of the double that holds `value`, most significant first."""
    return struct.pack(">Q", struct.unpack("<Q", struct.pack("<d", value))[0]).hex()


def bits64(word):
    """A 64-bit pattern as the sixteen digits of a value field."""
    return f"{word & 0xFFFFFFFFFFFFFFFF:016x}"


def constants(fmt):
    """Values worth trying, as the doubles that hold them."""
    top = fp.from_bits(fmt, fmt.inf - 1)                 # the largest finite
    low = fp.from_bits(fmt, 1)                           # the smallest subnormal
    normal = fmt.min_normal
    # a hair below the next power of two: rounds up to infinity.  A double cannot hold its own.
    over = (sys.float_info.max if fmt is fp.D
            else math.ldexp(1.0, fmt.emax + 1) - math.ldexp(1.0, fmt.emax - fmt.p + 1))
    values = [
        0.0, -0.0, math.inf, -math.inf, fp.NAN, -fp.NAN,
        fp.from_bits(fmt, fmt.inf | (1 << (fmt.p - 2)) | 0x155),   # quiet, with a payload
        fp.from_bits(fmt, fmt.sign | fmt.inf | (1 << (fmt.p - 2))),  # quiet and negative
        fp.from_bits(fmt, fmt.inf | 1),                            # signalling
        top, -top, fp.from_bits(fmt, fmt.inf - 2), -fp.from_bits(fmt, fmt.inf - 2), over, -over,
        normal, -normal,
        fp.from_bits(fmt, (1 << (fmt.p - 1)) - 1), -fp.from_bits(fmt, (1 << (fmt.p - 1)) - 1),  # largest subnormal
        fp.from_bits(fmt, 1 << (fmt.p - 2)), -fp.from_bits(fmt, 1 << (fmt.p - 2)),  # half the smallest normal
        low, -low, fp.from_bits(fmt, 2), -fp.from_bits(fmt, 2), fp.from_bits(fmt, 3), fp.from_bits(fmt, 7),
        fp.from_bits(fmt, fmt.inf - 1 - (1 << (fmt.p - 2))),      # a quarter below the top: ties up
        1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.0, 4.5, 10.0, 100.0, 0.1, 0.3, 1e-4,
        4095.0, 4096.0, 2047.5, 2048.5,
    ]
    # A register holds an exact value of its format.  Plain numbers such as 0.1
    # are rounded the way the assembler rounds a literal; to_bits on a value
    # that is not of the format would truncate, which the machine never asks.
    return [v if v != v or v in (0.0, -0.0) else fp.round_literal(fmt, v) for v in values]


def exact(fmt, x):
    """A value a register of the format could hold: the nearest one, the way
    the assembler rounds a literal.  A NaN, a zero or an infinity stays."""
    return fp.round_literal(fmt, x)


def sample(fmt, rng, pool):
    """An operand: a value of the format, corners included."""
    roll = rng.random()
    if roll < 0.42:
        return rng.choice(pool)                  # the pool is made of exact values
    if roll < 0.56:
        return exact(fmt, float(rng.choice([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15, 16, 17, 100, 255,
                                            256, 1000, 2047, 2048, 4096, 65504, 65505, 65520, 65519])))
    if roll < 0.68:
        # a few significant bits at any exponent the format has, and a little beyond
        spread = (fmt.emax + 8) - (fmt.emin - fmt.p - 8)
        e = fmt.emin - fmt.p - 8 + rng.randrange(spread)
        if fmt is fp.D:
            e = min(e, 1000)                      # the intermediate must stay inside a double
        x = math.ldexp(float(rng.randrange(1, 1 << rng.choice([1, 2, 3, 5, 8, 12, 17]))), e)
        return fp.convert(fmt, x)[0]
    return fp.from_bits(fmt, rng.getrandbits(fmt.bits))


def step_sample(fmt, rng, pool):
    """A step for the grid roundings: mostly plain numbers, sometimes odd."""
    if rng.random() < 0.30:
        return sample(fmt, rng, pool)
    return exact(fmt, float(rng.choice([0.5, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 0.25, 0.1, 0.3, 0.01, 1.5,
                                        2.5, 11, 13, 100, 1000, 0.001, 0.0625, 1 / 3, 1 / 7, 0.7,
                                        2 ** -3, 2 ** 6, 1e6])))


class Writer:
    def __init__(self):
        self.lines = []
        self.undefined = []      # operands the reference raises on: the port decides these
        self.raised = {}

    def call(self, op, fmt, fn, *operands):
        """fn(*operands), unless the reference cannot answer: see the module docstring's note."""
        try:
            return fn(fmt, *operands)
        except (OverflowError, ValueError, ZeroDivisionError) as error:
            kind = f"{op} to {fmt.name}: {type(error).__name__}: {error}"
            self.raised[kind] = self.raised.get(kind, 0) + 1
            self.undefined.append((op, fmt, fn, operands, error))
            return None

    def add(self, op, fmt, a=0.0, b=0.0, c=0.0, result=None, fmt2=None, sel=0):
        if result is None:
            return
        value, value2, flags = result
        fmt2 = fmt if fmt2 is None else fmt2
        self.lines.append(f"{OP[op]} {FMT[fmt.name]} {FMT[fmt2.name]} {sel} {hex64(a)} {hex64(b)} "
                          f"{hex64(c)} {hex64(value)} {hex64(value2)} {flags:02x}")

    def raw(self, op, fmt, word, value, sel=0):
        """A request whose operand a is a word of cells."""
        self.lines.append(f"{OP[op]} {FMT[fmt.name]} {FMT[fmt.name]} {sel} {bits64(word)} {hex64(0.0)} "
                          f"{hex64(0.0)} {hex64(value)} {hex64(0.0)} 00")

    def narrow(self, op, fmt, x, answer):
        """A request whose answer is a word of cells: the value x, and the cells it becomes."""
        self.lines.append(f"{OP[op]} {FMT[fmt.name]} {FMT[fmt.name]} 0 {hex64(x)} {hex64(0.0)} "
                          f"{hex64(0.0)} {bits64(answer)} {hex64(0.0)} 00")


def one(value, flags):
    return value, 0.0, flags


def generate(seed, count):
    rng = random.Random(seed)
    pools = {fmt.name: constants(fmt) for fmt in FORMATS}
    w = Writer()
    per = max(2, count // 150)

    for fmt in FORMATS:
        pool = pools[fmt.name]
        everything = [v for f in FORMATS for v in pools[f.name]] + pools["D"]

        # --- arithmetic, called exactly as machine.py calls it
        for fn, key in BINARY.items():
            for _ in range(per):
                x = sample(fmt, rng, pool)
                y = step_sample(fmt, rng, pool) if key.endswith("_TO") else sample(fmt, rng, pool)
                answer = w.call(key, fmt, fn, x, y)
                if answer:
                    w.add(key, fmt, x, y, result=one(*answer))
        for fn, key in UNARY.items():
            for _ in range(per):
                x = sample(fmt, rng, pool)
                answer = w.call(key, fmt, fn, x)
                if answer:
                    w.add(key, fmt, x, result=one(*answer))
        for _ in range(per):
            a, b, c = (sample(fmt, rng, pool) for _ in range(3))
            answer = w.call("FMA", fmt, fp.fma, a, b, c)
            if answer:
                w.add("FMA", fmt, a, b, c, one(*answer))

        # --- conversions: between every pair of formats, and from any double
        for dst in FORMATS:
            if dst is fmt:
                continue
            for _ in range(per):
                x = sample(fmt, rng, pool)
                value, flags = fp.convert(dst, x)
                w.add("CVT", dst, x, result=one(value, flags))
        for _ in range(per):
            x = sample(fp.D, rng, everything)
            value, flags = fp.convert(fmt, x)
            w.add("CVT", fmt, x, result=one(value, flags))

        # --- cells: widen and narrow
        for _ in range(per):
            # any pattern, or the pattern of a special value such as a NaN with a payload
            bits = (fp.to_bits(fmt, rng.choice(pool)) if rng.random() < 0.5
                    else rng.getrandbits(fmt.bits))
            # which cell of the word the value starts at: any for a half, 0 or 2 for a float
            sel = {1: rng.randrange(4), 2: rng.randrange(2) * 2, 4: 0}[fmt.cells]
            narrow = bits << (16 * (4 - fmt.cells - sel))     # into the cells it fills, from the top
            w.raw("FROMBITS", fmt, narrow, fp.from_bits(fmt, bits), sel)
            x = sample(fmt, rng, pool)
            answer = fp.to_bits(fmt, x)
            w.narrow("TOBITS", fmt, x, answer)

        # --- the casts that keep what a plain cast drops
        for dst in FORMATS:
            for _ in range(per):
                x = sample(fp.D, rng, everything)
                high, low, flags = fp.split(dst, x)
                w.add("SPLIT", dst, a=x, result=(high, low, flags))
        for dst in FORMATS:
            for src in FORMATS:
                for _ in range(max(2, per // 2)):
                    high = sample(src, rng, pools[src.name])
                    low = sample(src, rng, pools[src.name])
                    value, flags = fp.join(dst, high, low)
                    w.add("JOIN", dst, high, low, one(value, flags), fmt2=src)
        for mf in FORMATS:
            for ef in FORMATS:
                for _ in range(max(1, per // 3)):
                    x = sample(fp.D, rng, everything)
                    m, e = fp.mantissa(x)
                    m, fl = fp.convert(mf, m)
                    e, fl2 = fp.convert(ef, e)
                    w.add("MANT", mf, a=x, result=(m, e, fl | fl2), fmt2=ef)

    rng.shuffle(w.lines)
    return w


def port_defined(undefined):
    """Expected answers for the operands the reference raises on.

    The machine cannot crash, so the port takes the IEEE result: a finite
    number over infinity is a signed zero and exact, and a root too large for
    the format is infinity with overflow and inexact raised.
    """
    lines = []
    for op, fmt, fn, operands, error in undefined:
        if op == "DIV":
            a, b = operands
            if math.isinf(b) and a != 0 and not math.isinf(a):
                lines.append(f"{OP[op]} {FMT[fmt.name]} {FMT[fmt.name]} 0 {hex64(a)} {hex64(b)} {hex64(0.0)} "
                             f"{hex64(math.copysign(0.0, a) * math.copysign(1.0, b))} {hex64(0.0)} 00")
        elif op == "SQRT":
            (x,) = operands
            # the smallest value whose root rounds up to infinity: the overflow midpoint, squared
            edge = math.ldexp(1.0, fmt.emax + 1) - math.ldexp(1.0, fmt.emax - fmt.p + 1)
            if x > 0 and not math.isinf(x) and x >= edge * edge:
                lines.append(f"{OP[op]} {FMT[fmt.name]} {FMT[fmt.name]} 0 {hex64(x)} {hex64(0.0)} {hex64(0.0)} "
                             f"{hex64(math.inf)} {hex64(0.0)} {fp.OVERFLOW | fp.INEXACT:02x}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=120000, help="about how many requests")
    args = parser.parse_args()
    w = generate(args.seed, args.count)
    Path(args.out).write_text("\n".join(w.lines) + "\n")
    print(f"{args.out}: {len(w.lines)} requests")
    decided = port_defined(w.undefined)
    Path(args.out + ".port_defined").write_text("\n".join(decided) + "\n")
    for kind, n in sorted(w.raised.items()):
        print(f"  the reference raises on {n} sampled operand sets: {kind}")
    print(f"{args.out}.port_defined: {len(decided)} requests the reference raises on, answered as the port does")


if __name__ == "__main__":
    main()
