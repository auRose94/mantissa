#!/usr/bin/env python3
"""make_vectors.py -- test vectors for the GPU port's arithmetic unit.

    python3 make_vectors.py OUT.bin [--per-op N] [--seed S] [--ops ADD,DIV,...]
    python3 make_vectors.py OUT.bin --show INDEX [INDEX ...]

Writes test vectors: one operation, its operands, and the value and flags that
mantissa/fp.py gives.  gpu/src/fptest.cpp feeds them to the arithmetic unit in
gpu/src/fp.h, on the host and on the GPU, so it is checked against the
reference on random and awkward input, not only on what the golden traces
happen to touch.

Each record is 56 bytes, little-endian:

    op, fmt, src, 0     four bytes: the operation, the format of its result,
                        and for the casts the format of the operand
    0                   four bytes spare
    a, b, c             the operands, the 64 bits of the double that holds each
    value, value2       the answer, and its second half (SPLIT's low, MANT's e)
    flags, 0            the sticky flags, and four bytes spare

Operands are drawn from random bit patterns, the lowest binades, special
values (signed zeros, infinities, NaNs with payloads, the smallest and largest
subnormal and normal), small whole numbers, and neighbours of powers of two.
The second operand is often a neighbour of the first: its double, its half,
the next value of the format, one unit in the last place away.
"""
from __future__ import annotations

import argparse
import math
import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # the copy's root
sys.path.insert(0, str(ROOT))

from mantissa import fp  # noqa: E402

FORMATS = (fp.H, fp.F, fp.D)
OPS = ["ADD", "SUB", "MUL", "DIV", "SQRT", "FMA", "REM", "MOD", "MIN", "MAX", "CPSGN", "SCALB", "NEXT", "LT", "LE",
       "EQ", "CMP", "FLOOR_TO", "CEIL_TO", "TRUNC_TO", "ROUND_TO", "NEG", "ABS", "LOGB", "FLOOR", "CEIL", "TRUNC",
       "ROUND", "CLASS", "CONVERT", "SPLIT", "JOIN", "MANT"]
NUMBER = {name: number for number, name in enumerate(OPS)}
CASTS = ("CONVERT", "SPLIT", "JOIN", "MANT")
RECORD = struct.Struct("<BBBBIQQQQQII")


def bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def value_of(bits_):
    return struct.unpack("<d", struct.pack("<Q", bits_))[0]


def mk(fmt, value):
    """The fmt value nearest to `value`; a NaN passes as it is."""
    return fp.round_literal(fmt, float(value))  # float(): round_literal hands a whole-number literal back as it came


_SPECIALS = {}


def specials(fmt):
    """Values with a story: zeros, infinities, NaNs of several kinds, the edges of the ranges."""
    if fmt in _SPECIALS:
        return _SPECIALS[fmt]
    one = 1 << (fmt.p - 1)
    top = fmt.emax + fmt.emax
    bottom = fmt.emax + fmt.emin
    patterns = [0, 1, 2, 3, one - 1, one, one + 1, bottom << (fmt.p - 1)]
    patterns += [(bottom + k) << (fmt.p - 1) for k in (-1, 0, 1)]
    patterns += [top << (fmt.p - 1), (top << (fmt.p - 1)) | (one - 1), (top << (fmt.p - 1)) | (one - 2)]
    patterns += [fmt.inf, fmt.inf | 1, fmt.inf | (one >> 1), fmt.inf | (one - 1), fmt.inf | (one >> 1) | 1]
    for exponent in (fmt.emin - 1, fmt.emin, fmt.emin + 1, -2, -1, 0, 1, 2, 3, fmt.emax - 1):
        field = fmt.emax + exponent
        if 1 <= field < 2 * fmt.emax:
            patterns += [field << (fmt.p - 1), (field << (fmt.p - 1)) | 1, (field << (fmt.p - 1)) | (one - 1),
                         (field << (fmt.p - 1)) | (one >> 1)]
    _SPECIALS[fmt] = [fp.from_bits(fmt, pattern | sign) for pattern in patterns for sign in (0, fmt.sign)]
    return _SPECIALS[fmt]


def sampler(fmt, rng):
    """A function that hands out values of the format, by several recipes."""
    pool = specials(fmt)

    def low_binades():  # a random significand in one of the lowest binades, subnormals included
        field = rng.choice((0, 0, 1, 2, 3, 4, 7, 12))
        return fp.from_bits(fmt, (field << (fmt.p - 1)) | rng.getrandbits(fmt.p - 1)
                            | (rng.getrandbits(1) << (fmt.bits - 1)))

    def small():
        return mk(fmt, rng.choice((rng.randint(-32, 32), rng.randint(-4000, 4000),
                                   rng.randint(-3, 3) + rng.choice((0.0, 0.5, 0.25, -0.5)))))

    def near_power():
        return mk(fmt, math.ldexp(1.0 + rng.randint(-6, 6) * 2.0 ** -(fmt.p - 1),
                                  rng.randint(fmt.emin - 2, fmt.emax)))

    def one():
        roll = rng.random()
        if roll < 0.30:
            return fp.from_bits(fmt, rng.getrandbits(fmt.bits))
        if roll < 0.50:
            return rng.choice(pool)
        if roll < 0.65:
            return low_binades()
        if roll < 0.80:
            return small()
        return near_power()

    return one


def neighbour(fmt, a, rng):
    """A second operand with a reason to interest the first: rounded back to the format."""
    unit = 2.0 ** -(fmt.p - 1)
    finite = a == a and abs(a) != math.inf
    choice = rng.randrange(16)
    if choice == 0:
        b = a
    elif choice == 1:
        b = -a
    elif choice == 2:
        b = a * 2
    elif choice == 3:
        b = a / 2
    elif choice == 4:
        b = a * 3
    elif choice == 5:
        b = fp.nextafter(fmt, a, math.inf)[0] if finite else a
    elif choice == 6:
        b = fp.nextafter(fmt, a, -math.inf)[0] if finite else a
    elif choice == 7:
        b = a * (1.0 + unit)
    elif choice == 8:
        b = a * (1.0 - unit)
    elif choice == 9:
        b = a * 2.0 ** rng.randint(-3, 3)
    elif choice == 10:
        b = rng.choice(specials(fmt))
    elif choice == 11:
        b = a * rng.choice((1.5, 0.75, 7.0, 0.1, 10.0, 0.01, 1e10))
    elif choice == 12:
        b = rng.randint(-9, 9)
    elif choice == 13:
        b = -a * (1.0 + unit)
    elif choice == 14:
        b = a * 2.0 ** rng.randint(-fmt.p - 2, fmt.p + 2)
    else:
        b = a * rng.choice((rng.randint(-5, 5), 2.0 ** rng.randint(-20, 20)))
    return mk(fmt, b)


UNARY = ("SQRT", "NEG", "ABS", "LOGB", "FLOOR", "CEIL", "TRUNC", "ROUND", "CLASS")
BINARY = ("ADD", "SUB", "MUL", "DIV", "FMA", "REM", "MOD", "MIN", "MAX", "CPSGN", "LT", "LE", "EQ", "CMP",
          "SCALB", "NEXT", "FLOOR_TO", "CEIL_TO", "TRUNC_TO", "ROUND_TO", "JOIN")


def operands(op, fmt, src, rng):
    """(a, b, c) for one case: values of `fmt`, or of `src` for the casts."""
    if op in CASTS:
        return sampler(src, rng)(), 0.0, 0.0
    draw = sampler(fmt, rng)
    a = draw()
    if op in UNARY:
        if op == "SQRT" and rng.random() < 0.5:  # perfect squares, and squares of k + 1/2
            k = rng.choice((rng.randint(0, 2000), rng.randint(0, 1 << 12) + 0.5))
            a = mk(fmt, k * k)
        return a, 0.0, 0.0
    assert op in BINARY, op
    c = 0.0
    if op.endswith("_TO"):
        b = mk(fmt, rng.choice((0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 0.25, 0.3, 1e-10, 1e10,
                                2.0 ** rng.randint(fmt.emin - 2, fmt.emax), 0.1 ** rng.randint(1, 6))))
        if rng.random() < 0.5:  # a multiple of the step, just off one, or a long way off one
            a = mk(fmt, b * rng.choice((rng.randint(-500, 500), rng.randint(1, 1 << (fmt.p - 1)))))
            finite = a == a and abs(a) != math.inf
            nudge = rng.choice((0, 0, 1, -1, 2, -2, 3, -3))
            for _ in range(abs(nudge) if finite else 0):
                a = fp.nextafter(fmt, a, math.inf if nudge > 0 else -math.inf)[0]
            if finite and rng.random() < 0.25:
                a = mk(fmt, a * (1.0 + rng.choice((1, -1)) * 2.0 ** rng.randint(fmt.p, fmt.p + 30)))
    elif op == "SCALB":
        b = mk(fmt, rng.choice((rng.randint(-60, 60), rng.randint(-1100, 1100), 2200, -2200, 2199, -2199, 1075,
                                -1075, 1074, -1074, 1024, -1024, math.inf, -math.inf, 0.0, 0.5, -0.5, 2.5,
                                2.0 ** 40, -(2.0 ** 40), fp.NAN)))
    elif op == "NEXT":
        near = mk(fmt, fp.nextafter(fmt, a, rng.choice((math.inf, -math.inf)))[0]) if a == a and abs(a) != math.inf \
            else a
        b = near if rng.random() < 0.5 else draw()
    else:
        b = neighbour(fmt, a, rng) if rng.random() < 0.65 else draw()
    if op == "DIV" and rng.random() < 0.5:  # exact quotients, and quotients one unit off
        b = mk(fmt, rng.choice((rng.randint(1, 40), 2.0 ** rng.randint(fmt.emin, min(fmt.emax, 40)))))
        a = mk(fmt, b * rng.randint(-200, 200))
    if op == "FMA":
        roll = rng.random()
        if roll < 0.3:  # cancellation
            c = mk(fmt, -(a * b))
        elif roll < 0.5:
            c = mk(fmt, -(a * b) * (1.0 + rng.choice((2.0 ** -(fmt.p - 1), -(2.0 ** -(fmt.p - 1)), 1e-6))))
        else:
            c = draw()
    if op == "JOIN":  # halves that belong together, and terms far apart
        roll = rng.random()
        if roll < 0.4 and a == a:
            high, low, _ = fp.split(fmt, a * 2.0 ** rng.randint(-fmt.p, fmt.p))
            return high, low, 0.0
        if roll < 0.6:
            return mk(fmt, a * 2.0 ** rng.randint(60, 300)), draw(), 0.0
    return a, b, c


def answer(op, fmt, a, b, c):
    """(value, value2, flags) from mantissa/fp.py."""
    if op == "ADD":
        value, flags = fp.add(fmt, a, b)
    elif op == "SUB":
        value, flags = fp.sub(fmt, a, b)
    elif op == "MUL":
        value, flags = fp.mul(fmt, a, b)
    elif op == "DIV":
        value, flags = fp.div(fmt, a, b)
    elif op == "SQRT":
        value, flags = fp.sqrt(fmt, a)
    elif op == "FMA":
        value, flags = fp.fma(fmt, a, b, c)
    elif op == "REM":
        value, flags = fp.remainder(fmt, a, b)
    elif op == "MOD":
        value, flags = fp.fmod(fmt, a, b)
    elif op == "MIN":
        value, flags = fp.fmin(fmt, a, b)
    elif op == "MAX":
        value, flags = fp.fmax(fmt, a, b)
    elif op == "CPSGN":
        value, flags = fp.copysign(fmt, a, b)
    elif op == "SCALB":
        value, flags = fp.scaleb(fmt, a, b)
    elif op == "NEXT":
        value, flags = fp.nextafter(fmt, a, b)
    elif op == "LT":
        value, flags = fp.lt(fmt, a, b)
    elif op == "LE":
        value, flags = fp.le(fmt, a, b)
    elif op == "EQ":
        value, flags = fp.eq(fmt, a, b)
    elif op == "CMP":
        value, flags = fp.compare(fmt, a, b)
    elif op == "FLOOR_TO":
        value, flags = fp.floor_to(fmt, a, b)
    elif op == "CEIL_TO":
        value, flags = fp.ceil_to(fmt, a, b)
    elif op == "TRUNC_TO":
        value, flags = fp.trunc_to(fmt, a, b)
    elif op == "ROUND_TO":
        value, flags = fp.round_to(fmt, a, b)
    elif op == "NEG":
        value, flags = fp.neg(fmt, a)
    elif op == "ABS":
        value, flags = fp.fabs(fmt, a)
    elif op == "LOGB":
        value, flags = fp.logb(fmt, a)
    elif op == "FLOOR":
        value, flags = fp.floor(fmt, a)
    elif op == "CEIL":
        value, flags = fp.ceil(fmt, a)
    elif op == "TRUNC":
        value, flags = fp.trunc(fmt, a)
    elif op == "ROUND":
        value, flags = fp.round_even(fmt, a)
    elif op == "CLASS":
        value, flags = fp.classify(fmt, a)
    elif op == "CONVERT":
        value, flags = fp.convert(fmt, a)
    elif op == "SPLIT":
        value, value2, flags = fp.split(fmt, a)
        return value, value2, flags
    elif op == "JOIN":
        value, flags = fp.join(fmt, a, b)
    elif op == "MANT":
        value, value2 = fp.mantissa(a)
        return value, value2, 0
    else:
        raise ValueError(op)
    return value, 0, flags  # no second value, so its bits are zero


def one_case(op, number, fi, si, rng):
    fmt, src = FORMATS[fi], FORMATS[si]
    while True:
        a, b, c = operands(op, fmt, src, rng)
        try:
            value, value2, flags = answer(op, fmt, a, b, c)
        except (OverflowError, ZeroDivisionError, ValueError):  # Python refuses what the machine answers
            continue
        return RECORD.pack(number, fi, si, 0, 0, bits(a), bits(b), bits(c), bits(value), bits(value2), flags, 0)


def generate(per_op, seed, wanted):
    rng = random.Random(seed)
    records = []
    for name in wanted:
        number = NUMBER[name]
        combos = [(fi, si) for fi in range(3) for si in range(3)] if name in CASTS else [(fi, fi) for fi in range(3)]
        for fi, si in combos:
            for _ in range(per_op):
                records.append(one_case(name, number, fi, si, rng))
    return b"".join(records)


def show(path, indexes):
    data = Path(path).read_bytes()
    for index in indexes:
        start = index * RECORD.size
        op, fmt, src, _, _, a, b, c, value, value2, flags, _ = RECORD.unpack(data[start:start + RECORD.size])
        print(f"[{index}] {OPS[op]} fmt={FORMATS[fmt].name} src={FORMATS[src].name}")
        for label, word in (("a", a), ("b", b), ("c", c)):
            print(f"    {label} = {word:016x}  {value_of(word)!r}")
        print(f"    value = {value:016x}  {value_of(value)!r}")
        print(f"    value2 = {value2:016x}  {value_of(value2)!r}")
        print(f"    flags = {flags:#x} {fp.flag_names(flags)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", help="the file to write, or the file to read with --show")
    parser.add_argument("--per-op", type=int, default=4000, help="cases per operation, format and, for the casts, "
                        "operand format (default 4000)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ops", default=",".join(OPS), help="only these operations")
    parser.add_argument("--show", nargs="+", type=int, default=None, metavar="INDEX",
                        help="print records from an existing file instead of writing one")
    args = parser.parse_args(argv)
    if args.show is not None:
        show(args.out, args.show)
        return 0
    wanted = [name.strip().upper() for name in args.ops.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in NUMBER]
    if unknown:
        sys.exit(f"no such operation: {', '.join(unknown)}")
    data = generate(args.per_op, args.seed, wanted)
    Path(args.out).write_bytes(data)
    print(f"{args.out}: {len(data) // RECORD.size} cases in {len(data)} bytes, seed {args.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
