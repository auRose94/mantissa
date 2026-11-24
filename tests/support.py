"""Shared helpers: an exact reference for rounding, and a one-call program runner.

Nothing here calls into mantissa.fp to produce an expected value.  The
reference rounds with integer arithmetic on Fractions, reads bit patterns with
the struct module, and compares NaNs by their canonical bit pattern.
"""
import io
import math
import struct
from fractions import Fraction

from mantissa import asm
from mantissa.fp import D, F, FORMATS, H, INEXACT, OVERFLOW, UNDERFLOW
from mantissa.machine import Cpu, console, run
from mantissa.memory import Memory

STRUCT_CODE = {"H": "e", "F": "f", "D": "d"}

# Round to nearest, ties to even.  At or beyond this magnitude, the exact
# value rounds up to the next power of two, which is an overflow: it is the
# midpoint between the largest finite value and that power of two.
_THRESHOLD = {
    H: Fraction(65520),
    F: Fraction(2 ** 25 - 1) * Fraction(2) ** 103,
    D: Fraction(2 ** 54 - 1) * Fraction(2) ** 970,
}

CANONICAL_NAN32 = 0x7FC00000


def from_pattern(fmt, bits):
    """The double holding the fmt value whose bit pattern is `bits`."""
    return struct.unpack("<" + STRUCT_CODE[fmt.name], bits.to_bytes(fmt.bits // 8, "little"))[0]


def reference(fmt, exact):
    """(value, flags) for the rational `exact` rounded once to fmt, ties to even.

    The path is Python's correctly rounded Fraction to double conversion, then
    the struct module's double to fmt conversion.  Narrowing from a double is
    innocuous here because a double has more than twice the significand and
    far more exponent range than a half or a float (Figueroa, 1995).
    The caller supplies the sign when `exact` is zero.
    """
    if exact == 0:
        return 0.0, 0
    sign = -1.0 if exact < 0 else 1.0
    magnitude = abs(exact)
    if magnitude >= _THRESHOLD[fmt]:
        return sign * math.inf, OVERFLOW | INEXACT
    packed = struct.pack("<" + STRUCT_CODE[fmt.name], float(magnitude))
    value = sign * from_pattern(fmt, int.from_bytes(packed, "little"))
    flags = INEXACT if Fraction(abs(value)) != magnitude else 0
    if flags and abs(value) < fmt.min_normal:
        flags |= UNDERFLOW
    return value, flags


def same(got, expected):
    """Equality that also distinguishes the sign of zero; any two NaNs are equal."""
    if got != got or expected != expected:
        return got != got and expected != expected
    return got == expected and math.copysign(1.0, got) == math.copysign(1.0, expected)


def canonical(value):
    """True when value is the quiet NaN with sign 0 and no payload."""
    return value != value and struct.unpack("<I", struct.pack("<f", value))[0] == CANONICAL_NAN32


def sample(fmt, rng):
    """A random value of the format: bit patterns mostly, with special values mixed in."""
    roll = rng.random()
    if roll < 0.03:
        return rng.choice([0.0, -0.0, math.inf, -math.inf])
    if roll < 0.10:
        return float(rng.randrange(1 << fmt.p))
    return from_pattern(fmt, rng.getrandbits(fmt.bits))


def run_source(source, steps=200_000, cells=4096):
    """Assemble and run a program; returns (output, RunResult, memory, image)."""
    image = asm.assemble(source)
    memory = Memory(cells)
    memory.write(image.cells)
    out = io.StringIO()
    result = run(Cpu(pc=image.entry), memory, out=console(out.write), max_steps=steps)
    return out.getvalue(), result, memory, image
