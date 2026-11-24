"""The number formats and the arithmetic unit.

Every function here is pure: Python floats in, Python floats out.  A Python
float is an IEEE 754 binary64, so it can hold every value of all three machine
formats exactly.  A value never records its format; the caller tracks that
statically, through the register file it lives in or the suffix of the
instruction that produced it.

Arithmetic returns ``(value, flags)``, where flags is a mask of sticky IEEE 754
exception bits.  The flags are exact: "inexact" is decided from the true
mathematical result by integer arithmetic, never by a second rounding.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

# Sticky IEEE 754 exception flags.
INVALID = 1
DIVZERO = 2
OVERFLOW = 4
UNDERFLOW = 8
INEXACT = 16

FLAG_NAMES = (
    (INVALID, "invalid"),
    (DIVZERO, "divzero"),
    (OVERFLOW, "overflow"),
    (UNDERFLOW, "underflow"),
    (INEXACT, "inexact"),
)


def flag_names(flags):
    return [name for bit, name in FLAG_NAMES if flags & bit]


@dataclass(frozen=True)
class Format:
    """One of the machine's three, and only, data types."""

    name: str
    code: int          # 2-bit code in an instruction word; 0 is the system format
    p: int             # significand bits, counting the implicit leading 1
    emax: int          # exponent of the largest binade
    bits: int          # storage width
    cells: int         # addressable 16-bit cells occupied in memory
    emin: int          # exponent of the smallest normal binade
    min_normal: float  # 2.0 ** emin
    inf: int           # bit pattern of +infinity
    sign: int          # the sign bit, on its own


def _format(name, code, p, emax, bits, cells):
    emin = 1 - emax
    return Format(name, code, p, emax, bits, cells, emin, 2.0 ** emin,
                  ((1 << (bits - p)) - 1) << (p - 1), 1 << (bits - 1))


H = _format("H", 1, 11, 15, 16, 1)
F = _format("F", 2, 24, 127, 32, 2)
D = _format("D", 3, 53, 1023, 64, 4)
FORMATS = (H, F, D)
FORMAT_BY_CODE = {f.code: f for f in FORMATS}

# The quiet NaN every arithmetic operation produces: sign 0, payload 0.
NAN = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000))[0]
INF = math.inf

# Codes returned by classify().
SNAN, QNAN, NINF, NNORMAL, NSUBNORMAL, NZERO, PZERO, PSUBNORMAL, PNORMAL, PINF = range(10)

_D = struct.Struct("<d")
_Q = struct.Struct("<Q")
_F = struct.Struct("<f")
_I = struct.Struct("<I")
_TWO_53 = 9007199254740992.0  # beyond this every double is already whole


# --- cells and bit patterns ----------------------------------------------------


def from_bits(fmt, bits):
    """The value whose `fmt` encoding is `bits`, as a Python float."""
    if fmt is D:
        return _D.unpack(bits.to_bytes(8, "little"))[0]
    # NaNs are widened by hand: the host's own conversion is allowed to quiet
    # a signaling NaN, and a copy through a register must not change a bit.
    if fmt is F:
        if bits & 0x7F800000 == 0x7F800000 and bits & 0x7FFFFF:
            wide = (bits & 0x80000000) << 32 | 0x7FF0000000000000 | (bits & 0x7FFFFF) << 29
            return _D.unpack(_Q.pack(wide))[0]
        return _F.unpack(bits.to_bytes(4, "little"))[0]
    exp = bits >> 10 & 0x1F
    frac = bits & 0x3FF
    if exp == 0x1F:
        if frac == 0:
            return math.copysign(INF, -1.0 if bits & 0x8000 else 1.0)
        wide = (bits & 0x8000) << 48 | 0x7FF0000000000000 | frac << 42  # payload and quiet bit travel together
        return _D.unpack(_Q.pack(wide))[0]
    if exp == 0:
        value = math.ldexp(frac, -24)
    else:
        value = math.ldexp(frac + 1024, exp - 25)
    return -value if bits & 0x8000 else value


def to_bits(fmt, x):
    """The `fmt` encoding of `x`.  `x` must already be a value of that format."""
    if x != x:
        high = _Q.unpack(_D.pack(x))[0]
        sign = high & 0x8000000000000000
        if fmt is D:
            return high
        if fmt is F:
            return sign >> 32 | 0x7F800000 | high >> 29 & 0x7FFFFF
        return sign >> 48 | 0x7C00 | high >> 42 & 0x3FF
    if fmt is D:
        return _Q.unpack(_D.pack(x))[0]
    if fmt is F:
        return _I.unpack(_F.pack(x))[0]
    sign = fmt.sign if math.copysign(1.0, x) < 0 else 0
    if x == INF or x == -INF:
        return sign | fmt.inf
    if x == 0.0:
        return sign
    m, e = math.frexp(x)  # x = m * 2**e with 0.5 <= |m| < 1
    if e < -13:           # subnormal: a whole multiple of 2**-24
        return sign | int(math.ldexp(abs(x), 24))
    return sign | (e + 14) << 10 | abs(int(m * 2048.0)) - 1024


def pack(fmt, x):
    """The cells `x` occupies, most significant cell first."""
    bits = to_bits(fmt, x)
    n = fmt.cells
    return tuple(bits >> 16 * (n - 1 - i) & 0xFFFF for i in range(n))


def unpack(fmt, cells):
    bits = 0
    for cell in cells:
        bits = bits << 16 | cell
    return from_bits(fmt, bits)


# --- rounding --------------------------------------------------------------------


def _round_ratio(fmt, num, den):
    """Round the exact rational num/den to fmt, ties to even.

    Requires num != 0 and den > 0.  The flags can carry INEXACT, OVERFLOW and
    UNDERFLOW; tininess is judged after rounding.
    """
    p, emin = fmt.p, fmt.emin
    if num < 0:
        num = -num
        sign = -1.0
    else:
        sign = 1.0
    # 2**est <= num/den < 2**(est + 2), so the binary exponent is est or est + 1
    est = num.bit_length() - den.bit_length() - 1
    top = est + 1
    if top >= 0:
        if num >= den << top:
            est = top
    elif num << -top >= den:
        est = top
    # the quantum is 2**(est - p + 1), but never finer than the subnormal grid
    qe = max(est - p + 1, emin - p + 1)
    if qe >= 0:
        den <<= qe
        q, r = divmod(num, den)
    else:
        q, r = divmod(num << -qe, den)
    flags = INEXACT if r else 0
    if r:
        r += r
        if r > den or (r == den and q & 1):
            q += 1
            if q >> p:
                est += 1
    if est > fmt.emax:
        return math.copysign(INF, sign), flags | OVERFLOW | INEXACT
    if flags and est < emin and not q >> (p - 1):
        flags |= UNDERFLOW
    return math.ldexp(float(q), qe) * sign, flags


def round_literal(fmt, x):
    """The fmt value nearest to the double x: the assembler's rule for literals."""
    if fmt is D or x != x or x == 0.0 or x == INF or x == -INF:
        return x
    return _round_ratio(fmt, *x.as_integer_ratio())[0]


def convert(dst, x):
    """Move a value into the dst format, rounding once if that loses bits."""
    if x != x:
        return NAN, 0
    if dst is D or x == 0.0 or x == INF or x == -INF:
        return x, 0
    return _round_ratio(dst, *x.as_integer_ratio())


# --- arithmetic ------------------------------------------------------------------
#
# Narrow results are computed in double precision and rounded once more to the
# target format.  That is innocuous: a double has more than 2p + 2 significand
# bits and ample exponent range for both binary16 and binary32 (Figueroa's
# theorem), so the result equals rounding the exact value directly.  The exact
# integer tests below are what keep the flags honest.


def _two_sum(a, b):
    """RN(a + b), and the exact error a + b - RN(a + b).  The sum must be finite."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _narrow(fmt, s):
    """Round a finite double to fmt.  Only the rounding flags are reported."""
    if fmt is D or s == 0.0:
        return s, 0
    return _round_ratio(fmt, *s.as_integer_ratio())


def _finalize(fmt, value, flags, exact):
    if not exact:
        flags |= INEXACT
    if flags & INEXACT and -fmt.min_normal < value < fmt.min_normal:
        flags |= UNDERFLOW
    return value, flags


def _product_exact(p, a, b):
    np_, dp = p.as_integer_ratio()
    na, da = a.as_integer_ratio()
    nb, db = b.as_integer_ratio()
    return np_ * da * db == na * nb * dp


def _quotient_exact(a, q, b):
    nq, dq = q.as_integer_ratio()
    na, da = a.as_integer_ratio()
    nb, db = b.as_integer_ratio()
    return nq * da * nb == na * db * dq


def _root_exact(a, s):
    ns, ds = s.as_integer_ratio()
    na, da = a.as_integer_ratio()
    return ns * ns * da == na * ds * ds


def add(fmt, a, b):
    if a != a or b != b:
        return NAN, 0
    s, err = _two_sum(a, b)
    if s != s:  # inf - inf
        return NAN, INVALID
    if s == INF or s == -INF:
        return s, 0 if abs(a) == INF or abs(b) == INF else OVERFLOW | INEXACT
    return _finalize(fmt, *_narrow(fmt, s), err == 0.0)


def sub(fmt, a, b):
    return add(fmt, a, -b)


def mul(fmt, a, b):
    if a != a or b != b:
        return NAN, 0
    p = a * b
    if p != p:  # 0 * inf
        return NAN, INVALID
    if p == INF or p == -INF:
        return p, 0 if abs(a) == INF or abs(b) == INF else OVERFLOW | INEXACT
    # Two binary32 significands make at most 48 bits, so narrow products are
    # exact in a double and only the final rounding can lose information.
    exact = True if fmt is not D else _product_exact(p, a, b)
    return _finalize(fmt, *_narrow(fmt, p), exact)


def div(fmt, a, b):
    if a != a or b != b:
        return NAN, 0
    if b == 0.0:
        if a == 0.0 or abs(a) == INF:
            return NAN, INVALID
        return math.copysign(INF, a) * math.copysign(1.0, b), DIVZERO
    q = a / b
    if q != q:  # inf / inf
        return NAN, INVALID
    if q == INF or q == -INF:  # exact only when the dividend was already infinite
        return q, 0 if abs(a) == INF else OVERFLOW | INEXACT
    if a == 0.0:  # exact signed zero
        return q, 0
    v, fl = _narrow(fmt, q)
    if v == INF or v == -INF:  # overflowed on the way down to the format
        return v, fl
    return _finalize(fmt, v, fl, _quotient_exact(a, v, b))


def sqrt(fmt, a):
    if a != a:
        return NAN, 0
    if a == 0.0:  # sqrt(+-0) is +-0
        return a, 0
    if a < 0:
        return NAN, INVALID
    s = math.sqrt(a)
    if s == INF:
        return s, 0
    v, fl = _narrow(fmt, s)
    return _finalize(fmt, v, fl, _root_exact(a, v))


def fma(fmt, a, b, c):
    """a * b + c with a single rounding."""
    if a != a or b != b or c != c:
        return NAN, 0
    a_inf, b_inf, c_inf = abs(a) == INF, abs(b) == INF, abs(c) == INF
    if (a_inf and b == 0.0) or (a == 0.0 and b_inf):
        return NAN, INVALID
    if a_inf or b_inf:
        if c_inf and math.copysign(1.0, a) * math.copysign(1.0, b) != math.copysign(1.0, c):
            return NAN, INVALID
        return math.copysign(INF, a) * math.copysign(1.0, b), 0
    if c_inf:
        return c, 0
    if fmt is D:
        # a double product needs up to 106 bits, so round the exact rational
        na, da = a.as_integer_ratio()
        nb, db = b.as_integer_ratio()
        nc, dc = c.as_integer_ratio()
        num = na * nb * dc + nc * da * db
        value, fl = _round_ratio(D, num, da * db * dc)
        if num == 0:  # an exact zero gets its sign from IEEE rules, not from a ratio
            value = a * b + c
        return value, fl
    p = a * b  # exact: two narrow significands fit in a double
    s, err = _two_sum(p, c)
    if s == INF or s == -INF:
        return s, OVERFLOW | INEXACT
    return _finalize(fmt, *_narrow(fmt, s), err == 0.0)


def remainder(fmt, a, b):
    """The IEEE remainder: a - n * b with n the whole number nearest a / b."""
    if a != a or b != b:
        return NAN, 0
    if abs(a) == INF or b == 0.0:
        return NAN, INVALID
    if abs(b) == INF or a == 0.0:
        return a, 0
    return math.remainder(a, b), 0  # always exact


# --- sign bits, integral rounding, order ----------------------------------------


def copy(fmt, x):
    """A register-to-register move; the only way to copy within one type."""
    return x, 0


def neg(fmt, x):
    return -x, 0


def fabs(fmt, x):
    return math.fabs(x), 0


def copysign(fmt, x, y):
    return math.copysign(x, y), 0


def _integral(rounder, x):
    if x != x:
        return NAN, 0
    if abs(x) >= _TWO_53:  # already whole, or infinite
        return x, 0
    value = float(rounder(x))
    if value == 0.0:
        value = math.copysign(0.0, x)
    return value, 0 if value == x else INEXACT


def floor(fmt, x):
    return _integral(math.floor, x)


def ceil(fmt, x):
    return _integral(math.ceil, x)


def trunc(fmt, x):
    return _integral(math.trunc, x)


def round_even(fmt, x):
    return _integral(round, x)


def _on_grid(fmt, choose, x, step):
    """x moved to a multiple of |step|, the multiple picked by `choose`.

    The flags describe the one rounding of that multiple to fmt.
    """
    if x != x or step != step:
        return NAN, 0
    if step == 0.0 or abs(step) == INF or abs(x) == INF:
        return NAN, INVALID
    if x == 0.0:
        return x, 0
    nx, dx = x.as_integer_ratio()
    ns, ds = abs(step).as_integer_ratio()
    num, den = nx * ds, dx * ns  # x / |step|, exactly, with den > 0
    q, r = divmod(num, den)      # the floor of the quotient, and what is left
    q = choose(num, q, r, den)
    if q == 0:
        return math.copysign(0.0, x), INEXACT
    value, flags = _round_ratio(fmt, q * ns, ds)
    return value, flags | (INEXACT if r else 0)


def _floor(num, q, r, den):
    return q


def _ceil(num, q, r, den):
    return q + (1 if r else 0)


def _trunc(num, q, r, den):
    return q + (1 if num < 0 and r else 0)


def _half_even(num, q, r, den):
    return q + (1 if 2 * r > den or (2 * r == den and q & 1) else 0)


def floor_to(fmt, x, step):
    """The largest multiple of |step| that is not above x."""
    return _on_grid(fmt, _floor, x, step)


def ceil_to(fmt, x, step):
    """The smallest multiple of |step| that is not below x."""
    return _on_grid(fmt, _ceil, x, step)


def trunc_to(fmt, x, step):
    """The multiple of |step| nearest x on the way to zero."""
    return _on_grid(fmt, _trunc, x, step)


def round_to(fmt, x, step):
    """The nearest multiple of |step| to x; a tie goes to the even multiple."""
    return _on_grid(fmt, _half_even, x, step)


def fmin(fmt, a, b):
    if a != a:
        return (NAN, 0) if b != b else (b, 0)
    if b != b or a < b or (a == b and math.copysign(1.0, a) < math.copysign(1.0, b)):
        return a, 0
    return b, 0


def fmax(fmt, a, b):
    if a != a:
        return (NAN, 0) if b != b else (b, 0)
    if b != b or a > b or (a == b and math.copysign(1.0, a) > math.copysign(1.0, b)):
        return a, 0
    return b, 0


def lt(fmt, a, b):
    return (1.0 if a < b else 0.0), 0


def le(fmt, a, b):
    return (1.0 if a <= b else 0.0), 0


def eq(fmt, a, b):
    return (1.0 if a == b else 0.0), 0


def compare(fmt, a, b):
    """Three-valued: -1.0, 0.0 or 1.0 as a is below, equal to or above b."""
    if a != a or b != b:
        return NAN, 0
    return float((a > b) - (a < b)), 0


def fmod(fmt, a, b):
    """What is left after trunc division: a - b * trunc(a / b).  Always exact."""
    if a != a or b != b:
        return NAN, 0
    if abs(a) == INF or b == 0.0:
        return NAN, INVALID
    if abs(b) == INF or a == 0.0:
        return a, 0
    return math.fmod(a, b), 0


# --- exponent surgery ------------------------------------------------------------


def scaleb(fmt, x, n):
    """x * 2**n without any other rounding.  `n` is truncated to a whole number."""
    if x != x or n != n:
        return NAN, 0
    if x == 0.0 or abs(x) == INF:
        return x, 0
    if n == INF:  # anything past +-2200 can only overflow or vanish
        n = 2200
    elif n == -INF:
        n = -2200
    else:
        n = int(n)  # truncates toward zero
    if n > 2200:
        n = 2200
    elif n < -2200:
        n = -2200
    num, den = x.as_integer_ratio()
    if n >= 0:
        return _round_ratio(fmt, num << n, den)
    return _round_ratio(fmt, num, den << -n)


def logb(fmt, x):
    """The exponent of x as a number; a subnormal counts as the smallest normal."""
    if x != x:
        return NAN, 0
    if x == 0.0:
        return -INF, DIVZERO
    if abs(x) == INF:
        return INF, 0
    return float(max(math.frexp(x)[1] - 1, fmt.emin)), 0


def nextafter(fmt, x, y):
    """The fmt value one step from x in the direction of y."""
    if x != x or y != y:
        return NAN, 0
    if x == y:
        return y, 0
    if x == 0.0:
        return from_bits(fmt, 1 if y > 0.0 else fmt.sign | 1), 0
    bits = to_bits(fmt, x) + (1 if (y > x) == (x > 0.0) else -1)
    flags = OVERFLOW | INEXACT if bits & ~fmt.sign == fmt.inf else 0
    return from_bits(fmt, bits), flags


def classify(fmt, x):
    """Which of the ten IEEE 754 kinds of value x is; see the code constants."""
    if x != x:
        return float(QNAN), 0
    if abs(x) == INF:
        return float(PINF if x > 0 else NINF), 0
    if x == 0.0:
        return float(PZERO if math.copysign(1.0, x) > 0 else NZERO), 0
    if x < 0:
        return float(NSUBNORMAL if x > -fmt.min_normal else NNORMAL), 0
    return float(PSUBNORMAL if x < fmt.min_normal else PNORMAL), 0


# --- casts that keep what a plain cast would drop -----------------------------------


def split(dst, x):
    """(high, low, flags): x as two values of the dst format.

    high is x rounded to dst; low is what that rounding left out, rounded to
    dst in turn.  The flags are clear exactly when high + low is x.
    """
    if x != x or abs(x) == INF:
        return x, x, 0
    high, _ = convert(dst, x)
    low, _ = convert(dst, x - high)
    return high, low, 0 if high + low == x else INEXACT


def join(fmt, high, low):
    """high + low, rounded once to fmt."""
    if high != high or low != low:
        return NAN, 0
    if abs(high) == INF or abs(low) == INF:
        if high == -low and abs(high) == INF:
            return NAN, INVALID
        return high if abs(high) == INF else low, 0
    nh, dh = high.as_integer_ratio()
    nl, dl = low.as_integer_ratio()
    num = nh * dl + nl * dh
    if num == 0:  # IEEE rules decide the sign of an exact zero
        return high + low, 0
    return _round_ratio(fmt, num, dh * dl)


def mantissa(x):
    """(m, e) with x = m * 2**e and 0.5 <= |m| < 1: the number scaled into the
    first binade, where every format has its full precision, plus the scale."""
    if x != x:
        return NAN, NAN
    if x == 0.0 or abs(x) == INF:
        return x, 0.0
    m, e = math.frexp(x)
    return m, float(e)
