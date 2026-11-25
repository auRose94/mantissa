"""The number formats and the arithmetic unit.

Every function here is pure: values in, values out, never touching anything
else.  A Python float is an IEEE 754 binary64 and holds every half, float and
double exactly; quad values are the Quad and Qnan numbers at the bottom of
this file, exact by construction.  A value never records its format; the caller tracks that
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
from fractions import Fraction

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
    """One of the machine's data types: half, float, double, quad."""

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
Q = _format("Q", 4, 113, 16383, 128, 8)
object.__setattr__(Q, "min_normal", Fraction(1, 2) ** 16382)   # 2**-16382 underflows a float; exact instead
FORMATS = (H, F, D, Q)
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
    if fmt is Q:
        return q_from_bits(bits)
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
    if fmt is Q:
        return q_to_bits(x)
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
    if fmt is Q:
        return convert_into_q(x)[0]
    if fmt is D or x != x or x == 0.0 or x == INF or x == -INF:
        return x
    return _round_ratio(fmt, *x.as_integer_ratio())[0]


def convert(dst, x):
    """Move a value into the dst format, rounding once if that loses bits."""
    if isinstance(x, (Quad, Qnan)):
        return q_convert(dst, x)
    if dst is Q:
        return convert_into_q(x)
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
    if fmt is Q:
        return q_add(a, b)
    s, err = _two_sum(a, b)
    if s != s:  # inf - inf
        return NAN, INVALID
    if s == INF or s == -INF:
        return s, 0 if abs(a) == INF or abs(b) == INF else OVERFLOW | INEXACT
    return _finalize(fmt, *_narrow(fmt, s), err == 0.0)


def sub(fmt, a, b):
    if fmt is Q:
        return q_add(a, q_neg(b))
    return add(fmt, a, -b)


def mul(fmt, a, b):
    if a != a or b != b:
        return NAN, 0
    if fmt is Q:
        return q_mul(a, b)
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
    if fmt is Q:
        return q_div(a, b)

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
    if fmt is Q:
        return q_sqrt(a)

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
    if fmt is Q:
        return q_fma(a, b, c)
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
    if fmt is Q:
        return q_remainder(a, b)

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
    if fmt is Q:
        return x, 0

    """A register-to-register move; the only way to copy within one type."""
    return x, 0


def neg(fmt, x):
    if fmt is Q:
        return q_neg(x), 0

    return -x, 0


def fabs(fmt, x):
    if fmt is Q:
        return q_fabs(x), 0

    return math.fabs(x), 0


def copysign(fmt, x, y):
    if fmt is Q:
        return q_copysign(x, y), 0

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
    if fmt is Q:
        return q_integral(math.floor, x)
    return _integral(math.floor, x)


def ceil(fmt, x):
    if fmt is Q:
        return q_integral(math.ceil, x)
    return _integral(math.ceil, x)


def trunc(fmt, x):
    if fmt is Q:
        return q_integral(math.trunc, x)
    return _integral(math.trunc, x)


def round_even(fmt, x):
    if fmt is Q:
        return q_integral(round, x)
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
    if fmt is Q:
        return q_on_grid(_floor, x, step)
    return _on_grid(fmt, _floor, x, step)


def ceil_to(fmt, x, step):
    """The smallest multiple of |step| that is not below x."""
    if fmt is Q:
        return q_on_grid(_ceil, x, step)
    return _on_grid(fmt, _ceil, x, step)


def trunc_to(fmt, x, step):
    """The multiple of |step| nearest x on the way to zero."""
    if fmt is Q:
        return q_on_grid(_trunc, x, step)
    return _on_grid(fmt, _trunc, x, step)


def round_to(fmt, x, step):
    """The nearest multiple of |step| to x; a tie goes to the even multiple."""
    if fmt is Q:
        return q_on_grid(_half_even, x, step)
    return _on_grid(fmt, _half_even, x, step)


def fmin(fmt, a, b):
    if fmt is Q:
        return q_min(a, b)

    if a != a:
        return (NAN, 0) if b != b else (b, 0)
    if b != b or a < b or (a == b and math.copysign(1.0, a) < math.copysign(1.0, b)):
        return a, 0
    return b, 0


def fmax(fmt, a, b):
    if fmt is Q:
        return q_max(a, b)

    if a != a:
        return (NAN, 0) if b != b else (b, 0)
    if b != b or a > b or (a == b and math.copysign(1.0, a) > math.copysign(1.0, b)):
        return a, 0
    return b, 0


def lt(fmt, a, b):
    if fmt is Q:
        return Quad(0, 1 if q_order(a) < q_order(b) else 0), 0
    return (1.0 if a < b else 0.0), 0


def le(fmt, a, b):
    if fmt is Q:
        return Quad(0, 1 if q_order(a) <= q_order(b) else 0), 0
    return (1.0 if a <= b else 0.0), 0


def eq(fmt, a, b):
    if fmt is Q:
        return Quad(0, 1 if q_order(a) == q_order(b) else 0), 0
    return (1.0 if a == b else 0.0), 0


def compare(fmt, a, b):
    """Three-valued: -1.0, 0.0 or 1.0 as a is below, equal to or above b."""
    if fmt is Q:
        return q_compare(a, b)
    if a != a or b != b:
        return NAN, 0
    return float((a > b) - (a < b)), 0


def fmod(fmt, a, b):
    """What is left after trunc division: a - b * trunc(a / b).  Always exact."""
    if fmt is Q:
        return q_fmod(a, b)
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
    if fmt is Q:
        return q_scaleb(a=x, n=n)
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
    if fmt is Q:
        return q_logb(x)
    if x != x:
        return NAN, 0
    if x == 0.0:
        return -INF, DIVZERO
    if abs(x) == INF:
        return INF, 0
    return float(max(math.frexp(x)[1] - 1, fmt.emin)), 0


def nextafter(fmt, x, y):
    """The fmt value one step from x in the direction of y."""
    if fmt is Q:
        return q_nextafter(x, y)
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
    if fmt is Q:
        return q_classify(x)
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
    if dst is Q or isinstance(x, (Quad, Qnan)):
        quad_x = x if isinstance(x, (Quad, Qnan)) else convert_into_q(x)[0]
        return q_split(dst, quad_x)
    if x != x or abs(x) == INF:
        return x, x, 0
    high, _ = convert(dst, x)
    low, _ = convert(dst, x - high)
    return high, low, 0 if high + low == x else INEXACT


def join(fmt, high, low):
    """high + low, rounded once to fmt."""
    if fmt is Q or isinstance(high, (Quad, Qnan)) or isinstance(low, (Quad, Qnan)):
        high = high if isinstance(high, (Quad, Qnan)) else convert_into_q(high)[0]
        low = low if isinstance(low, (Quad, Qnan)) else convert_into_q(low)[0]
        return q_join(high, low)
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
    if isinstance(x, (Quad, Qnan)):
        return q_mantissa(x)
    if x != x:
        return NAN, NAN
    if x == 0.0 or abs(x) == INF:
        return x, 0.0
    m, e = math.frexp(x)
    return m, float(e)


# --- quad: binary128 -------------------------------------------------------------
#
# Quad values are not Python floats -- no float holds binary128.  A quad number
# is a Quad: (-1)**sign * m * 2**e, with m a whole number (0 for a zero,
# 2**112 <= m < 2**113 for a normal), inf marking an infinity, or a Qnan
# carrying its full 128-bit pattern.  Every operation computes the exact value
# and rounds once: quad does not need the double-precision shortcut the narrow
# formats use, because integer arithmetic is exact end to end.  The flags stay
# exact for the same reason.


@dataclass(frozen=True)
class Quad:
    sign: int = 0       # 0 positive, 1 negative; a Fraction cannot hold -0
    m: int = 0          # the significand as a whole number, 0 for zero and infinity
    e: int = 0          # the scale: value = (-1)**sign * m * 2**e
    inf: bool = False


@dataclass(frozen=True)
class Qnan:
    bits: int           # the full 128-bit pattern, sign, quiet bit and payload


QNAN_BITS = 0x7FFF8000000000000000000000000000
Q_CANON = Qnan(QNAN_BITS)          # what quad arithmetic answers with: quiet, payload 0


def is_qnan(x):
    return isinstance(x, Qnan)


def q_ratio(x):
    """(num, den), both positive, of a finite Quad's exact value."""
    if x.e >= 0:
        return x.m << x.e, 1
    return x.m, 1 << -x.e


def q_fraction(x):
    num, den = q_ratio(x)
    return Fraction(-num if x.sign else num, den)


def q_from_ratio(sign, num, den):
    """The Quad nearest the positive rational num/den, ties to even, with this
    sign for any zero it rounds to.  Same rule as _round_ratio, but the answer
    is a Quad: math.ldexp would lose quad to its own exponent range."""
    p, emin = Q.p, Q.emin
    est = num.bit_length() - den.bit_length() - 1
    top = est + 1
    if top >= 0:
        if num >= den << top:
            est = top
    elif num << -top >= den:
        est = top
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
    if est > Q.emax:
        return Quad(sign, 0, 0, True), flags | OVERFLOW | INEXACT
    if flags and est < emin and not q >> (p - 1):
        flags |= UNDERFLOW
    return Quad(sign, q, qe), flags


def _q_both(a, b):
    """Canonical NaN when either operand is one: quad arithmetic makes a new
    NaN, it never passes a payload through."""
    return Q_CANON, 0


def q_infs(a, b):
    return a.inf or b.inf


def q_add(a, b):
    if a.inf and b.inf and a.sign != b.sign:
        return Q_CANON, INVALID
    if a.inf:
        return a, 0
    if b.inf:
        return b, 0
    if a.m == 0 and b.m == 0:
        return Quad(a.sign & b.sign), 0     # (+0) + (-0) is +0, round to nearest
    na, da = q_ratio(a)
    nb, db = q_ratio(b)
    num = (-na if a.sign else na) * db + (-nb if b.sign else nb) * da
    if num == 0:
        return Quad(), 0                    # x + (-x) is +0, round to nearest
    return q_from_ratio(1 if num < 0 else 0, abs(num), da * db)


def q_mul(a, b):
    if a.inf and (b.inf or b.m == 0) or (a.m == 0 and b.inf):
        return Q_CANON, INVALID
    sign = a.sign ^ b.sign
    if a.inf or b.inf:
        return Quad(sign, 0, 0, True), 0
    if a.m == 0 or b.m == 0:
        return Quad(sign), 0
    na, da = q_ratio(a)
    nb, db = q_ratio(b)
    return q_from_ratio(sign, na * nb, da * db)


def q_div(a, b):
    if a.inf and b.inf or a.m == 0 and b.m == 0:
        return Q_CANON, INVALID
    sign = a.sign ^ b.sign
    if a.inf:
        return Quad(sign, 0, 0, True), 0
    if b.inf:
        return Quad(sign), 0
    if b.m == 0:
        return Quad(sign, 0, 0, True), DIVZERO
    if a.m == 0:
        return Quad(sign), 0
    na, da = q_ratio(a)
    nb, db = q_ratio(b)
    return q_from_ratio(sign, na * db, da * nb)


def q_sqrt(a):
    if a.inf and not a.sign:
        return a, 0
    if a.sign and not (a.m == 0 and a.sign and not a.inf):
        # -0 keeps its sign; every other negative is invalid
        return Q_CANON, INVALID
    if a.inf or a.m == 0:
        return a, 0
    na, da = q_ratio(a)
    # an integer square root with enough guard bits that the truncation
    # cannot flip the one rounding that follows
    k = Q.p + 8
    s = math.isqrt((na << (2 * k)) // da)
    value, flags = q_from_ratio(0, s, 1 << k)
    exact = not flags
    return value, flags


def q_fma(a, b, c):
    ab_sign = a.sign ^ b.sign
    if (a.inf and b.m == 0) or (a.m == 0 and b.inf):
        return Q_CANON, INVALID
    if a.inf or b.inf:
        if c.inf and ab_sign != c.sign:
            return Q_CANON, INVALID
        return Quad(ab_sign, 0, 0, True), 0
    if c.inf:
        return c, 0
    na, da = q_ratio(a)
    nb, db = q_ratio(b)
    nc, dc = q_ratio(c)
    num = (-1 if ab_sign else 1) * na * nb * dc + (-1 if c.sign else 1) * nc * da * db
    if num == 0:
        # an exact zero sum: -0 only when both addends were -0
        return Quad(ab_sign & c.sign), 0
    return q_from_ratio(1 if num < 0 else 0, abs(num), da * db * dc)


def q_remainder(a, b):
    """The IEEE remainder: always exact, so its flags are always clear."""
    if a.inf or b.m == 0:
        return Q_CANON, INVALID
    if b.inf or a.m == 0:
        return a, 0
    na, da = q_ratio(a)
    nb, db = q_ratio(b)
    # n = a/b, round half to even, as a whole number
    num, den = na * db, da * nb
    q, r = divmod(num, den)
    r2 = r + r
    if r2 > den or (r2 == den and q & 1):
        q += 1
    num_r = na * db - q * nb * da
    if num_r == 0:
        return Quad(a.sign), 0              # a zero remainder takes x's sign
    return q_from_ratio(1 if num_r < 0 else 0, abs(num_r), da * db)


def q_fmod(a, b):
    """MOD: what trunc division leaves over, always exact."""
    if a.inf or b.m == 0:
        return Q_CANON, INVALID
    if b.inf or a.m == 0:
        return a, 0
    na, da = q_ratio(a)
    nb, db = q_ratio(b)
    q = (na * db) // (da * nb)              # trunc: both sides are positive here
    num_r = na * db - q * nb * da
    if num_r == 0:
        return Quad(a.sign), 0
    return q_from_ratio(1 if num_r < 0 else 0, abs(num_r), da * db)


def q_order(x):
    """The exact signed value, for comparisons: inf sorts past every finite
    quad, whose largest is a whisker under 2**16385."""
    if x.inf:
        return -(2 ** 16385) if x.sign else 2 ** 16385
    num, den = q_ratio(x)
    return Fraction(-num if x.sign else num, den)


def q_min(a, b):
    if is_qnan(a):
        return (Q_CANON, 0) if is_qnan(b) else (b, 0)
    if is_qnan(b):
        return a, 0
    fa, fb = q_order(a), q_order(b)
    if fa < fb:
        return a, 0
    if fb < fa:
        return b, 0
    return (a if a.sign else b), 0          # MIN(+0, -0) is -0


def q_max(a, b):
    if is_qnan(a):
        return (Q_CANON, 0) if is_qnan(b) else (b, 0)
    if is_qnan(b):
        return a, 0
    fa, fb = q_order(a), q_order(b)
    if fa > fb:
        return a, 0
    if fb > fa:
        return b, 0
    return (b if a.sign else a), 0          # MAX(+0, -0) is +0


def q_compare(a, b):
    if is_qnan(a) or is_qnan(b):
        return Q_CANON, 0
    fa, fb = q_order(a), q_order(b)
    return Quad(0, (fa > fb) - (fa < fb)), 0


def q_neg(x):
    if is_qnan(x):
        return Qnan(x.bits ^ (1 << 127))
    return Quad(x.sign ^ 1, x.m, x.e, x.inf)


def q_fabs(x):
    if is_qnan(x):
        return Qnan(x.bits & ~(1 << 127))
    return Quad(0, x.m, x.e, x.inf)


def q_copysign(a, b):
    sign = b.sign if not is_qnan(b) else b.bits >> 127
    if is_qnan(a):
        return Qnan((a.bits & ~(1 << 127)) | sign << 127)
    return Quad(sign, a.m, a.e, a.inf)


def q_integral(choose, a):
    """floor, ceil, trunc or round-even of a finite quad: always exact, so a
    rounding that loses something says so, and a zero keeps x's sign."""
    if is_qnan(a):
        return Q_CANON, 0
    if a.inf:
        return a, 0
    num, den = q_ratio(a)
    signed = Fraction(-num if a.sign else num, den)
    value = choose(signed)
    if value == 0:
        return Quad(a.sign), INEXACT if signed != 0 else 0
    m = abs(int(value))
    return Quad(1 if value < 0 else 0, m, 0), 0 if signed == value else INEXACT


def q_on_grid(choose, x, step):
    """x moved to a multiple of |step|, the multiple picked by `choose`."""
    if is_qnan(x) or is_qnan(step):
        return Q_CANON, 0
    if step.m == 0 or step.inf or x.inf:
        return Q_CANON, INVALID
    if x.m == 0:
        return x, 0
    nx, dx = q_ratio(x)
    ns, ds = q_ratio(Quad(0, step.m, step.e))
    num, den = nx * ds, dx * ns
    q, r = divmod(num, den)
    q = choose(num, q, r, den)
    if q == 0:
        return Quad(x.sign), INEXACT
    value, flags = q_from_ratio(1 if q < 0 else 0, abs(q) * ns, ds)
    return value, flags | (INEXACT if r else 0)


def q_scaleb(a, n):
    """a * 2**n without any other rounding; n arrives as a double."""
    if is_qnan(a):
        return Q_CANON, 0
    if n != n:
        return Q_CANON, 0
    if a.m == 0 or a.inf:
        return a, 0
    if n == INF:
        n = 70000
    elif n == -INF:
        n = -70000
    else:
        n = int(n)
    n = max(-70000, min(70000, n))
    num, den = q_ratio(a)
    if n >= 0:
        return q_from_ratio(a.sign, num << n, den)
    return q_from_ratio(a.sign, num, den << -n)


def q_logb(a):
    if is_qnan(a):
        return Q_CANON, 0
    if a.m == 0:
        return Quad(1, 0, 0, True), DIVZERO
    if a.inf:
        return Quad(a.sign, 0, 0, True), 0
    e = a.e + Q.p - 1
    if a.m < 1 << (Q.p - 1):                # subnormal: counts as the smallest normal
        e = Q.emin
    return Quad(0, e), 0


def q_nextafter(a, b):
    if is_qnan(a) or is_qnan(b):
        return Q_CANON, 0
    fa, fb = q_order(a), q_order(b)
    if fa == fb:
        return b, 0
    if a.m == 0:
        # from a zero the next step is the smallest of the sign y points to
        return from_bits(Q, 1 if not b.sign else Q.sign | 1), 0
    bits = to_bits(Q, a) + (1 if (fb > fa) == (fa > 0) else -1)
    if bits & ~Q.sign == Q.inf:
        return from_bits(Q, bits), OVERFLOW | INEXACT
    return from_bits(Q, bits), 0


def q_classify(a):
    if is_qnan(a):
        return Quad(0, QNAN), 0
    if a.inf:
        return Quad(0, PINF if not a.sign else NINF), 0
    if a.m == 0:
        return Quad(0, NZERO if a.sign else PZERO), 0
    if a.sign:
        return Quad(0, NSUBNORMAL if a.m < 1 << (Q.p - 1) else NNORMAL), 0
    return Quad(0, PSUBNORMAL if a.m < 1 << (Q.p - 1) else PNORMAL), 0


def q_convert(dst, a):
    """A quad moved into dst: exact into quad, one rounding out of it."""
    if is_qnan(a):
        return NAN, 0
    if dst is Q:
        return a, 0
    if a.inf:
        return math.copysign(INF, -1.0 if a.sign else 1.0), 0
    num, den = q_ratio(a)
    value, flags = _round_ratio(dst, num, den)
    return math.copysign(value, -1.0 if a.sign else 1.0), flags


def convert_into_q(x):
    """A float moved into quad: every float is exact there."""
    if x != x:
        return Q_CANON, 0
    if x == 0.0:
        return Quad(1 if math.copysign(1.0, x) < 0 else 0), 0
    if x == INF:
        return Quad(0, 0, 0, True), 0
    if x == -INF:
        return Quad(1, 0, 0, True), 0
    num, den = x.as_integer_ratio()
    return q_from_ratio(1 if num < 0 else 0, abs(num), den)


def q_split(dst, x):
    """(high, low, flags): a quad x as two values of the dst format.  Into
    quad itself the split is exact: quad holds every double, so the low half
    is zero.  Out of quad, high rounds once and low carries the leavings."""
    if is_qnan(x):
        return Q_CANON, Q_CANON, 0
    if x.inf:
        return x, x, 0
    high, _ = q_convert(dst, x)
    if dst is Q:
        return high, Quad(), 0
    if isinstance(high, (Quad, Qnan)):
        rest = q_fraction(x) - q_fraction(high)
    else:
        rest = q_fraction(x) - Fraction(high)
    if rest == 0:
        low = Quad() if dst is Q else 0.0
        return high, low, 0
    if dst is Q:
        low, _ = convert_into_q(rest)
        exact = q_fraction(low) == rest
    else:
        low, flags = _round_ratio(dst, *rest.as_integer_ratio())
        exact = rest.as_integer_ratio()[0] * low.as_integer_ratio()[1] == \
                low.as_integer_ratio()[0] * rest.as_integer_ratio()[1] and not flags
    return high, low, 0 if exact else INEXACT


def q_join(high, low):
    """high + low, rounded once to quad."""
    if is_qnan(high) or is_qnan(low):
        return Q_CANON, 0
    if high.inf or low.inf:
        if high.inf and low.inf and high.sign != low.sign:
            return Q_CANON, INVALID
        return (high if high.inf else low), 0
    nh, dh = q_ratio(high)
    nl, dl = q_ratio(low)
    num = (-nh if high.sign else nh) * dl + (-nl if low.sign else nl) * dh
    if num == 0:
        return Quad(high.sign & low.sign), 0
    return q_from_ratio(1 if num < 0 else 0, abs(num), dh * dl)


def q_mantissa(a):
    """(m, e) with a = m * 2**e and 0.5 <= |m| < 1, both as quads."""
    if is_qnan(a):
        return Q_CANON, Q_CANON
    if a.inf or a.m == 0:
        return a, Quad()
    shift = a.m.bit_length()                # m * 2**(e - shift) lands in the first binade
    return Quad(a.sign, a.m, a.e - shift), Quad(0, shift)


def spell(x):
    """A quad spelled for a console: nan, or the exact decimal when it is
    short, else 34 significant digits -- binary128 carries about 34."""
    if is_qnan(x):
        return "nan"
    f = q_fraction(x)
    if f.denominator == 1 and len(str(f.numerator)) <= 40:
        return str(f.numerator)
    import decimal
    with decimal.localcontext() as ctx:
        ctx.prec = 34
        d = decimal.Decimal(f.numerator) / decimal.Decimal(f.denominator)
        text = format(d, "g")
    return text


def q_from_bits(bits):
    """The quad whose 128-bit encoding is `bits`."""
    sign, exp = bits >> 127, bits >> 112 & 0x7FFF
    frac = bits & ((1 << 112) - 1)
    if exp == 0x7FFF:
        if frac:
            return Qnan(bits)
        return Quad(sign, 0, 0, True)
    if exp == 0:
        return Quad(sign, frac, Q.emin - Q.p + 1)
    return Quad(sign, frac | 1 << 112, exp - 16383 - 112)


def q_to_bits(x):
    """The 128-bit encoding of a quad value."""
    if is_qnan(x):
        return x.bits
    sign = x.sign << 127
    if x.inf:
        return sign | Q.inf
    if x.m == 0:
        return sign
    if x.m >= 1 << (Q.p - 1):               # normal: the leading 1 is implicit
        e = x.e + 16383 + 112
        return sign | e << 112 | (x.m & ((1 << 112) - 1))
    return sign | x.m                       # subnormal: the scale is emin's
