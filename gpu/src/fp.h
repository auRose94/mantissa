// fp.h -- the arithmetic unit of the Mantissa-1.
//
// A port of mantissa/fp.py to C++ that compiles unchanged for the host and
// for an AMD GPU through HIP.  Every value is a double, as in the reference:
// a half or a float is the double that holds it, and a NaN carries whatever
// payload memory gave it.
//
// Values are made the way fp.py makes them: narrow results are computed in
// double and rounded once, straight from the double.  Flags are decided by
// exact integer arithmetic on the operands' bits, never from a second
// rounding, as in the reference.  Nothing here needs TwoSum.  Every number is
// written in canonical form, an odd integer times a power of two, and then
// each "was that exact" question is a comparison of two integers.
//
// Compile with -ffp-contract=off and without any fast-math.
#pragma once

#include <cstdint>
#include <cmath>

// M1_HD marks a function that HIP must build for the device as well as the host.
#if defined(__HIP__)
#define M1_HD __host__ __device__
#else
#define M1_HD
#endif

namespace m1 {

typedef unsigned __int128 u128;

constexpr uint64_t MASK52 = (1ULL << 52) - 1;
constexpr uint64_t SIGN64 = 0x8000000000000000ULL;
constexpr uint64_t QUIET_NAN = 0x7FF8000000000000ULL;

// Sticky IEEE 754 exception flags, the same bits as fp.py.
constexpr uint32_t F_INVALID = 1, F_DIVZERO = 2, F_OVERFLOW = 4, F_UNDERFLOW = 8, F_INEXACT = 16;

enum { HALF = 0, FLOAT = 1, DOUBLE = 2 };

struct Format {
    int p;       // significand bits, counting the implicit leading 1
    int emax;    // exponent of the largest binade
    int emin;    // exponent of the smallest normal binade
    int cells;   // 16-bit cells in memory
};
constexpr Format FORMATS[3] = {{11, 15, -14, 1}, {24, 127, -126, 2}, {53, 1023, -1022, 4}};

struct Result {
    double value;
    uint32_t flags;
};
struct Pair {  // SPLIT
    double high, low;
    uint32_t flags;
};

// Kinds of the operations that come in families.
enum { K_FLOOR = 0, K_CEIL = 1, K_TRUNC = 2, K_HALF_EVEN = 3 };

// --- bits -----------------------------------------------------------------------

M1_HD inline uint64_t bits_of(double d) {
    uint64_t u;
    __builtin_memcpy(&u, &d, 8);
    return u;
}
M1_HD inline double double_of(uint64_t u) {
    double d;
    __builtin_memcpy(&d, &u, 8);
    return d;
}
M1_HD inline bool nan_bits(uint64_t b) { return (b & ~SIGN64) > 0x7FF0000000000000ULL; }
M1_HD inline bool inf_bits(uint64_t b) { return (b & ~SIGN64) == 0x7FF0000000000000ULL; }
M1_HD inline bool is_nan(double x) { return nan_bits(bits_of(x)); }
M1_HD inline bool is_inf(double x) { return inf_bits(bits_of(x)); }
M1_HD inline bool sign_bit(double x) { return (bits_of(x) & SIGN64) != 0; }
M1_HD inline double m1_abs(double x) { return double_of(bits_of(x) & ~SIGN64); }

M1_HD inline double nan_value() { return double_of(QUIET_NAN); }
M1_HD inline double pos_inf() { return double_of(0x7FF0000000000000ULL); }
M1_HD inline double neg_inf() { return double_of(SIGN64 | 0x7FF0000000000000ULL); }
M1_HD inline double signed_zero(bool negative) { return double_of(negative ? SIGN64 : 0); }
M1_HD inline double signed_inf(bool negative) {
    return double_of(0x7FF0000000000000ULL | (negative ? SIGN64 : 0));
}

// |x| is below the smallest normal of fmt.  x is finite and not a NaN.
M1_HD inline bool below_min_normal(int fmt, double x) {
    return (bits_of(x) & ~SIGN64) < (uint64_t)(1023 + FORMATS[fmt].emin) << 52;
}

M1_HD inline int bitlen64(uint64_t x) { return 64 - __builtin_clzll(x); }  // x != 0
M1_HD inline int tzcnt64(uint64_t x) { return __builtin_ctzll(x); }        // x != 0
M1_HD inline int bitlen128(u128 x) {                                        // x != 0
    uint64_t hi = (uint64_t)(x >> 64);
    return hi ? 64 + bitlen64(hi) : bitlen64((uint64_t)x);
}
// A finite positive double is a whole number.
M1_HD inline bool whole_bits(uint64_t mag) {
    uint32_t e = (uint32_t)(mag >> 52);
    if (e < 1023) return mag == 0;
    if (e >= 1075) return true;
    return ((mag & MASK52) << (e - 1011)) == 0;  // the low 1075 - e bits of the significand are zero
}

// --- cells and bit patterns ---------------------------------------------------------

// The value whose `fmt` encoding is `bits`, as a double.  A NaN is widened by
// hand so that its payload and its quiet bit travel with it.
M1_HD inline double from_bits(int fmt, uint64_t bits) {
    if (fmt == DOUBLE) return double_of(bits);
    uint64_t sign, frac;
    uint32_t m, e;
    int field, len;
    if (fmt == FLOAT) {
        uint32_t b32 = (uint32_t)bits;
        sign = (uint64_t)(b32 >> 31) << 63;
        e = b32 >> 23 & 0xFF;
        m = b32 & 0x7FFFFF;
        if (e == 0xFF) {
            if (m == 0) return double_of(sign | 0x7FF0000000000000ULL);
            return double_of(sign | 0x7FF0000000000000ULL | (uint64_t)m << 29);
        }
        if (e > 0) return double_of(sign | (uint64_t)(e + 896) << 52 | (uint64_t)m << 29);
        if (m == 0) return double_of(sign);
        len = 32 - __builtin_clz(m);                      // m * 2**-149 = 1.xxx * 2**(len - 150)
        field = len + 873;
        frac = (uint64_t)((m << (24 - len)) & 0x7FFFFF) << 29;
    } else {
        uint32_t b16 = (uint32_t)bits & 0xFFFF;
        sign = (uint64_t)(b16 >> 15) << 63;
        e = b16 >> 10 & 0x1F;
        m = b16 & 0x3FF;
        if (e == 0x1F) {
            if (m == 0) return double_of(sign | 0x7FF0000000000000ULL);
            return double_of(sign | 0x7FF0000000000000ULL | (uint64_t)m << 42);
        }
        if (e > 0) return double_of(sign | (uint64_t)(e + 1008) << 52 | (uint64_t)m << 42);
        if (m == 0) return double_of(sign);
        len = 32 - __builtin_clz(m);                      // m * 2**-24 = 1.xxx * 2**(len - 25)
        field = len + 998;
        frac = (uint64_t)((m << (11 - len)) & 0x3FF) << 42;
    }
    return double_of(sign | (uint64_t)field << 52 | frac);
}

// The `fmt` encoding of `x`.  `x` must be a NaN or already a value of that format.
M1_HD inline uint64_t to_bits(int fmt, double x) {
    uint64_t b = bits_of(x);
    if (nan_bits(b)) {
        uint64_t sign = b & SIGN64;
        if (fmt == DOUBLE) return b;
        if (fmt == FLOAT) return sign >> 32 | 0x7F800000 | ((b >> 29) & 0x7FFFFF);
        return sign >> 48 | 0x7C00 | ((b >> 42) & 0x3FF);
    }
    if (fmt == DOUBLE) return b;
    const Format f = FORMATS[fmt];
    uint64_t out;
    uint32_t e = (uint32_t)(b >> 52) & 0x7FF;
    uint64_t m = b & MASK52;
    if (e == 0x7FF) {
        out = fmt == FLOAT ? 0x7F800000ULL : 0x7C00ULL;
    } else if ((b << 1) == 0) {
        out = 0;
    } else {
        int field = (int)e - 1023 + f.emax;  // a subnormal counts from the smallest normal
        if (field >= 1) {
            out = (uint64_t)field << (f.p - 1) | m >> (53 - f.p);
        } else {
            // a whole multiple of 2**(emin - p + 1); what does not fit is cut off, as fp.py's int() does
            out = ((1ULL << 52) | m) >> (52 + f.emin - f.p + 1024 - (int)e);
        }
    }
    return (b & SIGN64) ? ((b & SIGN64) >> (64 - f.cells * 16)) | out : out;
}

// --- rounding -----------------------------------------------------------------------

// q * 2**qe exactly, q nonzero and at most 53 significant bits, in range.
M1_HD inline double dyad_to_double(uint64_t q, int qe, bool neg) {
    int bq = bitlen64(q);
    int E = qe + bq - 1;
    uint64_t b = E >= -1022 ? (uint64_t)(E + 1023) << 52 | ((q << (53 - bq)) & MASK52) : q;
    return double_of(neg ? b | SIGN64 : b);
}

// Round plus or minus M * 2**exp to fmt, nearest, ties to even, with fp.py's
// _round_ratio flags.  M may be any 128-bit integer and exp any exponent.
M1_HD inline Result round_dyadic(int fmt, u128 M, int exp, bool neg) {
    const Format f = FORMATS[fmt];
    if (M == 0) return {signed_zero(neg), 0};
    int E = bitlen128(M) - 1 + exp;
    if (E > f.emax) return {signed_inf(neg), F_OVERFLOW | F_INEXACT};
    int qe = E - f.p + 1;
    if (qe < f.emin - f.p + 1) qe = f.emin - f.p + 1;
    int shift = qe - exp;
    uint64_t q;
    bool inexact = false;
    if (shift <= 0) {
        q = (uint64_t)(M << (unsigned)-shift);
    } else if (shift < 128) {
        int s = shift;
        q = (uint64_t)(M >> s);
        bool guard = ((M >> (s - 1)) & 1) != 0;
        bool sticky = (M & (((u128)1 << (s - 1)) - 1)) != 0;
        inexact = guard || sticky;
        if (guard && (sticky || (q & 1))) {
            q++;
            if (q == (1ULL << f.p) && ++E > f.emax) return {signed_inf(neg), F_OVERFLOW | F_INEXACT};  // carried a binade up
        }
    } else {
        q = 0;  // below a quarter of the least subnormal
        inexact = true;
    }
    uint32_t flags = inexact ? F_INEXACT : 0;
    if (q == 0) return {signed_zero(neg), flags | F_UNDERFLOW};
    if (inexact && q < (1ULL << (f.p - 1))) flags |= F_UNDERFLOW;  // subnormal and not exact
    return {dyad_to_double(q, qe, neg), flags};
}

// Round a finite double to fmt, once, straight from its bits.
M1_HD inline Result narrow(int fmt, double s) {
    if (fmt == DOUBLE) return {s, 0};
    uint64_t b = bits_of(s);
    bool neg = (b & SIGN64) != 0;
    if ((b << 1) == 0) return {s, 0};
    uint32_t e = (uint32_t)(b >> 52) & 0x7FF;
    uint64_t m = b & MASK52;
    if (fmt == FLOAT) {
        float f = (float)s;  // IEEE 754: nearest even, gradual underflow, overflow to infinity
        double v = (double)f;
        uint32_t flags = 0;
        if (bits_of(v) != b) flags |= F_INEXACT;
        if (is_inf(v)) return {v, F_OVERFLOW | F_INEXACT};
        if ((flags & F_INEXACT) && below_min_normal(FLOAT, v)) flags |= F_UNDERFLOW;
        return {v, flags};
    }
    // half: no instruction goes straight from a double, so it is done by hand
    uint32_t q, field;
    bool inexact;
    if (e >= 1009) {  // a normal half
        field = e - 1008;
        q = (1u << 10) | (uint32_t)(m >> 42);
        uint32_t guard = (uint32_t)(m >> 41) & 1u;
        bool sticky = (m & ((1ULL << 41) - 1)) != 0;
        inexact = guard != 0 || sticky;
        if (guard && (sticky || (q & 1u))) {
            q++;
            if (q == (1u << 11)) {  // carried out of the significand: one binade up
                q = 1u << 10;
                field++;
            }
        }
        if (field > 30) return {signed_inf(neg), F_OVERFLOW | F_INEXACT};
    } else if (e >= 997) {  // the subnormal range of a half, down to a quarter of its least subnormal
        uint32_t shift = 1051 - e;  // 43..54
        uint64_t sig = (1ULL << 52) | m;
        q = (uint32_t)(sig >> shift);
        uint32_t guard = (uint32_t)(sig >> (shift - 1)) & 1u;
        bool sticky = (sig & ((1ULL << (shift - 1)) - 1)) != 0;
        inexact = guard != 0 || sticky;
        field = 0;
        if (guard && (sticky || (q & 1u))) {
            q++;
            if (q == (1u << 10)) {  // rounded up to the smallest normal
                field = 1;
                q = 0;
            }
        }
    } else {  // less than a quarter of the least subnormal
        q = 0;
        field = 0;
        inexact = true;
    }
    uint32_t flags = inexact ? (field == 0 ? (F_INEXACT | F_UNDERFLOW) : F_INEXACT) : 0;
    return {from_bits(HALF, (neg ? 0x8000ULL : 0ULL) | (uint64_t)field << 10 | (q & 0x3FF)), flags};
}

// Move a value into the dst format, rounding once if that loses bits (fp.convert).
M1_HD inline Result convert(int dst, double x) {
    if (is_nan(x)) return {nan_value(), 0};
    if (dst == DOUBLE || x == 0.0 || is_inf(x)) return {x, 0};
    return narrow(dst, x);
}

// --- canonical form and exact flags ---------------------------------------------------

// value = odd * 2**exp, with odd odd; {0, 0} for zero.
struct Dyad {
    uint64_t odd;
    int exp;
};

M1_HD inline Dyad canon(double x) {
    uint64_t b = bits_of(x) & ~SIGN64;
    uint64_t sig;
    int exp;
    if ((b >> 52) == 0) {
        sig = b & MASK52;
        exp = -1074;
    } else {
        sig = (1ULL << 52) | (b & MASK52);
        exp = (int)(b >> 52) - 1075;
    }
    if (sig == 0) return {0, 0};
    int t = tzcnt64(sig);
    return {sig >> t, exp + t};
}

// The flags of rounding (+/-)mx * 2**ex (+/-)my * 2**ey to fmt, exactly.  `value` is
// the rounded result, finite; it only says whether that is tiny when the sum is too
// wide to form.  When the terms are 76 binades or more apart the sum is an odd
// integer with at least 128 significant bits and can never be exact.
M1_HD inline uint32_t sum_flags(int fmt, u128 mx, int ex, bool xneg, u128 my, int ey, bool yneg, double value) {
    if (mx == 0) return round_dyadic(fmt, my, ey, yneg).flags;
    if (my == 0) return round_dyadic(fmt, mx, ex, xneg).flags;
    int t0 = ex < ey ? ex : ey;
    int L = bitlen128(mx) + (ex - t0);
    int Ly = bitlen128(my) + (ey - t0);
    if (Ly > L) L = Ly;
    if (L > 127) return F_INEXACT | (is_inf(value) ? F_OVERFLOW : below_min_normal(fmt, value) ? F_UNDERFLOW : 0);
    u128 X = (u128)mx << (ex - t0);
    u128 Y = (u128)my << (ey - t0);
    u128 T;
    bool neg;
    if (xneg == yneg) {
        T = X + Y;
        neg = xneg;
    } else if (X >= Y) {
        T = X - Y;
        neg = xneg;
    } else {
        T = Y - X;
        neg = yneg;
    }
    return round_dyadic(fmt, T, t0, neg).flags;
}

// The flags of a / b delivered as `value`, a nonzero: INEXACT unless value * b is a exactly.
M1_HD inline uint32_t quotient_flags(int fmt, double a, double value, double b) {
    Dyad ca = canon(a), cv = canon(value), cb = canon(b);
    if (cv.odd != 0 && (u128)cv.odd * cb.odd == ca.odd && cv.exp + cb.exp == ca.exp) return 0;
    return F_INEXACT | (below_min_normal(fmt, value) ? F_UNDERFLOW : 0);
}

// --- arithmetic ---------------------------------------------------------------------

M1_HD inline Result op_add(int fmt, double a, double b) {
    if (is_nan(a) || is_nan(b)) return {nan_value(), 0};
    double s = a + b;
    uint64_t sb = bits_of(s);
    if (nan_bits(sb)) return {nan_value(), F_INVALID};  // inf - inf
    if (inf_bits(sb)) return {s, (is_inf(a) || is_inf(b)) ? 0 : (F_OVERFLOW | F_INEXACT)};
    double v = narrow(fmt, s).value;
    Dyad ca = canon(a), cb = canon(b);
    return {v, sum_flags(fmt, ca.odd, ca.exp, sign_bit(a), cb.odd, cb.exp, sign_bit(b), v)};
}

M1_HD inline Result op_sub(int fmt, double a, double b) { return op_add(fmt, a, -b); }

M1_HD inline Result op_mul(int fmt, double a, double b) {
    if (is_nan(a) || is_nan(b)) return {nan_value(), 0};
    double p = a * b;
    uint64_t pb = bits_of(p);
    if (nan_bits(pb)) return {nan_value(), F_INVALID};  // 0 * inf
    if (inf_bits(pb)) return {p, (is_inf(a) || is_inf(b)) ? 0 : (F_OVERFLOW | F_INEXACT)};
    double v = narrow(fmt, p).value;
    if (a == 0.0 || b == 0.0) return {v, 0};  // an exact signed zero
    Dyad ca = canon(a), cb = canon(b);
    return {v, round_dyadic(fmt, (u128)ca.odd * cb.odd, ca.exp + cb.exp, sign_bit(a) != sign_bit(b)).flags};
}

M1_HD inline Result op_div(int fmt, double a, double b) {
    if (is_nan(a) || is_nan(b)) return {nan_value(), 0};
    if (b == 0.0) {
        if (a == 0.0 || is_inf(a)) return {nan_value(), F_INVALID};
        return {signed_inf(sign_bit(a) != sign_bit(b)), F_DIVZERO};
    }
    double q = a / b;
    uint64_t qb = bits_of(q);
    if (nan_bits(qb)) return {nan_value(), F_INVALID};  // inf / inf
    if (inf_bits(qb)) return {q, is_inf(a) ? 0 : (F_OVERFLOW | F_INEXACT)};
    if (a == 0.0) return {q, 0};  // an exact signed zero
    Result v = narrow(fmt, q);
    if (is_inf(v.value)) return v;  // overflowed on the way down to the format
    return {v.value, quotient_flags(fmt, a, v.value, b)};
}

M1_HD inline Result op_sqrt(int fmt, double a) {
    if (is_nan(a)) return {nan_value(), 0};
    if (a == 0.0) return {a, 0};  // sqrt(+-0) is +-0
    if (sign_bit(a)) return {nan_value(), F_INVALID};
    double s = sqrt(a);
    if (is_inf(s)) return {s, 0};
    double v = narrow(fmt, s).value;
    Dyad ca = canon(a), cv = canon(v);
    bool exact = (u128)cv.odd * cv.odd == ca.odd && 2 * cv.exp == ca.exp;
    return {v, exact ? 0 : (F_INEXACT | (below_min_normal(fmt, v) ? F_UNDERFLOW : 0))};
}

// a * b + c with a single rounding.
M1_HD inline Result op_fma(int fmt, double a, double b, double c) {
    if (is_nan(a) || is_nan(b) || is_nan(c)) return {nan_value(), 0};
    bool ai = is_inf(a), bi = is_inf(b), ci = is_inf(c);
    if ((ai && b == 0.0) || (a == 0.0 && bi)) return {nan_value(), F_INVALID};
    uint64_t psign = (bits_of(a) ^ bits_of(b)) & SIGN64;
    if (ai || bi) {
        if (ci && psign != (bits_of(c) & SIGN64)) return {nan_value(), F_INVALID};
        return {double_of(psign | 0x7FF0000000000000ULL), 0};
    }
    if (ci) return {c, 0};
    if (a == 0.0 || b == 0.0) {  // the exact value is c, or a zero: -0 only if product and c both are
        if (c != 0.0) return {c, 0};
        return {double_of(psign & bits_of(c)), 0};
    }
    double s = fma(a, b, c);
    if (is_inf(s)) return {s, F_OVERFLOW | F_INEXACT};
    Result v = narrow(fmt, s);
    Dyad ca = canon(a), cb = canon(b), cc = canon(c);
    return {v.value, sum_flags(fmt, (u128)ca.odd * cb.odd, ca.exp + cb.exp, psign != 0, cc.odd, cc.exp,
                               sign_bit(c), v.value)};
}

// REM (kind 0): a - n * b with n the whole number nearest a / b, ties to even.
// MOD (kind 1): a - b * trunc(a / b).  Both are always exact.
M1_HD inline Result op_rem_mod(int fmt, int kind, double a, double b) {
    if (is_nan(a) || is_nan(b)) return {nan_value(), 0};
    if (is_inf(a) || b == 0.0) return {nan_value(), F_INVALID};
    if (is_inf(b) || a == 0.0) return {a, 0};
    Dyad ca = canon(a), cb = canon(b);
    int d = ca.exp - cb.exp;  // |a| / |b| = (ca.odd / cb.odd) * 2**d
    u128 s, B = 0, mag;
    int lo;
    bool up = false;
    if (d >= 0) {
        lo = cb.exp;
        uint64_t qparity;
        if (d <= 75) {
            u128 A = (u128)ca.odd << d;
            B = cb.odd;
            u128 q = A / B;
            s = A - q * B;
            qparity = (uint64_t)q & 1;
        } else {
            // ca.odd * 2**d mod cb.odd, one doubling at a time; the last carry is the parity of the quotient
            uint64_t r = ca.odd % cb.odd;
            qparity = 0;
            for (int i = 0; i < d; i++) {
                uint64_t r2 = r << 1;
                if (r2 >= cb.odd) {
                    r2 -= cb.odd;
                    qparity = 1;
                } else {
                    qparity = 0;
                }
                r = r2;
            }
            s = r;
            B = cb.odd;
        }
        if (kind == 0 && (s << 1) >= B && ((s << 1) > B || qparity)) up = true;
        mag = up ? B - s : s;
    } else {
        int j = -d;
        lo = ca.exp;
        s = ca.odd;
        if (j + bitlen64(cb.odd) <= 127) {  // otherwise the step is so much larger that the quotient is 0
            B = (u128)cb.odd << j;
            u128 q = s / B;
            s = s - q * B;
            if (kind == 0 && (s << 1) >= B && ((s << 1) > B || ((uint64_t)q & 1))) up = true;
        }
        mag = up ? B - s : s;
    }
    bool neg = sign_bit(a) != up;  // rounding up makes the remainder negative
    if (mag == 0) neg = sign_bit(a);
    // fp.py returns math.remainder or math.fmod of the doubles.  That is exact as a double, and for
    // a half or a float it can fall between the values of its own format; it is not rounded again.
    return {round_dyadic(DOUBLE, mag, lo, neg).value, 0};
}

M1_HD inline Result op_min(int fmt, double a, double b) {
    (void)fmt;  // fp.fmin takes the format and does not look at it
    if (is_nan(a)) return is_nan(b) ? Result{nan_value(), 0} : Result{b, 0};
    if (is_nan(b) || a < b || (a == b && sign_bit(a))) return {a, 0};
    return {b, 0};
}

M1_HD inline Result op_max(int fmt, double a, double b) {
    (void)fmt;
    if (is_nan(a)) return is_nan(b) ? Result{nan_value(), 0} : Result{b, 0};
    if (is_nan(b) || a > b || (a == b && !sign_bit(a))) return {a, 0};
    return {b, 0};
}

// x * 2**trunc(n) without any other rounding.  Anything past +-2200 can only overflow or vanish.
M1_HD inline Result op_scalb(int fmt, double x, double n) {
    if (is_nan(x) || is_nan(n)) return {nan_value(), 0};
    if (x == 0.0 || is_inf(x)) return {x, 0};
    int nn;
    if (is_inf(n)) {
        nn = n > 0.0 ? 2200 : -2200;
    } else {
        if (n > 2200.0) nn = 2200;
        else if (n < -2200.0) nn = -2200;
        else nn = (int)n;  // truncates toward zero
    }
    Dyad cx = canon(x);
    return round_dyadic(fmt, cx.odd, cx.exp + nn, sign_bit(x));
}

// The binary exponent of x as a number; a subnormal counts as the smallest normal.
M1_HD inline Result op_logb(int fmt, double x) {
    if (is_nan(x)) return {nan_value(), 0};
    if (x == 0.0) return {neg_inf(), F_DIVZERO};
    if (is_inf(x)) return {pos_inf(), 0};
    Dyad cx = canon(x);
    int E = bitlen64(cx.odd) - 1 + cx.exp;
    if (E < FORMATS[fmt].emin) E = FORMATS[fmt].emin;
    return {(double)E, 0};
}

// The fmt value one step from x in the direction of y.
M1_HD inline Result op_next(int fmt, double x, double y) {
    if (is_nan(x) || is_nan(y)) return {nan_value(), 0};
    if (x == y) return {y, 0};
    const Format f = FORMATS[fmt];
    uint64_t bits;
    if (x == 0.0) {
        bits = y > 0.0 ? 1 : ((uint64_t)1 << (f.cells * 16 - 1)) | 1;
    } else {
        bits = to_bits(fmt, x) + (((y > x) == (x > 0.0)) ? 1 : -1);
    }
    uint64_t mag = bits & ~((uint64_t)1 << (f.cells * 16 - 1));
    bool overflow = mag == (fmt == HALF ? 0x7C00ULL : fmt == FLOAT ? 0x7F800000ULL : 0x7FF0000000000000ULL);
    return {from_bits(fmt, bits), overflow ? (F_OVERFLOW | F_INEXACT) : 0};
}

// Whether the floor of x / |step| moves up one: floor, ceil, chop or round to the even multiple.
M1_HD inline bool grid_round_up(int kind, bool rnz, bool neg, bool gt_half, bool tie, bool qodd) {
    if (kind == K_FLOOR) return rnz && neg;
    if (kind == K_CEIL) return rnz && !neg;
    if (kind == K_HALF_EVEN) return gt_half || (tie && qodd);
    return false;
}

// x moved to a multiple of |step|; the multiple is chosen by `kind`.
M1_HD inline Result op_grid(int fmt, int kind, double x, double step) {
    if (is_nan(x) || is_nan(step)) return {nan_value(), 0};
    if (step == 0.0 || is_inf(step) || is_inf(x)) return {nan_value(), F_INVALID};
    if (x == 0.0) return {x, 0};
    bool neg = sign_bit(x);
    Dyad cx = canon(x), cs = canon(step);
    // x / |step|, correctly rounded.  From 2**(p + 2) upwards the step is at most a quarter of
    // the format's spacing at x, so both neighbouring multiples round to x itself.
    double rho = m1_abs(x) / m1_abs(step);
    double thresh = double_of((uint64_t)(1025 + FORMATS[fmt].p) << 52);
    if (!(rho < thresh)) {
        bool exact = false;
        if (is_inf(rho) || whole_bits(bits_of(rho))) {
            // x / |step| is a whole number exactly when the odd part of |step|'s significand
            // divides the odd part of x's, with no power of two left over below
            exact = cx.odd % cs.odd == 0 && cx.exp >= cs.exp;
        }
        uint32_t flags = exact ? 0 : F_INEXACT;
        if (!exact && below_min_normal(fmt, x)) flags |= F_UNDERFLOW;
        return {x, flags};
    }
    // Here the quotient is below 2**(p + 3) and everything fits in 128 bits.
    int k = cx.exp - cs.exp;
    u128 V = 0;
    int lo;
    u128 q = 0, m = 0;
    bool rnz = true, gt_half = false, tie = false;
    bool have = false;
    if (k >= 0) {
        u128 N = (u128)cx.odd << k;  // below 2**109
        u128 D = cs.odd;
        lo = cs.exp;
        q = N / D;
        u128 r = N - q * D;
        rnz = r != 0;
        gt_half = (r << 1) > D;
        tie = (r << 1) == D;
        m = grid_round_up(kind, rnz, neg, gt_half, tie, ((uint64_t)q & 1) != 0) ? q + 1 : q;
        V = m * D;      // at most N + D
        have = m != 0;
    } else {
        int j = -k;
        lo = cx.exp;
        if (j + bitlen64(cs.odd) <= 127) {  // otherwise |step| is so much larger than x that the quotient is 0
            u128 D = (u128)cs.odd << j;
            q = cx.odd / D;
            u128 r = cx.odd - q * D;
            rnz = r != 0;
            gt_half = (r << 1) > D;
            tie = (r << 1) == D;
            m = grid_round_up(kind, rnz, neg, gt_half, tie, ((uint64_t)q & 1) != 0) ? q + 1 : q;
            if (m != 0) {  // m * D is at most cx.odd + D, so this fits
                V = ((u128)((uint64_t)m * cs.odd)) << j;
                have = true;
            }
        } else if (grid_round_up(kind, true, neg, false, false, false)) {
            // |step| is so much larger than x that only 0 and 1 are candidates, and 1 is chosen:
            // the multiple is |step| itself, too wide to form at x's scale
            Result res = round_dyadic(fmt, cs.odd, cs.exp, neg);
            res.flags |= F_INEXACT;
            return res;
        }
    }
    if (!have) return {signed_zero(neg), F_INEXACT};  // the multiple chosen is zero
    Result res = round_dyadic(fmt, V, lo, neg);
    if (rnz) res.flags |= F_INEXACT;
    return res;
}

// Whole numbers: FLOOR, CEIL, CHOP and ROUND.
M1_HD inline Result op_integral(int kind, double x) {
    if (is_nan(x)) return {nan_value(), 0};
    uint64_t b = bits_of(x);
    uint64_t mag = b & ~SIGN64;
    if (mag >= 0x4340000000000000ULL || mag == 0) return {x, 0};  // already whole, or a signed zero
    bool neg = (b & SIGN64) != 0;
    uint64_t out;
    if (mag < 0x3FF0000000000000ULL) {  // |x| < 1
        bool up = (kind == K_CEIL && !neg) || (kind == K_FLOOR && neg) || (kind == K_HALF_EVEN && mag > 0x3FE0000000000000ULL);  // exactly one half is a tie, and goes to the even 0
        out = up ? 0x3FF0000000000000ULL : 0;
    } else {
        uint32_t e = (uint32_t)(mag >> 52);
        uint32_t shift = 1075 - e;  // fractional bits present, 0..52
        uint64_t sig = (1ULL << 52) | (mag & MASK52);
        uint64_t frac = sig & ((1ULL << shift) - 1);
        if (frac == 0) return {x, 0};
        uint64_t q = sig >> shift;
        bool up;
        if (kind == K_FLOOR) up = neg;
        else if (kind == K_CEIL) up = !neg;
        else if (kind == K_TRUNC) up = false;
        else {
            uint64_t half = 1ULL << (shift - 1);
            up = frac > half || (frac == half && (q & 1));
        }
        if (up) q++;
        int bq = bitlen64(q);  // q has 53 - shift bits, or one more after a carry
        out = (uint64_t)(e - 53 + shift + bq) << 52 | ((q << (53 - bq)) & MASK52);
    }
    return {double_of(neg ? out | SIGN64 : out), F_INEXACT};
}

// Which of the ten IEEE 754 kinds of value x is; see the codes in fp.py.
M1_HD inline Result op_classify(int fmt, double x) {
    uint64_t mag = bits_of(x) & ~SIGN64;
    double code;
    if (nan_bits(mag)) return {1.0, 0};  // a NaN is a value like any other here: its class is 1
    if (inf_bits(mag)) code = sign_bit(x) ? 2.0 : 9.0;
    else if (mag == 0) code = sign_bit(x) ? 5.0 : 6.0;
    else if (sign_bit(x)) code = below_min_normal(fmt, x) ? 4.0 : 3.0;
    else code = below_min_normal(fmt, x) ? 7.0 : 8.0;
    return {code, 0};
}

// --- casts that keep what a plain cast drops --------------------------------------

// x as two values of the dst format: high is x rounded to dst, low is what that
// rounding left out, rounded to dst in turn.  The flags are clear exactly when high + low is x.
M1_HD inline Pair op_split(int dst, double x) {
    if (is_nan(x) || is_inf(x)) return {x, x, 0};
    double high = convert(dst, x).value;
    double low = convert(dst, x - high).value;
    return {high, low, (high + low == x) ? 0 : F_INEXACT};
}

// high + low, rounded once to fmt.
M1_HD inline Result op_join(int fmt, double high, double low) {
    if (is_nan(high) || is_nan(low)) return {nan_value(), 0};
    bool hi = is_inf(high), li = is_inf(low);
    if (hi || li) {
        if (hi && li && (bits_of(high) ^ bits_of(low)) == SIGN64) return {nan_value(), F_INVALID};
        return {hi ? high : low, 0};
    }
    Dyad ch = canon(high), cl = canon(low);
    if (ch.odd == 0 && cl.odd == 0) return {high + low, 0};  // IEEE rules decide the sign of an exact zero
    if (ch.odd == 0) return round_dyadic(fmt, cl.odd, cl.exp, sign_bit(low));
    if (cl.odd == 0) return round_dyadic(fmt, ch.odd, ch.exp, sign_bit(high));
    int t0 = ch.exp < cl.exp ? ch.exp : cl.exp;
    int L = bitlen64(ch.odd) + (ch.exp - t0);
    int Ll = bitlen64(cl.odd) + (cl.exp - t0);
    if (L <= 127 && Ll <= 127) {
        u128 X = (u128)ch.odd << (ch.exp - t0);
        u128 Y = (u128)cl.odd << (cl.exp - t0);
        u128 T;
        bool neg;
        if (sign_bit(high) == sign_bit(low)) {
            T = X + Y;
            neg = sign_bit(high);
        } else if (X >= Y) {
            T = X - Y;
            neg = sign_bit(high);
        } else {
            T = Y - X;
            neg = sign_bit(low);
        }
        if (T == 0) return {high + low, 0};
        return round_dyadic(fmt, T, t0, neg);
    }
    // The terms are 76 binades or more apart, too wide to form.  The smaller one can only
    // matter where the larger is exactly on a rounding boundary of fmt: then the sum
    // falls on whichever side of it the smaller term lies.
    double s = high + low;
    if (is_inf(s)) return {s, F_OVERFLOW | F_INEXACT};
    Result r = narrow(fmt, s);
    double bb = s - high;
    double err = (high - (s - bb)) + (low - bb);  // the exact leavings of the double sum
    if (err == 0.0) return r;
    uint32_t flags = r.flags | F_INEXACT;
    double value = r.value;
    Dyad cs = canon(s);
    int p = FORMATS[fmt].p;
    int Es = bitlen64(cs.odd) - 1 + cs.exp;
    int qe = Es - p + 1;
    if (qe < FORMATS[fmt].emin - p + 1) qe = FORMATS[fmt].emin - p + 1;
    if (cs.exp == qe - 1) {  // s is exactly on a midpoint, or on the overflow threshold
        uint64_t fb = to_bits(fmt, value);
        if (err > 0.0) {
            fb += 1;
        } else if (fb != 0) {
            fb -= 1;
        }
        value = from_bits(fmt, (bits_of(s) & SIGN64) >> (64 - FORMATS[fmt].cells * 16) | fb);
        flags = F_INEXACT | (is_inf(value) ? F_OVERFLOW : 0) | (below_min_normal(fmt, value) ? F_UNDERFLOW : 0);
    } else if (below_min_normal(fmt, value)) {
        flags |= F_UNDERFLOW;
    }
    return {value, flags};
}

// (m, e) with x = m * 2**e and 0.5 <= |m| < 1, plus the scale, as a plain number.
M1_HD inline void op_mantissa(double x, double& m, double& e) {
    if (is_nan(x)) {
        m = e = nan_value();
        return;
    }
    uint64_t b = bits_of(x);
    if ((b & ~SIGN64) == 0 || inf_bits(b)) {
        m = x;
        e = 0.0;
        return;
    }
    uint64_t mag = b & ~SIGN64;
    uint32_t field = (uint32_t)(mag >> 52);
    uint64_t out;
    if (field > 0) {
        out = (b & (SIGN64 | MASK52)) | (uint64_t)1022 << 52;
        e = (double)((int)field - 1022);
    } else {
        int len = bitlen64(mag);
        out = (b & SIGN64) | (uint64_t)1022 << 52 | ((mag << (53 - len)) & MASK52);
        e = (double)(len - 1074);
    }
    m = double_of(out);
}

// --- sign bits, comparisons ---------------------------------------------------------

// 1.0 or 0.0; false if an operand is NaN.
M1_HD inline Result op_lt(int /*fmt*/, double a, double b) { return {a < b ? 1.0 : 0.0, 0}; }
M1_HD inline Result op_le(int /*fmt*/, double a, double b) { return {a <= b ? 1.0 : 0.0, 0}; }
M1_HD inline Result op_eq(int /*fmt*/, double a, double b) { return {a == b ? 1.0 : 0.0, 0}; }

// Three-valued: -1.0, 0.0 or 1.0 as a is below, equal to or above b; NaN if an operand is NaN.
M1_HD inline Result op_cmp(int /*fmt*/, double a, double b) {
    if (is_nan(a) || is_nan(b)) return {nan_value(), 0};
    return {(double)((a > b) - (a < b)), 0};
}

M1_HD inline double op_neg(double x) { return double_of(bits_of(x) ^ SIGN64); }
M1_HD inline double op_abs(double x) { return double_of(bits_of(x) & ~SIGN64); }
M1_HD inline double op_copysign(double x, double y) {
    return double_of((bits_of(x) & ~SIGN64) | (bits_of(y) & SIGN64));
}

}  // namespace m1
