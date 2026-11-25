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
#include <cstdio>
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

enum { HALF = 0, FLOAT = 1, DOUBLE = 2, QUAD = 3 };

struct Format {
    int p;       // significand bits, counting the implicit leading 1
    int emax;    // exponent of the largest binade
    int emin;    // exponent of the smallest normal binade
    int cells;   // 16-bit cells in memory
};
constexpr Format FORMATS[4] = {{11, 15, -14, 1}, {24, 127, -126, 2}, {53, 1023, -1022, 4},
                               {113, 16383, -16382, 8}};

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

// --- quad: binary128 -----------------------------------------------------------------
//
// A quad register holds the 128-bit pattern, as memory does.  Arithmetic
// decodes the pattern into the three-double shape of the value: the 113-bit
// significand splits into 53, 53 and what remains -- hi, mid, lo, every piece
// exact -- with the binary exponent carried beside it.  Products go through
// TwoProd on the machine's FMA (p plus the exact residual, both kept), and
// every sum is accumulated as exact dyadic pieces, m * 2**e with m a whole
// number, so an addition, a product or an fma is exact end to end and owes
// one rounding.  That rounding runs through an integer window: the top p + 2
// bits of the magnitude in a 128-bit integer, everything below folded into a
// signed tail that breaks ties at an exact midpoint.  A sum that cancels
// clean through its first window re-windows what is left, down as far as the
// value actually reaches; the dyad pieces ride out the whole quad exponent
// range, where bare doubles would underflow.  Division, square root, the
// remainders and the other ratio-shaped operations work on the integer
// significands directly -- u128 long division, a bit-at-a-time root, modular
// arithmetic -- because their exact answers reach far past any expansion, and
// the reference rounds them once from the exact value.
//
// Quirks of mantissa/fp.py are mirrored on purpose, the oracle being code: a
// rounding carry that lands the significand exactly on 2**113 leaves it there
// at the old quantum (q_from_ratio bumps est, not the quantum); MANT and the
// integral roundings hand Quad(m, 0) to q_to_bits, whose subnormal branch
// prints m itself as the bits; and LD of a quad reads only the four high
// cells, because memory._bits packs four.  Operations the reference cannot
// run at all -- its quad arithmetic raises on a NaN payload -- answer the
// sensible IEEE thing here: a canonical quiet NaN, payload 0.

constexpr int Q_P = 113, Q_EMAX = 16383, Q_EMIN = -16382, Q_CELLS = 8;
constexpr int Q_EMIN_P1 = Q_EMIN - Q_P + 1;  // -16494, the subnormal quantum
constexpr uint64_t QNAN_HI = 0x7FFF800000000000ULL, QNAN_LO = 0;
constexpr u128 Q_M112 = (u128)1 << 112, Q_M113 = (u128)1 << 113;
typedef __int128 s128;

struct QRes {
    uint64_t hi, lo;
    uint32_t flags;
};

M1_HD inline QRes q_qnan() { return QRes{QNAN_HI, QNAN_LO, 0}; }

// A decoded pattern: kind 0 zero, 1 finite, 2 infinity, 3 NaN.  A NaN loses
// its payload here; NEG, ABS and CPSGN work on the raw bits instead.
struct Qdec {
    int kind, sign;
    u128 m;
    int e;
};

M1_HD inline Qdec q_decode(uint64_t hi, uint64_t lo) {
    Qdec d;
    d.sign = (int)(hi >> 63);
    uint64_t mag = hi & ~SIGN64;
    uint32_t field = (uint32_t)(mag >> 48);
    u128 frac = ((u128)(mag & 0x0000FFFFFFFFFFFFULL) << 64) | lo;
    if (field == 0x7FFF) {
        d.kind = frac ? 3 : 2;
        d.m = 0;
        d.e = 0;
        return d;
    }
    if (field == 0) {
        d.kind = frac ? 1 : 0;
        d.m = frac;
        d.e = Q_EMIN_P1;
        return d;
    }
    d.kind = 1;
    d.m = frac | Q_M112;
    d.e = (int)field - 16383 - 112;
    return d;
}

// q_to_bits of fp.py, faithful to the non-canonical cases: m under 2**112
// prints as itself (the subnormal branch), m over prints with the field e
// carries -- which is how fp.py prints its 2**113 significand and its huge
// integral results.
M1_HD inline QRes q_encode_raw(int sign, u128 m, int e, uint32_t flags = 0) {
    QRes r;
    r.flags = flags;
    r.hi = (uint64_t)sign << 63;
    r.lo = 0;
    if (m == 0) return r;
    u128 bits;
    if ((s128)m >= (s128)Q_M112) {
        uint64_t field = (uint64_t)(e + 16383 + 112);
        bits = ((u128)field << 112) | (m & (Q_M112 - 1));
    } else {
        bits = m;  // subnormal print, or a negative significand as fp.py's | leaves it
    }
    if (sign) bits |= (u128)1 << 127;
    r.hi = (uint64_t)(bits >> 64);
    r.lo = (uint64_t)bits;
    return r;
}

M1_HD inline QRes q_inf_bits(int sign, uint32_t flags) {
    QRes r = q_encode_raw(sign, 0, 0);
    r.hi |= 0x7FFF000000000000ULL;
    r.flags = flags;
    return r;
}

M1_HD inline double q_dyad_value(u128 q, int e, bool neg) {
    if (q == 0) return signed_zero(neg);  // dyad_to_double wants a nonzero q
    return dyad_to_double((uint64_t)q, e, neg);
}

// An exact value m * 2**e, m at most 2**113, in the canonical shape
// q_from_ratio produces: significand in [2**112, 2**113), subnormals on the
// quantum.  The value must sit on the quad grid for the subnormal fold.
M1_HD inline QRes q_encode_canon(int sign, u128 m, int e, uint32_t flags = 0) {
    if (m == 0) return q_encode_raw(sign, 0, 0, flags);
    int bl = bitlen128(m);
    if (bl < Q_P) {
        int up = Q_P - bl;
        m <<= (unsigned)up;
        e -= up;
    }
    if (e < Q_EMIN_P1) {
        unsigned t = (unsigned)(Q_EMIN_P1 - e);
        m = t < 127 ? (m >> t) : 0;
        e = Q_EMIN_P1;
    }
    return q_encode_raw(sign, m, e, flags);
}

// --- exact dyadic pieces ---------------------------------------------------------------

// One piece of an exact sum: +-m * 2**e with m > 0.  Chunks carry at most 53
// bits; TwoProd residuals 53; a product piece is split before storing.
struct Qdy {
    u128 m;
    int e;
    bool neg;
};

constexpr int Q_ACC = 30;
struct Qacc {
    Qdy c[Q_ACC];
    int n;
};

M1_HD inline Qdy qdy_of(double v) {  // v finite and nonzero
    Dyad d = canon(v);
    Qdy q;
    q.m = (u128)d.odd;
    q.e = d.exp;
    q.neg = sign_bit(v);
    return q;
}

M1_HD inline void qacc_init(Qacc& a) { a.n = 0; }
M1_HD inline void qacc_push(Qacc& a, u128 m, int e, bool neg) {
    if (m == 0 || a.n >= Q_ACC) return;
    a.c[a.n].m = m;
    a.c[a.n].e = e;
    a.c[a.n].neg = neg;
    a.n++;
}

M1_HD inline void q_two_prod(double a, double b, double& p, double& err) {
    p = a * b;
    err = fma(a, b, -p);
}

// A finite nonzero quad as its three-double chunks, placed exactly.
M1_HD inline void q_chunks(const Qdec& d, Qacc& acc, bool neg) {
    int bl = bitlen128(d.m);
    int s1 = bl - 53;
    if (s1 > 0) {
        qacc_push(acc, d.m >> (unsigned)s1, d.e + s1, neg);
        u128 r1 = d.m & (((u128)1 << (unsigned)s1) - 1);
        if (r1) {
            int bl2 = bitlen128(r1);
            int s2 = bl2 - 53;
            if (s2 > 0) {
                qacc_push(acc, r1 >> (unsigned)s2, d.e + s2, neg);
                u128 r2 = r1 & (((u128)1 << (unsigned)s2) - 1);
                if (r2) qacc_push(acc, r2, d.e, neg);
            } else {
                qacc_push(acc, r1, d.e, neg);
            }
        }
    } else {
        qacc_push(acc, d.m, d.e, neg);
    }
}

// The three doubles of an operand, materialized for TwoProd.
M1_HD inline void q_doubles(const Qdec& d, double out[3]) {
    out[0] = out[1] = out[2] = 0.0;
    int bl = bitlen128(d.m);
    int s1 = bl - 53;
    if (s1 > 0) {
        out[0] = dyad_to_double((uint64_t)(d.m >> (unsigned)s1), d.e + s1, d.sign);
        u128 r1 = d.m & (((u128)1 << (unsigned)s1) - 1);
        if (r1) {
            int bl2 = bitlen128(r1);
            int s2 = bl2 - 53;
            if (s2 > 0) {
                out[1] = dyad_to_double((uint64_t)(r1 >> (unsigned)s2), d.e + s2, d.sign);
                u128 r2 = r1 & (((u128)1 << (unsigned)s2) - 1);
                if (r2) out[2] = dyad_to_double((uint64_t)r2, d.e, d.sign);
            } else {
                out[1] = dyad_to_double((uint64_t)r1, d.e, d.sign);
            }
        }
    } else {
        out[0] = dyad_to_double((uint64_t)d.m, d.e, d.sign);
    }
}

// A significand as up to three whole-number doubles -- the pure 53-bit chunks
// of m, no scale -- so TwoProd between two operands stays in double range at
// any quad scale.  The chunks' offsets within m come back through off[].
M1_HD inline void q_offset_doubles(const Qdec& d, double out[3], int off[3]) {
    out[0] = out[1] = out[2] = 0.0;
    int bl = bitlen128(d.m);
    int s1 = bl - 53;
    if (s1 > 0) {
        out[0] = (double)(uint64_t)(d.m >> (unsigned)s1);
        off[0] = s1;
        u128 r1 = d.m & (((u128)1 << (unsigned)s1) - 1);
        if (r1) {
            int bl2 = bitlen128(r1);
            int s2 = bl2 - 53;
            if (s2 > 0) {
                out[1] = (double)(uint64_t)(r1 >> (unsigned)s2);
                off[1] = s2;
                u128 r2 = r1 & (((u128)1 << (unsigned)s2) - 1);
                if (r2) {
                    out[2] = (double)(uint64_t)r2;
                    off[2] = 0;
                }
            } else {
                out[1] = (double)(uint64_t)r1;
                off[1] = 0;
            }
        }
    } else {
        out[0] = (double)(uint64_t)d.m;
        off[0] = 0;
    }
}

M1_HD inline void q_prod_pieces(const Qdec& a, const Qdec& b, Qacc& acc) {
    // the nine TwoProd pairs of the three-double shapes, exact: each product
    // is a whole number of at most 106 bits, kept whole (the window placement
    // normalizes widths, so wide pieces are fine)
    double da[3], db[3];
    int oa[3], ob[3];
    q_offset_doubles(a, da, oa);
    q_offset_doubles(b, db, ob);
    for (int i = 0; i < 3; i++) {
        if (da[i] == 0.0) continue;
        for (int j = 0; j < 3; j++) {
            if (db[j] == 0.0) continue;
            double p, err;
            q_two_prod(da[i], db[j], p, err);
            bool neg = a.sign != b.sign;
            if (p != 0.0) {
                Dyad d = canon(p);
                qacc_push(acc, (u128)d.odd, a.e + b.e + oa[i] + ob[j] + d.exp, neg);
            }
            if (err != 0.0) {
                // the residual rounds either way: it carries its own sign
                Dyad d = canon(err);
                bool eneg = neg ? !sign_bit(err) : sign_bit(err);
                qacc_push(acc, (u128)d.odd, a.e + b.e + oa[i] + ob[j] + d.exp, eneg);
            }
        }
    }
}

// --- one final rounding ----------------------------------------------------------------

struct QRounded {
    u128 q;
    int e;      // the quantum exponent: the result is q * 2**e
    uint32_t flags;
    bool zero;  // the exact value was zero
    bool neg;   // the sign of the value
    bool inf;   // overflowed: q is meaningless, the answer is the sign's infinity
};

M1_HD inline int q_top(const Qdy& k) { return k.e + bitlen128(k.m) - 1; }

// Compare r * 2**s against den, r and den positive, s any integer.
M1_HD inline int q_cmp_pow2(u128 r, int s, u128 den) {
    int br = r ? bitlen128(r) + s : 0;
    int bd = den ? bitlen128(den) : 0;
    if (br != bd) return br < bd ? -1 : 1;
    if (s >= 0) {
        if (s >= 127) return 1;
        u128 rr = r << (unsigned)s;
        return rr == den ? 0 : (rr < den ? -1 : 1);
    }
    unsigned t = (unsigned)-s;
    if (t >= 127) return -1;
    u128 dd = den >> t;
    if (r != dd) return r < dd ? -1 : 1;
    return (den & (((u128)1 << t) - 1)) ? 1 : 0;
}

// The sign of the exact sum of the pieces below the window: 0, or +-1.  A
// piece whose neighbours sit far below it outmasses everything after it, so
// the first wide gap settles the sign; nearer pieces fold exactly.
struct BelowLead {
    int sign;  // 0, or the sign of the exact sum
    u128 m;    // the leading piece's magnitude (dominates the rest)
    int e;
    bool neg;
    bool zero;
};
M1_HD inline BelowLead q_below_lead(Qacc& b) {
    BelowLead out;
    out.sign = 0;
    out.m = 0;
    out.e = 0;
    if (b.n == 0) return out;
    for (int i = 1; i < b.n; i++) {  // sort by top-bit position, descending
        Qdy k = b.c[i];
        int kp = q_top(k);
        int j = i - 1;
        while (j >= 0 && q_top(b.c[j]) < kp) {
            b.c[j + 1] = b.c[j];
            j--;
        }
        b.c[j + 1] = k;
    }
    bool have = false;
    u128 rm = 0;
    int re = 0;
    bool rneg = false;
    for (int i = 0; i < b.n; i++) {
        if (!have) {
            rm = b.c[i].m;
            re = b.c[i].e;
            rneg = b.c[i].neg;
            have = true;
            continue;
        }
        int gap = (re + bitlen128(rm) - 1) - q_top(b.c[i]);
        if (gap >= 2) break;
        u128 t = rm << (unsigned)gap;  // gap is 0 or 1
        u128 s;
        if (b.c[i].neg == rneg) {
            s = t + b.c[i].m;
        } else if (t >= b.c[i].m) {
            s = t - b.c[i].m;
        } else {
            s = b.c[i].m - t;
            rneg = b.c[i].neg;
        }
        if (s == 0) {
            have = false;
            continue;
        }
        rm = s;
        int tz = ((uint64_t)rm) ? tzcnt64((uint64_t)rm) : 64 + tzcnt64((uint64_t)(rm >> 64));
        rm >>= (unsigned)tz;
        re += tz;
    }
    out.zero = !have;
    out.m = rm;
    out.e = re;
    out.neg = rneg;
    out.sign = have ? (rneg ? -1 : 1) : 0;
    return out;
}

// Round (the sum of acc) plus a known below-mass of sign `tail` to fmt, once,
// nearest even.  The window keeps the top p + 2 bits of the magnitude; its
// two low bits are the guard and the round bit, and the mass under the window
// breaks ties.  A sum that cancels clean through its window re-windows what
// is left under it, until the value's real top turns up or nothing does.
M1_HD inline QRounded q_round(int fmt, Qacc& acc, int tail) {
    QRounded out;
    const bool quad = fmt == 3;
    const int p = quad ? Q_P : FORMATS[fmt].p;
    const int emax = quad ? Q_EMAX : FORMATS[fmt].emax;
    const int emin = quad ? Q_EMIN : FORMATS[fmt].emin;
    const int quantum = emin - p + 1;
    const int W = p + 2;
    out.flags = 0;
    out.inf = false;
    out.zero = false;
    out.neg = false;
    out.q = 0;
    out.e = 0;

    Qacc below;
    qacc_init(below);
    while (true) {
        int T = -(1 << 30);
        for (int i = 0; i < acc.n; i++) {
            int tp = q_top(acc.c[i]);
            if (tp > T) T = tp;
        }
        const int bot = T - W + 1;  // the window's bottom position
        qacc_init(below);
        s128 win = 0;
        for (int i = 0; i < acc.n; i++) {
            const Qdy k = acc.c[i];
            int bl = bitlen128(k.m);
            int off = T - q_top(k);  // the piece's top, in window steps
            if (off >= W) {
                qacc_push(below, k.m, k.e, k.neg);
                continue;
            }
            int lowb = off + bl - W;  // the piece's bits under the window bottom
            u128 inw;
            if (lowb < 0 && off >= -W + 1) {
                // the piece straddles the window bottom: its top part stays
                // in the window, its low part is under-window mass
                inw = k.m >> (unsigned)(-lowb);
                u128 cut = k.m & (((u128)1 << (unsigned)(-lowb)) - 1);
                if (cut) qacc_push(below, cut, k.e, k.neg);
            } else if (lowb < 0) {
                qacc_push(below, k.m, k.e, k.neg);   // entirely under the window
                continue;
            } else {
                // fully inside: shift the piece's top to its own position.
                // The top belongs at window index W-1+off; m's top is at
                // bl-1; the shift is W+off-bl -- the old code shifted by
                // W-off-bl, correct only for the piece at the very top and
                // silently dropping or corrupting every other piece.
                inw = k.m << (unsigned)(W + off - bl);
            }
            win += k.neg ? -(s128)inw : (s128)inw;
        }
        if (win != 0) {
            int bt = tail;
            BelowLead bl = q_below_lead(below);
            if (bt == 0) bt = bl.sign;
            bool wneg = win < 0;
            u128 mag = (u128)(wneg ? -win : win);
#ifdef M1_TRACE
#endif
            out.neg = wneg;
            // a window of exactly a power of two with mass taken out from
            // under it: the value's true top is one lower, and the honest
            // window there is 2*mag - ceil(2 * below / 2**bot) -- the borrow
            // written out, so the pieces under the window stay reachable.
            bool shaved = wneg ? bt > 0 : bt < 0;
            if (shaved && (mag & (mag - 1)) == 0) {
                int cmp_b = bl.zero ? -1 : q_cmp_pow2(bl.m, bl.e - bot + 1, 1);
                int c = cmp_b > 0 ? 2 : 1;  // ceil(2 * below / 2**bot)
                mag = 2 * mag - (u128)c;
                T -= 1;
                bt = wneg ? 1 : -1;  // what is left under the window adds back
                bl.sign = 0;
                bl.zero = true;
                shaved = true;
            }
            bool inexact = shaved || (mag & 3) != 0 || bt != 0;
            int E = T - (W - 1) + (bitlen128(mag) - 1);  // the value's binary exponent
            if (E > emax) {
                out.inf = true;
                out.flags = F_OVERFLOW | F_INEXACT;
                return out;
            }
            int shift = E - T + 2;  // mag >> shift is the p-bit truncation
            int e_out = E - p + 1;
            u128 q;
            if (e_out > quantum) {  // normal range
                q = shift >= 0 ? (mag >> (unsigned)(shift > 127 ? 127 : shift))
                               : (mag << (unsigned)(-shift > 113 ? 113 : -shift));
                u128 gbv = shift >= 2 ? ((mag >> (unsigned)(shift - 2)) & 3)
                                      : ((mag << (unsigned)(2 - shift)) & 3);
                uint32_t gb = (uint32_t)gbv;
                bool up;
                if (gb > 2) up = true;
                else if (gb == 1 && !bl.zero && bt > 0
                         && q_cmp_pow2(bl.m, bl.e - bot + 1, 0) >= 0) {
                    // the truncation sits a quarter low, but the under-window
                    // mass carries at least another quarter: past the midpoint
                    up = true;
                } else if (gb == 0 && !bl.zero && bt > 0
                           && q_cmp_pow2(bl.m, bl.e - bot + 1, 1) > 0) {
                    // three quarters or more of under-window mass: past the
                    // midpoint the same way
                    up = true;
                } else if (gb < 2) up = false;
                else {
                    // at the midpoint the under-window mass decides: mass that
                    // adds to the magnitude pushes past it, mass that shaves
                    // the magnitude holds it under, and a mass of exactly half
                    // is a true tie to the even
                    int half = bl.zero ? -1 : q_cmp_pow2(bl.m, bl.e - bot + 1, 1);
                    bool adds = half == 0 ? false : (wneg ? bt < 0 : bt > 0);
                    bool holds = half == 0 ? false : (wneg ? bt > 0 : bt < 0);
                    up = adds || (!adds && !holds && (q & 1));
                }
                if (up) q++;
                bool carried = q == ((u128)1 << p);
                out.flags = inexact ? F_INEXACT : 0;
                if (out.flags && e_out < emin && q < ((u128)1 << (p - 1))) out.flags |= F_UNDERFLOW;
                if (carried && quad) {
                    // fp.py keeps q = 2**p at the old quantum; mirror it
                    out.q = q;
                    out.e = e_out;
                    return out;
                }
                if (carried) {
                    q = (u128)1 << (p - 1);
                    e_out++;
                }
            } else {  // subnormal: re-round at the quantum, once
                int s2 = W - 1 - T + quantum;
                u128 low, half;
                if (s2 > 127) {
                    q = 0;
                    low = 0;
                    half = 2;  // the value is under a quarter quantum: no round
                    low = mag != 0 ? 1 : 0;  // up, but the magnitude tells inexact
                } else {
                    low = mag & (((u128)1 << (unsigned)s2) - 1);
                    q = mag >> (unsigned)s2;
                    half = (u128)1 << (unsigned)(s2 - 1);
                }
                bool up;
                if (low > half) up = true;
                else if (low < half) up = false;
                else up = bt > 0 || (bt == 0 && (q & 1));
                if (up) q++;
                inexact = inexact || low != 0;
                e_out = quantum;
                if (q == ((u128)1 << p)) {
                    if (quad) {  // the carry quirk reaches the boundary too
                        out.q = q;
                        out.e = e_out;
                        out.flags = inexact ? F_INEXACT : 0;
                        return out;
                    }
                    q = (u128)1 << (p - 1);
                    e_out++;
                }
                out.flags = 0;
                if (inexact) out.flags |= F_INEXACT;
                if (q == 0 || (inexact && q < ((u128)1 << (p - 1)))) out.flags |= F_UNDERFLOW;
            }
            out.q = q;
            out.e = e_out;
            out.zero = q == 0;
            return out;
        }
        // the window cancelled clean: whatever is under it is the value
        if (below.n == 0) {
            out.zero = true;
            out.neg = tail < 0;
            return out;
        }
        acc = below;
        qacc_init(below);
    }
}

// --- exact ratios: division, conversions, scaling ----------------------------------------

// Round (num/den) * 2**K to fmt, once, nearest even: fp.py's _round_ratio and
// q_from_ratio, with the division done on u128 pieces (num, den under 2**120,
// den > 0).  `sticky` marks exactness already known to be lost.
M1_HD inline u128 q_modmul(u128 x, u128 y, u128 m) {
    u128 acc = 0;
    int bl = bitlen128(x);
    for (int i = bl - 1; i >= 0; i--) {
        acc <<= 1;
        if ((x >> (unsigned)i) & 1) acc += y;
        if (acc >= m) acc -= m;
        if (acc >= m) acc -= m;
    }
    return acc;
}
M1_HD inline u128 q_modpow2(int d, u128 m) {  // 2**d mod m
    if (m == 1) return 0;
    u128 result = 1, base = 2 % m;
    while (d > 0) {
        if (d & 1) result = q_modmul(result, base, m);
        d >>= 1;
        if (d) base = q_modmul(base, base, m);
    }
    return result;
}

M1_HD inline QRounded q_ratio_round(int fmt, u128 num, u128 den, int K, bool sticky = false) {
    QRounded out;
    const bool quad = fmt == 3;
    const int p = quad ? Q_P : FORMATS[fmt].p;
    const int emax = quad ? Q_EMAX : FORMATS[fmt].emax;
    const int emin = quad ? Q_EMIN : FORMATS[fmt].emin;
    const int quantum = emin - p + 1;
    out.zero = false;
    out.neg = false;
    out.inf = false;
    out.flags = 0;
    int bln = num ? bitlen128(num) : 0, bld = bitlen128(den);
    int t = bln - bld;
    int est = t - 1;
    if (num) {
        if (t >= 0) {
            if (t < 128 && (num >> (unsigned)t) >= den) est++;
        } else {
            int k = -t;
            u128 fl = k < 128 ? (den >> (unsigned)k) : 0;
            bool rem = k < 128 && (den & (((u128)1 << (unsigned)k) - 1)) != 0;
            if (num >= fl + (rem ? 1 : 0)) est++;
        }
    }
    est += K;
    int qe = est - p + 1 > quantum ? est - p + 1 : quantum;
    int s = K - qe;
#ifdef M1_TRACE
#endif
    u128 q, r;
    int s_left = 0;
    bool inexact_tail;
    if (s >= 0) {
        q = den ? num / den : 0;
        r = den ? num % den : (num ? 1 : 0);
        unsigned room_step = (unsigned)(126 - bld);  // r << step must fit a u128
        while (s > 0) {
            unsigned step = (unsigned)s > 14 ? 14 : (unsigned)s;
            if (step > room_step) step = room_step;
            u128 num2 = r << step;
            q = (q << step) | (num2 / den);
            r = num2 % den;
            s -= (int)step;
        }
        s_left = 0;
        inexact_tail = r != 0;
    } else {
        int k = -s;
        // q = floor(num / (den * 2**k)): the low bits never reach the divisor
        u128 H = k < 128 ? (num >> (unsigned)k) : 0;
        q = H / den;
        if (den == 1) {
#ifdef M1_TRACE
#endif
            // my denominator folded the exponent, so fp.py's remainder
            // (num * 2**k) mod 2**(-K) is nonzero exactly when num carries
            // bits under the quantum's reach
            r = 0;
            int div_shift = qe - K;  // num must divide by 2**div_shift to be exact
            bool exact = div_shift <= 0
                         || (div_shift <= 126 && (num & ((((u128)1 << (unsigned)div_shift) - 1))) == 0);
            inexact_tail = !exact;
        } else {
            r = q_modmul(num % den, q_modpow2(k, den), den);  // fp.py's remainder frame
            inexact_tail = r != 0;
        }
        s_left = 0;
    }
    bool inexact = sticky || inexact_tail;
#ifdef M1_TRACE
#endif
    int cmp = q_cmp_pow2(r, s_left + 1, den);
    bool up = cmp > 0 || (cmp == 0 && (q & 1));
    if (up) q++;
    if (q >> p) est++;
    if (est > emax) {
        out.inf = true;
        out.flags = F_OVERFLOW | F_INEXACT;
        return out;
    }
    out.flags = inexact ? F_INEXACT : 0;
    if (out.flags && est < emin && (q >> (p - 1)) == 0) out.flags |= F_UNDERFLOW;
    out.q = q;
    out.e = qe;
    return out;
}

M1_HD inline Result q_ratio_narrow(int fmt, int sign, u128 num, u128 den, int K, bool sticky = false) {
    QRounded o = q_ratio_round(fmt, num, den, K, sticky);
    if (o.inf) return {signed_inf(sign), o.flags};
    return {q_dyad_value(o.q, o.e, sign), o.flags};
}
M1_HD inline QRes q_ratio_quad(int sign, u128 num, u128 den, int K, bool sticky = false) {
    QRounded o = q_ratio_round(3, num, den, K, sticky);
    if (o.inf) return q_inf_bits(sign, o.flags);
    return q_encode_raw(sign, o.q, o.e, o.flags);
}

// --- sign bits, order, comparisons --------------------------------------------------------

M1_HD inline QRes q_neg_bits(uint64_t hi, uint64_t lo) { return QRes{hi ^ SIGN64, lo, 0}; }
M1_HD inline QRes q_abs_bits(uint64_t hi, uint64_t lo) { return QRes{hi & ~SIGN64, lo, 0}; }
M1_HD inline QRes q_copysign_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t) {
    return QRes{(ah & ~SIGN64) | (bh & SIGN64), al, 0};
}

// The exact order of two non-NaN quads: -1, 0 or 1.
M1_HD inline int q_order3(const Qdec& a, const Qdec& b) {
    if (a.kind == 2 || b.kind == 2) {
        if (a.kind == 2 && b.kind == 2) return a.sign == b.sign ? 0 : (a.sign ? -1 : 1);
        if (a.kind == 2) return a.sign ? -1 : 1;
        return b.sign ? 1 : -1;
    }
    if (a.sign != b.sign) return a.sign ? -1 : 1;
    int mag;
    if (a.kind == 0 || b.kind == 0) {
        mag = a.kind == 0 ? (b.kind == 0 ? 0 : -1) : 1;
    } else {
        int pa = bitlen128(a.m) - 1 + a.e;
        int pb = bitlen128(b.m) - 1 + b.e;
        if (pa != pb) mag = pa < pb ? -1 : 1;
        else {
            int sh = a.e - b.e;
            u128 x, y;
            if (sh >= 0) {
                x = sh < 127 ? (a.m << (unsigned)sh) : 0;
                y = b.m;
            } else {
                x = a.m;
                y = -sh < 127 ? (b.m << (unsigned)-sh) : 0;
            }
            mag = x == y ? 0 : (x < y ? -1 : 1);
        }
    }
    return a.sign ? -mag : mag;
}

M1_HD inline QRes q_min_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3) return b.kind == 3 ? q_qnan() : QRes{bh, bl, 0};
    if (b.kind == 3) return QRes{ah, al, 0};
    int o = q_order3(a, b);
    if (o < 0) return QRes{ah, al, 0};
    if (o > 0) return QRes{bh, bl, 0};
    return a.sign ? QRes{ah, al, 0} : QRes{bh, bl, 0};  // MIN(+0, -0) is -0
}
M1_HD inline QRes q_max_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3) return b.kind == 3 ? q_qnan() : QRes{bh, bl, 0};
    if (b.kind == 3) return QRes{ah, al, 0};
    int o = q_order3(a, b);
    if (o > 0) return QRes{ah, al, 0};
    if (o < 0) return QRes{bh, bl, 0};
    return a.sign ? QRes{bh, bl, 0} : QRes{ah, al, 0};  // MAX(+0, -0) is +0
}

M1_HD inline QRes q_cmp_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3 || b.kind == 3) return q_qnan();
    int o = q_order3(a, b);
    // fp.py's Quad(0, -1) carries a negative significand and its bits go
    // negative, past what a trace can print; the same value as a 128-bit
    // pattern is its two's complement.
    return q_encode_raw(o < 0 ? 1 : 0, o == 0 ? 0 : 1, 0);
}
M1_HD inline QRes q_lt_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3 || b.kind == 3) return q_encode_raw(0, 0, 0);  // no oracle: NaN never orders
    return q_encode_raw(0, q_order3(a, b) < 0 ? 1 : 0, 0);
}
M1_HD inline QRes q_le_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3 || b.kind == 3) return q_encode_raw(0, 0, 0);
    return q_encode_raw(0, q_order3(a, b) <= 0 ? 1 : 0, 0);
}
M1_HD inline QRes q_eq_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3 || b.kind == 3) return q_encode_raw(0, 0, 0);
    return q_encode_raw(0, q_order3(a, b) == 0 ? 1 : 0, 0);
}

// --- sums and products ---------------------------------------------------------------------

// The exact a + b (b's sign given separately, so subtraction reuses this),
// settled past the infinities, zeros and NaNs.  Every piece of the lower
// operand shifts down exactly; the window sees to what still matters.
M1_HD inline QRounded q_sum_engine(const Qdec& a, const Qdec& b, bool bneg) {
    bool swap = a.e < b.e;
    const Qdec& hi = swap ? b : a;
    const Qdec& lo = swap ? a : b;
    bool hineg = swap ? bneg : a.sign;
    bool lneg = swap ? a.sign : bneg;
    Qacc acc;
    qacc_init(acc);
    q_chunks(hi, acc, hineg);
    if (lo.kind != 0) q_chunks(lo, acc, lneg);  // pieces carry absolute exponents already
    return q_round(3, acc, 0);
}

M1_HD inline QRes q_add_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3 || b.kind == 3) return q_qnan();
    if (a.kind == 2 && b.kind == 2 && a.sign != b.sign) return q_qnan();  // inf - inf
    if (a.kind == 2) return QRes{ah, al, 0};
    if (b.kind == 2) return QRes{bh, bl, 0};
    if (a.kind == 0 && b.kind == 0) return q_encode_raw(a.sign & b.sign, 0, 0);
    QRounded o = q_sum_engine(a, b, b.sign);
    if (o.zero && o.flags == 0) return q_encode_raw(0, 0, 0);  // x + (-x) is +0
    if (o.inf) return q_inf_bits(o.neg, o.flags);
    return q_encode_raw(o.neg, o.q, o.e, o.flags);
}
M1_HD inline QRes q_sub_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    // q_add(a, q_neg(b)): the zero-sign rules fall out as fp.py's do
    return q_add_bits(ah, al, bh ^ SIGN64, bl);
}

M1_HD inline QRes q_mul_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3 || b.kind == 3) return q_qnan();
    int sign = a.sign ^ b.sign;
    if ((a.kind == 2 && (b.kind == 2 || b.kind == 0)) || (a.kind == 0 && b.kind == 2)) return q_qnan();
    if (a.kind == 2 || b.kind == 2) return q_inf_bits(sign, 0);
    if (a.kind == 0 || b.kind == 0) return q_encode_raw(sign, 0, 0);
    Qacc acc;
    qacc_init(acc);
    q_prod_pieces(a, b, acc);
    QRounded o = q_round(3, acc, 0);
    if (o.zero && o.flags == 0) return q_encode_raw(sign, 0, 0);
    if (o.inf) return q_inf_bits(sign, o.flags);
    return q_encode_raw(o.neg, o.q, o.e, o.flags);
}

M1_HD inline QRes q_fma_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl, uint64_t ch, uint64_t cl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl), c = q_decode(ch, cl);
    if (a.kind == 3 || b.kind == 3 || c.kind == 3) return q_qnan();
    int ab_sign = a.sign ^ b.sign;
    if ((a.kind == 2 && b.kind == 0) || (a.kind == 0 && b.kind == 2)) return q_qnan();
    if (a.kind == 2 || b.kind == 2) {
        if (c.kind == 2 && ab_sign != c.sign) return q_qnan();
        return q_inf_bits(ab_sign, 0);
    }
    if (c.kind == 2) return QRes{ch, cl, 0};
    if (a.kind == 0 || b.kind == 0) {
        if (c.kind != 0) return QRes{ch, cl, 0};
        return q_encode_raw(ab_sign & c.sign, 0, 0);
    }
    Qacc acc;
    qacc_init(acc);
    q_prod_pieces(a, b, acc);
    if (c.kind != 0) q_chunks(c, acc, c.sign);  // absolute exponents; the window aligns
    QRounded o = q_round(3, acc, 0);
    if (o.zero && o.flags == 0) return q_encode_raw(ab_sign & c.sign, 0, 0);
    if (o.inf) return q_inf_bits(o.neg, o.flags);
    return q_encode_raw(o.neg, o.q, o.e, o.flags);
}

// --- division -------------------------------------------------------------------------------

M1_HD inline QRes q_div_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3 || b.kind == 3) return q_qnan();
    int sign = a.sign ^ b.sign;
    if ((a.kind == 2 && b.kind == 2) || (a.kind == 0 && b.kind == 0)) return q_qnan();
    if (a.kind == 2) return q_encode_raw(sign, 0, 0);
    if (b.kind == 2) return q_encode_raw(sign, 0, 0);
    if (b.kind == 0) return q_inf_bits(sign, F_DIVZERO);
    if (a.kind == 0) return q_encode_raw(sign, 0, 0);
    return q_ratio_quad(sign, a.m, b.m, a.e - b.e);
}

// --- square root: a bit-at-a-time root of the significand -----------------------------------

// floor(sqrt(m << S)) with a bit-at-a-time root over four limbs
M1_HD inline u128 q_isqrt(u128 m, int S) {
    uint64_t R[4] = {0, 0, 0, 0};
    R[0] = (uint64_t)m;
    R[1] = (uint64_t)(m >> 64);
    for (int k = 0; k < S; k++) {
        R[3] = (R[3] << 1) | (R[2] >> 63);
        R[2] = (R[2] << 1) | (R[1] >> 63);
        R[1] = (R[1] << 1) | (R[0] >> 63);
        R[0] <<= 1;
    }
    int hp = 0;
    for (int limb = 3; limb >= 0; limb--) {
        if (R[limb]) {
            hp = limb * 64 + 63 - __builtin_clzll(R[limb]);
            break;
        }
    }
    int npairs = (hp >> 1) + 1;
    u128 rem = 0, root = 0;
    for (int k = npairs - 1; k >= 0; k--) {
        uint64_t pair = (R[k >> 5] >> ((k & 31) * 2)) & 3;
        rem = (rem << 2) | pair;
        u128 t = (root << 2) + 1;
        if (rem >= t) {
            rem -= t;
            root = (root << 1) | 1;
        } else {
            root <<= 1;
        }
    }
    return root;
}

M1_HD inline QRes q_sqrt_bits(uint64_t hi, uint64_t lo) {
    Qdec a = q_decode(hi, lo);
    if (a.kind == 3) return q_qnan();
    if (a.kind == 2 && !a.sign) return QRes{hi, lo, 0};
    if (a.sign && a.kind != 0) return QRes{QNAN_HI, QNAN_LO, F_INVALID};
    if (a.kind == 0 || a.kind == 2) return QRes{hi, lo, 0};
    // fp.py: s = isqrt((na << 242) // da), the answer s / 2**121, once.  The
    // radicand is a.m * 2**(a.e + 242), floored to a whole number, so deep
    // values truncate to nothing and answer zero with the flags clear.
    int E = a.e + 242;
    u128 mag = a.m;
    if (E < 0) {
        int t = -E;
        u128 T = t < 127 ? (mag >> (unsigned)t) : 0;
        if (T == 0) return q_encode_raw(0, 0, 0);
        u128 root = q_isqrt(T, 0);
        return q_ratio_quad(0, root, (u128)1 << 121, 0);
    }
    if (E & 1) {
        mag <<= 1;
        E -= 1;
    }
    int S = 238 - bitlen128(mag);
    S &= ~1;
    if (S > E) S = E & ~1;
    if (S < 0) S = 0;
    u128 root = q_isqrt(mag, S);
    int g = (E - S) / 2;
    return q_ratio_quad(0, root, (u128)1 << 121, g);
}

// x * y mod m, x and y under m <= 2**113: schoolbook, one bit at a time.
M1_HD inline QRes q_rem_bits(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl, int kind) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3 || b.kind == 3) return q_qnan();
    if (a.kind == 2 || b.kind != 1) return QRes{QNAN_HI, QNAN_LO, F_INVALID};  // a inf, or b zero-or-inf
    if (a.kind == 0) return QRes{ah, al, 0};
    int d = a.e - b.e;
    u128 mag;
    int scale;
    bool up = false;
    if (d >= 0) {
        scale = b.e;
        if (d <= 14) {
            u128 shifted = a.m << (unsigned)d;  // fits: d <= 14, m < 2**113
            mag = shifted % b.m;
            if (kind == 0) {  // REM rounds to the nearest multiple
                u128 qq = shifted / b.m;
                u128 r2 = mag << 1;
                up = r2 > b.m || (r2 == b.m && (qq & 1));
            }
        } else {
            mag = q_modmul(a.m % b.m, q_modpow2(d, b.m), b.m);
            if (kind == 0) {
                u128 m2 = b.m << 1;
                u128 par = q_modmul(a.m % m2, q_modpow2(d, m2), m2);
                u128 r2 = mag << 1;
                up = r2 > b.m || (r2 == b.m && (par >= b.m));
            }
        }
        if (up) mag = b.m - mag;
    } else {
        // |a| < |b|: the quotient is 0 and the floor remainder is a itself;
        // REM rounds to b instead only when a is past b/2
        int K = -d;
        // up <=> |a| past |b|/2 <=> b.m * 2**K under 2 * a.m
        up = kind == 0 && q_cmp_pow2(b.m, K, a.m << 1) < 0;
        if (up) {
            // mag = b.m * 2**K - a.m; up implies b.m * 2**(K-1) < a.m < 2**113,
            // so b.m << K fits comfortably
            mag = (b.m << (unsigned)K) - a.m;
        } else {
            mag = a.m;
        }
        scale = a.e;
    }
    if (mag == 0) return q_encode_raw(a.sign, 0, 0);  // a zero remainder takes a's sign
    // fp.py's num_r is built from magnitudes: rounding up is what makes the
    // printed remainder negative, never a's own sign
    return q_encode_canon(up ? 1 : 0, mag, scale);
}

// --- exponent surgery, neighbors, classification --------------------------------------------

M1_HD inline int q_trunc_exp(const Qdec& n) {  // the quad n as a whole exponent, clamped
    if (n.kind == 0) return 0;
    if (n.kind == 2) return n.sign ? -70000 : 70000;
    if (n.e >= 0) return n.sign ? -70000 : 70000;  // a normal is at least 2**112
    int k = -n.e;
    u128 v = k < 127 ? (n.m >> (unsigned)k) : 0;
    if (v > 70000) return n.sign ? -70000 : 70000;
    int r = (int)v;
    return n.sign ? -r : r;
}

M1_HD inline QRes q_scalb_bits(uint64_t xh, uint64_t xl, uint64_t nh, uint64_t nl) {
    Qdec x = q_decode(xh, xl), n = q_decode(nh, nl);
    if (x.kind == 3) return q_qnan();
    if (x.kind == 0 || x.kind == 2) return QRes{xh, xl, 0};
    if (n.kind == 3) return q_qnan();
    int nn = q_trunc_exp(n);
    return q_ratio_quad(x.sign, x.m, 1, x.e + nn);
}

M1_HD inline QRes q_logb_bits(uint64_t hi, uint64_t lo) {
    Qdec a = q_decode(hi, lo);
    if (a.kind == 3) return q_qnan();
    if (a.kind == 0 || a.kind == 2) return q_inf_bits(1, F_DIVZERO);  // fp.py: a.m == 0 first
    int e = a.e + bitlen128(a.m) - 1;
    if (bitlen128(a.m) <= Q_P - 1) e = Q_EMIN;  // a subnormal counts as the smallest normal
    return q_encode_raw(0, (u128)e, 0);  // Quad(0, e): prints as the subnormal its code is
}

M1_HD inline QRes q_next_bits(uint64_t xh, uint64_t xl, uint64_t yh, uint64_t yl) {
    Qdec x = q_decode(xh, xl), y = q_decode(yh, yl);
    if (x.kind == 3 || y.kind == 3) return q_qnan();
    if (q_order3(x, y) == 0) return QRes{yh, yl, 0};
    if (x.m == 0)  // zero and infinity alike: the step is the least subnormal
        return q_encode_raw(0, y.sign ? ((u128)1 << 127) | 1 : 1, 0);
    // to_bits(Q, a) +- 1, overflow where fp.py says so
    bool xp = !x.sign;
    bool bits_inc = (q_order3(y, x) > 0) == xp;  // (fb > fa) == (fa > 0)
    u128 bits = ((u128)xh << 64) | xl;
    bits += bits_inc ? 1 : ~(u128)0;
    QRes r;
    r.hi = (uint64_t)(bits >> 64);
    r.lo = (uint64_t)bits;
    r.flags = ((bits & ~((u128)1 << 127)) == ((u128)0x7FFF << 112)) ? (uint32_t)(F_OVERFLOW | F_INEXACT) : 0;
    return r;
}

M1_HD inline QRes q_class_bits(uint64_t hi, uint64_t lo) {
    Qdec a = q_decode(hi, lo);
    u128 code;
    if (a.kind == 3) code = 1;  // QNAN
    else if (a.kind == 2) code = a.sign ? 2 : 9;  // NINF, PINF
    else if (a.kind == 0) code = a.sign ? 5 : 6;  // NZERO, PZERO
    else if (a.sign) code = bitlen128(a.m) <= Q_P - 1 ? 4 : 3;  // NSUBNORMAL, NNORMAL
    else code = bitlen128(a.m) <= Q_P - 1 ? 7 : 8;  // PSUBNORMAL, PNORMAL
    return q_encode_raw(0, code, 0);
}

// --- integral roundings: exact, with fp.py's raw prints -------------------------------------

M1_HD inline QRes q_integral_bits(int kind, uint64_t hi, uint64_t lo) {
    Qdec a = q_decode(hi, lo);
    if (a.kind == 3) return q_qnan();
    if (a.kind == 2 || a.kind == 0) return QRes{hi, lo, 0};
    bool sx = a.sign;
    u128 q;
    bool inexact;
    if (a.e >= 0) {  // already whole: Quad(m * 2**e, 0), printed raw
        int bl = bitlen128(a.m) + a.e;
        u128 bits;
        if (bl >= 113) {
            u128 lowm = a.e >= 112 ? 0 : ((a.m << (unsigned)a.e) & (Q_M112 - 1));
            bits = ((u128)(uint64_t)16495 << 112) | lowm;
        } else {
            bits = a.m << (unsigned)a.e;
        }
        if (sx) bits |= (u128)1 << 127;
        QRes r;
        r.hi = (uint64_t)(bits >> 64);
        r.lo = (uint64_t)bits;
        r.flags = 0;
        return r;
    }
    int k = -a.e;
    if (k >= 128) {
        q = 0;
        inexact = a.m != 0;
        if (kind == 0 && sx && inexact) q = 1;        // floor
        else if (kind == 1 && !sx && inexact) q = 1;  // ceil
        // trunc and round-even stay at zero: the value is under half
    } else {
        q = a.m >> (unsigned)k;
        u128 r = a.m & (((u128)1 << (unsigned)k) - 1);
        inexact = r != 0;
        u128 den = (u128)1 << (unsigned)k;
        if (kind == 0) {  // floor
            if (sx && r) q++;
        } else if (kind == 1) {  // ceil
            if (!sx && r) q++;
        } else if (kind == 3) {  // round even
            u128 r2 = r << 1;
            if (r2 > den || (r2 == den && (q & 1))) q++;
        }  // trunc: nothing
    }
    if (q == 0) return q_encode_raw(sx, 0, 0, F_INEXACT);  // a nonzero x lost everything
    return q_encode_raw(sx, q, 0, inexact ? F_INEXACT : 0);
}

// --- on-grid roundings ----------------------------------------------------------------------

M1_HD inline int q_floorlog_ratio(u128 num, u128 den) {  // floor(log2(num/den)), both > 0
    int t = bitlen128(num) - bitlen128(den);
    int est = t - 1;
    if (t >= 0) {
        if (t < 128 && (num >> (unsigned)t) >= den) est++;
    } else {
        int k = -t;
        u128 fl = k < 128 ? (den >> (unsigned)k) : 0;
        bool rem = k < 128 && (den & (((u128)1 << (unsigned)k) - 1)) != 0;
        if (num >= fl + (rem ? 1 : 0)) est++;
    }
    return est;
}

M1_HD inline QRes q_grid_bits(int kind, uint64_t xh, uint64_t xl, uint64_t sh, uint64_t sl) {
    Qdec x = q_decode(xh, xl), st = q_decode(sh, sl);
    if (x.kind == 3 || st.kind == 3) return q_qnan();
    if (st.kind == 0 || st.kind == 2 || x.kind == 2) return QRes{QNAN_HI, QNAN_LO, F_INVALID};
    if (x.kind == 0) return QRes{xh, xl, 0};
    bool sx = x.sign;
    // the quotient |x| / |step| = (x.m / st.m) * 2**(x.e - st.e), exact
    int fl = q_floorlog_ratio(x.m, st.m) + (x.e - st.e);
    if (fl >= 126) {
        // the quotient is giant: fp.py rounds the exact multiple once and its
        // rounding carries past 2**113, which the quirk then prints at the old
        // quantum -- half of x, fraction zero, for both the exact and the
        // inexact multiple.  A vanishing quotient takes x's sign instead.
        int x_exp = x.e + bitlen128(x.m) - 1;
        int d = x.e - st.e;
        bool exact;
        if (d >= 0) {
            u128 rem = d <= 14 ? ((x.m << (unsigned)d) % st.m)
                               : q_modmul(x.m % st.m, q_modpow2(d, st.m), st.m);
            exact = rem == 0;
        } else {
            int k = -d;  // unreachable: mx/ms < 2**113 bounds d below; kept for safety
            u128 rem = k <= 14 ? (x.m % (st.m << (unsigned)k)) : x.m;
            exact = rem == 0;
        }
        if (exact || x.m != Q_M112)
            return QRes{xh & ~SIGN64, xl, exact ? 0 : F_INEXACT};  // the multiple rounds back to x
        // x is a power of two and the multiple sits just under it: fp.py's
        // floor log drops, the quantum halves, and the rounding carries into
        // 2**113 -- which the quirk prints at the old quantum, halved
        return q_encode_raw(0, Q_M113, x_exp - 113, F_INEXACT);
    }
    // the exact quotient, every bit of it
    u128 q, r;
    int s = x.e - st.e;
    if (s >= 0) {
        q = x.m / st.m;
        r = x.m % st.m;
        while (s > 0) {
            unsigned step = (unsigned)s > 14 ? 14 : (unsigned)s;
            u128 num2 = r << step;
            q = (q << step) | (num2 / st.m);
            r = num2 % st.m;
            s -= (int)step;
        }
    } else {
        int k = -s;
        if (k <= 14) {
            u128 dd = st.m << (unsigned)k;
            q = x.m / dd;
            r = x.m % dd;
        } else {
            // the quotient is a fraction: r is nonzero and below the divisor
            q = 0;
            r = x.m;  // 0 < r < den, by the divisor's own scale
        }
    }
    // choose the multiple by kind, on the magnitude: fp.py's choose sees the
    // positive fraction, so the multiple is positive and only an exact zero
    // of it takes x's sign
    u128 mq = q;
    int cmp2 = q_cmp_pow2(st.m, s < 0 ? -s : 0, x.m << 1);  // 2r vs den, any scale
    if (kind == 1) {  // ceil
        if (r) mq++;
    } else if (kind == 3) {  // nearest, ties to the even multiple
        if (cmp2 < 0) mq++;
        else if (cmp2 == 0 && (q & 1)) mq++;
    }  // floor, trunc: the floor of a positive fraction
    if (mq == 0) return q_encode_raw(sx, 0, 0, F_INEXACT);
    if (bitlen128(mq) + bitlen128(st.m) <= 126)
        return q_ratio_quad(0, mq * st.m, 1, st.e, r != 0);
    // the multiple needs more bits than a u128: it is within two steps of x,
    // and a step is under a quarter ulp of x there, so x is the rounding
    return QRes{xh & ~SIGN64, xl, r ? F_INEXACT : 0u};
}

// --- mantissa, split, join, conversions -----------------------------------------------------

// MANT's m and e as fp.py's q_mantissa builds them, raw prints included.
M1_HD inline QRes q_mant_m_bits(uint64_t hi, uint64_t lo) {
    Qdec a = q_decode(hi, lo);
    if (a.kind == 3) return q_qnan();
    if (a.kind == 0 || a.kind == 2) return QRes{hi, lo, 0};  // zero and infinity pass through
    int shift = bitlen128(a.m);
    return q_encode_raw(a.sign, a.m, a.e - shift);
}
M1_HD inline QRes q_mant_e_bits(uint64_t hi, uint64_t lo) {
    Qdec a = q_decode(hi, lo);
    if (a.kind == 3) return q_qnan();
    if (a.kind == 0 || a.kind == 2) return q_encode_raw(0, 0, 0);
    return q_encode_raw(0, (u128)bitlen128(a.m), 0);
}

// SPLIT of a quad into a narrow format: the low half is x - high, exact.
M1_HD inline Result q_split_low(int dst, uint64_t xh, uint64_t xl, double high, uint32_t& flags) {
    Qdec x = q_decode(xh, xl);
    if (is_inf(high)) {  // fp.py raises here (Fraction of an infinity); answer -inf
        flags = F_INEXACT;
        return {signed_inf(!sign_bit(high)), 0};
    }
    Qacc acc;
    qacc_init(acc);
    q_chunks(x, acc, x.sign);
    Dyad hd = canon(high);
    qacc_push(acc, (u128)hd.odd, hd.exp, sign_bit(high));
    QRounded o = q_round(dst, acc, 0);
    flags = o.flags & F_INEXACT;
    if (o.inf) return {signed_inf(o.neg), o.flags};
    return {q_dyad_value(o.q, o.e, o.neg), o.flags};
}

// JOIN with a quad operand: the exact sum, rounded once to dst.  (Into a
// narrow format fp.py stores the quad object itself in the register, which a
// 64-bit cell cannot hold; this rounds the same exact sum once to dst.)
M1_HD inline QRes q_join_bits(int dst, uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    Qdec a = q_decode(ah, al), b = q_decode(bh, bl);
    if (a.kind == 3 || b.kind == 3) {
        if (dst == 3) return q_qnan();
        return QRes{QUIET_NAN, 0, 0};
    }
    if (a.kind == 2 || b.kind == 2) {
        if (a.kind == 2 && b.kind == 2 && a.sign != b.sign) {
            if (dst == 3) return q_qnan();
            return QRes{QUIET_NAN, 0, F_INVALID};
        }
        uint64_t ih = a.kind == 2 ? ah : bh, il = a.kind == 2 ? al : bl;
        if (dst == 3) return QRes{ih, il, 0};
        return QRes{bits_of(signed_inf(sign_bit(ih))), 0, 0};
    }
    Qacc acc;
    qacc_init(acc);
    if (a.kind != 0) q_chunks(a, acc, a.sign);
    if (b.kind != 0) q_chunks(b, acc, b.sign);
    QRounded o = q_round(dst, acc, 0);
    if (o.zero && o.flags == 0) return q_encode_raw(a.sign & b.sign, 0, 0);
    if (o.inf) return q_inf_bits(o.neg, o.flags);
    if (dst == 3) return q_encode_raw(o.neg, o.q, o.e, o.flags);
    return QRes{bits_of(q_dyad_value(o.q, o.e, o.neg)), 0, o.flags};
}

// A quad moved into a narrow format, rounding once (fp.q_convert).
M1_HD inline Result q_to_fmt(int dst, uint64_t hi, uint64_t lo) {
    Qdec a = q_decode(hi, lo);
    if (a.kind == 3) return {nan_value(), 0};
    if (a.kind == 2) return {signed_inf(a.sign), 0};
    if (a.kind == 0) return {signed_zero(a.sign), 0};
    return q_ratio_narrow(dst, a.sign, a.m, 1, a.e);
}

// A double moved into quad: every double is exact there (fp.convert_into_q).
M1_HD inline QRes q_from_double(double x) {
    if (is_nan(x)) return q_qnan();
    if (x == 0.0) return q_encode_raw(sign_bit(x), 0, 0);
    if (is_inf(x)) return q_inf_bits(sign_bit(x), 0);
    Dyad d = canon(x);
    return q_encode_canon(sign_bit(x), (u128)d.odd, d.exp);
}

}  // namespace m1
