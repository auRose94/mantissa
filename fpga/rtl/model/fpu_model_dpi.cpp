// The C library behind the FPU simulation model (rtl/model/fpu_model.sv).
//
// Python's math module calls the same libm functions for the same arguments,
// so these are what mantissa/fp.py computes.  Nothing here may be built with
// fast-math: fma must round once and a * b must round twice.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "quad_fp.h"

namespace {

// narrow bit patterns, as the machine stores them: a half or a float held in
// a 64-bit word's low bits
double narrow_value(int fmt, uint64_t bits) {
  if (fmt == 0) {  // half
    uint16_t h = static_cast<uint16_t>(bits);
    _Float16 f = *reinterpret_cast<_Float16*>(&h);
    return static_cast<double>(f);
  }
  if (fmt == 1) {  // float
    uint32_t u = static_cast<uint32_t>(bits);
    float f;
    std::memcpy(&f, &u, sizeof f);
    return static_cast<double>(f);
  }
  return m1::double_of(bits);  // double
}

uint64_t narrow_bits(int fmt, double x) {
  if (fmt == 0) {  // half: one rounding
    _Float16 h = static_cast<_Float16>(x);
    uint16_t u;
    std::memcpy(&u, &h, sizeof u);
    return u;
  }
  if (fmt == 1) {  // float: one rounding
    float f = static_cast<float>(x);
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
  }
  return m1::bits_of(x);
}

m1::QRes quad_from_narrow(int fmt, uint64_t bits) {
  return m1::q_from_double(narrow_value(fmt, bits));  // every narrow value is exact in quad
}

}  // namespace

extern "C" {

double c_fma(double x, double y, double z) { return std::fma(x, y, z); }
double c_ldexp(double x, int e) { return std::ldexp(x, e); }
double c_frexp(double x, int* e) { return std::frexp(x, e); }
double c_fmod(double x, double y) { return std::fmod(x, y); }
double c_remainder(double x, double y) { return std::remainder(x, y); }
double c_floor(double x) { return std::floor(x); }
double c_ceil(double x) { return std::ceil(x); }
double c_trunc(double x) { return std::trunc(x); }
double c_rint(double x) { return std::rint(x); }
double c_fabs(double x) { return std::fabs(x); }
double c_copysign(double x, double y) { return std::copysign(x, y); }

// The float pattern nearest to x, in one rounding.
uint32_t c_fbits(double x) {
  volatile float converted = static_cast<float>(x);
  const float f = converted;
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

// The half pattern nearest to x, in one rounding: the compiler's own
// conversion truncates straight from the double.
uint16_t c_hbits(double x) {
  volatile _Float16 converted = static_cast<_Float16>(x);
  const _Float16 h = converted;
  uint16_t u;
  std::memcpy(&u, &h, sizeof u);
  return u;
}

// ---- quad: binary128 --------------------------------------------------------
//
// mantissa/fp.py works quads on exact integers; the widest of those is the
// 113-bit significand, and every shift it needs is bounded by the quantum it
// rounds onto, so unsigned __uint128_t carries the arithmetic.  These mirror
// fp.convert into quad (exact: a widening is a bit shuffle) and q_convert out
// of quad (fp._round_ratio over the 113-bit significand, ties to even,
// subnormals, tininess judged after rounding).

const uint64_t Q_NAN_HI = 0x7FFF800000000000ull;   // QNAN_BITS, most significant half

// fp.convert_into_q: every double is exact in quad, so this is surgery on the
// bits.  A NaN becomes quad arithmetic's shared quiet NaN: the payload does
// not travel.
void c_q_widen(uint64_t a, uint64_t* hi, uint64_t* lo, int* flags) {
  const uint64_t sign = a >> 63;
  const unsigned e11 = static_cast<unsigned>(a >> 52) & 0x7ff;
  const uint64_t m = a & ((1ull << 52) - 1);
  unsigned __int128 q;
  *flags = 0;
  if (e11 == 0x7ff && m != 0) {
    q = (static_cast<unsigned __int128>(Q_NAN_HI) << 64);          // Q_CANON
  } else if (e11 == 0x7ff) {
    q = (static_cast<unsigned __int128>(sign) << 127)
        | (static_cast<unsigned __int128>(0x7fff) << 112);         // an infinity keeps its sign
  } else if (e11 == 0 && m == 0) {
    q = static_cast<unsigned __int128>(sign) << 127;               // a zero keeps its sign
  } else if (e11 != 0) {
    // a normal double: the 11-bit biased exponent grows to 15 bits unbiased
    // (e - 1023 -> e - 1023 + 16383), the 52-bit fraction moves left to the
    // bottom of quad's 112.
    q = (static_cast<unsigned __int128>(sign) << 127)
        | (static_cast<unsigned __int128>(e11 + 15360) << 112)
        | (static_cast<unsigned __int128>(m) << 60);
  } else {
    // a subnormal double is a normal quad (2**-1074 is far above quad's
    // floor): normalise the fraction onto the 113-bit significand.
    int lead = 63 - __builtin_clzll(m);
    q = (static_cast<unsigned __int128>(sign) << 127)
        | (static_cast<unsigned __int128>(lead + 15309) << 112)
        | ((static_cast<unsigned __int128>(m) << (112 - lead))
           & (((static_cast<unsigned __int128>(1)) << 112) - 1));
  }
  *hi = static_cast<uint64_t>(q >> 64);
  *lo = static_cast<uint64_t>(q);
}

// One rounding of the exact rational m * 2**e onto dst (0 half, 1 float,
// 2 double), ties to even.  m is at most 113 bits with its leading 1 at
// `lead`; returns the value and the flags: INEXACT, and OVERFLOW or
// UNDERFLOW as fp._round_ratio raises them, tininess judged after rounding.
// The quantum 2**qe is never finer than quad's exponent makes it: the shift
// t = qe - e is always to the right and at least lead - p + 1, so only the
// guard and sticky bits of m survive, never a wide integer.
static double q_round(int dst, unsigned __int128 m, int lead, int e, int neg, int* flags) {
  static const int P[3] = {11, 24, 53};
  static const int EMAX[3] = {15, 127, 1023};
  const int p = P[dst], emax = EMAX[dst], emin = -emax;
  int est = e + lead;                             // 2**est <= |value| < 2**(est + 1)
  int qe = est - p + 1;                           // the quantum, on the subnormal grid at least
  if (qe < emin - p + 1) qe = emin - p + 1;
  *flags = 0;
  int t = qe - e;                                 // how far the significand shifts right
  unsigned __int128 q;
  int guard, sticky;
  if (t > lead) {                                 // everything falls below the quantum
    q = 0;
    guard = 0;
    sticky = m != 0;
  } else {
    q = m >> t;
    guard = static_cast<int>((m >> (t - 1)) & 1);
    sticky = (m & (((static_cast<unsigned __int128>(1)) << (t - 1)) - 1)) != 0;
  }
  if (guard && (sticky || (q & 1))) {
    q += 1;
    if (q >> p) est += 1;                         // the rounding carried into the next binade
  }
  if (est > emax) {
    *flags = 0x14;                                // OVERFLOW | INEXACT
    return std::copysign(std::numeric_limits<double>::infinity(), neg ? -1.0 : 1.0);
  }
  const int inexact = guard || sticky;
  if (inexact) *flags |= 0x10;                    // INEXACT
  if (inexact && est < emin && !(q >> (p - 1))) *flags |= 0x08;   // UNDERFLOW
  return std::ldexp(static_cast<double>(static_cast<uint64_t>(q)), qe) * (neg ? -1.0 : 1.0);
}

// fp.q_convert: a quad narrowed to dst, rounded once.  A quad NaN answers the
// shared double NaN; an infinity keeps its sign; the flags ride with the
// rounding exactly as scalar CVT's do.
void c_q_narrow(uint64_t ahi, uint64_t alo, int dst, uint64_t* y, int* flags) {
  const unsigned __int128 a = (static_cast<unsigned __int128>(ahi) << 64) | alo;
  const int neg = static_cast<int>(ahi >> 63);
  const unsigned e15 = static_cast<unsigned>(ahi >> 48) & 0x7fff;
  const unsigned __int128 frac = a & (((static_cast<unsigned __int128>(1)) << 112) - 1);
  double v;
  if (e15 == 0x7fff && frac != 0) {
    v = std::numeric_limits<double>::quiet_NaN();
    *flags = 0;
  } else if (e15 == 0x7fff) {
    v = std::copysign(std::numeric_limits<double>::infinity(), neg ? -1.0 : 1.0);
    *flags = 0;
  } else if (e15 == 0 && frac == 0) {
    v = neg ? -0.0 : 0.0;                          // a zero keeps its sign; a power
    *flags = 0;                                    // of two has frac == 0 too and is normal
  } else if (e15 != 0) {
    v = q_round(dst, (static_cast<unsigned __int128>(1) << 112) | frac, 112,
                static_cast<int>(e15) - 16495, neg, flags);
  } else {
    // a subnormal quad sits near 2**-16494, farther down than half of any
    // narrow format's smallest subnormal: it always rounds to a signed zero.
    v = neg ? -0.0 : 0.0;
    *flags = 0x18;                                // INEXACT | UNDERFLOW
  }
  double d = v;
  uint64_t bits;
  std::memcpy(&bits, &d, sizeof bits);
  *y = bits;
}


// Quad arithmetic: the whole operation dispatches to the software binary128
// -- the same engine the HIP port runs (copied to quad_fp.h above), so the
// two ports answer round for round, and the fpu vectors hold both to the
// reference.  Operands arrive as hi/lo pairs; a narrow operand in a mixed
// cast arrives in its low word with fmt telling the truth, and every narrow
// value is exact in quad, so widening is a value copy.
void c_quad_op(int op, int f, int f2,
               unsigned long long ahi, unsigned long long alo,
               unsigned long long bhi, unsigned long long blo,
               unsigned long long chi, unsigned long long clo,
               unsigned long long* yhi, unsigned long long* ylo,
               unsigned long long* y2hi, unsigned long long* y2lo, int* flags) {
  m1::QRes r{m1::QNAN_HI, m1::QNAN_LO, 0};
  uint64_t m2 = 0;        // the second answer (SPLIT's low, MANT's exponent)
  uint64_t m2_hi = 0;     // its high word: quads answer in full
  switch (op) {
    case 0:  r = m1::q_add_bits(ahi, alo, bhi, blo); break;
    case 1:  r = m1::q_sub_bits(ahi, alo, bhi, blo); break;
    case 2:  r = m1::q_mul_bits(ahi, alo, bhi, blo); break;
    case 3:  r = m1::q_div_bits(ahi, alo, bhi, blo); break;
    case 4:  r = m1::q_sqrt_bits(ahi, alo); break;
    case 5:  r = m1::q_fma_bits(ahi, alo, bhi, blo, chi, clo); break;  // a*b + c
    case 6:  r = m1::q_rem_bits(ahi, alo, bhi, blo, 0); break;
    case 7:  r = m1::q_rem_bits(ahi, alo, bhi, blo, 1); break;
    case 8:  r = m1::q_min_bits(ahi, alo, bhi, blo); break;
    case 9:  r = m1::q_max_bits(ahi, alo, bhi, blo); break;
    case 10: r = m1::q_neg_bits(ahi, alo); break;
    case 11: r = m1::q_abs_bits(ahi, alo); break;
    case 12: r = m1::q_copysign_bits(ahi, alo, bhi, blo); break;
    case 13: r = m1::q_integral_bits(0, ahi, alo); break;
    case 14: r = m1::q_integral_bits(1, ahi, alo); break;
    case 15: r = m1::q_integral_bits(2, ahi, alo); break;
    case 16: r = m1::q_integral_bits(3, ahi, alo); break;
    case 17: r = m1::q_grid_bits(0, ahi, alo, bhi, blo); break;
    case 18: r = m1::q_grid_bits(1, ahi, alo, bhi, blo); break;
    case 19: r = m1::q_grid_bits(2, ahi, alo, bhi, blo); break;
    case 20: r = m1::q_grid_bits(3, ahi, alo, bhi, blo); break;
    case 21: r = m1::q_scalb_bits(ahi, alo, bhi, blo); break;
    case 22: r = m1::q_logb_bits(ahi, alo); break;
    case 23: r = m1::q_next_bits(ahi, alo, bhi, blo); break;
    case 24: r = m1::q_class_bits(ahi, alo); break;
    case 25: r = m1::q_lt_bits(ahi, alo, bhi, blo); break;
    case 26: r = m1::q_le_bits(ahi, alo, bhi, blo); break;
    case 27: r = m1::q_eq_bits(ahi, alo, bhi, blo); break;
    case 28: r = m1::q_cmp_bits(ahi, alo, bhi, blo); break;
    case 32: {  // SPLIT: f names the halves' file
      const bool a_exp_all_ones = ((ahi >> 48) & 0x7fff) == 0x7fff;
      const bool a_is_qnan = a_exp_all_ones && ((ahi & 0x0000ffffffffffffULL) || alo);
      if (a_is_qnan) {
        // fp.py: a NaN splits into the quad canonical NaN, both halves, no flags
        r = m1::QRes{m1::QNAN_HI, m1::QNAN_LO, 0};
        m2_hi = m1::QNAN_HI; m2 = m1::QNAN_LO;
        break;
      }
      const bool a_is_qinf = a_exp_all_ones
                             && (ahi & 0x0000ffffffffffffULL) == 0 && alo == 0;
      if (a_is_qinf) {
        // an infinity splits into itself, both halves, no flags
        const unsigned long long inf_bits = (ahi >> 63) ? 0xfff0000000000000ULL
                                                        : 0x7ff0000000000000ULL;
        if (f == 3) r = m1::QRes{ahi, alo, 0};
        else { r = m1::QRes{0, inf_bits, 0}; m2 = inf_bits; }
        break;
      }
      if (f == 3) {  // into quad: exact, the low half is the quad zero
        r = m1::QRes{ahi, alo, 0};
        m2_hi = 0; m2 = 0;
        break;
      }
      // out of quad: high is one rounding, low narrows the exact leavings;
      // the split's flag is exact exactly when narrowing the low lost
      // nothing and the leavings did not run away to an infinity
      uint64_t hb = 0;
      int hf = 0;
      c_q_narrow(ahi, alo, f, &hb, &hf);
      m1::QRes wide_back = quad_from_narrow(f, hb);
      m1::QRes rest = m1::q_sub_bits(ahi, alo, wide_back.hi, wide_back.lo);
      uint64_t lb = 0;
      int lf = 0;
      c_q_narrow(rest.hi, rest.lo, f, &lb, &lf);
      const bool leavings_finite = ((rest.hi >> 48) & 0x7fff) != 0x7fff;
      const bool exact = (lf & 0x10) == 0 && leavings_finite;
      r = m1::QRes{0, hb, exact ? 0u : 0x10u};
      m2 = lb;
      break;
    }
    case 33: {  // JOIN: f names the destination's file; the parts may be narrow or quad
      m1::QRes hi_part = (f2 == 3) ? m1::QRes{ahi, alo, 0} : quad_from_narrow(f2, ahi);
      m1::QRes lo_part = (f2 == 3) ? m1::QRes{bhi, blo, 0} : quad_from_narrow(f2, bhi);
      r = m1::q_join_bits(f, hi_part.hi, hi_part.lo, lo_part.hi, lo_part.lo);
      break;
    }
    case 34: {  // MANT: m in f's format, e in f2's
      m1::QRes m = m1::q_mant_m_bits(ahi, alo);
      m1::QRes e = m1::q_mant_e_bits(ahi, alo);
      r = m;
      if (f2 == 3) {
        m2_hi = e.hi; m2 = e.lo;
      } else {
        uint64_t eb = 0; int ef = 0;
        c_q_narrow(e.hi, e.lo, f2, &eb, &ef);
        m2 = eb;
      }
      break;
    }
    case 35: r = m1::QRes{ahi, alo, 0}; break;  // MOV
    default: r = m1::q_qnan(); break;
  }
  *yhi = r.hi;
  *ylo = r.lo;
  *y2hi = m2_hi;
  *y2lo = m2;
  *flags = static_cast<int>(r.flags);
}

}  // extern "C"
