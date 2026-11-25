// The C library behind the FPU simulation model (rtl/model/fpu_model.sv).
//
// Python's math module calls the same libm functions for the same arguments,
// so these are what mantissa/fp.py computes.  Nothing here may be built with
// fast-math: fma must round once and a * b must round twice.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

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

}  // extern "C"
