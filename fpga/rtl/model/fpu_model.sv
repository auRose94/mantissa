// The floating point unit as a simulation model: the semantics of mantissa/fp.py.
//
// This file is for co-simulation only.  It uses `real`, which is a C double
// in Verilator, exactly what a Python float is, so the IEEE operations are the
// same operations the reference performs: add, multiply, divide, square root
// and the libm calls (fma, fmod, remainder, floor, ceil, trunc, rint, frexp,
// ldexp) that fp.py makes through the math module.  The parts of fp.py that
// work on exact integers (round_ratio, the exactness of a product, quotient
// or root, the sum of two values, a value moved onto a grid) are transcribed
// here with wide integers.  Every denominator the reference divides by is a
// power of two, so a shift and a mask stand in for its divmod.
//
// The interface is the one the core talks to, mantissa_pkg::fpu_op_e, and it
// is the same interface rtl/synth/fpu_syn.sv implements for synthesis.
//
//   start    one cycle, with op, fmt, fmt2, sel and the operands
//   done     one cycle, one cycle after start, with y, y2 and the flags
//
// Across the interface a value travels as the bits that hold it: a half or a
// float as the 64 bits of the double that holds it, a quad as its own 128
// bits (operand a and result y; quad CVT is this slice's business, quad
// arithmetic is not the model's and answers the canonical quad NaN).
`timescale 1ns / 1ps

module mantissa_fpu
  import mantissa_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  input  logic [5:0]   op,      // fpu_op_e
  input  logic [2:0]   fmt,     // fmt_e
  input  logic [2:0]   fmt2,
  input  logic [1:0]   sel,
  input  logic [127:0] a,       // a quad travels here in full; the narrow formats in the low 64
  input  logic [63:0]  b,
  input  logic [63:0]  c,
  output logic         done,
  output logic [127:0] y,       // a quad result in full; the narrow formats in the low 64
  output logic [63:0]  y2,
  output logic [4:0]   flags
);

  // ---- the C library, which is what Python's math module calls -----------------
  import "DPI-C" function real c_fma(real x, real y, real z);
  import "DPI-C" function real c_ldexp(real x, int e);
  import "DPI-C" function real c_frexp(real x, output int e);
  import "DPI-C" function real c_fmod(real x, real y);
  import "DPI-C" function real c_remainder(real x, real y);
  import "DPI-C" function real c_floor(real x);
  import "DPI-C" function real c_ceil(real x);
  import "DPI-C" function real c_trunc(real x);
  import "DPI-C" function real c_rint(real x);
  import "DPI-C" function real c_fabs(real x);
  import "DPI-C" function real c_copysign(real x, real y);
  import "DPI-C" function int unsigned c_fbits(real x);        // the float pattern nearest to x
  import "DPI-C" function shortint unsigned c_hbits(real x);   // the half pattern nearest to x
  // the quad conversions, in C++ integer math (fp.convert_into_q and fp.q_convert)
  import "DPI-C" function void c_q_widen(input longint unsigned a, output longint unsigned hi,
                                         output longint unsigned lo, output int flags);
  import "DPI-C" function void c_q_narrow(input longint unsigned ahi, input longint unsigned alo,
                                          input int dst, output longint unsigned yv,
                                          output int flags);

  localparam real TWO_53 = 9007199254740992.0;

  // Wide enough for every integer the reference forms from doubles.  The
  // largest is an exact product-sum brought to a common power of two: a
  // 106-bit product whose exponent lies up to 3119 places below the addend's,
  // so 3225 bits.  A quotient of two values times a significand, or a
  // significand scaled onto a grid, needs no more than 2151.
  localparam int W = 3264;
  typedef logic [W-1:0] big_t;

  // ---- small helpers -----------------------------------------------------------

  function automatic real r(input logic [63:0] x);
    return $bitstoreal(x);
  endfunction

  function automatic logic [63:0] bits(input real x);
    return $realtobits(x);
  endfunction

  function automatic real nan();
    return $bitstoreal(NAN_BITS);
  endfunction

  function automatic real inf();
    return $bitstoreal(64'h7FF0000000000000);
  endfunction

  function automatic logic is_nan(input real x);
    return (x != x);
  endfunction

  function automatic logic is_inf(input real x);
    return (x == inf()) || (x == -inf());
  endfunction

  function automatic logic signbit(input real x);
    logic [63:0] bx;
    bx = $realtobits(x);
    return bx[63];
  endfunction

  function automatic real min_normal(input logic [2:0] f);
    return c_ldexp(1.0, 1 - f_emax(f));
  endfunction

  // ---- exact integers ------------------------------------------------------------

  // int.bit_length(): the number of bits it takes to write x down
  function automatic int bit_length(input big_t x);
    logic [63:0] v;
    bit_length = 0;
    for (int w = 0; w < W / 64; w++) begin
      v = x[64*w +: 64];
      // $clog2 is a ceiling: one below the bit length, except for a power of two
      if (v != 64'd0) bit_length = 64 * w + $clog2(v) + (((v & (v - 64'd1)) == 64'd0) ? 1 : 0);
    end
  endfunction

  // x modulo 2**s
  function automatic big_t lo_bits(input big_t x, input int s);
    if (s <= 0) return big_t'(0);
    if (s >= W) return x;
    return x & ((big_t'(1) << s) - big_t'(1));
  endfunction

  // fp._round_ratio.  Round the exact rational +-num * 2**-den_exp to the
  // format, ties to even.  num is not 0.  The flags are INEXACT, OVERFLOW and
  // UNDERFLOW; tininess is judged after rounding.
  function automatic void round_ratio(input logic [2:0] f, input big_t num, input logic neg,
                                      input int den_exp, output real value, output logic [4:0] fl);
    int p, emax, emin, nbits, est, qe, s;
    big_t q, rm, num2;
    p     = f_p(f);
    emax  = f_emax(f);
    emin  = 1 - emax;
    nbits = bit_length(num);
    est   = nbits - den_exp - 1;            // 2**est <= num * 2**-den_exp < 2**(est + 1)
    // the quantum is 2**(est - p + 1), but never finer than the subnormal grid
    qe = ((est - p + 1) > (emin - p + 1)) ? (est - p + 1) : (emin - p + 1);
    if (qe >= 0) begin
      s  = den_exp + qe;
      q  = (s >= W) ? big_t'(0) : (num >> s);
      rm = lo_bits(num, s);
    end else begin
      num2 = num << (-qe);
      s    = den_exp;
      q    = (s >= W) ? big_t'(0) : (num2 >> s);
      rm   = lo_bits(num2, s);
    end
    fl = (rm != big_t'(0)) ? F_INEXACT : 5'd0;
    if (rm != big_t'(0)) begin
      // twice the remainder against the quantum 2**s: above it rounds away,
      // exactly on it rounds to even
      if ((s >= 1) && (s < W) && rm[s-1]
          && (((s >= 2) && (lo_bits(rm, s - 1) != big_t'(0))) || q[0])) begin
        q = q + big_t'(1);
        if (q[p]) est = est + 1;
      end
    end
    if (est > emax) begin
      value = neg ? -inf() : inf();
      fl = fl | F_OVERFLOW | F_INEXACT;
    end else begin
      if (fl[4] && (est < emin) && !q[p-1]) fl = fl | F_UNDERFLOW;
      value = c_ldexp(real'(q[63:0]), qe);
      if (neg) value = -value;
    end
  endfunction

  // x = mant * 2**expo exactly, for a finite x.  mant < 2**53.
  function automatic big_t mant_of(input real x, output int expo);
    logic [63:0] bx;
    bx = $realtobits(x);
    if (bx[62:52] == 11'd0) begin
      expo = -1074;
      return big_t'(bx[51:0]);
    end
    expo = int'(bx[62:52]) - 1075;
    return big_t'({1'b1, bx[51:0]});
  endfunction

  // the same, in 64 bits, for the exactness tests
  function automatic logic [63:0] parts(input real x, output int expo);
    logic [63:0] bx;
    bx = $realtobits(x);
    if (bx[62:52] == 11'd0) begin
      expo = -1074;
      return {12'd0, bx[51:0]};
    end
    expo = int'(bx[62:52]) - 1075;
    return {11'd0, 1'b1, bx[51:0]};
  endfunction

  // m1 * 2**e1 == m2 * 2**e2, exactly
  function automatic logic same_value(input logic [63:0] m1, input int e1,
                                      input logic [255:0] m2, input int e2);
    logic [255:0] p;
    if (e1 >= e2) begin
      if (e1 - e2 >= 256) return (m1 == 64'd0) && (m2 == 256'd0);
      p = {192'd0, m1} << (e1 - e2);
      return p == m2;
    end
    if (e2 - e1 >= 256) return (m1 == 64'd0) && (m2 == 256'd0);
    p = m2 << (e2 - e1);
    return p == {192'd0, m1};
  endfunction

  // fp._product_exact: p == a * b exactly
  function automatic logic product_exact(input real p, input real a, input real b);
    int ep, ea, eb;
    logic [63:0] mp, ma, mb;
    logic [127:0] prod;
    mp = parts(p, ep);
    ma = parts(a, ea);
    mb = parts(b, eb);
    prod = {64'd0, ma} * {64'd0, mb};
    return same_value(mp, ep, {128'd0, prod}, ea + eb);
  endfunction

  // fp._quotient_exact: q == a / b exactly
  function automatic logic quotient_exact(input real a, input real q, input real b);
    int ea, eq, eb;
    logic [63:0] ma, mq, mb;
    logic [127:0] prod;
    ma = parts(a, ea);
    mq = parts(q, eq);
    mb = parts(b, eb);
    prod = {64'd0, mq} * {64'd0, mb};
    return same_value(ma, ea, {128'd0, prod}, eq + eb);
  endfunction

  // fp._root_exact: s * s == a exactly
  function automatic logic root_exact(input real a, input real s);
    int ea, es;
    logic [63:0] ma, ms;
    logic [127:0] prod;
    ma = parts(a, ea);
    ms = parts(s, es);
    prod = {64'd0, ms} * {64'd0, ms};
    return same_value(ma, ea, {128'd0, prod}, 2 * es);
  endfunction

  // The exact sum of +-m1 * 2**e1 and +-m2 * 2**e2.
  function automatic void add_terms(input big_t m1, input int e1, input logic n1,
                                    input big_t m2, input int e2, input logic n2,
                                    output big_t mag, output logic neg);
    int e0;
    big_t t1, t2;
    e0 = (e1 < e2) ? e1 : e2;
    t1 = m1 << (e1 - e0);
    t2 = m2 << (e2 - e0);
    if (n1 == n2) begin
      mag = t1 + t2; neg = n1;
    end else if (t1 >= t2) begin
      mag = t1 - t2; neg = n1;
    end else begin
      mag = t2 - t1; neg = n2;
    end
  endfunction

  // ---- cells and bit patterns: fp.from_bits, fp.to_bits -------------------------

  function automatic real from_bits(input logic [2:0] f, input logic [63:0] w);
    int e;
    real m;
    logic sgn;
    sgn = 1'b0;
    case (f)
      FMT_H: begin
        sgn = w[15];
        e = int'(w[14:10]);
        if (e == 31) begin
          if (w[9:0] == 10'd0) return sgn ? -inf() : inf();
          // widened by hand, so a NaN keeps its payload and its quiet bit
          return $bitstoreal({sgn, 11'h7FF, w[9:0], 42'd0});
        end
        if (e == 0) m = c_ldexp(real'(w[9:0]), -24);
        else m = c_ldexp(real'({2'b0, w[9:0]} + 11'd1024), e - 25);
      end
      FMT_F: begin
        sgn = w[31];
        e = int'(w[30:23]);
        if (e == 255) begin
          if (w[22:0] == 23'd0) return sgn ? -inf() : inf();
          return $bitstoreal({sgn, 11'h7FF, w[22:0], 29'd0});
        end
        if (e == 0) m = c_ldexp(real'(w[22:0]), -149);
        else m = c_ldexp(real'({1'b1, w[22:0]}), e - 150);
      end
      default: return $bitstoreal(w);
    endcase
    return sgn ? -m : m;
  endfunction

  // The cells x occupies, most significant cell first, as 64 bits.  x is a
  // value of the format, so rounding it once more changes nothing.
  function automatic logic [127:0] to_bits(input logic [2:0] f, input real x);
    logic [63:0] bx;
    bx = $realtobits(x);
    if (is_nan(x)) begin
      case (f)
        FMT_H:   return {48'd0, bx[63], 5'b11111, bx[51:42]};
        FMT_F:   return {32'd0, bx[63], 8'hFF, bx[51:29]};
        default: return bx;
      endcase
    end
    if (f == FMT_H) return {48'd0, c_hbits(x)};
    if (f == FMT_F) return {32'd0, c_fbits(x)};
    return bx;
  endfunction

  // ---- conversions ---------------------------------------------------------------

  // fp.convert: move a value into the dst format, rounding once.
  function automatic void convert(input logic [2:0] dst, input real x, output real v, output logic [4:0] fl);
    int ex;
    big_t mx;
    if (is_nan(x)) begin
      v = nan(); fl = 5'd0;
    end else if ((dst == FMT_D) || (x == 0.0) || is_inf(x)) begin
      v = x; fl = 5'd0;
    end else begin
      mx = mant_of(x, ex);
      if (ex >= 0) round_ratio(dst, mx << ex, x < 0.0, 0, v, fl);
      else round_ratio(dst, mx, x < 0.0, -ex, v, fl);
    end
  endfunction

  // fp._narrow: round a finite double to the format.  Only the rounding flags.
  function automatic void narrow(input logic [2:0] f, input real s, output real v, output logic [4:0] fl);
    int ex;
    big_t mx;
    if ((f == FMT_D) || (s == 0.0)) begin
      v = s; fl = 5'd0;
    end else begin
      mx = mant_of(s, ex);
      if (ex >= 0) round_ratio(f, mx << ex, s < 0.0, 0, v, fl);
      else round_ratio(f, mx, s < 0.0, -ex, v, fl);
    end
  endfunction

  // fp._finalize
  function automatic void finalize(input logic [2:0] f, input real value, input logic [4:0] fl,
                                   input logic exact, output real v, output logic [4:0] out);
    real mn;
    v = value;
    out = fl;
    if (!exact) out = out | F_INEXACT;
    mn = min_normal(f);
    if (out[4] && (-mn < value) && (value < mn)) out = out | F_UNDERFLOW;
  endfunction

  // ---- arithmetic -------------------------------------------------------------------

  // fp._two_sum: RN(a + b), and the exact error a + b - RN(a + b).
  function automatic void two_sum(input real a, input real b, output real s, output real err);
    real bb;
    s = a + b;
    bb = s - a;
    err = (a - (s - bb)) + (b - bb);
  endfunction

  function automatic void f_add(input logic [2:0] f, input real a, input real b, output real v,
                                output logic [4:0] fl);
    real s, err, nv;
    logic [4:0] nfl;
    if (is_nan(a) || is_nan(b)) begin
      v = nan(); fl = 5'd0;
    end else begin
      two_sum(a, b, s, err);
      if (is_nan(s)) begin                          // inf - inf
        v = nan(); fl = F_INVALID;
      end else if (is_inf(s)) begin
        v = s;
        fl = (is_inf(a) || is_inf(b)) ? 5'd0 : (F_OVERFLOW | F_INEXACT);
      end else begin
        narrow(f, s, nv, nfl);
        finalize(f, nv, nfl, err == 0.0, v, fl);
      end
    end
  endfunction

  function automatic void f_mul(input logic [2:0] f, input real a, input real b, output real v,
                                output logic [4:0] fl);
    real p, nv;
    logic [4:0] nfl;
    logic exact;
    if (is_nan(a) || is_nan(b)) begin
      v = nan(); fl = 5'd0;
    end else begin
      p = a * b;
      if (is_nan(p)) begin                          // 0 * inf
        v = nan(); fl = F_INVALID;
      end else if (is_inf(p)) begin
        v = p;
        fl = (is_inf(a) || is_inf(b)) ? 5'd0 : (F_OVERFLOW | F_INEXACT);
      end else begin
        // two narrow significands make at most 48 bits, so only a double
        // product can lose information
        exact = (f != FMT_D) ? 1'b1 : product_exact(p, a, b);
        narrow(f, p, nv, nfl);
        finalize(f, nv, nfl, exact, v, fl);
      end
    end
  endfunction

  function automatic void f_div(input logic [2:0] f, input real a, input real b, output real v,
                                output logic [4:0] fl);
    real q, nv;
    logic [4:0] nfl;
    if (is_nan(a) || is_nan(b)) begin
      v = nan(); fl = 5'd0;
    end else if (b == 0.0) begin
      if ((a == 0.0) || is_inf(a)) begin
        v = nan(); fl = F_INVALID;
      end else begin
        v = (signbit(a) != signbit(b)) ? -inf() : inf();     // copysign(inf, a) * copysign(1, b)
        fl = F_DIVZERO;
      end
    end else begin
      q = a / b;
      if (is_nan(q)) begin                          // inf / inf
        v = nan(); fl = F_INVALID;
      end else if (is_inf(q)) begin
        v = q;
        fl = is_inf(a) ? 5'd0 : (F_OVERFLOW | F_INEXACT);    // exact only from an infinite dividend
      end else if ((a == 0.0) || is_inf(b)) begin
        // an exact signed zero.  So is a finite number over infinity: the
        // reference goes on to ask whether that quotient is exact and dies
        // on the way, and a machine cannot, so this is the IEEE answer.
        v = q; fl = 5'd0;
      end else begin
        narrow(f, q, nv, nfl);
        if (is_inf(nv)) begin                       // overflowed on the way down to the format
          v = nv; fl = nfl;
        end else begin
          finalize(f, nv, nfl, quotient_exact(a, nv, b), v, fl);
        end
      end
    end
  endfunction

  function automatic void f_sqrt(input logic [2:0] f, input real a, output real v, output logic [4:0] fl);
    real s, nv;
    logic [4:0] nfl;
    if (is_nan(a)) begin
      v = nan(); fl = 5'd0;
    end else if (a == 0.0) begin
      v = a; fl = 5'd0;                             // sqrt(+-0) is +-0
    end else if (a < 0.0) begin
      v = nan(); fl = F_INVALID;
    end else begin
      s = $sqrt(a);
      if (is_inf(s)) begin
        v = s; fl = 5'd0;
      end else begin
        narrow(f, s, nv, nfl);
        if (is_inf(nv)) begin
          // A root too large for the format.  The reference goes on to ask
          // whether infinity is an exact root and dies on the way; a machine
          // cannot, so this is the IEEE answer.
          v = nv; fl = nfl;
        end else begin
          finalize(f, nv, nfl, root_exact(a, nv), v, fl);
        end
      end
    end
  endfunction

  // fp.fma: a * b + c with a single rounding
  function automatic void f_fma(input logic [2:0] f, input real a, input real b, input real c,
                                output real v, output logic [4:0] fl);
    real p, s, err, nv;
    logic [4:0] nfl;
    logic a_inf, b_inf, c_inf, neg;
    int ea, eb, ec;
    big_t ma, mb, mc, prod, mag;
    if (is_nan(a) || is_nan(b) || is_nan(c)) begin
      v = nan(); fl = 5'd0;
    end else begin
      a_inf = is_inf(a);
      b_inf = is_inf(b);
      c_inf = is_inf(c);
      if ((a_inf && (b == 0.0)) || ((a == 0.0) && b_inf)) begin
        v = nan(); fl = F_INVALID;
      end else if (a_inf || b_inf) begin
        if (c_inf && ((signbit(a) != signbit(b)) != signbit(c))) begin
          v = nan(); fl = F_INVALID;
        end else begin
          v = ((signbit(a) != signbit(b))) ? -inf() : inf();
          fl = 5'd0;
        end
      end else if (c_inf) begin
        v = c; fl = 5'd0;
      end else if (f == FMT_D) begin
        // a double product needs up to 106 bits, so round the exact rational
        ma = mant_of(a, ea);
        mb = mant_of(b, eb);
        mc = mant_of(c, ec);
        prod = ma * mb;
        add_terms(prod, ea + eb, signbit(a) != signbit(b), mc, ec, signbit(c), mag, neg);
        if (mag == big_t'(0)) begin
          v = a * b + c; fl = 5'd0;                 // an exact zero takes the IEEE sign
        end else begin
          round_ratio(FMT_D, mag, neg, -((ea + eb < ec) ? ea + eb : ec), v, fl);
        end
      end else begin
        // two narrow significands fit in a double, so the product is exact
        p = a * b;
        two_sum(p, c, s, err);
        if (is_inf(s)) begin
          v = s; fl = F_OVERFLOW | F_INEXACT;
        end else begin
          narrow(f, s, nv, nfl);
          finalize(f, nv, nfl, err == 0.0, v, fl);
        end
      end
    end
  endfunction

  // fp.remainder: a - n * b with n the whole number nearest a / b
  function automatic void f_rem(input logic [2:0] f, input real a, input real b, output real v,
                                output logic [4:0] fl);
    if (is_nan(a) || is_nan(b)) begin
      v = nan(); fl = 5'd0;
    end else if (is_inf(a) || (b == 0.0)) begin
      v = nan(); fl = F_INVALID;
    end else if (is_inf(b) || (a == 0.0)) begin
      v = a; fl = 5'd0;
    end else begin
      v = c_remainder(a, b); fl = 5'd0;             // always exact
    end
  endfunction

  // fp.fmod: what is left after trunc division
  function automatic void f_mod(input logic [2:0] f, input real a, input real b, output real v,
                                output logic [4:0] fl);
    if (is_nan(a) || is_nan(b)) begin
      v = nan(); fl = 5'd0;
    end else if (is_inf(a) || (b == 0.0)) begin
      v = nan(); fl = F_INVALID;
    end else if (is_inf(b) || (a == 0.0)) begin
      v = a; fl = 5'd0;
    end else begin
      v = c_fmod(a, b); fl = 5'd0;                  // always exact
    end
  endfunction

  // ---- sign bits, integral rounding, order ------------------------------------------

  function automatic void f_min(input real a, input real b, output real v, output logic [4:0] fl);
    fl = 5'd0;
    if (is_nan(a)) v = is_nan(b) ? nan() : b;
    else if (is_nan(b) || (a < b) || ((a == b) && (signbit(a) && !signbit(b)))) v = a;
    else v = b;
  endfunction

  function automatic void f_max(input real a, input real b, output real v, output logic [4:0] fl);
    fl = 5'd0;
    if (is_nan(a)) v = is_nan(b) ? nan() : b;
    else if (is_nan(b) || (a > b) || ((a == b) && (!signbit(a) && signbit(b)))) v = a;
    else v = b;
  endfunction

  // fp._integral: 0 floor, 1 ceil, 2 trunc, 3 round to even
  function automatic void integral(input int which, input real x, output real v, output logic [4:0] fl);
    real value;
    if (is_nan(x)) begin
      v = nan(); fl = 5'd0;
    end else if (c_fabs(x) >= TWO_53) begin         // already whole, or infinite
      v = x; fl = 5'd0;
    end else begin
      case (which)
        0: value = c_floor(x);
        1: value = c_ceil(x);
        2: value = c_trunc(x);
        default: value = c_rint(x);
      endcase
      if (value == 0.0) value = c_copysign(0.0, x);
      v = value;
      fl = (value == x) ? 5'd0 : F_INEXACT;
    end
  endfunction

  // fp._on_grid: x moved to a multiple of |step|.  which picks the multiple as
  // in integral().  |x / step| is num over den, exactly: q0 is its floor and
  // rm its remainder, so rm is to den as the fraction is to 1.  This is the
  // one place the reference divides by something that is not a power of two.
  function automatic void on_grid(input logic [2:0] f, input int which, input real x, input real step,
                                  output real v, output logic [4:0] fl);
    real ax, as;
    int ex, es;
    big_t mx, ms, den, num, q0, rm, rp, rp2, qm;
    logic qneg, r_nz, r_eq_half, r_gt_half, inc;
    if (is_nan(x) || is_nan(step)) begin
      v = nan(); fl = 5'd0;
    end else if ((step == 0.0) || is_inf(step) || is_inf(x)) begin
      v = nan(); fl = F_INVALID;
    end else if (x == 0.0) begin
      v = x; fl = 5'd0;
    end else begin
      ax = c_fabs(x);
      as = c_fabs(step);
      mx = mant_of(ax, ex);
      ms = mant_of(as, es);
      if (ex >= es) begin
        num = mx << (ex - es);
        den = ms;
      end else begin
        num = mx;
        den = ms << (es - ex);
      end
      // for the magnitude of x: the floor of the quotient and what is left
      q0 = num / den;
      rm = num % den;
      // Python's divmod floors, so for a negative x the quotient is one below
      // and the remainder is what is missing from the next multiple
      qneg = (x < 0.0);
      r_nz = (rm != big_t'(0));
      rp = (qneg && r_nz) ? (den - rm) : rm;            // the remainder Python hands back
      qm = qneg ? q0 + big_t'(r_nz ? 1 : 0) : q0;       // the magnitude of that quotient
      rp2 = rp << 1;                                    // twice the remainder, against den
      r_eq_half = (rp2 == den);
      r_gt_half = (rp2 > den);
      case (which)
        0: inc = 1'b0;                                  // floor: q
        1: inc = r_nz;                                  // ceil: q + 1 if r
        2: inc = qneg && r_nz;                          // trunc: q + 1 if num < 0 and r
        default: inc = r_gt_half || (r_eq_half && qm[0]);  // half even
      endcase
      qm = qneg ? (qm - big_t'(inc ? 1 : 0)) : (qm + big_t'(inc ? 1 : 0));
      if (qm == big_t'(0)) begin
        v = c_copysign(0.0, x); fl = F_INEXACT;
      end else begin
        num = qm * ms;
        if (es >= 0) round_ratio(f, num << es, qneg, 0, v, fl);   // q * |step| exactly
        else round_ratio(f, num, qneg, -es, v, fl);
        if (r_nz) fl = fl | F_INEXACT;
      end
    end
  endfunction

  // Python's int(n) for a double: toward zero.  Done on the bits: a cast
  // rounds to nearest in SystemVerilog, and Verilator's $rtoi, right on a
  // port, came back as -2**31 for 2**23 on a function's argument.
  function automatic int trunc_int(input real x);
    logic [63:0] bx;
    logic [52:0] mag;
    logic [10:0] t;
    logic [5:0]  sh;
    int          whole;
    bx = $realtobits(x);
    if (bx[62:52] < 11'd1023) trunc_int = 0;                  // below 1 in magnitude
    else if (bx[62:52] > 11'd1075) trunc_int = bx[63] ? -2201 : 2201;
    else begin
      t = 11'd1075 - bx[62:52];
      sh = t[5:0];
      mag = {1'b1, bx[51:0]} >> sh;
      whole = {1'b0, mag[30:0]};
      if (mag[52:31] != 22'd0) trunc_int = bx[63] ? -2201 : 2201;   // 2**31 or more: past any clamp
      else trunc_int = bx[63] ? -whole : whole;
    end
  endfunction

  // ---- exponent surgery -------------------------------------------------------------

  function automatic void scaleb(input logic [2:0] f, input real x, input real n, output real v,
                                 output logic [4:0] fl);
    int ni, ex, est, p, emax, emin;
    big_t mx;
    if (is_nan(x) || is_nan(n)) begin
      v = nan(); fl = 5'd0;
    end else if ((x == 0.0) || is_inf(x)) begin
      v = x; fl = 5'd0;
    end else begin
      if (is_inf(n)) ni = (n > 0.0) ? 2200 : -2200;    // anything past +-2200 can only overflow or vanish
      else ni = trunc_int(n);
      if (ni > 2200) ni = 2200;
      else if (ni < -2200) ni = -2200;
      mx = mant_of(x, ex);
      est = bit_length(mx) - 1 + ex + ni;              // the exponent of x * 2**ni
      p = f_p(f); emax = f_emax(f); emin = 1 - emax;
      if (est > emax) begin
        v = (x < 0.0) ? -inf() : inf();
        fl = F_OVERFLOW | F_INEXACT;
      end else if (est < emin - p - 1) begin           // under half the smallest subnormal
        v = c_copysign(0.0, x);
        fl = F_INEXACT | F_UNDERFLOW;
      end else if (ex + ni >= 0) begin
        round_ratio(f, mx << (ex + ni), x < 0.0, 0, v, fl);
      end else begin
        round_ratio(f, mx, x < 0.0, -(ex + ni), v, fl);
      end
    end
  endfunction

  function automatic void logb(input logic [2:0] f, input real x, output real v, output logic [4:0] fl);
    int e;
    real m;
    if (is_nan(x)) begin
      v = nan(); fl = 5'd0;
    end else if (x == 0.0) begin
      v = -inf(); fl = F_DIVZERO;
    end else if (is_inf(x)) begin
      v = inf(); fl = 5'd0;
    end else begin
      m = c_frexp(x, e);
      v = real'((e - 1 > 1 - f_emax(f)) ? e - 1 : 1 - f_emax(f));
      fl = 5'd0;
    end
  endfunction

  function automatic void nextafter(input logic [2:0] f, input real x, input real yv, output real v,
                                    output logic [4:0] fl);
    logic [63:0] w, mag;
    logic up;
    if (is_nan(x) || is_nan(yv)) begin
      v = nan(); fl = 5'd0;
    end else if (x == yv) begin
      v = yv; fl = 5'd0;
    end else if (x == 0.0) begin
      // the smallest value of the format, on y's side of zero
      case (f)
        FMT_H:   w = (yv > 0.0) ? 64'd1 : 64'h8001;
        FMT_F:   w = (yv > 0.0) ? 64'd1 : 64'h80000001;
        default: w = (yv > 0.0) ? 64'd1 : 64'h8000000000000001;
      endcase
      v = from_bits(f, w);
      fl = 5'd0;
    end else begin
      // the reference's rule, kept as it is including its signed zeros: one
      // step in the bit pattern, up when y is on the far side of x from zero
      w = to_bits(f, x);
      up = ((yv > x) == (x > 0.0));
      w = up ? (w + 64'd1) : (w - 64'd1);
      case (f)
        FMT_H:   mag = {49'd0, w[14:0]};
        FMT_F:   mag = {33'd0, w[30:0]};
        default: mag = {1'b0, w[62:0]};
      endcase
      case (f)
        FMT_H:   fl = (mag == 64'h7C00) ? (F_OVERFLOW | F_INEXACT) : 5'd0;
        FMT_F:   fl = (mag == 64'h7F800000) ? (F_OVERFLOW | F_INEXACT) : 5'd0;
        default: fl = (mag == 64'h7FF0000000000000) ? (F_OVERFLOW | F_INEXACT) : 5'd0;
      endcase
      v = from_bits(f, w);
    end
  endfunction

  function automatic void classify(input logic [2:0] f, input real x, output real v, output logic [4:0] fl);
    real mn;
    fl = 5'd0;
    mn = min_normal(f);
    if (is_nan(x)) v = 1.0;
    else if (is_inf(x)) v = (x > 0.0) ? 9.0 : 2.0;
    else if (x == 0.0) v = signbit(x) ? 5.0 : 6.0;
    else if (x < 0.0) v = (x > -mn) ? 4.0 : 3.0;
    else v = (x < mn) ? 7.0 : 8.0;
  endfunction

  // ---- the casts that keep what a plain cast drops -------------------------------------

  // fp.split: (high, low, flags)
  function automatic void split(input logic [1:0] dst, input real x, output real high, output real low,
                                output logic [4:0] fl);
    logic [4:0] dummy;
    if (is_nan(x) || is_inf(x)) begin
      high = x; low = x; fl = 5'd0;
    end else begin
      convert(dst, x, high, dummy);
      convert(dst, x - high, low, dummy);
      fl = (high + low == x) ? 5'd0 : F_INEXACT;
    end
  endfunction

  // fp.join: high + low, rounded once
  function automatic void f_join(input logic [2:0] f, input real high, input real low, output real v,
                               output logic [4:0] fl);
    int eh, el;
    big_t mh, ml, mag;
    logic neg;
    if (is_nan(high) || is_nan(low)) begin
      v = nan(); fl = 5'd0;
    end else if (is_inf(high) || is_inf(low)) begin
      if ((high == -low) && is_inf(high)) begin
        v = nan(); fl = F_INVALID;
      end else begin
        v = is_inf(high) ? high : low;
        fl = 5'd0;
      end
    end else begin
      mh = mant_of(high, eh);
      ml = mant_of(low, el);
      add_terms(mh, eh, signbit(high), ml, el, signbit(low), mag, neg);
      if (mag == big_t'(0)) begin
        v = high + low; fl = 5'd0;                  // IEEE rules decide the sign of an exact zero
      end else begin
        round_ratio(f, mag, neg, -((eh < el) ? eh : el), v, fl);
      end
    end
  endfunction

  // fp.mantissa: (m, e) with x = m * 2**e and 0.5 <= |m| < 1
  function automatic void mantissa(input real x, output real m, output real e);
    int ei;
    if (is_nan(x)) begin
      m = nan(); e = nan();
    end else if ((x == 0.0) || is_inf(x)) begin
      m = x; e = 0.0;
    end else begin
      m = c_frexp(x, ei);
      e = real'(ei);
    end
  endfunction

  // ---- comparisons ----------------------------------------------------------------------

  function automatic void compare(input fpu_op_e which, input real a, input real b, output real v,
                                  output logic [4:0] fl);
    fl = 5'd0;
    case (which)
      F_LT: v = (a < b) ? 1.0 : 0.0;                 // false if an operand is NaN
      F_LE: v = (a <= b) ? 1.0 : 0.0;
      F_EQ: v = (a == b) ? 1.0 : 0.0;
      default: begin
        if (is_nan(a) || is_nan(b)) v = nan();
        else v = real'((a > b) ? 1 : ((a < b) ? -1 : 0));
      end
    endcase
  endfunction

  // ---- the operation ----------------------------------------------------------------------

  function automatic void compute(input fpu_op_e o, input logic [2:0] f, input logic [2:0] f2,
                                  input logic [1:0] cells, input logic [127:0] ab, input logic [63:0] bb,
                                  input logic [63:0] cb, output logic [127:0] yv, output logic [63:0] y2v,
                                  output logic [4:0] fl);
    real av, bv, cv, v, v2, tm, te;
    logic [4:0] fl2, qi;
    logic [63:0] word, qn_y;
    longint unsigned qw_hi, qw_lo;
    av = r(ab[63:0]);
    bv = r(bb);
    cv = r(cb);
    v2 = 0.0;
    fl2 = 5'd0;
    fl = 5'd0;
    case (o)
      F_ADD:      f_add(f, av, bv, v, fl);
      F_SUB:      f_add(f, av, -bv, v, fl);
      F_MUL:      f_mul(f, av, bv, v, fl);
      F_DIV:      f_div(f, av, bv, v, fl);
      F_SQRT:     f_sqrt(f, av, v, fl);
      F_FMA:      f_fma(f, av, bv, cv, v, fl);       // a * b + c: rd = ra * rb + rd
      F_REM:      f_rem(f, av, bv, v, fl);
      F_MOD:      f_mod(f, av, bv, v, fl);
      F_MIN:      f_min(av, bv, v, fl);
      F_MAX:      f_max(av, bv, v, fl);
      F_NEG:      begin v = -av; fl = 5'd0; end
      F_ABS:      begin v = c_fabs(av); fl = 5'd0; end
      F_CPSGN:    begin v = c_copysign(av, bv); fl = 5'd0; end
      F_FLOOR:    integral(0, av, v, fl);
      F_CEIL:     integral(1, av, v, fl);
      F_CHOP:     integral(2, av, v, fl);
      F_ROUND:    integral(3, av, v, fl);
      F_FLOOR_TO: on_grid(f, 0, av, bv, v, fl);
      F_CEIL_TO:  on_grid(f, 1, av, bv, v, fl);
      F_CHOP_TO:  on_grid(f, 2, av, bv, v, fl);
      F_ROUND_TO: on_grid(f, 3, av, bv, v, fl);
      F_SCALB:    scaleb(f, av, bv, v, fl);
      F_LOGB:     logb(f, av, v, fl);
      F_NEXT:     nextafter(f, av, bv, v, fl);
      F_CLASS:    classify(f, av, v, fl);
      F_LT, F_LE, F_EQ, F_CMP: compare(o, av, bv, v, fl);
      F_MOV:      begin v = av; fl = 5'd0; end
      F_CVT: begin
        if (f == FMT_Q) begin
          // into quad: every double is exact there, so a widening is a bit
          // shuffle -- sign, an 11-bit biased exponent grown to 15 bits, the
          // 52-bit fraction at the bottom of quad's 112.  A NaN loses its
          // payload: quad arithmetic answers with the one shared NaN.
          c_q_widen(ab[63:0], qw_hi, qw_lo, qi);
          yv = {qw_hi, qw_lo};
          fl = qi;
        end else if (f2 == FMT_Q) begin
          // out of quad: one rounding over the 113-bit significand
          c_q_narrow(ab[127:64], ab[63:0], int'(f), qn_y, qi);
          yv = {64'd0, qn_y};
          fl = qi;
        end else begin
          convert(f, av, v, fl);
        end
      end
      F_FROMBITS: begin
        word = 64'd0;
        fl = 5'd0;
        case (f)
          FMT_H:   word = {48'd0, ab[16 * (3 - int'(cells)) +: 16]};
          FMT_F:   word = {32'd0, ab[16 * (2 - int'(cells)) +: 32]};
          default: word = ab[63:0];
        endcase
        v = from_bits(f, word);
      end
      F_TOBITS: begin
        v = r(to_bits(f, av)[63:0]);
        fl = 5'd0;
      end
      F_SPLIT:    split(f, av, v, v2, fl);           // x is always a; only FMA reads c
      F_JOIN:     f_join(f, av, bv, v, fl);
      F_MANT: begin
        mantissa(av, tm, te);
        convert(f, tm, v, fl);                       // m, in the format of the first operand
        convert(f2, te, v2, fl2);                    // e, in the format of the second
        fl = fl | fl2;
      end
      default:    v = av;
    endcase
    if (!(o == F_CVT && (f == FMT_Q || f2 == FMT_Q))) begin
      yv = {64'd0, bits(v)};
    end
    y2v = bits(v2);
  endfunction

  // ---- the interface ------------------------------------------------------------------------

  fpu_op_e      op_r;
  fmt_e         fmt_r, fmt2_r;
  logic [1:0]   sel_r;
  logic [127:0] a_r;
  logic [63:0]  b_r, c_r;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      done  <= 1'b0;
      op_r  <= F_ADD;
      fmt_r <= fmt_D;
      fmt2_r <= fmt_D;
      sel_r <= 2'd0;
      a_r   <= 128'd0;
      b_r   <= 64'd0;
      c_r   <= 64'd0;
    end else begin
      done <= start;
      if (start) begin
        op_r   <= fpu_op_e'(op);
        fmt_r  <= fmt_e'(fmt);
        fmt2_r <= fmt_e'(fmt2);
        sel_r  <= sel;
        a_r    <= a;
        b_r    <= b;
        c_r    <= c;
      end
    end
  end

  // the request is registered, so this runs once per request
  always_comb begin
    // quad: this slice has the casts and the moves, not the arithmetic.  A
    // quad MOV hands the 128 bits on, FROMBITS and TOBITS are the identity on
    // a quad register's pattern (the core runs a quad load or store straight
    // through memory and never asks), CVT answers exactly in both directions
    // through compute, and every other operation on quad values answers the
    // canonical quad NaN, or the shared double NaN when the answer stays
    // narrow -- a defined answer nothing feeds: the core has no quad
    // arithmetic either, and the fuzzer refuses operand codes 96 and up.
    // Only MANT reads its second format, so fmt2 == Q routes by it alone:
    // the port carries whatever the other operand field decoded, and the
    // core keeps it honest for CVT by registering it.
    if (fmt_r == FMT_Q && (op_r inside {F_MOV, F_FROMBITS, F_TOBITS})) begin
      y = a_r;
      y2 = 64'd0;
      flags = 5'd0;
    end else if ((fmt_r == FMT_Q || (fmt2_r == FMT_Q && op_r == F_MANT)) && op_r != F_CVT) begin
      // no quad arithmetic yet
      y = (fmt_r == FMT_Q) ? mantissa_pkg::QNAN_BITS : {64'd0, NAN_BITS};
      y2 = 64'd0;
      flags = 5'd0;
    end else begin
      compute(op_r, fmt_r, fmt2_r, sel_r, a_r, b_r, c_r, y, y2, flags);
    end
  end

endmodule
