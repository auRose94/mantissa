// The C library behind the FPU simulation model (rtl/model/fpu_model.sv).
//
// Python's math module calls the same libm functions for the same arguments,
// so these are what mantissa/fp.py computes.  Nothing here may be built with
// fast-math: fma must round once and a * b must round twice.
#include <cmath>
#include <cstdint>
#include <cstring>

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

}  // extern "C"
