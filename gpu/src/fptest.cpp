// fptest.cpp -- the arithmetic unit of gpu/src/fp.h against the reference's answers.
//
//   fptest VECTORS.bin [--details N]
//
// VECTORS.bin comes from gpu/python/make_vectors.py: one operation per record
// with the value and flags mantissa/fp.py gives.  Built natively (clang++ or
// g++) the records are checked on the host; built with hipcc the same run()
// executes in a kernel, which is what tells a bug in the unit from a
// divergence of the GPU's floating point.
#include "fp.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef __HIP__
#include <hip/hip_runtime.h>
#include "align.h"
void* operator new(std::size_t n) { return mantissa_aligned_new(n); }
#define CK(x)                                                                                               \
    do {                                                                                                    \
        hipError_t error_ = (x);                                                                            \
        if (error_ != hipSuccess) {                                                                         \
            std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(error_));                                \
            return 2;                                                                                       \
        }                                                                                                   \
    } while (0)
#endif

#ifdef __HIP__
#define M1_HDH __host__ __device__  // for the record loaders, which both sides use
#else
#define M1_HDH
#endif

namespace {

constexpr size_t RECORD = 56;
constexpr int NOPS = 33;
const char* OPS[NOPS] = {"ADD",       "SUB",     "MUL",       "DIV",     "SQRT",     "FMA",     "REM",
                         "MOD",       "MIN",     "MAX",       "CPSGN",   "SCALB",    "NEXT",    "LT",
                         "LE",        "EQ",      "CMP",       "FLOOR_TO", "CEIL_TO", "TRUNC_TO", "ROUND_TO",
                         "NEG",       "ABS",     "LOGB",      "FLOOR",   "CEIL",     "TRUNC",   "ROUND",
                         "CLASS",     "CONVERT", "SPLIT",     "JOIN",    "MANT"};
const char* FNAMES[3] = {"H", "F", "D"};
constexpr uint32_t FLAG_BITS[5] = {m1::F_INVALID, m1::F_DIVZERO, m1::F_OVERFLOW, m1::F_UNDERFLOW, m1::F_INEXACT};
const char* FLAG_NAMES[5] = {"invalid", "divzero", "overflow", "underflow", "inexact"};

struct Rec {
    uint32_t op, fmt, src;
    uint64_t a, b, c, value, value2;
    uint32_t flags;
};
struct Got {
    uint64_t value, value2;
    uint32_t flags;
};
constexpr int MAX_DETAILS = 64;
struct Detail {
    uint64_t index;
    Got got;
};

M1_HDH inline uint64_t load64(const uint8_t* p) {
    uint64_t u;
    __builtin_memcpy(&u, p, 8);
    return u;
}
M1_HDH inline uint32_t load32(const uint8_t* p) {
    uint32_t u;
    __builtin_memcpy(&u, p, 4);
    return u;
}
M1_HDH inline Rec load(const uint8_t* p) {
    Rec r;
    r.op = p[0];
    r.fmt = p[1];
    r.src = p[2];
    r.a = load64(p + 8);
    r.b = load64(p + 16);
    r.c = load64(p + 24);
    r.value = load64(p + 32);
    r.value2 = load64(p + 40);
    r.flags = load32(p + 48);
    return r;
}

// One case through the unit under test.
M1_HD inline Got run(const Rec& r) {
    using namespace m1;
    const int fmt = (int)r.fmt;
    const double a = double_of(r.a), b = double_of(r.b), c = double_of(r.c);
    Result x{};
    switch (r.op) {
        case 0: x = op_add(fmt, a, b); break;
        case 1: x = op_sub(fmt, a, b); break;
        case 2: x = op_mul(fmt, a, b); break;
        case 3: x = op_div(fmt, a, b); break;
        case 4: x = op_sqrt(fmt, a); break;
        case 5: x = op_fma(fmt, a, b, c); break;
        case 6: x = op_rem_mod(fmt, 0, a, b); break;
        case 7: x = op_rem_mod(fmt, 1, a, b); break;
        case 8: x = op_min(fmt, a, b); break;
        case 9: x = op_max(fmt, a, b); break;
        case 10: return {bits_of(op_copysign(a, b)), 0, 0};
        case 11: x = op_scalb(fmt, a, b); break;
        case 12: x = op_next(fmt, a, b); break;
        case 13: x = op_lt(fmt, a, b); break;
        case 14: x = op_le(fmt, a, b); break;
        case 15: x = op_eq(fmt, a, b); break;
        case 16: x = op_cmp(fmt, a, b); break;
        case 17: x = op_grid(fmt, K_FLOOR, a, b); break;
        case 18: x = op_grid(fmt, K_CEIL, a, b); break;
        case 19: x = op_grid(fmt, K_TRUNC, a, b); break;
        case 20: x = op_grid(fmt, K_HALF_EVEN, a, b); break;
        case 21: return {bits_of(op_neg(a)), 0, 0};
        case 22: return {bits_of(op_abs(a)), 0, 0};
        case 23: x = op_logb(fmt, a); break;
        case 24: x = op_integral(K_FLOOR, a); break;
        case 25: x = op_integral(K_CEIL, a); break;
        case 26: x = op_integral(K_TRUNC, a); break;
        case 27: x = op_integral(K_HALF_EVEN, a); break;
        case 28: x = op_classify(fmt, a); break;
        case 29: x = convert(fmt, a); break;
        case 30: {
            Pair p = op_split(fmt, a);
            return {bits_of(p.high), bits_of(p.low), p.flags};
        }
        case 31: x = op_join(fmt, a, b); break;
        default: {
            double m, e;
            op_mantissa(a, m, e);
            return {bits_of(m), bits_of(e), 0};
        }
    }
    return {bits_of(x.value), 0, x.flags};
}

M1_HD inline bool agrees(const Rec& r, const Got& g) {
    return g.value == r.value && g.value2 == r.value2 && g.flags == r.flags;
}

std::string flags_text(uint32_t flags) {
    std::string out;
    for (int i = 0; i < 5; i++) {
        if (flags & FLAG_BITS[i]) {
            if (!out.empty()) out += ",";
            out += FLAG_NAMES[i];
        }
    }
    if (flags & ~31u) {
        if (!out.empty()) out += ",";
        out += "?";
    }
    return out.empty() ? "-" : out;
}
void show_double(const char* label, uint64_t word) {
    double d;
    __builtin_memcpy(&d, &word, 8);
    std::printf("%s=%016llx(%.17g)", label, (unsigned long long)word, d);
}
void describe(size_t index, const Rec& r, const Got& g) {
    const bool two = r.op == 30 || r.op == 32;                 // SPLIT and MANT answer with two values
    const bool has_b = !(r.op == 4 || r.op == 32 || (r.op >= 21 && r.op <= 30));  // not the unary ops and casts
    std::printf("  case %zu: %s.%s", index, OPS[r.op < NOPS ? r.op : 0], FNAMES[r.fmt < 3 ? r.fmt : 0]);
    if (r.op >= 29) std::printf(" from %s", FNAMES[r.src < 3 ? r.src : 0]);
    std::printf("  ");
    show_double("a", r.a);
    if (has_b) {
        std::printf(" ");
        show_double("b", r.b);
    }
    if (r.op == 5) {
        std::printf(" ");
        show_double("c", r.c);
    }
    std::printf("\n      want ");
    show_double("value", r.value);
    if (two) {
        std::printf(" ");
        show_double("value2", r.value2);
    }
    std::printf(" flags=%s\n      got  ", flags_text(r.flags).c_str());
    show_double("value", g.value);
    if (two) {
        std::printf(" ");
        show_double("value2", g.value2);
    }
    std::printf(" flags=%s\n", flags_text(g.flags).c_str());
}

}  // namespace

#ifdef __HIP__
__global__ void test_kernel(const uint8_t* data, size_t count, uint32_t* bad_total, uint32_t* bad_ops,
                            uint32_t* slots, Detail* details) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        Rec r = load(data + i * RECORD);
        Got g = run(r);
        if (!agrees(r, g)) {
            atomicAdd(bad_total, 1u);
            atomicAdd(&bad_ops[r.op < NOPS ? r.op : 0], 1u);
            uint32_t slot = atomicAdd(slots, 1u);
            if (slot < MAX_DETAILS) details[slot] = {i, g};
        }
    }
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fptest VECTORS.bin [--details N]\n");
        return 1;
    }
    int wanted_details = 16;
    for (int i = 2; i + 1 < argc; i++) {
        if (!std::strcmp(argv[i], "--details")) wanted_details = std::atoi(argv[i + 1]);
    }
    FILE* handle = std::fopen(argv[1], "rb");
    if (!handle) {
        std::perror(argv[1]);
        return 1;
    }
    std::vector<uint8_t> data;
    uint8_t buffer[1 << 16];
    size_t got_bytes;
    while ((got_bytes = std::fread(buffer, 1, sizeof buffer, handle)) > 0) data.insert(data.end(), buffer, buffer + got_bytes);
    std::fclose(handle);
    size_t count = data.size() / RECORD;
    if (count == 0 || data.size() % RECORD) {
        std::fprintf(stderr, "%s: not a whole number of %zu-byte records\n", argv[1], RECORD);
        return 1;
    }

    uint32_t bad_total = 0;
    uint32_t bad_ops[NOPS] = {0};
    const char* where;
    std::vector<std::pair<size_t, Got>> shown;
#ifdef __HIP__
    where = "GPU";
    uint8_t* d_data; CK(hipMalloc(&d_data, data.size()));
    CK(hipMemcpy(d_data, data.data(), data.size(), hipMemcpyHostToDevice));
    uint32_t* d_counters; CK(hipMalloc(&d_counters, (2 + NOPS) * 4));
    CK(hipMemset(d_counters, 0, (2 + NOPS) * 4));
    Detail* d_details; CK(hipMalloc(&d_details, MAX_DETAILS * sizeof(Detail)));
    hipLaunchKernelGGL(test_kernel, dim3(320), dim3(256), 0, 0, d_data, count, d_counters, d_counters + 2,
                       d_counters + 1, d_details);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(&bad_total, d_counters, 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(bad_ops, d_counters + 2, NOPS * 4, hipMemcpyDeviceToHost));
    std::vector<Detail> details(MAX_DETAILS);
    if (bad_total) {
        CK(hipMemcpy(details.data(), d_details, MAX_DETAILS * sizeof(Detail), hipMemcpyDeviceToHost));
    }
    CK(hipFree(d_data));
    CK(hipFree(d_counters));
    CK(hipFree(d_details));
    for (uint32_t i = 0; i < bad_total && i < MAX_DETAILS && (int)shown.size() < wanted_details; i++) {
        shown.push_back({details[i].index, details[i].got});
    }
#else
    where = "host";
    for (size_t i = 0; i < count; i++) {
        Rec r = load(data.data() + i * RECORD);
        Got g = run(r);
        if (!agrees(r, g)) {
            bad_total++;
            bad_ops[r.op < NOPS ? r.op : 0]++;
            if ((int)shown.size() < wanted_details) shown.push_back({i, g});
        }
    }
#endif
    std::printf("fptest on the %s: %zu cases, %u disagree with the reference\n", where, count, bad_total);
    if (bad_total) {
        std::printf("  by operation:");
        for (int op = 0; op < NOPS; op++) {
            if (bad_ops[op]) std::printf(" %s=%u", OPS[op], bad_ops[op]);
        }
        std::printf("\n");
        for (size_t k = 0; k < shown.size() && (int)k < wanted_details; k++) {
            describe(shown[k].first, load(data.data() + shown[k].first * RECORD), shown[k].second);
        }
    }
    return bad_total ? 1 : 0;
}
