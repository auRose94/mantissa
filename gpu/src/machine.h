// machine.h -- the processor of the Mantissa-1: fetch, decode and execute.
//
// A port of mantissa/machine.py and mantissa/memory.py.  It follows the
// reference instruction for instruction, in the same order, so that a fault
// is raised at the same point and changes nothing: every check comes before
// the first write.  An instruction is fetched from memory on every step, so a
// store is seen by the very next fetch.
//
// The program counter is kept as a cell index rather than a double.  That is
// the same thing: every value the reference assigns to it has just been
// checked to be a whole cell number on a four-cell boundary, in range.
#pragma once

#include "fp.h"

namespace m1 {

// Opcodes, by number.
enum : uint32_t {
    NOP = 1,
    HALT = 2,
    MOV = 10,
    LDI = 11,
    LD = 12,
    ST = 13,
    CVT = 14,
    SPLIT = 15,
    JOIN = 16,
    ADD = 20,
    SUB = 21,
    MUL = 22,
    DIV = 23,
    SQRT = 24,
    FMA = 25,
    REM = 26,
    MOD = 27,
    MIN = 28,
    MAX = 29,
    NEG = 30,
    ABS = 31,
    CPSGN = 32,
    FLOOR = 33,
    CEIL = 34,
    CHOP = 35,
    ROUND = 36,
    FLOOR_TO = 37,
    CEIL_TO = 38,
    CHOP_TO = 39,
    ROUND_TO = 40,
    SCALB = 41,
    LOGB = 42,
    NEXT = 43,
    CLASS = 44,
    MANT = 45,
    LT = 50,
    LE = 51,
    EQ = 52,
    CMP = 53,
    BR = 60,
    BSGN = 61,
    BEQ = 62,
    BNE = 63,
    BLT = 64,
    BLE = 65,
    BZ = 66,
    BNZ = 67,
    JAL = 68,
    JMP = 69,
    PUT = 70,
    PUTC = 71,
    VADD2 = 72,
    VSUB2 = 73,
    VMUL2 = 74,
    VDIV2 = 75,
    VFMA2 = 76,
    VMIN2 = 77,
    VMAX2 = 78,
    VADD4 = 79,
    VSUB4 = 80,
    VMUL4 = 81,
    VDIV4 = 82,
    VFMA4 = 83,
    VMIN4 = 84,
    VMAX4 = 85
};

// The vector instructions work on 2 or 4 consecutive registers of one file.
M1_HD inline uint32_t lanes_of(uint32_t op) {
    return (op == VADD2 || op == VSUB2 || op == VMUL2 || op == VDIV2 || op == VFMA2
            || op == VMIN2 || op == VMAX2) ? 2 : 4;
}

constexpr uint32_t NREG = 96;  // h0-h31 are 0-31, f0-f31 are 32-63, d0-d31 are 64-95

constexpr uint64_t bit(uint32_t n) { return (uint64_t)1 << n; }
constexpr uint64_t range(uint32_t lo, uint32_t hi) {  // bits lo..hi, inclusive
    return (((hi - lo + 1) == 64) ? ~(uint64_t)0 : ((bit(hi - lo + 1) - 1))) << lo;
}

// The opcode bitmaps.  Opcodes run up to 71, and shifting a 64-bit word by 64
// or more is undefined, so each bitmap lives in a low half for opcodes 0-63
// and a high half for 64-71, and bitmap() picks the half.  The halves are
// written in their own coordinates: a mask like range(60, 71) shifted into
// place would overflow 64 bits and lose its top bits, which is exactly the
// bug that made every opcode from BLT to PUTC look illegal.
constexpr uint64_t VALID_LO = range(1, 2) | range(10, 16) | range(20, 45) | range(50, 53) | range(60, 63);
constexpr uint64_t VALID_HI = range(0, 21);  // bit k is opcode 64 + k: 64 to 85
constexpr uint64_t USED_A_LO = VALID_LO & ~(bit(NOP) | bit(HALT) | bit(BR));
constexpr uint64_t USED_A_HI = VALID_HI;
constexpr uint64_t USED_B_LO = bit(MOV) | bit(LD) | bit(ST) | bit(CVT) | range(15, 16) | range(20, 24) | bit(FMA) |
                               range(26, 40) | range(41, 44) | bit(MANT) | range(50, 53) | range(62, 63);
constexpr uint64_t USED_B_HI = range(0, 1) | range(8, 21);
// BLT and BLE, then the vector instructions, 72 to 85
constexpr uint64_t USED_C_LO = range(15, 16) | range(20, 23) | bit(FMA) | range(26, 29) | bit(CPSGN) | range(37, 40) |
                               bit(SCALB) | bit(NEXT) | range(50, 53) | bit(MANT);
constexpr uint64_t USED_C_HI = range(8, 21);  // and their third
constexpr uint64_t SAME_TYPE_LO = bit(MOV) | range(20, 40) | range(41, 44) | range(50, 53) | range(62, 63);
constexpr uint64_t SAME_TYPE_HI = range(0, 1) | range(8, 21);
// BLT, BLE, and the vector instructions share their file

M1_HD inline bool bitmap(uint64_t lo, uint64_t hi, uint32_t op) {
    return ((op < 64 ? lo >> op : hi >> (op - 64)) & 1) != 0;
}

enum { RUNNING = 0, HALTED = 1, FAULTED = 2 };
enum { NO_FAULT = 0, FAULT_ADDRESS = 1, FAULT_ILLEGAL = 2, FAULT_CHARACTER = 3 };
constexpr const char* FAULT_NAMES[4] = {"", "address", "illegal-instruction", "not-a-character"};

// One machine.  `regs` and `mem` may live in shared memory or in global memory.
struct Ctx {
    uint64_t* regs;   // the 96 registers, as the bits of their doubles
    uint32_t* tags;   // which NaN object a register holds; only the trace can tell them apart
    uint8_t* mem;     // the cells, two bytes each, most significant cell of a value first
    uint32_t size;    // how many cells
    uint32_t pc;      // as a cell index
    uint32_t flags;   // the sticky IEEE 754 flags
    uint32_t status;  // RUNNING, HALTED or FAULTED
    uint32_t fault;   // FAULT_ADDRESS, FAULT_ILLEGAL or FAULT_CHARACTER
    uint32_t serial;  // counts the NaN objects that loads make, for the trace
    uint64_t steps;   // how many instructions have run
};

// What one step did besides writing registers.
struct Effects {
    bool stored;
    uint32_t address, cells;
    uint64_t stored_value;  // the bits of the double that was stored, as the reference reports it
    uint32_t printed;       // 0 nothing, 1 a value (PUT), 2 a character (PUTC)
    uint64_t value;
    uint32_t code;
};

// --- memory --------------------------------------------------------------------------

// The reference judges an address as memory._index does: not a NaN, not negative, not past the
// end, a whole number of cells, and on a multiple of the width if it must be.
M1_HD inline bool address(double addr, uint32_t width, bool aligned, uint32_t size, uint32_t& index) {
    uint64_t b = bits_of(addr);
    if (nan_bits(b)) return false;
    if ((b & SIGN64) && (b << 1)) return false;  // negative; -0 is an address, of cell 0
    if (addr + (double)width > (double)size) return false;
    uint32_t e = (uint32_t)(b >> 52) & 0x7FF;
    if (e < 1023) {  // zero
        index = 0;
        return true;
    }
    uint32_t shift = 1075 - e;  // here 1023 <= e < 1075: a larger address was past the end
    uint64_t m = b & MASK52;
    if ((m << (64 - shift)) != 0) return false;  // not a whole number of cells
    index = (uint32_t)((m | (1ULL << 52)) >> shift);
    return !aligned || index % width == 0;
}

// A value's cells sit in memory most significant cell first, and a cell is a
// 16-bit word in the machine's own byte order.  So a double read straight from
// four cells comes out with its cells the wrong way round: these put them back.
M1_HD inline uint64_t cells64(uint64_t x) {  // its own inverse
    uint64_t y = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
    return (y << 32) | (y >> 32);
}
M1_HD inline uint32_t cells32(uint32_t x) { return (x << 16) | (x >> 16); }

M1_HD inline uint64_t load64(const uint8_t* mem, uint32_t index) {  // four cells
    uint64_t u;
    __builtin_memcpy(&u, mem + 2 * (size_t)index, 8);
    return cells64(u);
}
M1_HD inline void store64(uint8_t* mem, uint32_t index, uint64_t value) {
    value = cells64(value);
    __builtin_memcpy(mem + 2 * (size_t)index, &value, 8);
}
M1_HD inline uint64_t load_value(int fmt, const uint8_t* mem, uint32_t index) {
    if (fmt == DOUBLE) return load64(mem, index);
    if (fmt == FLOAT) {
        uint32_t u;
        __builtin_memcpy(&u, mem + 2 * (size_t)index, 4);
        return cells32(u);
    }
    uint16_t u;
    __builtin_memcpy(&u, mem + 2 * (size_t)index, 2);
    return u;
}
M1_HD inline void store_value(int fmt, uint8_t* mem, uint32_t index, uint64_t value) {
    if (fmt == DOUBLE) {
        store64(mem, index, value);
    } else if (fmt == FLOAT) {
        uint32_t u = cells32((uint32_t)value);
        __builtin_memcpy(mem + 2 * (size_t)index, &u, 4);
    } else {
        uint16_t u = (uint16_t)value;
        __builtin_memcpy(mem + 2 * (size_t)index, &u, 2);
    }
}

// --- decode --------------------------------------------------------------------------

constexpr uint32_t file_of(uint32_t code) { return code >> 5; }  // 0 half, 1 float, 2 double

// `word` is the double the machine fetched.  It must be a whole number below 100,000,000 whose
// opcode exists, whose unused fields are 00 and whose used fields name registers.  For an
// instruction whose operands must share a type, they must.
M1_HD inline bool decode(uint64_t word, uint32_t& op, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& spec) {
    if (word >> 63) return false;
    uint32_t e = (uint32_t)(word >> 52) & 0x7FF;
    if (e < 1023 || e > 1049) return false;  // below 1, or 2**27 and above
    uint32_t shift = 1075 - e;
    uint64_t m = word & MASK52;
    if ((m << (64 - shift)) != 0) return false;  // not a whole number
    uint32_t n = (uint32_t)((m | (1ULL << 52)) >> shift);
    op = n / 1000000u;
    if (op > 85 || !bitmap(VALID_LO, VALID_HI, op)) return false;
    a = n / 10000u % 100u;
    b = n / 100u % 100u;
    c = n % 100u;
    uint32_t used = (bitmap(USED_A_LO, USED_A_HI, op) | (bitmap(USED_B_LO, USED_B_HI, op) << 1)
                     | (bitmap(USED_C_LO, USED_C_HI, op) << 2));
    uint32_t ka = a >> 5, kb = b >> 5, kc = c >> 5;  // 3 names nothing
    if (!(((used & 1) ? ka < 3 : a == 0) && ((used & 2) ? kb < 3 : b == 0) && ((used & 4) ? kc < 3 : c == 0)))
        return false;
    spec = used | (uint32_t)(bitmap(SAME_TYPE_LO, SAME_TYPE_HI, op) << 3);
    if (spec & 8) {  // all the used operands must be registers of one type
        uint32_t kinds = ((used & 1) ? bit(ka) : 0) | ((used & 2) ? bit(kb) : 0) | ((used & 4) ? bit(kc) : 0);
        if (kinds & (kinds - 1)) return false;
    }
    return true;
}

// How many doubles of operand follow the instruction at the cell index `pc`.  The reference
// reads them with memory._index(4, pc + k, False): an address fault if they run past the end.
M1_HD inline bool operand(Ctx& c, uint32_t pc, uint32_t k, uint64_t& value) {
    if (pc + k + 4 > c.size) return false;
    value = load64(c.mem, pc + k);
    return true;
}

// --- one step ------------------------------------------------------------------------

M1_HD inline uint32_t fail(Ctx& c, uint32_t kind) {
    c.status = FAULTED;
    c.fault = kind;
    return FAULTED;
}

template <bool TRACE>
M1_HD uint32_t step(Ctx& c, Effects& fx) {
    uint32_t pc = c.pc;
    if ((pc & 3) || pc + 4 > c.size) return fail(c, FAULT_ADDRESS);  // memory.fetch
    uint32_t op, a, b, d, spec;
    if (!decode(load64(c.mem, pc), op, a, b, d, spec)) return fail(c, FAULT_ILLEGAL);

    uint64_t* R = c.regs;
    uint32_t next = pc + 4;
    uint32_t fa = file_of(a);  // the format of the first operand's register
    uint64_t target;

    auto fresh = [&]() -> uint32_t {  // a NaN that comes out of memory is a new object every time
        return ++c.serial;
    };
    auto wr = [&](uint32_t code, double value, uint32_t tag) {
        R[code] = bits_of(value);
        if (TRACE) c.tags[code] = tag;
    };
#define M1_GO(newpc)                    \
    do {                                \
        c.pc = (newpc);                 \
        return RUNNING; \
    } while (0)

    switch (op) {
        case NOP:
            M1_GO(next);
        case HALT:
            c.status = HALTED;  // the pc stays where the HALT is
            return HALTED;

        // Two or three registers of one type.
        case ADD: case SUB: case MUL: case DIV: case REM: case MOD: case MIN: case MAX: case CPSGN:
        case FLOOR_TO: case CEIL_TO: case CHOP_TO: case ROUND_TO: case SCALB: case NEXT:
        case LT: case LE: case EQ: case CMP: {
            double x = double_of(R[b]), y = double_of(R[d]);
            Result r;
            switch (op) {
                case ADD: r = op_add(fa, x, y); break;
                case SUB: r = op_sub(fa, x, y); break;
                case MUL: r = op_mul(fa, x, y); break;
                case DIV: r = op_div(fa, x, y); break;
                case REM: r = op_rem_mod(fa, 0, x, y); break;
                case MOD: r = op_rem_mod(fa, 1, x, y); break;
                case MIN: r = op_min(fa, x, y); break;
                case MAX: r = op_max(fa, x, y); break;
                case CPSGN: r = Result{op_copysign(x, y), 0}; break;
                case FLOOR_TO: r = op_grid(fa, K_FLOOR, x, y); break;
                case CEIL_TO: r = op_grid(fa, K_CEIL, x, y); break;
                case CHOP_TO: r = op_grid(fa, K_TRUNC, x, y); break;
                case ROUND_TO: r = op_grid(fa, K_HALF_EVEN, x, y); break;
                case SCALB: r = op_scalb(fa, x, y); break;
                case NEXT: r = op_next(fa, x, y); break;
                case LT: r = op_lt(fa, x, y); break;
                case LE: r = op_le(fa, x, y); break;
                case EQ: r = op_eq(fa, x, y); break;
                default: r = op_cmp(fa, x, y); break;
            }
            // A NaN that MIN or MAX hands back is the operand itself, object and all.
            uint32_t tag = 0;
            if (TRACE && (op == MIN || op == MAX) && is_nan(r.value) && !(is_nan(x) && is_nan(y))) tag = c.tags[b];
            c.flags |= r.flags;
            wr(a, r.value, tag);
            M1_GO(next);
        }

        // One register of one type.
        case SQRT: case NEG: case ABS: case LOGB: case FLOOR: case CEIL: case CHOP: case ROUND: case CLASS: {
            double x = double_of(R[b]);
            Result r;
            uint32_t tag = 0;  // NEG, ABS and CPSGN copy a NaN as it is, as a new object
            switch (op) {
                case SQRT: r = op_sqrt(fa, x); break;
                case NEG: r = Result{op_neg(x), 0}; tag = fresh(); break;
                case ABS: r = Result{op_abs(x), 0}; tag = fresh(); break;
                case LOGB: r = op_logb(fa, x); break;
                case FLOOR: r = op_integral(K_FLOOR, x); break;
                case CEIL: r = op_integral(K_CEIL, x); break;
                case CHOP: r = op_integral(K_TRUNC, x); break;
                case ROUND: r = op_integral(K_HALF_EVEN, x); break;
                default: r = op_classify(fa, x); break;
            }
            c.flags |= r.flags;
            wr(a, r.value, tag);
            M1_GO(next);
        }

        case FMA: {  // rd = ra * rb + rd, rounded once
            Result r = op_fma(fa, double_of(R[b]), double_of(R[d]), double_of(R[a]));
            c.flags |= r.flags;
            wr(a, r.value, 0);
            M1_GO(next);
        }

        case VADD2: case VSUB2: case VMUL2: case VDIV2: case VMIN2: case VMAX2:
        case VADD4: case VSUB4: case VMUL4: case VDIV4: case VMIN4: case VMAX4: {
            uint32_t lanes = lanes_of(op);
            if (((a & 31) + lanes > 32) || ((b & 31) + lanes > 32) || ((d & 31) + lanes > 32))
                return fail(c, FAULT_ILLEGAL);  // a group runs past the end of its file
            for (uint32_t lane = 0; lane < lanes; lane++) {
                double x = double_of(R[b + lane]), y = double_of(R[d + lane]);
                Result r;
                switch (op) {
                    case VADD2: case VADD4: r = op_add(fa, x, y); break;
                    case VSUB2: case VSUB4: r = op_sub(fa, x, y); break;
                    case VMUL2: case VMUL4: r = op_mul(fa, x, y); break;
                    case VDIV2: case VDIV4: r = op_div(fa, x, y); break;
                    case VMIN2: case VMIN4: r = op_min(fa, x, y); break;
                    default: r = op_max(fa, x, y); break;
                }
                c.flags |= r.flags;
                wr(a + lane, r.value, 0);
            }
            M1_GO(next);
        }

        case VFMA2: case VFMA4: {
            uint32_t lanes = lanes_of(op);
            if (((a & 31) + lanes > 32) || ((b & 31) + lanes > 32) || ((d & 31) + lanes > 32))
                return fail(c, FAULT_ILLEGAL);
            for (uint32_t lane = 0; lane < lanes; lane++) {
                Result r = op_fma(fa, double_of(R[b + lane]), double_of(R[d + lane]),
                                  double_of(R[a + lane]));
                c.flags |= r.flags;
                wr(a + lane, r.value, 0);
            }
            M1_GO(next);
        }

        case MOV:  // the same object passes from one register to the other
            wr(a, double_of(R[b]), TRACE ? c.tags[b] : 0);
            M1_GO(next);

        case LDI: {  // the inline operand is rounded once, the way the register it loads into needs
            if (!operand(c, pc, 4, target)) return fail(c, FAULT_ADDRESS);
            Result r = convert(fa, double_of(target));
            c.flags |= r.flags;
            wr(a, r.value, 0);
            M1_GO(pc + 8);
        }

        case LD: {  // the address may come from any file: a pointer's type is its reach
            uint32_t index;
            uint32_t width = (uint32_t)FORMATS[fa].cells;
            if (!address(double_of(R[b]), width, true, c.size, index)) return fail(c, FAULT_ADDRESS);
            wr(a, from_bits(fa, load_value(fa, c.mem, index)), fresh());
            M1_GO(next);
        }

        case ST: {
            uint32_t index;
            uint32_t width = (uint32_t)FORMATS[file_of(b)].cells;
            if (!address(double_of(R[a]), width, true, c.size, index)) return fail(c, FAULT_ADDRESS);
            store_value(file_of(b), c.mem, index, to_bits(file_of(b), double_of(R[b])));
            fx.stored = true;
            fx.address = index;
            fx.cells = width;
            fx.stored_value = R[b];
            M1_GO(next);
        }

        case CVT: {  // changes type, so its registers must differ in type
            if (fa == file_of(b)) return fail(c, FAULT_ILLEGAL);
            Result r = convert(fa, double_of(R[b]));
            c.flags |= r.flags;
            wr(a, r.value, 0);
            M1_GO(next);
        }

        case SPLIT: {  // SPLIT high, low, x: two registers of one type
            if (fa != file_of(b)) return fail(c, FAULT_ILLEGAL);
            Pair p = op_split(fa, double_of(R[d]));
            uint32_t tag = TRACE ? c.tags[d] : 0;  // a NaN goes into both halves, as the same object
            wr(a, p.high, tag);
            wr(b, p.low, tag);
            c.flags |= p.flags;
            M1_GO(next);
        }

        case JOIN: {  // JOIN x, high, low: the two parts are read from registers of one type
            if (file_of(b) != file_of(d)) return fail(c, FAULT_ILLEGAL);
            Result r = op_join(fa, double_of(R[b]), double_of(R[d]));
            c.flags |= r.flags;
            wr(a, r.value, 0);
            M1_GO(next);
        }

        case MANT: {  // MANT m, e, x: the two results are each rounded to their own register
            double m, e;
            op_mantissa(double_of(R[d]), m, e);
            Result rm = convert(fa, m), re = convert(file_of(b), e);
            wr(a, rm.value, 0);
            wr(b, re.value, 0);
            c.flags |= rm.flags | re.flags;
            M1_GO(next);
        }

        case PUT:
            fx.printed = 1;
            fx.value = R[a];
            M1_GO(next);

        case PUTC: {
            double value = double_of(R[a]);
            if (!whole_bits(bits_of(value) & ~SIGN64) || sign_bit(value) || value > 1048575.0 ||
                (value >= 55296.0 && value <= 57343.0))
                return fail(c, FAULT_CHARACTER);  // not a Unicode code point
            fx.printed = 2;
            fx.code = (uint32_t)value;
            M1_GO(next);
        }

        case BZ: case BNZ: {  // the target is read before the condition is tested
            if (!operand(c, pc, 4, target)) return fail(c, FAULT_ADDRESS);
            if ((R[a] << 1) == 0 == (op == BZ)) {  // +0 and -0 are both zero; a NaN is not
                uint32_t index;
                if (!address(double_of(target), 4, true, c.size, index)) return fail(c, FAULT_ADDRESS);
                M1_GO(index);
            }
            M1_GO(pc + 8);
        }

        case JMP: {
            if (file_of(a) != 1) return fail(c, FAULT_ILLEGAL);
            uint32_t index;
            if (!address(double_of(R[a]), 4, true, c.size, index)) return fail(c, FAULT_ADDRESS);
            M1_GO(index);
        }

        case JAL: {  // leaves pc + 8 in a float register
            if (file_of(a) != 1) return fail(c, FAULT_ILLEGAL);
            if (!operand(c, pc, 4, target)) return fail(c, FAULT_ADDRESS);
            uint32_t index;
            if (!address(double_of(target), 4, true, c.size, index)) return fail(c, FAULT_ADDRESS);
            wr(a, (double)(pc + 8), 0);
            M1_GO(index);
        }

        case BR: {
            if (!operand(c, pc, 4, target)) return fail(c, FAULT_ADDRESS);
            uint32_t index;
            if (!address(double_of(target), 4, true, c.size, index)) return fail(c, FAULT_ADDRESS);
            M1_GO(index);
        }

        case BSGN: {  // ternary: negative takes the first target, positive the second, zero (or a NaN) falls through
            double value = double_of(R[a]);
            if (value < 0.0) {
                if (!operand(c, pc, 4, target)) return fail(c, FAULT_ADDRESS);
            } else if (value > 0.0) {
                if (!operand(c, pc, 8, target)) return fail(c, FAULT_ADDRESS);
            } else {
                M1_GO(pc + 12);
            }
            uint32_t index;
            if (!address(double_of(target), 4, true, c.size, index)) return fail(c, FAULT_ADDRESS);
            M1_GO(index);
        }

        default: {  // BEQ, BNE, BLT, BLE; BNE is taken for NaN
            double x = double_of(R[a]), y = double_of(R[b]);
            bool taken = op == BEQ ? x == y : op == BNE ? x != y : op == BLT ? x < y : x <= y;
            if (taken) {
                if (!operand(c, pc, 4, target)) return fail(c, FAULT_ADDRESS);
                uint32_t index;
                if (!address(double_of(target), 4, true, c.size, index)) return fail(c, FAULT_ADDRESS);
                M1_GO(index);
            }
            M1_GO(pc + 8);
        }
    }
#undef M1_GO
}

// --- the trace -------------------------------------------------------------------------

enum { E_STEP = 1, E_REG, E_STORE, E_PUT, E_PUTC, E_NOTHING, E_HALT, E_FAULT, E_LIMIT };

struct Event {
    uint32_t kind;
    uint32_t step;
    uint32_t pc;
    uint32_t c;  // a register code; an address in its top 24 bits and a width in its low 8; a character; a fault kind
    uint64_t d;  // the 64 bits of a value
};

struct Sink {
    Event* out;
    uint32_t capacity;
    uint32_t count;
    bool overflowed;
};

M1_HD inline void emit(Sink& sink, uint32_t kind, uint32_t step, uint32_t pc, uint32_t c, uint64_t d) {
    if (sink.count < sink.capacity) {
        Event& e = sink.out[sink.count++];
        e.kind = kind;
        e.step = step;
        e.pc = pc;
        e.c = c;
        e.d = d;
    } else {
        sink.overflowed = true;
    }
}

// A register counts as changed when its bits differ, or when it is a NaN and not the same NaN
// object as before.  That is what trace.py's _same does with `is`.
M1_HD inline bool changed(uint64_t was, uint32_t was_tag, uint64_t now, uint32_t now_tag) {
    return was != now || (nan_bits(now) && was_tag != now_tag);
}

// --- the run loop -----------------------------------------------------------------------

// Step until the machine stops on its own or reaches max_steps.  With TRACE set, every step and
// what it changed goes to the sink; `before` and `before_tags` hold 96 entries each.
template <bool TRACE>
M1_HD void run(Ctx& c, uint64_t max_steps, Sink* sink, uint64_t* before, uint32_t* before_tags) {
    uint64_t steps = 0;
    while (c.status == RUNNING && steps < max_steps) {
        uint32_t pc = c.pc;
        if (TRACE) {
            for (uint32_t i = 0; i < NREG; i++) {
                before[i] = c.regs[i];
                before_tags[i] = c.tags[i];
            }
        }
        Effects fx;
        fx.stored = false;
        fx.printed = 0;
        step<TRACE>(c, fx);
        if (TRACE) {
            emit(*sink, E_STEP, (uint32_t)steps, pc, 0, 0);
            bool anything = false;
            for (uint32_t i = 0; i < NREG; i++) {  // file by file, by register number
                if (changed(before[i], before_tags[i], c.regs[i], c.tags[i])) {
                    emit(*sink, E_REG, (uint32_t)steps, pc, i, c.regs[i]);
                    anything = true;
                }
            }
            if (fx.stored) {
                emit(*sink, E_STORE, (uint32_t)steps, pc, fx.address << 8 | fx.cells, fx.stored_value);
                anything = true;
            }
            if (fx.printed == 1) {
                emit(*sink, E_PUT, (uint32_t)steps, pc, 0, fx.value);
                anything = true;
            } else if (fx.printed == 2) {
                emit(*sink, E_PUTC, (uint32_t)steps, pc, fx.code, 0);
                anything = true;
            }
            if (!anything) emit(*sink, E_NOTHING, (uint32_t)steps, pc, 0, 0);
        }
        steps++;
    }
    c.steps = steps;
    if (TRACE) {
        if (c.status == HALTED) emit(*sink, E_HALT, 0, 0, 0, 0);
        else if (c.status == FAULTED) emit(*sink, E_FAULT, 0, 0, c.fault, 0);
        else emit(*sink, E_LIMIT, 0, 0, 0, 0);
    }
}

// Set the registers to zero, f31 to `sp`, and the pc to `entry`, as the contract says.
M1_HD inline void reset(Ctx& c, uint32_t entry, uint32_t sp) {
    for (uint32_t i = 0; i < NREG; i++) {
        c.regs[i] = 0;
        if (c.tags) c.tags[i] = 0;
    }
    c.regs[63] = bits_of((double)sp);  // f31
    c.pc = entry;
    c.flags = 0;
    c.status = RUNNING;
    c.fault = NO_FAULT;
    c.serial = 0;
    c.steps = 0;
}

}  // namespace m1
