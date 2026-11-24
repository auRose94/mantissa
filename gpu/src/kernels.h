// kernels.h -- the two HIP kernels of the port.
//
// cosim_kernel runs one machine and writes a trace of what it did, to be
// compared line for line with the reference.  array_kernel runs many machines
// at once, one work-item each, every one with its own registers and its own
// private copy of memory.
#pragma once

#include "machine.h"

namespace m1 {

// How the cosim kernel lays out its shared memory, in bytes.
constexpr uint32_t COSIM_LDS = 96 * 8 + 96 * 4 + 96 * 8 + 96 * 4;  // regs, tags, before, before tags

// The array kernel keeps each work-item's registers in shared memory.  The
// per-item stride is 98 eight-byte words rather than 96: with a stride that
// is 2 modulo 4 in four-byte banks, the sixteen lanes of a wavefront that
// read the same register hit sixteen different pairs of banks.
constexpr uint32_t REG_STRIDE = 98 * 8;
constexpr uint32_t ARRAY_BLOCK = 64;  // two wavefronts: 64 * 784 bytes is under the 64 KiB limit

struct ArrayHead {
    uint32_t instances, cells;
    uint64_t max_steps;
    uint32_t n_inputs, n_outputs, image_cells, org, entry, sp;
};
struct Patch {  // a value the host puts into one cell of one machine before it runs
    uint32_t cell, fmt;
    uint64_t bits;
};
struct Pick {  // a cell the host reads back after the machine has run
    uint32_t cell, fmt;
};

// Give one machine its image and its starting state.
M1_HD inline void start(Ctx& c, const uint16_t* image, const ArrayHead& h) {
    uint16_t* cells = (uint16_t*)c.mem;
    uint32_t end = h.org + h.image_cells;
    if ((h.org & 1) == 0 && (h.cells & 1) == 0) {  // two cells at a time
        for (uint32_t k = 0; k < h.cells; k += 2) {
            uint32_t lo = k >= h.org && k < end ? image[k - h.org] : 0;
            uint32_t hi = k + 1 >= h.org && k + 1 < end ? image[k + 1 - h.org] : 0;
            *(uint32_t*)(cells + k) = lo | hi << 16;
        }
    } else {
        for (uint32_t k = 0; k < h.cells; k++) cells[k] = k >= h.org && k < end ? image[k - h.org] : 0;
    }
    reset(c, h.entry, h.sp);
}

// One machine, one trace.  `memory` holds `cells` cells and is already loaded.
__global__ void cosim_kernel(uint32_t entry, uint32_t cells, uint32_t sp, uint64_t max_steps, uint8_t* memory,
                             Event* events, uint32_t capacity, uint32_t* result) {
    extern __shared__ __align__(16) uint8_t lds[];
    if (threadIdx.x != 0) return;
    Ctx c;
    c.regs = (uint64_t*)lds;
    c.tags = (uint32_t*)(lds + 96 * 8);
    c.mem = memory;
    c.size = cells;
    reset(c, entry, sp);
    Sink sink{events, capacity, 0, false};
    run<true>(c, max_steps, &sink, (uint64_t*)(lds + 96 * 12), (uint32_t*)(lds + 96 * 20));
    result[0] = sink.count;
    result[1] = sink.overflowed ? 1 : 0;
    result[2] = c.status;
    result[3] = c.fault;
    *(uint64_t*)(result + 4) = c.steps;
}

// Many machines at once.  memories holds instances * cells cells; every
// work-item writes its own copy of the image into its own before it runs.
__global__ void array_kernel(ArrayHead h, const uint16_t* image, const Patch* patches, const Pick* picks,
                             uint8_t* memories, uint64_t* out_steps, uint8_t* out_status, uint64_t* out_values) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h.instances) return;
    extern __shared__ __align__(16) uint8_t lds[];
    Ctx c;
    c.regs = (uint64_t*)(lds + threadIdx.x * REG_STRIDE);
    c.tags = nullptr;
    c.mem = memories + (size_t)i * h.cells * 2;
    c.size = h.cells;
    start(c, image, h);
    const Patch* mine = patches + (size_t)i * h.n_inputs;
    for (uint32_t k = 0; k < h.n_inputs; k++) store_value((int)mine[k].fmt, c.mem, mine[k].cell, mine[k].bits);
    run<false>(c, h.max_steps, nullptr, nullptr, nullptr);
    const Pick* take = picks + (size_t)i * h.n_outputs;
    uint64_t* values = out_values + (size_t)i * h.n_outputs;
    for (uint32_t k = 0; k < h.n_outputs; k++) {
        // out_values holds the 64 bits of each picked value, as the result file does.
        values[k] = bits_of(from_bits((int)take[k].fmt, load_value((int)take[k].fmt, c.mem, take[k].cell)));
    }
    out_steps[i] = c.steps;
    out_status[i] = (uint8_t)(c.status | c.fault << 2 | (c.flags & 31) << 4);
}

}  // namespace m1
