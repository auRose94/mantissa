// align.h -- a workaround for a crash in the installed ROCm runtime.  See gpu/NOTES.md.
//
// amd::roc::Device::createVirtualDevice allocates a 0x4c0-byte object with plain
// operator new, which promises 16-byte alignment, and its constructor then
// zero-fills it with 64-byte aligned stores.  When malloc hands back an address
// that is not 64-byte aligned, the first kernel launch dies with a
// general-protection fault, reported as SIGSEGV.  Whether that happens depends
// only on the heap layout: on the executable's path, its name and its content.
//
// Defining the global operator new here makes every such allocation 64-byte
// aligned.  The runtime's calls bind to this definition, because the executable
// comes first in the symbol lookup order.  Nothing on the system is changed.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

inline void* mantissa_aligned_new(std::size_t n) {
    if (void* p = std::aligned_alloc(64, (n + 63) & ~std::size_t(63))) return p;
    throw std::bad_alloc();
}
