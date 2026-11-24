# Notes on the GPU port

Things that are not where the brief or the contract said they would be, and
decisions taken here.  The reference (`mantissa/`) is always right; where
this file says otherwise, it is recording what the reference *does*.

## The machine this was built on

- Radeon 8060S, `gfx1151`, 20 compute units, wavefront 32, 64 KiB of LDS per
  workgroup (`hipGetDeviceProperties`).  `rocminfo` prints 40 for the compute
  unit count; the runtime's 20 is the number HIP reports and what the
  throughput figures are consistent with.
- HIP 7.16.0 (package `hip-runtime-amd 10.1`), clang 24.0.0git.

## A crash in the ROCm runtime that is not ours

Any HIP program here can die with SIGSEGV inside `hipLaunchKernel`.  It is
not random and it is not the kernel: whether it happens depends on the heap
layout, so it changes with the executable's path, its name and its content.
Over 24 program names, 10 crashed and 14 ran.

The cause, from the disassembly of `/opt/rocm/lib/libamdhip64.so.7`:
`amd::roc::Device::createVirtualDevice` allocates a 0x4c0-byte `VirtualGPU`
with plain `operator new`, which promises 16-byte alignment, and the
constructor then zero-fills it with `vmovdqa64`, which needs 64 bytes.  When
`malloc` hands back an address that is 16- or 32-byte aligned but not 64, the
store raises a general-protection fault, which Linux reports as SIGSEGV with
fault address 0.

The workaround lives in `gpu/src/align.h`: the host program defines the global
`operator new` to return 64-byte-aligned memory.  The runtime's allocations
bind to it (27005 calls in one run), so its aligned stores always have an
aligned target.  With it, 0 of 24 names crash.  Nothing on the system is
changed.  Setting `AMD_LOG_LEVEL=4` looked like a fix at first; it only moved
the heap layout.

## Contract changes applied during the port

Applied as commit "Trace: PUT and PUTC hand over values": `P step pc value`
and `C step pc code` replace the `O` lines, and the two `R` lines of `SPLIT`
and `MANT` come out file by file, not in operand order.  The code in
`mantissa/trace.py` always scanned `h`, then `f`, then `d`; only the prose was
wrong.
