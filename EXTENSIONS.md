# Extensions: SIMD, a screen, a compiler, and how far this goes

The machine is deliberately small; these are the extensions it can grow
without changing what it is.  Each one is specified here well enough to build.
"Specified" is not "built": what exists today is the reference simulator, the
HIP and FPGA ports, the compiler in `compiler/`, and the game.

## SIMD: two- and four-lane register operations

The natural SIMD for Mantissa-1 is not a new register file -- operand fields
are two decimal digits and every code from 00 to 95 is taken.  It is
*consecutive registers*: one field names the first register of a group, and
the operation covers the group.  `VADD4 d8, d16, d24` is

    d8..d11  = d8..d11  + d16..d19,  d24..d27 rounding as doubles

The machine already thinks in vectors of a sort -- memory is a row of cells,
arrays are runs of doubles, and the game's pixel machines are the fleet form
of the same idea.  What was missing is doing four of them with one fetch and
one decode.

    opcode  operation                      lanes
    72      VADD2 rd, ra, rb               rd..rd+1  = ra..ra+1 + rb..rb+1
    73      VSUB2 rd, ra, rb
    74      VMUL2 rd, ra, rb
    75      VDIV2 rd, ra, rb
    76      VFMA2 rd, ra, rb               rd..rd+1 = ra..ra+1 * rb..rb+1 + rd..rd+1
    77      VMIN2 rd, ra, rb
    78      VMAX2 rd, ra, rb
    79      VADD4 rd, ra, rb
    80      VSUB4 rd, ra, rb
    81      VMUL4 rd, ra, rb
    82      VDIV4 rd, ra, rb
    83      VFMA4 rd, ra, rb               rd..rd+3 = ra..ra+3 * rb..rb+3 + rd..rd+3
    84      VMIN4 rd, ra, rb
    85      VMAX4 rd, ra, rb

What it buys, honestly:

- **FPGA**: four lanes through four FPUs -- the one place it is a real
  hardware win, and the reason to build it.
- **HIP**: the interpreter executes four scalar ops; the fleet is already the
  vector unit (1,536 machines are a 1,536-lane SIMD).  Measured: the vector
  Mandelbrot (`gpu/python/mandel_vector.py`) renders the same picture as the
  scalar one from 4 times fewer machines with about 4 percent more total
  instructions -- on the fleet, SIMD buys machine count, not speed.  The
  fetch-and-decode savings are real on CPUs and the FPGA core.
- **Reference**: four calls to the same arithmetic, verified the same way.
- **The predicate trap**: an escaped Mandelbrot lane never returns, so the
  fresh predicate needs no frozen copy -- but it must be computed from the
  distance BEFORE the lanes are updated, and the natural scratch block is
  also where the next iteration's products want to live.  The ordering is
  the whole game; `gpu/python/mandel_vector.py` documents the working one.

## The screen: a framebuffer device

The machine has no output but PUT.  A screen is a memory convention plus a
console that watches it:

    cells [FB, FB + W*H)   one packed colour per pixel (r*65536 + g*256 + b),
                           row major, top left first

- **Cosim mode** (one machine): the console maps those cells to a window and
  refreshes when they change.  The game's ffmpeg pipe is this, minus the
  picks -- the machine writes straight to the watched region.
- **Array mode** (one machine per pixel): there is no shared memory to watch;
  the console reads one cell per machine, which is what the game already does.
  The extension here is batching: a result file that *is* the framebuffer, so
  the console can blit it without decoding.
- **3D textures**: memory layout conventions, not hardware.  A volume is
  `base + (z*Y + y)*X + x`; the compiler can know this and emit the address
  arithmetic.  Nothing in the machine changes.

The useful rule for all of these: the machine computes, the console watches.
No instruction ever touches a window.

## Optimisation patterns, found and measured

Patterns that make Mantissa-1 programs faster, with where they came from:

1. **Load constants once** (the compiler's literal pool).  Inside a loop, an
   inline operand costs its load every iteration; a pooled register costs it
   once.
2. **Dead initialisation** (the compiler's dead-store pass).  `var x = 0.0`
   followed immediately by an assignment emits nothing.
3. **Unroll by writing less** (`smc_emit.mant`).  Self-modifying code can
   write the unrolled loop at run time; the emitted loop has no branches at
   all.
4. **Specialise by patching** (`smc_edit.mant`).  A value that changes once
   per hundred iterations can live *in* an instruction instead of in memory:
   the loop body reads it for free.
5. **Count in limbs** (`bignum.mant`).  Small registers stay exact; carry is
   add and MOD.
6. **SCALB instead of multiply** where the multiplier is a power of two:
   same instruction count, but exact at every magnitude -- it is how the
   compiler will implement `x * 2^k` when it grows constant exponents.
7. **Compare early, branch once**: comparisons write numbers, `BZ` tests
   them; `CMP` + `BSGN` settles three-way choices in two instructions.
8. **The walls are data-dependent** (`stall.mant`): a loop that counts down
   past a wall freezes just as one counting up does.  There is no trick --
   the fix is limbs or a wider format, which is the lesson.

## The compiler

`compiler/m4c.py` compiles a small float-only language to assembly, with
constant folding, `x*1`/`x-0` identities, literal pooling, dead-store
elimination and the wall warnings.  Its tests compile, run on the reference,
and compare against answers computed independently.

The road from here, in order of value:

1. ~~**Arrays**~~: built.  `a[i]` is the base plus `MUL`/`ADD`; arrays are
   zero-filled, aligned, and there are no bounds checks because the machine
   faults on its own.
2. ~~**Functions**~~: built.  `fn name(params) { return expr }`, recursion
   included.  The caller pushes the link, its variable registers and any
   temporaries it still holds, then the arguments; the callee pops its
   arguments and answers in `d0`; the caller parks the answer before
   restoring.  The link goes in deepest, because the callee pops off the top.
3. **Peephole at the assembly level**: `LDI` folded into the instruction that
   uses it (the assembler's literal handling already rounds; the compiler can
   decide format), `MOV` elision across assignments, branch to branch.
4. **Vector intrinsics**: `vadd(a, b)` maps one source line to one `VADD4`.

## Self-programming, intelligence, and how far this goes

The machine's programs are numbers, and it can write them: `smc_emit.mant`
is a program that writes programs.  The honest extrapolations, in order of
how soon they are buildable:

- **Search over programs on the fleet.**  Mutation is arithmetic on
  instruction numbers; fitness is a trace comparison; the population is the
  array-mode fleet.  Every piece exists today (the fuzzer in `fpga/tools/`
  is this, minus the mutation loop).  This is evolution, not learning, and it
  would run *inside* the machine.
- **Gradient learning** needs arrays and matrix products: doubles carry the
  arithmetic, `VFMA4` (or the fleet, one weight per machine) carries the
  parallelism, and the walls bound single-precision accumulation.  Possible,
  and slow, and it would be learning *in* the machine rather than on it.
- **Self-emulation**: a Mantissa-1 interpreter written in Mantissa-1.  Nothing
  forbids it -- code is data, `LD` reads instructions.  Expect a slowdown of
  a few hundred times, which is the traditional price.
- **Self-hosting**: the compiler compiles the compiler.  The language needs
  arrays first; after that it is work, not invention.

How far it goes: the ceiling on a single machine is its own fetch-decode-execute
loop -- about 60 million instructions a second simulated on one CPU core,
about 200,000 a second per machine on the GPU fleet, and on the FPGA, whatever
a 100 MHz soft core with pipelined floating point gives (hundreds of times the
fleet, in one machine).  The ceiling on the *idea* is further out: it is a
complete, self-modifying, float-only computer whose every value -- code,
pixels, physics, and any model you train in it -- is the same kind of number.
