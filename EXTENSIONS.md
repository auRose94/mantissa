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

The address ramp (2026-11-24) is what makes the fat end of this real: `LD`
and `ST` now read their address from any register, and the pointer's type is
its reach -- a half reaches 2,048 cells, a float 2^24 (32 MiB), a double
2^53, about 18,014 TB of cells.  A 4K framebuffer at one double per pixel is
33 million cells -- past the float wall, a double's everyday errand.  A
512^3 voxel volume is 134 million cells; same story.  A `SPLIT` pair carries
a cell number past the float wall and `JOIN` hands it to a double pointer:
far pointers, with `CVT` as the cast.  Jump targets stay floats -- a half
link that rounds a return address is a crash, not a pointer.  The OOP door
this opens is ordinary: objects at double pointers, fields as base+offset
pairs, vtables as float function pointers under the 32 MiB of code.

The useful rule for all of these: the machine computes, the console watches.
No instruction ever touches a window.

## Quad: the guard band became a register file

The operand fields are two decimal digits -- a hundred codes -- and three
files of 32 registers fill 00 to 95.  Codes 96 to 99 were the guard band;
they are now `q0` to `q3`, four registers of IEEE 754 binary128 (113-bit
significand, 16 bytes, 8 cells).  Because instructions take their type from
their registers, every opcode lights up for quad without a new encoding:
add, mul, sqrt, fma, remainder, min, the casts, nextafter, classify.  The
reference computes quad exactly -- integer arithmetic end to end, rounded
once -- which the narrow formats cannot do and quad does not need to fake.
The counting wall is 2^113: `quad.mant` squares 3,037,000,500 exactly, past
what any double names.  An inline operand is still a double, so `LDI.Q`
sees a double literal's true value.

The ports: the fleet's mutation engine refuses quad fields until
`gpu/src/machine.h` can run them (its reference evaluator would disagree
with the ISA otherwise); `gpu/test.sh` and `fpga/test.sh` skip
`golden/quad.trace` with a message, by the same pattern simd used.  The
GPU has no native float128 -- the plan there is software binary128 over
three doubles, ~159 internal bits rounded once to 113, because two doubles
carry ~106 and that is 7 bits short of honest.  The FPGA's format code is
two bits with no spare slot, so quad there starts with widening.

Octuple is real: IEEE 754 has specified binary256 as an interchange format
since 2008 (236-bit significand, 19-bit exponent, about 71 decimal digits),
with no required arithmetic -- a machine that implements it defines what
implementing means.  It has no room in the two-digit fields, so octuple is
ISA v3: twelve-digit instructions (`ooo aaa bbb ccc`, still exact in a
double), fields carrying up to 999 codes.

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

## The self-interpreter: `programs/selfemu.mant`

The machine interpreting itself, in hand assembly -- the self-emulation the
next section extrapolates, built.  The target program sits in the
interpreter's own memory, between the
labels `target` and `end`, and its loads and stores address that memory
directly, so a self-modifying target works without doing anything about it.
The interpreter walks it: fetch the instruction number with `LD`, chop and
mod it into opcode, operand fields and two dispatch digits, compare-chain
through class and unit, and perform the effect.

- **The register file** is 96 doubles at `regs`, one per register code,
  every value widened.  Honesty about types is one CVT pair: before
  arithmetic both operands are narrowed to the target's own format, the
  operation runs there, and the result is widened back -- one rounding, in
  the target's type, which is the machine's rounding.  Loads, stores, MOV
  and LDI round only where the machine does, because a stored value is
  always the exact widening of its format.
- **PUT and PUTC** of the target are performed by the interpreter's own
  PUT and PUTC of the same widened value; the console sees identical bits,
  so the comparison against the reference is exact.
- **Coverage**, in any mix of the three types: NOP HALT MOV LDI LD ST ADD
  SUB MUL DIV, BR BSGN BEQ BNE BLT BLE BZ BNZ JAL JMP, PUT PUTC.  Anything
  else stops the interpreter if a target uses it.
- **Faults**: a pc off `[target, end)` or off a four-cell line, an unknown
  opcode, or an operand field over 95 stops the interpreter with its own
  HALT.  A finishing target and a faulting one are indistinguishable from
  the console -- the reference keeps them apart, and the tests use legal
  targets.  The type rules are not re-checked; the assembler refuses such
  targets first.

`tests/test_selfemu.py` runs seven targets both ways -- directly, and
spliced in at the `target` line -- and compares the console output value
for value, both runs ending in HALT.  Measured slowdown, interpreted steps
per direct step:

| target | direct steps | interpreted | slowdown |
|---|---|---|---|
| demo string printer | 200 | 8,531 | 42.7x |
| hello string | 240 | 10,235 | 42.6x |
| counting loop | 56 | 2,692 | 48.1x |
| branch tour | 30 | 1,241 | 41.4x |
| load/store round trip | 23 | 976 | 42.4x |
| subroutine, BSGN | 28 | 1,158 | 41.4x |
| rounding in half and float | 20 | 882 | 44.1x |

About 40 to 50 times, not the traditional few hundred: one target step costs
roughly 30 instructions of fetch, decode and dispatch plus a handful in the
handler, and nothing here emulates what is already the same machine.  The
natural next rungs are the conversions this subset leaves out -- CVT, SPLIT
and JOIN are dispatch entries, not inventions -- and a compiler that emits
this style of interpreter around a larger subset.


## The search: evolution over programs on the fleet

The first extrapolation of the section below is now built, host-driven:
`gpu/python/evolve.py` evolves Mantissa-1 programs, the population is the
array fleet, and one generation is one `mantissa-gpu array` run.

The task: a fixed skeleton reads a case's x into d0, passes it through four
evolved instruction slots, and stores d4, looping over the cases patched
into the machine.  A candidate is four whole instruction numbers, one
d0 -> d4 function, and fitness is the sum of squared error of the picks
against the reference's exact answer for x^2 + x.  One candidate is one
machine; the population rides to the fleet as the job's patch block, so the
genome *is* the machine's input patches.  The host scores, keeps the
elites, and sends mutated children plus a few fresh random bodies as the
next generation's patches.

Mutation is the SMC arithmetic of `smc_edit.mant`, done on the host:
`+1000000*k` retargets the opcode, `+10000*k` the first operand field,
`+100*k` the second, `+k` the third.  Every edit must still decode
(`mantissa.isa.decode`) and stay four cells, or it is redrawn.  The ISA's
decimal layout does half the work: the arithmetic opcodes are consecutive
(20 ADD, 21 SUB, 22 MUL) and neighbouring registers have neighbouring codes
(d0 is 64, d1 is 65), so small arithmetic walks the ISA semantically.

**The float wall cuts through the instruction numbers.**  20010203 is past
16,777,216, where a float counts by twos -- in fact no float holds it at
all: the nearest float is 20010204, and `fp.convert(fp.F, 20010203.0)`
returns exactly that with the inexact flag as the only trace.  Arithmetic
done in floats lands one register off before anything is added.  The host
therefore mutates on the integer the whole double names and stores back as
a double, which counts by ones to 2^53 and is exact for every eight-digit
instruction.  A program doing this to itself needs `.D` registers, which is
what `smc_edit.mant` already does.  `test_evolve.py` asserts the exact path
against the float one.

Measured, at the defaults (population 256, 8 cases, 4 slots, seed 6) on the
Radeon 8060S: sum of squared error 0.0 -- all eight outputs exact, and equal
to the reference's -- at generation 32 of a 900 budget, in 2.5 s, about
75 ms a generation, of which the kernel is 1 ms: the fleet does the running,
the host does the thinking, and a generation costs a process spawn, not a
kernel.  The visible descent (1438.5, 1302.5, ..., 112.5, 70.5, 46.5, 2.75,
0) is the graded landscape working.  Six of the seven other seeds tried in
the same budget converged too -- generations 44, 371, 651, 51, 791 and 47,
the slowest in 58 s -- and one stalled at squared error 70 for the whole
run.  The winners are the programs a human would write (`DIV d4,
d0, d1` then `FMA d4, d0, d0`; or `MUL d4, d0, d4` after `ADD d4, d0, d1`),
reached by arithmetic on numbers.  The same loop runs unchanged on the
reference simulator alone (`--demo --cpu`), so the GPU is speed, not a
requirement.

What this is not: learning, and not yet *inside* the machine.  The host
drives the loop; the machines only run what they are patched with.  A
population as cells, mutating its own fitness in place -- evolution the
machine performs on itself -- is still road, not building.

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
