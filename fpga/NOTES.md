# Notes on the reference, the contract and the tools

Things found while building the port that are not in `README.md` or
`PORTS.md`, or read differently there than the reference behaves.  In every
case the port does what `mantissa/` does.  The code is right; this is where it
surprised me.

## The contract as applied

The branch carries the coordinator's contract patch (`b8f6b3f`): no `O` lines;
`P step pc value` and `C step pc code` instead; `R` lines file by file; `f31`
starts at the header's `sp`; the reference runs at about 350,000 instructions a
second.  It was applied cleanly with `git am` and the reference's own 156
tests pass on it.

## A value in a trace is written little-endian

`PORTS.md` says a value is "the 64 bits of the double that holds it, as
sixteen hex digits".  The reference writes it with
`struct.pack("<d", value).hex()` (`trace.bits`), so the sixteen digits are the
double's eight bytes *least significant byte first*.  `1.0` is
`000000000000f03f`, not `3ff0000000000000`.  The image cells in the header are
the other way round, most significant first.  The testbench prints values the
reference's way.

## Whether a register "changed" depends on Python object identity, for NaNs

`trace._same(a, b)` is `a is b or (a == b and same sign of zero)`.  For every
value but a NaN that is "same bits".  For a NaN, `a == b` is false, so a
register counts as changed unless the Python object in it is the very same
object:

- every NaN that arithmetic produces is one shared object, `fp.NAN`
  (`7ff8000000000000`); writing it over itself in the same register is *no*
  change, an `X` line;
- `LD` makes a new object every time, so loading the same NaN into the same
  register twice gives an `R` line each time, with the same sixteen digits;
- `MOV` and `SPLIT` hand the object on (the destination then holds the source's
  object), `MANT` does not (it goes through `convert`, which returns `fp.NAN`);
- `NEG`, `ABS` and `CPSGN` make a new object, so `ABS h0, h0` on a NaN gives an
  `R` line whose value is the same as before;
- `LDI` goes through `convert`, so a NaN literal is the shared object.

A machine has no objects.  The core therefore says, for each register it
writes, where a NaN came from (`ev_wprov`: the shared NaN, inherited from
register `ev_wsrc`, or a fresh one), and the testbench keeps a tag per
register to tell the objects apart.  A fresh NaN gets a tag no register holds,
the destination's old tag included.  That is the only place the port keeps
state that is not the machine's, and it is in the testbench, not in the core's
architecture.  On a board with no trace the provenance bits carry no meaning.

## The reference raises on two operand sets instead of running

`python -m mantissa run` dies with a traceback (an uncaught `OverflowError`,
not a fault) for:

- `DIV` in every format when the divisor is ±infinity and the dividend is a
  finite nonzero number: `fp.div` goes on to `_quotient_exact`, which calls
  `as_integer_ratio()` on infinity;
- `SQRT` to half or to float of a number whose root rounds up to infinity in
  that format (from 4,292,870,400 for a half), by the same route through
  `_root_exact`.  No register can hold such an operand legitimately (a half or
  float register holds a half or float value, whose root is small), so this one
  is practically unreachable.

A machine cannot crash, so the port answers as IEEE 754 would: a signed zero,
exact, for the first; infinity with overflow and inexact for the second.  No
trace of the reference exists for these, so they are not compared against it.
`tools/gen_fpu_vectors.py` lists the operand sets it finds in
`build/fpu_vectors.txt.port_defined`, and `tb_fpu` runs them as the port
defines them.  A random program that meets one is counted as "the reference
could not run" by the fuzzer and skipped.  In 44,000 random programs it
happened 26 times.

## `REM` and `MOD` are whatever the C library does

`fp.remainder` and `fp.fmod` call `math.remainder` and `math.fmod`, which are
libm.  The model calls the same libm through DPI, so it agrees with the
reference on this machine by construction.  For the synthesizable unit I
probed what that libm (glibc 2.44) computes: over 400,000 random operand
pairs, 189,345 of them with quotients above 2^52, `remainder` matched the exact
definition (x − n·y with n the whole number nearest x/y, ties to even) every
time.  So the hardware implements that definition.

## `JAL` writes a double into a float register without rounding

`JAL` leaves `pc + 8` in a float register as a Python float, not converted to
the float format as `LDI` and every arithmetic result are.  For any memory a
float can address up to 2^27 cells the value happens to be exact, so it
changes nothing in practice; beyond that the "float" register would hold a
number that is not a float.  The port keeps all registers as the 64-bit double
and writes the link unrounded, so it behaves the same if it ever matters.

## `to_bits` truncates a value that is not of the format

`fp.to_bits` is only ever given values of its format, where rounding and
truncating agree.  Given another double, the half path truncates
(`int(m * 2048.0)`) and the float path rounds once (a C cast).  The port
narrows once in both cases; only the `JAL` case above could tell them apart.

## `NEXT` around zero

`fp.nextafter` steps the bit pattern by one toward larger magnitude when `y`
is on the far side of `x` from zero.  From the smallest subnormal toward zero
this lands on +0 even for a negative `x`, and from −0 toward a positive `y` it
lands on the largest finite value (0x8000 − 1).  The port keeps the rule as it
is; a hardware designer would write `nextafter` differently, and should not.

## The step limit is not in a trace file

`L` says the limit was reached, not what it was.  The testbench replays a file
that ends in `L` to as many steps as the file has `S` lines, and any other
file to 1,000,000 (the reference's default), unless told otherwise.

## What Verilator 5.054 does

- `shortreal` promoted to `real` is refused, so the model narrows a double to
  a float through DPI.
- `longint'(x)` on a real rounds to nearest (ties away from zero), where
  Python's `int(x)` truncates; and `$rtoi(x)`, right on a module port, came
  back as −2^31 for 2^23 on a function's argument.  The model truncates on
  the bits.
- `$clog2` on a signal is a ceiling, as specified; `bit_length` is one more
  than that only for a power of two.

## What Yosys 0.66 reads

This build takes a smaller subset of SystemVerilog than its documentation
suggests.  The synthesizable RTL is written to it:

- no `return` in a function: assign to the function's name; early exits become
  `if`/`else`;
- no `import` in a module body and no `module m import pkg::*; (...)` header:
  names are spelled `mantissa_pkg::name`;
- no `case inside`, no `$clog2` of a signal, no `casez` ranges needed;
- package functions cannot refer to the package's enum members by name.

`proc` turns an incompletely assigned `always_comb` variable into a latch and
says so, which is how one was caught in the core (`cell_off`); Verilator had
said nothing at the warning levels in use.  `make lint` elaborates with Yosys
for that reason.

## Bugs the golden traces did not show

The thirteen golden traces and the Mandelbrot passed while the port still had
these.  `tools/fuzz_programs.py` found the first three on its first runs:

- the stack pointer was put in the register with code 95, which is `d31`; `f31`
  is code 63;
- the cell of the word a load or a store starts at was taken from the low bits
  of the address double instead of from the cell number;
- a float stored at a cell congruent to 2 modulo 4 wrote cells 1 and 2 of the
  word, not cells 2 and 3 (no golden trace stores a float there);
- `NEG`, `ABS` and `CPSGN` labelled a NaN they wrote as the shared one.

It also found two faults in the testbench's own bookkeeping: its copy of the
registers started `f31` at zero rather than at the stack pointer, and a fresh
NaN could be given the tag of the NaN it replaced.

One more was found by reading `PORTS.md` rather than by a failing program, and
is now covered by the fuzzer: the machine has the number of cells the header
says, not the number the build provides, so a trace recorded with 1,000 cells
must fault for address 1,000.

## The quad arithmetic dispatch: the window's below-mass sign (WIP, on the branch `fpga-quad-arith`)

The model dispatches quad arithmetic to a software binary128 (`quad_fp.h`,
the HIP port's engine copied over), and `c_quad_op` answers op by op.  It is
right for everything except a near-tie family: when a product's exact value
sits within about half a quantum of the truncation, the window's under-mass
accounting can carry the wrong sign, and the answer lands one ulp off.
Repros (from the fpu vectors, seed 4):
  - MUL 9bbaafbd619699cfe1988ad9f06c144a * f8377019c381e88f38c0c8fd8712b8bc
    (the engine rounds down; the reference rounds up),
  - FMA 0000ffff... * 52a6670d... + 0000ffff... (the same family),
  - ROUND_TO: exact multiples of the step answer one step high (39 cases at
    the 150k vector count).
The window placement itself was corrected tonight (each piece's shift used
W-off-bl, correct only for the piece at the very top of the window; it is now
W+off-bl), the compare/logb/scalb/split operand orders and sign conventions
are fixed, and fp.py grew the matching guards.  What remains is the residual
sign through q_prod_pieces/q_chunks' borrow and the below-mass comparison at
in-window remainders of 0 and 1 -- the branch's vectors file reproduces all
of it.

## The vector instructions

Built: opcodes 72 to 85 (VADD, VSUB, VMUL, VDIV, VFMA, VMIN and VMAX over 2
or 4 consecutive registers; see EXTENSIONS.md) run on the core, lane by lane
through the FPU, and `golden/simd.trace` replays with the rest.  Four things
came up on the way:

- **the event capture grew to four register writes a step** (a VADD4 writes
  four), with the values held in their own registers and copied to the event
  on the last lane, so the event pulse still carries the whole step;
- **the last lane is its own exception**: it completes the very cycle the
  event is built, so its slot reads the sources (`r_a + r_lane`, `fpu_y`)
  instead of the holding register it is writing that same edge.  Reading the
  holding register emitted the previous vector instruction's last lane.
- **the lane count needs its width**: `lanes_of` first returned two bits, so
  a four-lane op wrapped to zero, and in `r_lane == lanes_of - 1` the unsized
  literal widened the expression to 32 bits -- `0 - 1` is not `3` there.  The
  core spun inside the first VADD4 forever; a three-bit return fixed every
  use site at once.
- **the predicate ordering** that bit the GPU kernel (the distance must be
  read before the lanes are updated -- see EXTENSIONS.md) applies here too,
  but the core escaped it: each lane's FPU call runs while every register
  still holds its pre-instruction value.

The fuzzer exercises the vector instructions along with everything else.

## The q file on the core

Quad is format 3: the format field widened from two bits to three, the four
register codes 96 to 99 became q0 to q3, and `st_cells = 1 << fb` lands on
eight -- but only once `st_cells` grows a fourth bit.  `3'd1 << 3'd3` is
`3'd0`: the lane-width lesson again, caught this time before the run.

The slice is memory and casts.  A quad store writes its two words of cells on
two cycles and a quad load reads them the same way, most significant word
first, no FPU and no `fpu_done` between them -- the state gate must not wait
on `done` for a quad store, because none was raised and `done` never comes.
LD.Q retires in ST_OPND, and the fetch the advance assumes has to go out
there too: without it the next decode reads whatever the load's second word
left in `mem_rdata`, which in quadmem was eight cells of data, and the core
faulted on an instruction that was never fetched.

A quad operand read for the 64-bit paths -- addresses, PUTC, the sources of
the arithmetic the core does not have -- is the double the quad's value
narrows to, one rounding, plus an exactness flag.  Any address a quad can
name is small and whole and narrows exactly; a narrowing that loses bits is
no address either (the true value is past the end, or not a whole number),
so ST and LD fold the flag into their address checks and fault the same way
the reference would.  PUTC does not fold it in: the reference narrows to a
double there too and asks its questions of that.  The narrowing lives in
`q_slice64`, the same mathematics as the model's `c_q_narrow` into D minus
the flags; the reference crashes outright on an address held in a q register
(`float()` of a Quad raises), so this is the port's own answer and nothing
compares it.

`fpu_fmt2` in ST_OPND is whatever the second operand field decodes from
`mem_rdata` -- which is loaded data by then, not the instruction.  The core
now registers `r_fmt2` at decode and the model routes on `fmt2 == Q` only
for MANT, the one operation that reads its second format.  Before both of
those, an `LD` of a half could answer the canonical NaN.

## The reference loads a quad wrong

`Memory._bits` has cases for one, two and four cells and falls through to
the four-cell formula for everything else, so an eight-cell load assembles
four cells into 64 bits and `q_from_bits` reads that as a subnormal quad.
Every nonzero quad loaded from memory comes back as junk; a store is fine
(it walks `fmt.cells`), so is a load of eight zero cells, which both
machines read as +0.  `golden/quad.trace` never loads, which is why nobody
met it.  `mantissa/` is not this port's to fix; the core implements the
eight-cell assembly the machine specifies, and `tools/quadmem.mant` loads
only zeroed cells.  When the reference is fixed, nothing here changes.

## Quads compare by value, NaN tags and all

The reference's quads are dataclasses and its `_same` is `a == b` for them,
so a quad register changed when the 128 bits in it changed -- no object
identity, no NaN tags.  The testbench keeps four quad registers in its
shadow file and compares bits.  That only works because every quad value
this slice writes is encoding-canonical: the widenings and the loads go
through `q_from_bits` and `convert_into_q`, whose answers always are, so
bit equality and dataclass equality never disagree.  The core still labels
quad writes with provenance (FRESH for a load, INHERIT for a MOV, CANON for
a cast) for the day arithmetic arrives; the testbench ignores it.

## What the event port grew

`ev_wval0`, `ev_sval` and `ev_putval` are 128 bits; `ev_wval1..3` stay at
64 because only the first write of a step can be a quad, and a quad store
or PUT is the only other thing that moves 128 bits.  `ev_putq` says when a
PUT hands over a quad, so the testbench knows to print 32 hex digits: a
quad +0 is all zeros and cannot be told from a double by looking.  An
eight-cell store's width already rides in the M line's cell count.

## What the model answers for quad

Quad CVT goes through DPI: `c_q_widen` is a bit shuffle (a NaN loses its
payload to the shared quad NaN; a subnormal double is a normal quad, far
above quad's floor), `c_q_narrow` is `_round_ratio` over the 113-bit
significand -- the shift to the quantum is always rightward and at least
lead - p + 1, so guard and sticky bits are all that survive and no wide
integer is needed.  A quad MOV, FROMBITS and TOBITS are the identity on the
128 bits; everything else answers the canonical quad NaN, or the shared
double NaN when the answer stays narrow.  Nothing feeds that: the core has
no quad arithmetic, and the fuzzer refuses operand codes 96 and up (the
ISA is legal; the file just is not on the core yet).  The fuzzer also
refuses quad *values* in inline operands -- `fp.FORMATS` grew a Q, and a
Quad cannot ride in a double literal.
