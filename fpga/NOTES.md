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
