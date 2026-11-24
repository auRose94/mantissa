# Mantissa-1

A fictional computer with no integers and no bytes, and a simulator for it.
The name is a placeholder.

Every value in memory is an IEEE 754 **half**, **float** or **double**.  There
is nothing else to put in a register: no integers, no characters, no
pointers.  Addresses are floats, the program counter is a float, a loop
counter is a float, and so is every instruction.

An instruction is a whole number, eight decimal digits long, stored as a
double.  A program can read its own instructions, do arithmetic on them and
write them back.

## Try it

Needs Python 3.10 or newer and nothing else.  From this directory:

    python -m mantissa run programs/hello.mant
    python -m mantissa run programs/stall.mant --steps 36 --panel
    python -m mantissa run programs/smc_edit.mant
    python -m mantissa run programs/smc_emit.mant
    python -m mantissa run programs/bignum.mant
    python -m mantissa run programs/pairs.mant
    python -m mantissa disasm programs/smc_edit.mant
    python programs/mandel_zoom.py
    python -m unittest discover -s tests -t .

Program output goes to stdout, the simulator's remarks to stderr, so a program
can be piped.  `--trace` shows each instruction before it runs, `--panel`
shows registers and flags at the end, `disasm` lists a program next to the
numbers its instructions are.  `PORTS.md` describes `trace`, which records a
run for another implementation to replay.

## The machine

| type | bits | significand | exponent | cells | counts every whole number up to |
|---|---|---|---|---|---|
| `.H` half | 16 | 11 | −14 to 15 | 1 | 2,048 |
| `.F` float | 32 | 24 | −126 to 127 | 2 | 16,777,216 |
| `.D` double | 64 | 53 | −1022 to 1023 | 4 | 9,007,199,254,740,992 |

**Memory** is a row of 16-bit cells and nothing smaller.  A half is one cell,
a float two, a double four, most significant cell first.  Data must sit at a
multiple of its own width.

**Registers**: `h0`–`h31`, `f0`–`f31`, `d0`–`d31`, each holding only its own
type.  By convention `sp` is `f31` (the runner starts it at the top of
memory) and `lr` is `f30`.

**Addresses** are floats holding cell numbers, which caps the address space
at 2^24 cells: 24 bits is all a float's significand has.

**Instructions are numbers.**  An instruction is a double whose value is an
eight-digit decimal number:

    oo aa bb cc      oo   the opcode
                     aa   the first operand
                     bb   the second
                     cc   the third

An operand is a register code: `00`–`31` is `h0`–`h31`, `32`–`63` is
`f0`–`f31`, `64`–`95` is `d0`–`d31`.  A field the instruction does not use
must be `00`; opcode `00` is kept back, so zeroed memory and small numbers do
not run.  Any number that does not decode stops the machine.

So `20010203` is `ADD h1, h2, h3`, and arithmetic edits it:

    20010203 + 10000    20020203   ADD h2, h2, h3    the first operand moves one register on
    20010203 + 320000   20330203   ADD f1, h2, h3    ... or from the half registers to the floats
    20010203 + 5000000  25010203   FMA h1, h2, h3    ... or the opcode itself

Doubles hold these numbers exactly, and chop and mod take them apart again.

An instruction is 4 cells, at a multiple of 4.  An inline operand is a double
in the next 4 cells, so an instruction with one operand is 8 cells and `BSGN`,
with two, is 12.

**Flags**: the five sticky IEEE 754 flags (`invalid`, `divzero`, `overflow`,
`underflow`, `inexact`).  `--panel` shows them.  No instruction reads them.

**Faults** stop the machine; there are no handlers.  `address`, for an
address that is not a whole cell number in range or is misaligned;
`illegal-instruction`, for a number that is not an instruction, or one whose
operands break its type rules; `not-a-character`, for `PUTC` of something
that is not a Unicode code point.

## Instruction set

Typed instructions use registers of one type.  The suffix `.H`, `.F` or `.D`
is optional: every register already says what type it holds.

| opcode | | |
|---|---|---|
| 01 | `NOP` | |
| 02 | `HALT` | |
| 10 | `MOV r, r` | copy |
| 11 | `LDI r, value` | load an inline operand |
| 12 | `LD r, [fa]` | `fa` is a float register holding an address |
| 13 | `ST [fa], r` | |
| 14 | `CVT r, r` | change type, rounding once if bits are lost |
| 15 | `SPLIT r, r, r` | `SPLIT high, low, x` |
| 16 | `JOIN r, r, r` | `JOIN x, high, low` |
| 20–23 | `ADD`, `SUB`, `MUL`, `DIV` `r, r, r` | IEEE 754 arithmetic |
| 24 | `SQRT r, r` | |
| 25 | `FMA r, r, r` | `rd = ra * rb + rd`, rounded once |
| 26 | `REM r, r, r` | IEEE remainder |
| 27 | `MOD r, r, r` | what trunc division leaves over, with the sign of its left operand |
| 28, 29 | `MIN`, `MAX` `r, r, r` | a NaN operand is ignored; `MIN(+0, -0)` is `-0` |
| 30, 31 | `NEG r, r`, `ABS r, r` | sign-bit operations; a NaN passes through unchanged |
| 32 | `CPSGN r, r, r` | magnitude of the first, sign of the second |
| 33, 34 | `FLOOR r, r`, `CEIL r, r` | whole numbers |
| 35 | `CHOP r, r` | shave off everything right of the point |
| 36 | `ROUND r, r` | to a whole number, ties to even |
| 33–36 | `FLOOR`, `CEIL`, `CHOP`, `ROUND` `r, r, step` | the same, to a multiple of `step` (opcodes 37–40) |
| 41 | `SCALB r, r, r` | `ra * 2**trunc(rb)` |
| 42 | `LOGB r, r` | binary exponent of `ra` as a number |
| 43 | `NEXT r, r, r` | next value of the format from `ra` toward `rb` |
| 44 | `CLASS r, r` | 1 quiet NaN, 2 −inf, 3 −normal, 4 −subnormal, 5 −0, 6 +0, 7 +subnormal, 8 +normal, 9 +inf |
| 45 | `MANT r, r, r` | `MANT m, e, x`: `x = m * 2**e` with `0.5 <= \|m\| < 1` |
| 50–52 | `LT`, `LE`, `EQ` `r, r, r` | `1.0` or `0.0`; false if an operand is NaN |
| 53 | `CMP r, r, r` | three-valued: `-1.0`, `0.0` or `1.0`; NaN if an operand is NaN |
| 60 | `BR target` | |
| 61 | `BSGN r, if_negative, if_positive` | ternary: negative jumps to the first target, positive to the second, zero falls through |
| 62–65 | `BEQ`, `BNE`, `BLT`, `BLE` `r, r, target` | `BNE` is taken for NaN |
| 66, 67 | `BZ r, target`, `BNZ r, target` | |
| 68 | `JAL fa, target` | leaves `pc + 8` in `fa` |
| 69 | `JMP fa` | |
| 70, 71 | `PUT r`, `PUTC r` | print the value, or one code point |

A target is a cell address on a four-cell boundary.  Arithmetic needs all of
its registers to be of one type; `CVT` needs two different types; `SPLIT`
writes its two halves to one type and `JOIN` reads its two parts from one
type; every other register of these casts may be of any type.

## Two casts that keep what a plain cast drops

**`SPLIT` and `JOIN` extend precision.**  `SPLIT high, low, x` rounds `x` to
the narrow type for `high` and hands the rounding's leavings to `low`, so
that `high + low` is `x`.  Every whole number up to 2^49 + 2^24, about
5.6 × 10^14, survives a round trip through a pair of floats; a float alone
stops at 16,777,215.  A pair has float's *range*, not double's: too small a
number and both halves underflow away.  The `inexact` flag says when a pair
fell short.

**`MANT` and `SCALB` extend range.**  `MANT` scales a number into the first
binade, where every format has its full precision, and hands back the power
of two it used, as a plain number.  No half can hold 6.02e23; a half holds
its mantissa, and a half holds its exponent, and `SCALB` puts the number
back.

## Self-modifying code

Code and data are the same cells, and nothing stops a program from writing
where it is about to fetch.  Three idioms, one program each:

- **Edit an instruction in place.**  `smc_edit.mant`: the loop's `PUT`
  instruction is its own counter; each pass adds 10000 to move the register
  it prints one along.
- **Write a program.**  `smc_emit.mant`: a subroutine builds instruction
  numbers from an opcode and three operand digits and appends them to a
  buffer; the main program writes an unrolled loop with it, prints what it
  wrote, and runs it.
- **Read a program.**  `selflist.mant`: chop and mod take the machine's own
  instructions apart into opcode and operands.

There is no protection.  Most accidents are loud, because a number with a
stray digit is usually not an instruction; intentions are not.

## Assembly

    ; a comment
    loop:   LD     h0, [f0]       ; label, mnemonic, operands
            LDI.F  f1, buf + 4    ; numbers, labels, + and -, inf, nan
            PUTC.H h0             ; the suffix is optional, and must match
            ROUND  h2, h2, h3     ; round h2 to a multiple of h3
            BSGN   f6, lower, higher
            DATA.H 1, 2           ; constants of one type
            TEXT   "hi\n"         ; one half per code point
            SPACE  16             ; cells held in reserve

Execution starts at `main` if there is one, otherwise at the first
instruction.  A literal is rounded to its register's format the way C rounds
one, so `LDI.H h0, 4095` quietly loads 4096.

## What it is like

Each of these is shown by a program in `programs/` or a test in `tests/`.

**Counters freeze.**  `x + 1` and `x - 1` are exact only while `x` is below
2048 for a half and 16,777,216 for a float.  Past that, `i += 1` leaves `i`
where it was, a loop that waits for `i` to arrive never ends, and the only
trace is the `inexact` flag, which nothing reads.  (`stall.mant`,
`wall24.mant`)

**So integers are built from small ones.**  Four floats of 12 bits each make
a 48-bit counter that walks through the wall, and prints itself in decimal
with nothing but chop and mod.  (`bignum.mant`)

**Addresses are significands.**  Float addresses reach 2^24 cells and no
further.

**There are no bit operations.**  A number has a sign, an exponent and a
significand, and the machine offers exactly those.

**Text is numbers.**  A character is its code point in a half, which covers
U+0000 to U+07FF; anything wider needs a float.

**Code is numbers.**  A program can compute an instruction from four
numbers, and take one apart again.

**Precision is a decision with visible consequences.**  Adding 0.1 a
thousand times gives 105.1875 in half, 99.9990463256836 in float and
99.9999999999986 in double; the same float arithmetic with the rounding error
carried in a second register gives 100.0.  (`tenths.mant`)

**The same program, three machines.**  `mandel_zoom.py` renders one
Mandelbrot program written for each type.  Out of 1,408 characters:

| window width | half against float | float against double |
|---|---|---|
| 2.6, the whole set | 14 differ | 0 |
| 0.02 | 107 | 0 |
| 0.0008 | 392, half has collapsed into blocks | 2 |
| 0.000008 | 1,406 | 46 |
| 0.0000002 | 1,293 | 426, float has collapsed into stripes |

## The simulator

`mantissa/fp.py` holds the formats and the arithmetic, and everything in it is
a pure function over Python floats.  Results for the narrow types are computed
in double precision and rounded once more, which is innocuous because a double
has more than twice the significand and far more exponent range.  The flags
are decided from the true result by exact integer arithmetic, never from a
second rounding.

`step(cpu, memory)` returns a new `Cpu`; the assembler and the disassembler
are functions from text to cells and back.

The tests compare the arithmetic against an exact reference built from
`Fraction` and the `struct` module, about 31,000 random operations per run,
plus every one of the 65,536 half bit patterns.  They also run the Mandelbrot
in all three types against an independent simulation.

The simulator runs about 350,000 instructions a second in CPython 3.14.

`PUT` hands a value over; the command line prints it the way Python prints
a float.  That spelling belongs to the console, not to the machine: what a
port must reproduce is the value.

## Decisions still open

- **Flags**: visible on the panel only.  A program could be allowed to test
  them.
- **Literals**: rounded silently.  The assembler could refuse a literal its
  type cannot hold.
- **Memory protection**: none.  Code could be write-protected, at a price.
- **No transcendental instructions, no input, no traps, no compiler.**

## Layout

    mantissa/   fp, isa, memory, machine, asm, disasm, trace, and the command line
    programs/   the demos, and the Mandelbrot zoom tour
    compiler/   m4c.py: a compiler for a small float-only language
    gpu/        the HIP port: cosim, sim (60M instructions a second on the host),
                array mode, and Mantissa Dodge
    fpga/       the soft core: Verilator-verified, Yosys-elaborated, fuzzed
    tests/      the arithmetic against exact arithmetic, then everything else
    golden/     recorded runs for other implementations to replay; see PORTS.md
    EXTENSIONS.md  SIMD, a screen device, optimisation patterns, and the road

The name is Mantissa.  It stuck.
