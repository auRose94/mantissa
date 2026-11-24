# Porting the Mantissa-1

A port is another implementation of the same machine: the same cells go in,
and the same things have to happen.  This file is the contract.  The
reference is the Python simulator in `mantissa/`; `README.md` is the
instruction set.

## What a port must do

1. Load an image into cell memory at address `org`, set the registers to
   zero, set `f31` to `sp`, and set the pc to `entry`.
2. Fetch, decode and execute exactly as `mantissa/machine.py` does, instruction
   for instruction, until `HALT`, a fault, or a step limit.
3. Produce the same trace as the reference, line for line.

## The trace file

`python -m mantissa trace prog.mant --out prog.trace` records a run.  The file
starts with the image and goes on with the trace.  The header:

    # mantissa-1 trace
    cells 65536      how many cells the machine has
    org 0            where the image starts
    entry 0          the first pc
    sp 65536         what f31 starts as
    image 120        how many cells the image has
    0000 0000 ...    the cells, four hex digits each, most significant first, 16 to a line

Then the line `run`, and after it one line per step that begins with `S`,
followed by the lines for what that step changed:

    S 0 0                        step 0 begins at pc 0
    R 0 0 f5 3ff0000000000000    a register changed
    M 4 16 32 4 3ff0000000000000  a store
    P 5 20 4024000000000000       PUT handed a value over
    C 6 24 33                     PUTC printed a character
    X 3 12                       nothing changed
    H                            halted

- `R step pc register value`.  The register is `h0` to `d31`.  `SPLIT` and
  `MANT` write two registers, so they give two `R` lines: file by file, all
  half registers first, then the floats, then the doubles, and by register
  number within a file.  Not in the order of the instruction.
- `M step pc address cells value`.  `cells` is 1, 2 or 4.
- `P step pc value`.  `PUT` hands a value over.  How a console spells it is
  not part of the machine.
- `C step pc code`.  `PUTC` prints the character with that Unicode code
  point, as a decimal number.
- A step that changed nothing gives one `X` line instead.

A value is always the 64 bits of the double that holds it, as sixteen hex
digits.  A half or a float is widened to a double first.  A NaN that
arithmetic produced is the quiet NaN `7ff8000000000000`.

The last line is `H` for a halt, `F <kind>` for a fault (`address`,
`illegal-instruction`, `not-a-character`), or `L` when the step limit was
reached.

Stores alone tell memory apart from the image, so memory is not listed again.
Flags are not traced: they are not part of what a program can observe.

## The places a port goes wrong

- **Fused multiply and add.**  Compilers like to turn a multiply followed by
  an add into one `fma`.  `MUL` and `ADD` are two instructions here and must
  round twice; only `FMA` rounds once.  With HIP and Clang, compile with
  `-ffp-contract=off`.
- **Division and square root.**  They must be correctly rounded IEEE 754.
  Relaxed or approximate modes are wrong.
- **Half and float arithmetic.**  Compute it in double and round once at the
  end, or use hardware half and single arithmetic that is IEEE 754 with
  gradual underflow.  Either is right: a double has more than twice the
  significand of a float, so the two agree.
- **Narrowing from a double.**  Round once, straight from the double.  Going
  double, float, half rounds twice and is sometimes wrong.
- **NaNs.**  Arithmetic on a NaN gives the quiet NaN with no payload, in every
  format.  `MOV`, `NEG`, `ABS` and `CPSGN` copy a NaN as it is.
- **Signed zero.**  `-0.0` is a value of its own, and traces show it.
- **The order of a decimal decode.**  An instruction number must be a whole
  number from 0 to 99,999,999, its unused fields 00, its used fields under 96,
  and its opcode one that exists.  Anything else is an `illegal-instruction`
  fault before anything executes.  The operand type rules are faults the same
  way; `machine.py` has them all.
- **A fault changes nothing.**  Registers keep their values from before the
  instruction that faulted, and so does memory.
- **Self-modification.**  A store is seen by the very next fetch.  Never cache
  an instruction across a store.

## Golden traces

`golden/` holds traces recorded by the reference, one per demo program.
Between them they execute every instruction, and they include a run that
halts, one that faults and one that stops at the step limit.  The tests check
that they still match the reference and still cover every opcode.  The
Mandelbrot is too long to keep and is recorded when needed:

    python -m mantissa trace programs/mandel.mant --out /tmp/mandel.trace
