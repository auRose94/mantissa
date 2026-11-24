# The GPU port

The Mantissa-1 running on an AMD Radeon 8060S (Strix Halo, gfx1151) through
HIP.  One C++17 source tree, two programs, one Python helper directory.

## What is here

- `src/fp.h`: the arithmetic unit, device and host (`M1_HD`).  Every operation
  of `mantissa/fp.py`: round to nearest even, gradual underflow, signed
  zeros, a canonical NaN from arithmetic, CHOP/FLOOR/CEIL/ROUND with and
  without a step, MOD, REM, CMP, SPLIT, JOIN, MANT, SCALB, LOGB, NEXT,
  CLASS, CVT.  Narrow results are computed in double and rounded once.
- `src/machine.h`: the processor: cell memory, the decimal instruction
  decode with the operand type rules, one `step` function, and a run loop
  that records what a step changed.  A store is seen by the next fetch.
- `src/kernels.h`: `cosim_kernel` (one machine, traced) and `array_kernel`
  (many machines, one work-item each, private memory, values patched in and
  picked out).
- `src/main.cpp`: the host: `cosim`, `array`, `info`.
- `src/fptest.cpp`: feeds test vectors through the arithmetic unit on the
  host and on the GPU and compares against the reference's answers.
- `src/align.h`: works around a SIGSEGV inside this ROCm release (see
  NOTES.md).
- `python/make_vectors.py`: test vectors from `mantissa/fp.py`.
- `python/make_job.py`: builds array jobs; `python/render.py` renders their
  results and replays a sample in the reference.
- `kernel/mandel_pixel.mant`: the per-pixel kernel, as documentation.

## Build and test

    bash gpu/test.sh

That builds both programs and runs everything:

- 49,200 arithmetic cases against the reference, on the GPU: 0 disagree.
- Co-simulation of every trace in `golden/` plus the Mandelbrot
  (298,595 steps, 597,191 trace lines): all PASS, line for line.
- The array demo: 1,536 machines, one per pixel, drawing the Mandelbrot set
  in about a millisecond of kernel time (~0.3 giga machine-instructions a
  second in aggregate), with 60 instances replayed in the Python reference:
  all agree.

## Usage

    mantissa-gpu cosim TRACE.trace [--out OUT.trace] [--steps N]
    mantissa-gpu array JOB.bin [--out RESULT.bin]
    mantissa-gpu info

`cosim` runs the image in a reference trace file and writes the same trace
format, so the two can be diffed; given the reference's own run it says
PASS or FAIL and where they differ.  `array` runs the job file that
`python/make_job.py` writes: many machines, one work-item each, values
patched into private memory before the run and picked out after it.

## Compile flags that matter

- `-ffp-contract=off`: without it the compiler fuses `MUL` then `ADD` into
  one `fma`, and `MUL`/`ADD` are two instructions here that must round twice.
- No fast-math: division and square root must be correctly rounded.

## A game on the machine

`python3 gpu/python/run_game.py --frames 180 --record dodge.mp4` plays
Mantissa Dodge: a checkerboard corridor, three obstacle balls coming down the
lanes, and a white ball the player steers.  Every pixel of every frame is
computed by its own Mantissa-1 machine on the GPU (a ray cast per pixel:
`SQRT` for the ray, `CHOP` and `MOD` for the checkerboard, the quadratic
formula for the balls, a packed colour out); a second, single-machine job is
the referee, which advances the obstacles and tests collision; the host is
only the console, carrying alive and score between frames.  `test.sh` runs a
few frames of it.

## What the numbers mean

An interpreter on a GPU is many slow machines in parallel, not one fast one.
One machine on one work-item does about 170,000 instructions a second (the
Python reference does about 350,000); the same kernel across 1,536 machines
moved about 0.32 giga machine-instructions a second in aggregate.  The unit
of speed is the fleet, not the machine.
