# The FPGA port

The Mantissa-1 as a soft core in SystemVerilog, verified instruction for
instruction against the reference with Verilator, and elaborated by Yosys.
The architecture, the cycle counts and the resource numbers are in
`NOTES.md`, next to everything the tools did surprisingly.

## What is here

- `rtl/mantissa_core.sv`: the core: cell memory interface, the three register
  files, fetch, decimal decode with the operand type rules, execute.  Every
  instruction, every fault, and a store is seen by the next fetch.
- `rtl/mantissa_cell_ram.sv`: 16-bit cell memory a vendor tool can map to
  block RAM.
- `rtl/mantissa_top.sv`, `rtl/mantissa_pkg.sv`: the demo top and the shared
  types.
- `rtl/model/fpu_model.sv` + `rtl/model/fpu_model_dpi.cpp`: the floating
  point, as a simulation model with exactly the reference's semantics.  This
  is what every trace replay runs against.
- `tb/tb_main.cpp`, `tb/tb_fpu.cpp`: the testbenches.  `tb_main` reads a
  reference trace file, runs the core, and writes a trace in the same format
  to diff against.
- `tools/gen_fpu_vectors.py`: test vectors from `mantissa/fp.py`.
- `tools/fuzz_programs.py`: generates random Mantissa-1 programs, runs each
  on the core and on the reference, compares.  It found real core bugs the
  golden traces had not (see NOTES.md).
- `synth/`: Yosys scripts; `make lint` elaborates with Yosys to catch what
  Verilator forgives.

## Run everything

    bash fpga/test.sh                 # builds, replays every trace, fuzzes
    bash fpga/test.sh --quick         # no Yosys, a short fuzz
    bash fpga/test.sh --fuzz 5000     # more random programs

In order: the arithmetic unit against reference vectors, operation by
operation; then every trace in `golden/` and the Mandelbrot replayed on the
core, line for line against the reference; then (if `rtl/synth/fpu_syn.sv`
exists, the synthesizable FPU) the same again; Yosys cell counts; and a few
thousand random programs differentially tested.

## The line between simulated and synthesizable

The core and every non-floating-point operation are synthesizable and are
what Yosys elaborates.  The floating point runs in simulation through the
model in `rtl/model/` (Verilator `real` arithmetic plus a DPI `fma`), behind
the same module interface a synthesizable unit would take.  `NOTES.md` lists
exactly what a hardware FPU must implement to drop in, and what Yosys 0.66
cannot read, so the path from here to a real board with vendor floating-point
IP is written down rather than guessed.
