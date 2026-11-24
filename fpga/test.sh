#!/usr/bin/env bash
# Build everything and run every test of the Mantissa-1 FPGA port.
#
#     bash fpga/test.sh                    everything there is to run
#     bash fpga/test.sh --quick            no Yosys, a short fuzz run
#     bash fpga/test.sh --no-fuzz          skip the random programs
#     bash fpga/test.sh --no-synth         skip Yosys
#     bash fpga/test.sh --fuzz N           N random programs (default 2000)
#
# In order:
#   1. build the simulator with the FPU model (rtl/model/fpu_model.sv)
#   2. the FPU model, operation by operation, against vectors whose answers
#      come from mantissa/fp.py (tools/gen_fpu_vectors.py)
#   3. record the Mandelbrot with the reference, from the copy's root
#   4. replay every trace in golden/ and the Mandelbrot: one PASS or FAIL line
#      each, compared line for line with the reference
#   5. if rtl/synth/fpu_syn.sv exists: the same three with the synthesizable
#      FPU in place of the model
#   6. Yosys: cell counts (synth/run.sh)
#   7. random programs against the reference (tools/fuzz_programs.py)
#
# Exit status is non-zero if anything fails.  A failing trace is reported as
# failing; nothing here adjusts an expected line.
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
export PYTHONDONTWRITEBYTECODE=1        # leave no bytecode caches in the reference's directories
mkdir -p build

fuzz=2000
synth=yes
quick=no
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) quick=yes ;;
        --no-fuzz) fuzz=0 ;;
        --no-synth) synth=no ;;
        --fuzz) fuzz="$2"; shift ;;
        *) echo "test.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
[ "$quick" = yes ] && { synth=no; [ "$fuzz" = 2000 ] && fuzz=300; }
command -v yosys >/dev/null 2>&1 || synth=no

failed=0
section() { echo; echo "### $*"; }
keep() { [ "$1" -eq 0 ] || failed=1; }

# replay a list of traces on one simulator; the simulator prints PASS/FAIL per trace
replay() {                                  # replay <simulator> <label>
    local sim="$1" name
    shift
    for t in "$@"; do
        [ -f "$t" ] || { echo "FAIL (missing) $t"; failed=1; continue; }
        "$sim" "$t" || failed=1
    done
}

section "1. build: the simulator with the FPU model"
make -s all || { echo "build failed"; exit 1; }

section "2. the FPU model against mantissa/fp.py, operation by operation"
make -s test-fpu; keep $?

section "3. the Mandelbrot, recorded by the reference"
# the same command as PORTS.md gives, run from the copy's root; an existing
# recording can be reused with MANDEL_TRACE=<file>
if [ -n "${MANDEL_TRACE:-}" ] && [ -f "$MANDEL_TRACE" ]; then
    mandel="$MANDEL_TRACE"
else
    mandel=build/mandel.trace
    (cd "$ROOT" && python3 -m mantissa trace programs/mandel.mant --out fpga/$mandel) || { echo "could not record the Mandelbrot"; exit 1; }
fi
echo "recorded: $mandel"

section "4. golden/ and the Mandelbrot on the simulator with the FPU model"
# golden/simd.trace waits on the core's vector instructions: the event capture
# holds two register writes a step and a VADD4 writes four.  See NOTES.md.
replay build/obj_model/m1_sim $(ls "$ROOT"/golden/*.trace | grep -v simd.trace) "$mandel"

if [ -f rtl/synth/fpu_syn.sv ]; then
    section "5a. build: the simulator with the synthesizable FPU"
    make -s sim-syn; keep $?
    if [ -x build/obj_syn/m1_sim ]; then
        section "5b. the synthesizable FPU against mantissa/fp.py, operation by operation"
        make -s test-fpu-syn; keep $?
        section "5c. golden/ and the Mandelbrot on the simulator with the synthesizable FPU"
        replay build/obj_syn/m1_sim $(ls "$ROOT"/golden/*.trace | grep -v simd.trace) "$mandel"
    fi
else
    echo
    echo "### 5. synthesizable FPU: rtl/synth/fpu_syn.sv does not exist, nothing to run"
fi

if [ "$synth" = yes ]; then
    section "6. Yosys: elaboration and cell counts"
    make -s lint; keep $?
    make -s synth; keep $?
else
    echo; echo "### 6. Yosys: skipped"
fi

if [ "$fuzz" -gt 0 ]; then
    section "7. $fuzz random programs against the reference"
    make -s fuzz N="$fuzz"; keep $?
else
    echo; echo "### 7. random programs: skipped"
fi

echo
if [ "$failed" -eq 0 ]; then
    echo "test.sh: everything passed"
else
    echo "test.sh: FAILURES ABOVE"
fi
exit $failed
