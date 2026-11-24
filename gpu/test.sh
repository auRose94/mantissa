#!/usr/bin/env bash
# Builds the GPU port and checks it against the reference:
# the arithmetic on test vectors, every golden trace plus the Mandelbrot by
# co-simulation, and the array demo with a sample replayed in the reference.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/gpu/src"
BUILD="$ROOT/gpu/build"
WORK="${TMPDIR:-/tmp}/mantissa-gpu-test"
mkdir -p "$BUILD" "$WORK"
fail=0

say() { printf '%s\n' "$*"; }
check() { [ "$1" -eq 0 ] || fail=1; }

say "== building =="
hipcc -O2 -ffp-contract=off -std=c++17 -I "$SRC" -o "$BUILD/mantissa-gpu" "$SRC/main.cpp" 2> "$WORK/cc.log"
check $?
hipcc -O2 -ffp-contract=off -std=c++17 -I "$SRC" -o "$BUILD/fptest" "$SRC/fptest.cpp" 2>> "$WORK/cc.log"
check $?
grep -E "error" "$WORK/cc.log" | head -5
if [ ! -x "$BUILD/mantissa-gpu" ] || [ ! -x "$BUILD/fptest" ]; then
    say "build failed"
    exit 1
fi

say "== arithmetic against the reference =="
python3 "$ROOT/gpu/python/make_vectors.py" "$WORK/vectors.bin" --per-op 400 > /dev/null || exit 1
"$BUILD/fptest" "$WORK/vectors.bin" | tee "$WORK/fptest.log"
check $?
grep -q " 0 disagree" "$WORK/fptest.log" || fail=1

say "== co-simulation: every golden trace, and the Mandelbrot =="
( cd "$ROOT" && python3 -m mantissa trace programs/mandel.mant --out "$WORK/mandel.trace" > /dev/null ) || exit 1
pass=0
failcosim=0
for t in "$ROOT"/golden/*.trace "$WORK/mandel.trace"; do
    if "$BUILD/mantissa-gpu" cosim "$t" 2>&1 | grep -q "cosim PASS"; then
        pass=$((pass + 1))
    else
        failcosim=$((failcosim + 1))
        say "FAIL: $(basename "$t")"
    fi
done
say "cosim: $pass PASS, $failcosim FAIL"
check "$failcosim"

say "== the array demo: a Mandelbrot set, one machine per pixel =="
python3 "$ROOT/gpu/python/make_job.py" "$WORK/mandel.bin" --width 64 --height 24 --steps 1000 > /dev/null || exit 1
"$BUILD/mantissa-gpu" array "$WORK/mandel.bin" --out "$WORK/mandel.result" > "$WORK/array.log"
check $?
grep -E "kernel|instructions ran" "$WORK/array.log"
python3 "$ROOT/gpu/python/render.py" "$WORK/mandel.result" --width 64 --height 24 --crosscheck 60
check $?

say "== the game: a frame loop with a referee and per-pixel machines =="
python3 "$ROOT/gpu/python/run_game.py" --frames 6 --pngs "$WORK/game" > "$WORK/game.log" 2>&1
check $?
grep -E "frames in" "$WORK/game.log" | tail -1
grep -q "score" "$WORK/game.log" || fail=1

if [ "$fail" -eq 0 ]; then
    say "== ALL PASS =="
else
    say "== FAILURES ABOVE =="
fi
exit "$fail"
