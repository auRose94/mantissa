#!/usr/bin/env bash
# Run every Yosys script here that has something to run, and print each one's
# cell count.  Logs go to build/yosys/<script>.log.
set -u
cd "$(dirname "$0")/.."
mkdir -p build/yosys
status=0
for script in core_blackbox cell_ram; do
    [ -f "synth/$script.ys" ] || continue
    log="build/yosys/$script.log"
    if yosys -q -l "$log" -s "synth/$script.ys" > /dev/null 2>&1; then
        echo "== $script: $(grep -c . "$log") log lines, $log"
        # the last stat block: the count of each kind of cell
        awk '/^=== /{block=""} {block=block $0 "\n"} END{printf "%s", block}' "$log" | grep -v "^$" | tail -40
    else
        echo "== $script: FAILED (see $log)"
        grep -m5 "ERROR" "$log"
        status=1
    fi
done
exit $status
