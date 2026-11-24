# Mantissa-1 on an FPGA: build, test, synthesize.
#
#   make            build the simulator with the FPU model      (build/obj_model/m1_sim)
#   make test       everything: see test.sh
#   make lint       Verilator lint and a Yosys elaboration of the synthesizable RTL
#   make test-fpu   the FPU model against the vectors tools/gen_fpu_vectors.py writes from fp.py
#   make fuzz N=..  random programs against the reference (smoke test; N defaults to 300)
#   make synth      Yosys cell counts, in build/yosys/
#   make clean
#
# Everything this makes goes under build/.

PYTHON    ?= python3
VERILATOR ?= verilator
YOSYS     ?= yosys
BUILD     := build
export PYTHONDONTWRITEBYTECODE := 1     # never leave bytecode caches in the reference's directories

RTL    := rtl/mantissa_pkg.sv rtl/mantissa_core.sv rtl/mantissa_cell_ram.sv rtl/mantissa_top.sv
MODEL  := rtl/model/fpu_model.sv rtl/model/fpu_model_dpi.cpp
VFLAGS := -Wno-fatal -Wno-DECLFILENAME -Wno-VARHIDDEN -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC

.PHONY: all test lint test-fpu fuzz synth clean help
.DELETE_ON_ERROR:

all: $(BUILD)/obj_model/m1_sim

# ---- simulators -------------------------------------------------------------------
# The core is the same in both; only the file that defines mantissa_fpu differs.
$(BUILD)/obj_model/m1_sim: $(RTL) $(MODEL) tb/tb_main.cpp
	$(VERILATOR) --cc --exe --build --timing -j 0 $(VFLAGS) --top-module mantissa_top \
	    -Mdir $(dir $@) $(RTL) $(MODEL) tb/tb_main.cpp -o m1_sim

$(BUILD)/obj_fpu_model/tb_fpu: rtl/mantissa_pkg.sv $(MODEL) tb/tb_fpu.cpp
	$(VERILATOR) --cc --exe --build --timing -j 0 $(VFLAGS) --top-module mantissa_fpu \
	    -Mdir $(dir $@) rtl/mantissa_pkg.sv $(MODEL) tb/tb_fpu.cpp -o tb_fpu

# ---- tests ------------------------------------------------------------------------
$(BUILD)/fpu_vectors.txt: tools/gen_fpu_vectors.py
	$(PYTHON) tools/gen_fpu_vectors.py --out $@ --count 150000

test-fpu: $(BUILD)/obj_fpu_model/tb_fpu $(BUILD)/fpu_vectors.txt
	$(BUILD)/obj_fpu_model/tb_fpu $(BUILD)/fpu_vectors.txt
	$(BUILD)/obj_fpu_model/tb_fpu $(BUILD)/fpu_vectors.txt.port_defined

test:
	bash test.sh

lint:
	$(VERILATOR) --lint-only -Wall $(VFLAGS) --top-module mantissa_top $(RTL) $(MODEL)
	$(YOSYS) -q -l $(BUILD)/yosys_elaborate.log -s synth/elaborate.ys

fuzz: $(BUILD)/obj_model/m1_sim
	$(PYTHON) tools/fuzz_programs.py --count $(or $(N),300) --seed $$((RANDOM * 32768 + RANDOM)) \
	    --out-dir $(BUILD)/fuzz_smoke --dirty 0.5

# ---- synthesis ------------------------------------------------------------------
synth:
	bash synth/run.sh

clean:
	rm -rf $(BUILD)/obj_model $(BUILD)/obj_fpu_model $(BUILD)/obj_dbg $(BUILD)/obj_fpu_syn $(BUILD)/obj_syn \
	       $(BUILD)/fpu_vectors*.txt $(BUILD)/yosys $(BUILD)/fuzz* $(BUILD)/mandel.trace $(BUILD)/golden.list \
	       $(BUILD)/obj_fpu* $(BUILD)/obj_rambb $(BUILD)/yosys_*.log

help:
	@sed -n '2,12p' $(firstword $(MAKEFILE_LIST))
