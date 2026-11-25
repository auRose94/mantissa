// A whole Mantissa-1 machine: the core, the cell memory and the floating
// point unit.
//
// The FPU is whatever file defines `mantissa_fpu`:
//   rtl/model/fpu_model.sv   the simulation model, for co-simulation
//   rtl/synth/fpu_syn.sv     the synthesizable unit
// The core and this file are the same in both builds.
//
// Cells are loaded one at a time through cfg_we while rst_n is low.
`default_nettype none
`timescale 1ns / 1ps

module mantissa_top
#(parameter int unsigned AW = 16)                    // cells: 2**AW
(
  input  logic                     clk,
  input  logic                     rst_n,

  input  logic [63:0]              boot_pc,           // the first pc, as a double
  input  logic [63:0]              boot_sp,           // what f31 starts as, as a double
  input  logic [31:0]              max_steps,         // the step limit; 0 is no limit
  input  logic [AW:0]              cells,             // how many cells this machine has, 1 to 2**AW

  input  logic                     cfg_we,            // write one cell
  input  logic [AW-1:0]            cfg_addr,
  input  logic [15:0]              cfg_data,

  output logic                     halted,
  output logic                     fault,
  output logic [1:0]              fault_kind,        // mantissa_pkg::fault_e
  output logic                     step_limit,
  output logic [31:0]              step_count,

  output logic                     ev_valid,
  output logic [1:0]               ev_end,
  output logic [1:0]              ev_fault,          // mantissa_pkg::fault_e
  output logic [63:0]              ev_pc,
  output logic [2:0]               ev_nwr,
  output logic [6:0]               ev_wcode0,
  output logic [6:0]               ev_wcode1,
  output logic [6:0]               ev_wcode2,
  output logic [6:0]               ev_wcode3,
  output logic [127:0]             ev_wval0,
  output logic [63:0]              ev_wval1,
  output logic [63:0]              ev_wval2,
  output logic [63:0]              ev_wval3,
  output logic [1:0]               ev_wprov0,
  output logic [1:0]               ev_wprov1,
  output logic [1:0]               ev_wprov2,
  output logic [1:0]               ev_wprov3,
  output logic [6:0]               ev_wsrc0,
  output logic [6:0]               ev_wsrc1,
  output logic [6:0]               ev_wsrc2,
  output logic [6:0]               ev_wsrc3,
  output logic [63:0]              capacity,          // the number of cells, as a double
  output logic                     ev_store,
  output logic [AW-1:0]            ev_saddr,
  output logic [2:0]               ev_sfmt,
  output logic [127:0]             ev_sval,
  output logic                     ev_put,
  output logic [127:0]             ev_putval,
  output logic                     ev_putq,
  output logic                     ev_putc,
  output logic [20:0]              ev_putcode
);

  // the core's memory port
  logic [AW-3:0] mem_addr;
  logic            mem_en, mem_we;
  logic [3:0]      mem_wmask;
  logic [63:0]     mem_wdata, mem_rdata;
  // what the memory sees: the core, or a cell being loaded
  logic [AW-3:0] ram_addr;
  logic            ram_en, ram_we;
  logic [3:0]      ram_wmask;
  logic [63:0]     ram_wdata;
  // the core's FPU port
  logic                  fpu_start, fpu_done;
  logic [5:0]            fpu_op;      // mantissa_pkg::fpu_op_e
  logic [2:0]            fpu_fmt, fpu_fmt2;
  logic [1:0]            fpu_sel;
  logic [127:0]          fpu_a, fpu_y;
  logic [127:0]          fpu_b, fpu_c, fpu_y2;
  logic [4:0]            fpu_flags;

  mantissa_core
  #(.AW(AW))
  u_core
  (
    .clk(clk), .rst_n(rst_n),
    .boot_pc(boot_pc), .boot_sp(boot_sp), .max_steps(max_steps), .cells(cells),
    .mem_addr(mem_addr), .mem_en(mem_en), .mem_we(mem_we), .mem_wmask(mem_wmask),
    .mem_wdata(mem_wdata), .mem_rdata(mem_rdata),
    .fpu_start(fpu_start), .fpu_op(fpu_op), .fpu_fmt(fpu_fmt), .fpu_fmt2(fpu_fmt2),
    .fpu_sel(fpu_sel), .fpu_a(fpu_a), .fpu_b(fpu_b), .fpu_c(fpu_c),
    .fpu_done(fpu_done), .fpu_y(fpu_y), .fpu_y2(fpu_y2), .fpu_flags(fpu_flags),
    .halted(halted), .fault(fault), .fault_kind(fault_kind), .step_limit(step_limit),
    .step_count(step_count),
    .ev_valid(ev_valid), .ev_end(ev_end), .ev_fault(ev_fault), .ev_pc(ev_pc), .ev_nwr(ev_nwr),
    .ev_wcode0(ev_wcode0), .ev_wcode1(ev_wcode1), .ev_wcode2(ev_wcode2), .ev_wcode3(ev_wcode3),
    .ev_wval0(ev_wval0), .ev_wval1(ev_wval1), .ev_wval2(ev_wval2), .ev_wval3(ev_wval3),
    .ev_wprov0(ev_wprov0), .ev_wprov1(ev_wprov1), .ev_wprov2(ev_wprov2), .ev_wprov3(ev_wprov3),
    .ev_wsrc0(ev_wsrc0), .ev_wsrc1(ev_wsrc1), .ev_wsrc2(ev_wsrc2), .ev_wsrc3(ev_wsrc3),
    .ev_store(ev_store), .ev_saddr(ev_saddr), .ev_sfmt(ev_sfmt), .ev_sval(ev_sval),
    .ev_put(ev_put), .ev_putval(ev_putval), .ev_putq(ev_putq), .ev_putc(ev_putc), .ev_putcode(ev_putcode)
  );

  mantissa_fpu
  u_fpu
  (
    .clk(clk), .rst_n(rst_n),
    .start(fpu_start), .op(fpu_op), .fmt(fpu_fmt), .fmt2(fpu_fmt2), .sel(fpu_sel),
    .a(fpu_a), .b(fpu_b), .c(fpu_c),
    .done(fpu_done), .y(fpu_y), .y2(fpu_y2), .flags(fpu_flags)
  );

  // a cell being loaded goes through the same port as everything else
  always_comb begin
    ram_en    = mem_en | cfg_we;
    ram_we    = cfg_we ? 1'b1 : mem_we;
    ram_addr  = cfg_we ? cfg_addr[AW-1:2] : mem_addr;
    ram_wmask = cfg_we ? (4'd8 >> cfg_addr[1:0]) : mem_wmask;
    ram_wdata = cfg_we ? {4{cfg_data}} : mem_wdata;
  end

  mantissa_cell_ram
  #(.AW(AW))
  u_ram
  (
    .clk(clk), .addr(ram_addr), .en(ram_en), .we(ram_we), .wmask(ram_wmask), .wdata(ram_wdata),
    .rdata(mem_rdata)
  );

  // how many cells there are: what a stack pointer starts as
  localparam logic [63:0] CAPACITY_BITS = (64'd1023 + 64'(AW)) << 52;   // 2**AW as a double
  assign capacity = CAPACITY_BITS;

endmodule

`default_nettype wire
