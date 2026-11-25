// Shared definitions for the Mantissa-1 FPGA port.
//
// Everything here is written in the SystemVerilog subset that Yosys 0.66 reads
// and Verilator simulates: no `return` in functions, no `import` inside a
// module body (modules spell names as mantissa_pkg::name), no `case inside`.
`default_nettype none
`timescale 1ns / 1ps

package mantissa_pkg;

  // ---- values ---------------------------------------------------------------
  // Every value in the machine is an IEEE 754 half, float, double or quad.
  // Inside the core and across the FPU interface a value travels as the bits
  // that hold it: a half or a float as the 64 bits of the double that holds
  // it (widened first, a NaN keeping its sign and payload in the double's
  // positions), a quad as its own 128 bits.  Only cell memory holds the
  // narrow encodings, and only the FPU converts.
  localparam logic [63:0] NAN_BITS = 64'h7FF8000000000000;   // arithmetic's quiet NaN: sign 0, no payload
  localparam logic [63:0] ONE_BITS  = 64'h3FF0000000000000;   // 1.0
  localparam logic [63:0] NONE_BITS = 64'hBFF0000000000000;   // -1.0
  localparam logic [127:0] QNAN_BITS = 128'h7FFF8000000000000000000000000000;
                                             // quad arithmetic's quiet NaN: sign 0, no payload

  // ---- formats and register files -------------------------------------------
  // The same number names a format and the register file that holds it: an
  // operand field's register code is (file * 32) + index, so quad, the file
  // the two-digit guard band became, is file 3 -- four registers, codes 96
  // to 99.
  localparam logic [2:0] FMT_H = 3'd0;    // half:   11 significand bits, emax 15,     1 cell
  localparam logic [2:0] FMT_F = 3'd1;    // float:  24 significand bits, emax 127,    2 cells
  localparam logic [2:0] FMT_D = 3'd2;    // double: 53 significand bits, emax 1023,   4 cells
  localparam logic [2:0] FMT_Q = 3'd3;    // quad:   113 significand bits, emax 16383, 8 cells

  typedef enum logic [2:0] { fmt_H = 3'd0, fmt_F = 3'd1, fmt_D = 3'd2, fmt_Q = 3'd3 } fmt_e;

  // significand bits, counting the implicit leading 1
  function automatic int unsigned f_p(input logic [2:0] f);
    if (f == FMT_H) f_p = 11;
    else if (f == FMT_F) f_p = 24;
    else if (f == FMT_D) f_p = 53;
    else f_p = 113;
  endfunction

  // exponent of the largest binade
  function automatic int f_emax(input logic [2:0] f);
    if (f == FMT_H) f_emax = 15;
    else if (f == FMT_F) f_emax = 127;
    else if (f == FMT_D) f_emax = 1023;
    else f_emax = 16383;
  endfunction

  // cells a value of the format occupies in memory
  function automatic int unsigned f_cells(input logic [2:0] f);
    f_cells = 1 << f;                       // quad's eight fall out of the shift
  endfunction

  // ---- sticky IEEE 754 flags ------------------------------------------------
  localparam logic [4:0] F_INVALID  = 5'b00001;
  localparam logic [4:0] F_DIVZERO  = 5'b00010;
  localparam logic [4:0] F_OVERFLOW = 5'b00100;
  localparam logic [4:0] F_UNDERFLOW = 5'b01000;
  localparam logic [4:0] F_INEXACT  = 5'b10000;

  // ---- faults ---------------------------------------------------------------
  typedef enum logic [1:0] {
    FAULT_ADDRESS = 2'd0, FAULT_ILLEGAL = 2'd1, FAULT_NOT_A_CHARACTER = 2'd2
  } fault_e;

  // ---- the FPU interface ----------------------------------------------------
  // One request, one answer.  The core raises `start` for one cycle with the
  // operation and its operands; the unit answers with `done` for one cycle,
  // later by however long the operation takes.  Every value operation the
  // machine has goes through here, conversions between the narrow cell
  // encodings and the 64-bit form included, so the core knows no format
  // arithmetic at all.
  typedef enum logic [5:0] {
    F_ADD = 6'd0, F_SUB, F_MUL, F_DIV, F_SQRT, F_FMA, F_REM, F_MOD, F_MIN, F_MAX,
    F_NEG, F_ABS, F_CPSGN, F_FLOOR, F_CEIL, F_CHOP, F_ROUND,
    F_FLOOR_TO, F_CEIL_TO, F_CHOP_TO, F_ROUND_TO, F_SCALB, F_LOGB, F_NEXT, F_CLASS,
    F_LT, F_LE, F_EQ, F_CMP,
    F_CVT, F_FROMBITS, F_TOBITS, F_SPLIT, F_JOIN, F_MANT, F_MOV
  } fpu_op_e;

  // ---- what a register write says about a NaN --------------------------------
  // The reference is Python, and its trace calls a register changed only when
  // the Python object in it is another object.  For every value but a NaN
  // that is the same as "different bits".  A NaN is another object unless it
  // is the very same one, so a write has to say where its NaN came from.
  localparam logic [1:0] PROV_CANON   = 2'd0;  // arithmetic's shared quiet NaN, or not a NaN at all
  localparam logic [1:0] PROV_INHERIT = 2'd1;  // the NaN another register holds (wsrc names it)
  localparam logic [1:0] PROV_FRESH   = 2'd2;  // a NaN made by this instruction: a load, or a sign change

  // ---- how a step ended -----------------------------------------------------
  localparam logic [1:0] EV_RETIRE = 2'd0;    // the instruction ran
  localparam logic [1:0] EV_HALT   = 2'd1;    // it was HALT
  localparam logic [1:0] EV_FAULT  = 2'd2;    // it stopped the machine; ev_fault says why

endpackage

`default_nettype wire
