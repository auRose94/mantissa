// The floating point unit as a black box: the interface in mantissa_pkg, with
// no logic.  For taking cell counts of the core alone, and for a vendor flow
// that binds its own floating point IP to the port.
`default_nettype none
`timescale 1ns / 1ps

(* blackbox *)
module mantissa_fpu
(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [5:0]  op,      // mantissa_pkg::fpu_op_e
  input  logic [2:0]  fmt,     // mantissa_pkg: H, F, D or Q
  input  logic [2:0]  fmt2,
  input  logic [1:0]  sel,
  input  logic [127:0] a,      // a quad in full, the narrow formats in the low 64
  input  logic [63:0] b,
  input  logic [63:0] c,
  output logic        done,
  output logic [127:0] y,      // a quad in full, the narrow formats in the low 64
  output logic [63:0] y2,
  output logic [4:0]  flags
);
endmodule

`default_nettype wire
