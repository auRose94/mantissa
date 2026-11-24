// Cell memory: 2**AW cells of 16 bits and nothing smaller.
//
// One access moves a four-cell word, most significant cell first, and a write
// can touch any one of the four cells.  The memory is four banks of one cell
// each, every bank a plain synchronous single-port RAM with a write enable:
// the shape every vendor tool maps to block RAM.
//
//   addr    word address: cells addr*4 .. addr*4+3
//   en      read, or write with we
//   wmask   for a write, one bit per cell, bit 0 the most significant cell
//   rdata   one cycle after en; a read in the same cycle as a write to the
//           same word returns what was there before
`default_nettype none
`timescale 1ns / 1ps

module mantissa_cell_ram
#(parameter int unsigned AW = 16)
(
  input  logic             clk,
  input  logic [AW-3:0]    addr,
  input  logic             en,
  input  logic             we,
  input  logic [3:0]       wmask,
  input  logic [63:0]      wdata,
  output logic [63:0]      rdata
);

  localparam int unsigned WORDS = (1 << AW) >> 2;

  for (genvar i = 0; i < 4; i++) begin : g_bank
    logic [15:0] bank [0:WORDS-1];
    always_ff @(posedge clk) begin
      if (en) begin
        if (we && wmask[i]) bank[addr] <= wdata[16*i +: 16];
        rdata[16*i +: 16] <= bank[addr];
      end
    end
  end

endmodule

`default_nettype wire
