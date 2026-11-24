// Mantissa-1 core: cells, registers, pc, decode and control.
//
// The core knows nothing about floating point.  It moves 64-bit values
// (mantissa_pkg: the bits of the double that holds a value), checks that an
// address or an instruction number is a whole number in range, takes an
// instruction number apart into decimal fields, and runs a small state
// machine.  Every value operation, conversions between the narrow cell
// encodings and the 64-bit form included, is a request to the FPU.
//
// Timing, with a memory that answers one cycle after a request and an FPU that
// answers one cycle after `start` (the synthesizable unit takes longer for
// some operations and the core simply waits on `done`):
//
//   NOP, PUT, PUTC, JMP                 1 cycle
//   arithmetic, MOV, CVT, branches      2 cycles
//   LDI, LD, ST, a positive BSGN        3 cycles
//
// A store is written at the end of its cycle and the next fetch is issued in
// the cycle after that, so no instruction is ever cached across a store.
//
// Written in the SystemVerilog subset that Yosys 0.66 reads.
`default_nettype none
`timescale 1ns / 1ps

module mantissa_core
#(parameter int unsigned AW = 16)                    // cells: 2**AW
(
  input  logic                     clk,
  input  logic                     rst_n,

  // where the machine starts, taken once after reset
  input  logic [63:0]              boot_pc,           // the first pc, as a double
  input  logic [63:0]              boot_sp,           // what f31 starts as, as a double
  input  logic [31:0]              max_steps,         // the step limit; 0 is no limit
  input  logic [AW:0]              cells,             // how many cells this machine has: 1 to 2**AW

  // cell memory: one four-cell word per access, most significant cell first
  output logic [AW-3:0]          mem_addr,          // word address: cells addr*4 .. addr*4+3
  output logic                     mem_en,
  output logic                     mem_we,            // with mem_en: a write
  output logic [3:0]               mem_wmask,         // which of the four cells a write touches
  output logic [63:0]              mem_wdata,
  input  logic [63:0]              mem_rdata,         // one cycle after mem_en

  // the floating point unit
  output logic                     fpu_start,
  output logic [5:0]              fpu_op,            // mantissa_pkg::fpu_op_e
  output logic [1:0]              fpu_fmt,           // mantissa_pkg: the format of the operation
  output logic [1:0]              fpu_fmt2,          // the second format: JOIN's parts, MANT's exponent
  output logic [1:0]               fpu_sel,           // for F_FROMBITS: which cell of a word the value is in
  output logic [63:0]              fpu_a,
  output logic [63:0]              fpu_b,
  output logic [63:0]              fpu_c,
  input  logic                     fpu_done,
  input  logic [63:0]              fpu_y,
  input logic [63:0]               fpu_y2,            // SPLIT's low half, MANT's exponent
  input  logic [4:0]               fpu_flags,

  // status
  output logic                     halted,
  output logic                     fault,
  output logic [1:0]              fault_kind,        // mantissa_pkg::fault_e
  output logic                     step_limit,
  output logic [31:0]              step_count,

  // trace port: one pulse per step, with what the step changed
  output logic                     ev_valid,
  output logic [1:0]               ev_end,            // mantissa_pkg::EV_*
  output logic [1:0]              ev_fault,          // mantissa_pkg::fault_e
  output logic [63:0]              ev_pc,             // the pc the step began at
  output logic [1:0]               ev_nwr,            // 0, 1 or 2 registers written
  output logic [6:0]               ev_wcode0,         // the first register written, 0..95
  output logic [6:0]               ev_wcode1,         // and the second, when there is one
  output logic [63:0]              ev_wval0,          // the values written
  output logic [63:0]              ev_wval1,
  output logic [1:0]               ev_wprov0,         // where a written NaN came from (mantissa_pkg::PROV_*)
  output logic [1:0]               ev_wprov1,
  output logic [6:0]               ev_wsrc0,          // for an inherited NaN: the register it came from
  output logic [6:0]               ev_wsrc1,
  output logic                     ev_store,
  output logic [AW-1:0]            ev_saddr,          // cell address
  output logic [1:0]               ev_sfmt,           // the format stored: 1, 2 or 4 cells
  output logic [63:0]              ev_sval,           // the value the register held
  output logic                     ev_put,
  output logic [63:0]              ev_putval,
  output logic                     ev_putc,
  output logic [20:0]              ev_putcode         // the Unicode code point
);

  localparam int unsigned R_SP    = 63;      // f31, where sp lives by convention: file 1 * 32 + 31

  typedef enum logic [3:0] {
    ST_BOOT = 4'd0,       // one cycle after reset: take boot_pc and boot_sp
    ST_DECODE,            // mem_rdata holds the instruction at pc
    ST_FPU,               // waiting on the FPU
    ST_OPND,              // mem_rdata holds an inline operand, a loaded word, or a branch target
    ST_POST,              // a store was written last cycle; fetch the next instruction now
    ST_FETCH_FAULT,       // the next pc is not a place an instruction could be fetched from
    ST_HALT, ST_FAULT, ST_LIMIT
  } state_e;

  typedef mantissa_pkg::fpu_op_e fpu_op_e;
  typedef mantissa_pkg::fault_e fault_e;

  // what kind of instruction it is
  typedef enum logic [3:0] {
    C_ILLEGAL = 4'd0, C_NOP, C_HALT, C_ALU, C_ST, C_LDI, C_LD, C_BR, C_JMP, C_JAL,
    C_BSGN, C_BZ, C_BCMP, C_PUT, C_PUTC
  } cls_e;

  // what to do when the FPU answers
  localparam logic [1:0] RET_ALU = 2'd0, RET_ST = 2'd1, RET_BRANCH = 2'd2;
  // what mem_rdata is in ST_OPND
  localparam logic [1:0] O_LDI = 2'd0, O_LD = 2'd1, O_TARGET = 2'd2, O_JAL = 2'd3;

  // opcode numbers
  localparam logic [6:0] OP_CVT   = 7'd14, OP_SPLIT = 7'd15, OP_JOIN  = 7'd16;
  localparam logic [6:0] OP_BSGN  = 7'd61, OP_BEQ   = 7'd62, OP_BNE   = 7'd63;
  localparam logic [6:0] OP_BLT   = 7'd64, OP_BLE   = 7'd65, OP_BZ    = 7'd66, OP_BNZ = 7'd67;

  localparam logic [63:0] ONE_BITS  = mantissa_pkg::ONE_BITS;    // 1.0
  localparam logic [63:0] NONE_BITS = mantissa_pkg::NONE_BITS;   // -1.0

  // ---- state ----------------------------------------------------------------
  state_e            state;
  logic [63:0]       pc;
  logic [63:0]       regs [0:95];          // h0..h31, f0..f31, d0..d31
  logic [4:0]        flags;                // the five sticky flags
  logic              limit_hit;            // the step limit was reached; nothing more runs

  // the decode of the current instruction, kept while the FPU or memory works
  logic [6:0]        r_op;
  logic [6:0]        r_a, r_b;             // destination register codes
  logic [1:0]        r_ret;                // RET_*
  logic [3:0]        r_skip;               // cells the instruction takes when it falls through
  logic [1:0]        r_oret;               // O_*
  logic [1:0]        r_fmt;
  logic [1:0]        r_opnd_off;           // the operand a branch wants is at pc + 4, or at pc + 8
  logic              r_wb2;                // the instruction writes a second register
  logic              r_setflags;           // the instruction's flags are sticky
  logic [1:0]        r_prov0, r_prov1;     // where a written NaN came from
  logic [6:0]        r_src0, r_src1;
  // a store or a load in progress
  logic [1:0]        st_fmt, st_sel;
  logic [2:0]        st_cells;              // 1, 2 or 4
  logic [AW-1:0]     st_idx;
  logic [63:0]       st_val;

  assign halted     = (state == ST_HALT);
  assign fault      = (state == ST_FAULT);
  assign step_limit = (state == ST_LIMIT);

  // ---- functions on the bits of a double --------------------------------------

  function automatic logic is_nan(input logic [63:0] x);
    is_nan = (&x[62:52]) && (|x[51:0]);
  endfunction

  // The whole number a double holds, by magnitude.  is_int is false for a NaN;
  // an infinity is whole but saturates far out of range, and so does anything
  // from 2**53 up.  The sign is left to the caller: -0.0 is 0.
  function automatic logic [54:0] dbl_int(input logic [63:0] x, output logic is_int);
    logic [10:0] e;
    logic [51:0] m;
    logic [10:0] t;
    logic [5:0]  sh;
    e = x[62:52];
    m = x[51:0];
    if (&e && |m) begin
      is_int  = 1'b0;
      dbl_int = {55{1'b1}};
    end else if (e < 11'd1023) begin
      is_int  = !(|{e, m});                   // below 1 in magnitude: only zero is whole
      dbl_int = 55'd0;
    end else if (e > 11'd1075) begin
      is_int  = 1'b1;
      dbl_int = {55{1'b1}};
    end else begin
      t = 11'd1075 - e;                       // 52 .. 0
      sh = t[5:0];
      is_int  = (m & ((52'd1 << sh) - 52'd1)) == 52'd0;
      dbl_int = {2'b00, 1'b1, m} >> sh;
    end
  endfunction

  // Can `width` cells be reached at x?  With `aligned`, x must be a multiple
  // of width.  This is memory._index in the reference: NaN, negative, past
  // the end, not whole and misaligned are all `address` faults.  The end is
  // `cells`, the size this machine was given, not all the memory built.
  function automatic logic addr_valid(input logic [63:0] x, input logic [2:0] width,
                                      input logic aligned);
    logic [54:0] v;
    logic        is_int;
    logic [AW:0] past;
    v = dbl_int(x, is_int);
    past = {1'b0, v[AW-1:0]} + {{(AW-2){1'b0}}, width};
    addr_valid = is_int && !(x[63] && (|x[62:0])) && (v[54:AW] == {(55-AW){1'b0}}) && (past <= cells)
                 && (!aligned || (width == 3'd1) || ((width == 3'd2) ? !v[0] : (v[1:0] == 2'd0)));
  endfunction

  // The cell index of a double that addr_valid has accepted.
  function automatic logic [AW+3:0] addr_index(input logic [63:0] x);
    logic [54:0] v;
    logic        is_int;
    v = dbl_int(x, is_int);
    addr_index = v[AW+3:0];
  endfunction

  // The double that holds the whole number v.  v is below 2**(AW+4), far
  // below 2**53, so this is exact: normalised, the leading 1 is the one that
  // is not stored, and the bits below it are the fraction.
  function automatic logic [63:0] int_to_dbl(input logic [AW+3:0] v);
    logic [5:0]  t;                           // one more than the position of the leading 1
    logic [10:0] eb;
    logic [51:0] frac;
    t = 6'd0;
    for (int i = AW + 3; i >= 0; i--) begin
      if (v[i] && t == 6'd0) t = i[5:0] + 6'd1;
    end
    if (t == 6'd0) int_to_dbl = 64'd0;
    else begin
      eb = 11'd1022 + {5'b0, t};
      frac = {{(48-AW){1'b0}}, v} << (6'd53 - t);
      int_to_dbl = {1'b0, eb, frac};
    end
  endfunction

  // ---- decode: the instruction number and its decimal fields ------------------
  // An instruction is the double at pc, and it must be a whole number from 0
  // to 99,999,999.  Read from the left, two decimal digits each: the opcode
  // and the operands a, b and c.
  logic [10:0] ie;
  logic [51:0] im;
  logic [10:0] it;
  logic [5:0]  ish;
  logic [52:0] ishifted;
  logic        ins_whole;
  logic [26:0] ins_n;
  logic        ins_range;
  logic [52:0] pm100, pm10k, pm1m;
  logic [26:0] n100, n10k, n1m, rem_a, rem_b, rem_c;
  logic [6:0]  d_op, d_a, d_b, d_c;
  cls_e        d_cls;
  logic [2:0]  d_use;            // bit 0: field a is used, bit 1: b, bit 2: c
  logic        d_same;           // every register used must be of one type
  logic [3:0]  d_skip;           // cells the instruction takes when it falls through
  fpu_op_e d_fop;
  logic        d_fields, d_types, d_num_ok, d_ok;
  logic [6:0]  d_ra, d_rb, d_rc;
  logic [1:0]  fa, fb, fc;

  always_comb begin
    ie = mem_rdata[62:52];
    im = mem_rdata[51:0];
    it = 11'd1075 - ie;
    ish = it[5:0];
    // whole: not negative, at least 1 (so ie >= 1023), under 2**27 (so the
    // low 1075 - ie fraction bits are clear); 99,999,999 is narrower still
    ins_whole = !mem_rdata[63] && (ie >= 11'd1023) && (ie <= 11'd1049)
                && ((im & ((52'd1 << ish) - 52'd1)) == 52'd0);
    ishifted  = {1'b1, im} >> ish;
    ins_n     = ishifted[26:0];
    ins_range = (ins_n < 27'd100000000);

    // decimal fields.  n / 100 is (n * ceil(2**32 / 100)) >> 32 for every n
    // below 2**30, so three multiplications peel the fields off from the right.
    pm100 = ins_n * 27'd42949673;
    n100  = {6'd0, pm100[52:32]};
    pm10k = {26'd0, n100} * 27'd42949673;
    n10k  = {6'd0, pm10k[52:32]};
    pm1m  = {26'd0, n10k} * 27'd42949673;
    n1m   = {6'd0, pm1m[52:32]};
    rem_a = n10k - n1m  * 27'd100;
    rem_b = n100 - n10k * 27'd100;
    rem_c = ins_n - n100 * 27'd100;
    d_op  = n1m[6:0];
    d_a   = rem_a[6:0];
    d_b   = rem_b[6:0];
    d_c   = rem_c[6:0];
  end

  // What the opcode says about its fields.  This is the table in isa.py.
  always_comb begin
    d_cls  = C_ILLEGAL;
    d_use  = 3'b000;
    d_same = 1'b0;
    d_skip = 4'd4;
    d_fop  = mantissa_pkg::F_ADD;
    case (d_op)
      7'd1:   d_cls = C_NOP;
      7'd2:   d_cls = C_HALT;
      7'd10:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_MOV; end
      7'd11:  begin d_cls = C_LDI; d_use = 3'b001; d_skip = 4'd8; end
      7'd12:  begin d_cls = C_LD;  d_use = 3'b011; end
      7'd13:  begin d_cls = C_ST;  d_use = 3'b011; end
      7'd14:  begin d_cls = C_ALU; d_use = 3'b011; d_fop = mantissa_pkg::F_CVT; end
      7'd15:  begin d_cls = C_ALU; d_use = 3'b111; d_fop = mantissa_pkg::F_SPLIT; end
      7'd16:  begin d_cls = C_ALU; d_use = 3'b111; d_fop = mantissa_pkg::F_JOIN; end
      7'd20:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_ADD; end
      7'd21:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_SUB; end
      7'd22:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_MUL; end
      7'd23:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_DIV; end
      7'd24:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_SQRT; end
      7'd25:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_FMA; end
      7'd26:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_REM; end
      7'd27:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_MOD; end
      7'd28:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_MIN; end
      7'd29:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_MAX; end
      7'd30:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_NEG; end
      7'd31:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_ABS; end
      7'd32:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_CPSGN; end
      7'd33:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_FLOOR; end
      7'd34:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_CEIL; end
      7'd35:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_CHOP; end
      7'd36:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_ROUND; end
      7'd37:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_FLOOR_TO; end
      7'd38:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_CEIL_TO; end
      7'd39:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_CHOP_TO; end
      7'd40:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_ROUND_TO; end
      7'd41:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_SCALB; end
      7'd42:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_LOGB; end
      7'd43:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_NEXT; end
      7'd44:  begin d_cls = C_ALU; d_use = 3'b011; d_same = 1'b1; d_fop = mantissa_pkg::F_CLASS; end
      7'd45:  begin d_cls = C_ALU; d_use = 3'b111; d_fop = mantissa_pkg::F_MANT; end
      7'd50:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_LT; end
      7'd51:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_LE; end
      7'd52:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_EQ; end
      7'd53:  begin d_cls = C_ALU; d_use = 3'b111; d_same = 1'b1; d_fop = mantissa_pkg::F_CMP; end
      7'd60:  begin d_cls = C_BR;   d_skip = 4'd8; end
      7'd61:  begin d_cls = C_BSGN; d_use = 3'b001; d_skip = 4'd12; end
      7'd62, 7'd63, 7'd64, 7'd65:
              begin d_cls = C_BCMP; d_use = 3'b011; d_same = 1'b1; d_skip = 4'd8; end
      7'd66, 7'd67:
              begin d_cls = C_BZ;   d_use = 3'b001; d_skip = 4'd8; end
      7'd68:  begin d_cls = C_JAL;  d_use = 3'b001; end
      7'd69:  begin d_cls = C_JMP;  d_use = 3'b001; end
      7'd70:  begin d_cls = C_PUT;  d_use = 3'b001; end
      7'd71:  begin d_cls = C_PUTC; d_use = 3'b001; end
      default: d_cls = C_ILLEGAL;
    endcase
  end

  // A field the instruction does not use must be 00; one it uses must name a
  // register, 00 to 95.  Register code = file * 32 + index.
  always_comb begin
    fa = d_a[6:5];
    fb = d_b[6:5];
    fc = d_c[6:5];
    d_fields = ((d_use[0] ? (d_a < 7'd96) : (d_a == 7'd0))
                && (d_use[1] ? (d_b < 7'd96) : (d_b == 7'd0))
                && (d_use[2] ? (d_c < 7'd96) : (d_c == 7'd0)));
    d_ra = {fa, d_a[4:0]};
    d_rb = {fb, d_b[4:0]};
    d_rc = {fc, d_c[4:0]};
    d_num_ok = ins_whole && ins_range && d_fields && (d_cls != C_ILLEGAL);
  end

  // The operand type rules of machine.py.  Breaking one is an
  // illegal-instruction fault, raised before anything runs.
  always_comb begin
    d_types = 1'b1;
    case (d_cls)
      C_ALU: begin
        if (d_same)                                     d_types = (fa == fb) && (!d_use[2] || (fa == fc));
        else if (d_op == OP_CVT)                        d_types = (fa != fb);
        else if (d_op == OP_SPLIT)                      d_types = (fa == fb);
        else if (d_op == OP_JOIN)                       d_types = (fb == fc);
      end
      C_LD:                                            d_types = (fb == mantissa_pkg::FMT_F);
      C_ST:                                            d_types = (fa == mantissa_pkg::FMT_F);
      C_JMP, C_JAL:                                    d_types = (fa == mantissa_pkg::FMT_F);
      C_BCMP:                                          d_types = (fa == fb);
      default: ;
    endcase
    d_ok = d_num_ok && d_types;
  end

  // ---- operands ----------------------------------------------------------------
  logic [63:0] ra, rb, rc;              // the registers the fields name

  always_comb begin
    ra = regs[(fa == 2'd3) ? 7'd0 : d_ra];      // 3 names no file; the decode will fault
    rb = regs[(fb == 2'd3) ? 7'd0 : d_rb];
    rc = regs[(fc == 2'd3) ? 7'd0 : d_rc];
  end

  // ---- control -----------------------------------------------------------------
  logic dec_go;
  fault_e dec_fault;
  logic y_one, y_zero, y_negone, taken, bsgn_pos;
  logic adv_now, target_now, jmp_now, retire_now, limit_new;
  logic [3:0] adv_k;
  logic fetch_now, opnd_here;
  logic [2:0] cell_off;
  logic [AW+3:0] fetch_idx;

  // ---- addresses ---------------------------------------------------------------
  logic [AW+3:0] pc_i, boot_i, pc4_i, pc8_i, pcn, st_i, ld_i, jmp_i, target_i;
  logic [63:0]   pcn_d, link_d;
  logic [54:0]   putc_i;
  logic          putc_is_int;
  logic          boot_ok, opnd4_ok, opnd8_ok, pcn_ok;
  logic          st_ok, ld_ok, jmp_ok, target_ok, putc_ok;
  logic [2:0]    st_width, ld_width;

  always_comb begin
    pc_i = addr_index(pc);
    boot_i = addr_index(boot_pc);
    boot_ok = addr_valid(boot_pc, 3'd4, 1'b1);

    pc4_i = pc_i + 4;
    pc8_i = pc_i + 8;
    // an inline operand at pc + k must fit: pc + k + 4 <= cells
    // an inline operand at pc + k, or the instruction at the next pc, must fit: + 4 <= cells
    opnd4_ok = ({1'b0, pc4_i[AW-1:0]} + {{AW{1'b0}}, 1'b1} <= cells);
    opnd8_ok = ({1'b0, pc8_i[AW-1:0]} + {{AW{1'b0}}, 1'b1} <= cells);

    pcn = pc_i + {{(AW){1'b0}}, adv_k};
    pcn_ok = ({1'b0, pcn[AW-1:0]} + {{AW{1'b0}}, 1'b1} <= cells);
    pcn_d = int_to_dbl(pcn);
    link_d = int_to_dbl(pc8_i);

    st_width = {2'b00, 1'b1} << fb;
    ld_width = {2'b00, 1'b1} << fa;
    st_ok = addr_valid(ra, st_width, 1'b1);
    ld_ok = addr_valid(rb, ld_width, 1'b1);
    jmp_ok = addr_valid(ra, 3'd4, 1'b1);
    st_i = addr_index(ra);
    ld_i = addr_index(rb);
    jmp_i = addr_index(ra);

    target_ok = addr_valid(mem_rdata, 3'd4, 1'b1);
    target_i = addr_index(mem_rdata);

    // a character: a whole number from 0 to 0x10FFFF that is not a surrogate
    putc_i = dbl_int(ra, putc_is_int);
    putc_ok = putc_is_int && !(ra[63] && (|ra[62:0])) && (putc_i <= 55'd1114111)
              && !((putc_i >= 55'd55296) && (putc_i <= 55'd57343));
  end

  // A decode is good when the number is an instruction and its operands break
  // no rule.  Then the three instructions that reach memory at once, and
  // PUTC, are checked before anything runs.
  always_comb begin
    dec_go = 1'b0;
    dec_fault = mantissa_pkg::FAULT_ILLEGAL;
    if (!d_ok) dec_fault = mantissa_pkg::FAULT_ILLEGAL;
    else if (((d_cls == C_ST) && !st_ok) || ((d_cls == C_LD) && !ld_ok) || ((d_cls == C_JMP) && !jmp_ok))
      dec_fault = mantissa_pkg::FAULT_ADDRESS;
    else if ((d_cls == C_PUTC) && !putc_ok)
      dec_fault = mantissa_pkg::FAULT_NOT_A_CHARACTER;
    else
      dec_go = 1'b1;
  end

  // The answer to a comparison, as the three values the FPU returns.
  always_comb begin
    y_one    = (fpu_y == ONE_BITS);
    y_negone = (fpu_y == NONE_BITS);
    y_zero   = (fpu_y[62:0] == 63'd0);
    taken = 1'b0;
    bsgn_pos = 1'b0;
    case (r_op)
      OP_BZ:   taken = y_one;               // x == 0
      OP_BNZ:  taken = y_zero;              // x != 0, and a NaN is not equal to 0
      OP_BEQ:  taken = y_one;
      OP_BNE:  taken = y_zero;              // taken for a NaN too
      OP_BLT:  taken = y_one;
      OP_BLE:  taken = y_one;
      OP_BSGN: begin                        // negative or positive jump; zero and NaN fall through
        bsgn_pos = y_one;
        taken = y_one || y_negone;
      end
      default: ;
    endcase
  end

  // What this cycle does.  A cycle that ends an instruction either names the
  // next pc (advance by adv_k, jump, or take a target) and issues the fetch
  // for it, or hands over to a state that will.
  always_comb begin
    adv_now = 1'b0;
    adv_k = 4'd4;
    target_now = 1'b0;
    jmp_now = 1'b0;
    case (state)
      ST_DECODE: begin
        if (dec_go && !limit_hit) begin
          if ((d_cls == C_NOP) || (d_cls == C_PUT) || (d_cls == C_PUTC)) adv_now = 1'b1;
          else if (d_cls == C_JMP) jmp_now = 1'b1;
        end
      end
      ST_FPU: begin
        if (fpu_done && (r_ret == RET_ALU)) begin
          adv_now = 1'b1;
          adv_k = r_skip;                         // 8 for LDI, 4 for the rest
        end else if (fpu_done && (r_ret == RET_BRANCH) && !taken) begin
          adv_now = 1'b1;
          adv_k = r_skip;                         // past the operand, or both of BSGN's
        end
      end
      ST_OPND: target_now = (r_oret == O_TARGET) || (r_oret == O_JAL);
      ST_POST: adv_now = 1'b1;
      default: ;
    endcase
    target_now = target_now && (r_opnd_off[0] ? opnd8_ok : opnd4_ok) && target_ok;
  end

  // A step retires when its instruction has run, or when it has faulted; the
  // reference counts a faulting step as a step.
  always_comb begin
    retire_now = 1'b0;
    case (state)
      ST_DECODE: begin
        if (!dec_go) retire_now = !limit_hit;
        else if ((d_cls == C_NOP) || (d_cls == C_HALT) || (d_cls == C_PUT)
                 || (d_cls == C_PUTC) || (d_cls == C_JMP)) retire_now = 1'b1;
      end
      ST_FPU: begin
        if (fpu_done) retire_now = (r_ret == RET_ALU) || ((r_ret == RET_BRANCH) && !taken);
      end
      ST_OPND:  retire_now = (r_oret == O_TARGET) || (r_oret == O_JAL);
      ST_POST:  retire_now = 1'b1;
      ST_FETCH_FAULT: retire_now = !limit_hit;
      default: ;
    endcase
    limit_new = retire_now && (max_steps != 32'd0) && ((step_count + 32'd1) == max_steps);
  end

  always_comb begin
    fetch_now = 1'b0;
    fetch_idx = {(AW+4){1'b0}};
    case (state)
      ST_BOOT: begin
        if (boot_ok) begin
          fetch_now = 1'b1;
          fetch_idx = boot_i;
        end
      end
      ST_POST: begin
        fetch_now = 1'b1;                            // the store has landed: fetch what follows
        fetch_idx = pcn;
      end
      ST_DECODE: begin
        if (jmp_now) begin
          fetch_now = 1'b1;
          fetch_idx = jmp_i;
        end else if (adv_now && pcn_ok) begin
          fetch_now = 1'b1;
          fetch_idx = pcn;
        end
      end
      ST_FPU: begin
        if (adv_now && pcn_ok) begin
          fetch_now = 1'b1;
          fetch_idx = pcn;
        end
      end
      ST_OPND: begin
        if (target_now) begin
          fetch_now = 1'b1;
          fetch_idx = target_i;
        end
      end
      default: ;
    endcase
  end

  // The memory port.  A cycle either fetches, reads an operand (the target of
  // a branch is read while the comparison runs: reading has no effect, and
  // its address is only held against the machine if the branch is taken), or
  // writes a store.
  always_comb begin
    opnd_here = (d_cls == C_LDI) || (d_cls == C_BR) || (d_cls == C_JAL)
                || (d_cls == C_BSGN) || (d_cls == C_BZ) || (d_cls == C_BCMP);
    mem_en = 1'b0;
    mem_we = 1'b0;
    mem_wmask = 4'd0;
    mem_wdata = 64'd0;
    mem_addr = {(AW-2){1'b0}};
    // the shift that puts a value's cells, most significant first, into the
    // word at cell st_sel: 16 * (4 - cells - sel)
    cell_off = 3'd4 - st_cells - {1'b0, st_sel};
    if (fetch_now) begin
      mem_en = 1'b1;
      mem_addr = fetch_idx[AW-1:2];
    end else begin
      case (state)
        ST_DECODE: begin
          if (dec_go && (d_cls == C_LD)) begin          // the word a load reads
            mem_en = 1'b1;
            mem_addr = ld_i[AW-1:2];
          end else if (dec_go && opnd_here && opnd4_ok) begin
            mem_en = 1'b1;
            mem_addr = pc4_i[AW-1:2];
          end
        end
        ST_FPU: begin
          if (fpu_done && (r_ret == RET_ST)) begin
            mem_en = 1'b1;
            mem_we = 1'b1;
            mem_addr = st_idx[AW-1:2];
            // bit 3 is the first cell of the word: a half at sel covers that cell, a
            // float (sel is 0 or 2) covers sel and sel + 1, a double all four
            mem_wmask = (st_fmt == mantissa_pkg::FMT_H) ? (4'd8 >> st_sel)
                        : (st_fmt == mantissa_pkg::FMT_F) ? (4'd12 >> st_sel) : 4'd15;
            mem_wdata = fpu_y << {cell_off, 4'b0};
          end else if (fpu_done && (r_ret == RET_BRANCH) && taken && bsgn_pos && opnd8_ok) begin
            mem_en = 1'b1;                                            // BSGN's second target
            mem_addr = pc8_i[AW-1:2];
          end
        end
        default: ;
      endcase
    end
  end

  // The FPU port.  A request is raised for one cycle, and what it carries is
  // valid in that same cycle: it comes straight from the decode and the
  // register files, or from the operand the memory has just delivered.
  always_comb begin
    fpu_start = 1'b0;
    fpu_op = mantissa_pkg::F_ADD;
    fpu_fmt = mantissa_pkg::fmt_e'(fa);
    fpu_fmt2 = mantissa_pkg::fmt_e'(fb);
    fpu_sel = 2'd0;
    fpu_a = 64'd0;
    fpu_b = 64'd0;
    fpu_c = 64'd0;
    case (state)
      ST_DECODE: begin
        if (dec_go) fpu_start = (d_cls == C_ALU) || (d_cls == C_ST) || (d_cls == C_BSGN)
                               || (d_cls == C_BZ) || (d_cls == C_BCMP);
        case (d_cls)
          C_ALU: begin
            fpu_op = d_fop;
            fpu_a = ((d_fop == mantissa_pkg::F_SPLIT) || (d_fop == mantissa_pkg::F_MANT)) ? rc : rb;
            fpu_b = rc;
            fpu_c = ra;                          // FMA's addend is its own destination
          end
          C_ST: begin
            fpu_op = mantissa_pkg::F_TOBITS;           // the value, b, as the cells it becomes
            fpu_fmt = mantissa_pkg::fmt_e'(fb);
            fpu_a = rb;
          end
          C_BSGN: begin
            fpu_op = mantissa_pkg::F_CMP;
            fpu_fmt = mantissa_pkg::FMT_D;
            fpu_a = ra;
            fpu_b = 64'd0;
          end
          C_BZ: begin
            fpu_op = mantissa_pkg::F_EQ;
            fpu_fmt = mantissa_pkg::FMT_D;
            fpu_a = ra;
            fpu_b = 64'd0;
          end
          C_BCMP: begin
            fpu_fmt = mantissa_pkg::FMT_D;
            fpu_a = ra;
            fpu_b = rb;
            case (d_op)
              OP_BEQ, OP_BNE: fpu_op = mantissa_pkg::F_EQ;
              OP_BLT:         fpu_op = mantissa_pkg::F_LT;
              default:        fpu_op = mantissa_pkg::F_LE;
            endcase
          end
          default: ;
        endcase
      end
      ST_OPND: begin
        fpu_start = (r_oret == O_LDI) || (r_oret == O_LD);
        if (r_oret == O_LDI) begin
          fpu_op = mantissa_pkg::F_CVT;
          fpu_fmt = mantissa_pkg::fmt_e'(r_fmt);
          fpu_a = mem_rdata;
        end else if (r_oret == O_LD) begin
          fpu_op = mantissa_pkg::F_FROMBITS;
          fpu_fmt = mantissa_pkg::fmt_e'(r_fmt);
          fpu_sel = st_sel;
          fpu_a = mem_rdata;
        end
      end
      default: ;
    endcase
  end

  // ---- the state machine ---------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= ST_BOOT;
      pc <= 64'd0;
      flags <= 5'd0;
      step_count <= 32'd0;
      limit_hit <= 1'b0;
      fault_kind <= mantissa_pkg::FAULT_ADDRESS;
      for (int i = 0; i < 96; i++) regs[i] <= 64'd0;
      r_ret <= 2'd0;
      r_oret <= 2'd0;
      r_skip <= 4'd4;
      r_op <= 7'd0;
      r_a <= 7'd0;
      r_b <= 7'd0;
      r_fmt <= 2'd0;
      r_opnd_off <= 2'd0;
      r_wb2 <= 1'b0;
      r_setflags <= 1'b0;
      r_prov0 <= 2'd0;
      r_prov1 <= 2'd0;
      r_src0 <= 7'd0;
      r_src1 <= 7'd0;
      st_fmt <= 2'd0;
      st_cells <= 3'd1;
      st_sel <= 2'd0;
      st_idx <= {AW{1'b0}};
      st_val <= 64'd0;
      ev_valid <= 1'b0;
      ev_end <= mantissa_pkg::EV_RETIRE;
      ev_fault <= mantissa_pkg::FAULT_ADDRESS;
      ev_pc <= 64'd0;
      ev_nwr <= 2'd0;
      ev_wcode0 <= 7'd0;
      ev_wcode1 <= 7'd0;
      ev_wval0 <= 64'd0;
      ev_wval1 <= 64'd0;
      ev_wprov0 <= 2'd0;
      ev_wprov1 <= 2'd0;
      ev_wsrc0 <= 7'd0;
      ev_wsrc1 <= 7'd0;
      ev_store <= 1'b0;
      ev_saddr <= {AW{1'b0}};
      ev_sfmt <= 2'd0;
      ev_sval <= 64'd0;
      ev_put <= 1'b0;
      ev_putval <= 64'd0;
      ev_putc <= 1'b0;
      ev_putcode <= 21'd0;
    end else begin
      // the event pulse lasts one cycle
      ev_valid <= retire_now;
      ev_end <= mantissa_pkg::EV_RETIRE;
      ev_fault <= mantissa_pkg::FAULT_ADDRESS;
      ev_pc <= pc;
      ev_nwr <= 2'd0;
      ev_wcode0 <= 7'd0;
      ev_wcode1 <= 7'd0;
      ev_wval0 <= 64'd0;
      ev_wval1 <= 64'd0;
      ev_wprov0 <= 2'd0;
      ev_wprov1 <= 2'd0;
      ev_wsrc0 <= 7'd0;
      ev_wsrc1 <= 7'd0;
      ev_store <= 1'b0;
      ev_saddr <= {AW{1'b0}};
      ev_sfmt <= 2'd0;
      ev_sval <= 64'd0;
      ev_put <= 1'b0;
      ev_putval <= 64'd0;
      ev_putc <= 1'b0;
      ev_putcode <= 21'd0;
      limit_hit <= limit_new;

      case (state)
        ST_BOOT: begin
          pc <= boot_pc;
          regs[R_SP] <= boot_sp;
          if (boot_ok) state <= ST_DECODE;
          else state <= ST_FETCH_FAULT;
        end

        ST_FETCH_FAULT: begin
          // the step at this pc cannot fetch: nothing changes
          if (limit_hit) state <= ST_LIMIT;
          else begin
            ev_end <= mantissa_pkg::EV_FAULT;
            ev_fault <= mantissa_pkg::FAULT_ADDRESS;
            fault_kind <= mantissa_pkg::FAULT_ADDRESS;
            step_count <= step_count + 32'd1;
            state <= ST_FAULT;
          end
        end

        ST_DECODE: begin
          if (limit_hit) state <= ST_LIMIT;
          else if (!dec_go) begin
            // not an instruction, or operands that break a type rule, or an
            // address that is not a place: nothing runs
            ev_end <= mantissa_pkg::EV_FAULT;
            ev_fault <= dec_fault;
            fault_kind <= dec_fault;
            step_count <= step_count + 32'd1;
            state <= ST_FAULT;
          end else begin
            case (d_cls)
              C_NOP: step_count <= step_count + 32'd1;

              C_HALT: begin
                ev_end <= mantissa_pkg::EV_HALT;
                step_count <= step_count + 32'd1;
                state <= ST_HALT;
              end

              C_PUT: begin
                ev_put <= 1'b1;
                ev_putval <= ra;
                step_count <= step_count + 32'd1;
              end

              C_PUTC: begin
                ev_putc <= 1'b1;
                ev_putcode <= putc_i[20:0];
                step_count <= step_count + 32'd1;
              end

              C_JMP: step_count <= step_count + 32'd1;      // the tail takes the target

              C_ALU: begin
                r_ret <= RET_ALU;
                r_skip <= d_skip;
                r_op <= d_op;
                r_a <= d_ra;
                r_b <= d_rb;
                r_fmt <= fa;
                r_wb2 <= (d_fop == mantissa_pkg::F_SPLIT) || (d_fop == mantissa_pkg::F_MANT);
                r_setflags <= 1'b1;
                // MOV passes a NaN on, and SPLIT passes x on when x is a NaN;
                // NEG, ABS and CPSGN make a new NaN with the sign changed
                r_prov0 <= ((d_fop == mantissa_pkg::F_MOV)
                            || ((d_fop == mantissa_pkg::F_SPLIT) && is_nan(rc))) ? mantissa_pkg::PROV_INHERIT
                           : ((d_fop == mantissa_pkg::F_NEG) || (d_fop == mantissa_pkg::F_ABS)
                              || (d_fop == mantissa_pkg::F_CPSGN)) ? mantissa_pkg::PROV_FRESH
                           : mantissa_pkg::PROV_CANON;
                r_src0 <= (d_fop == mantissa_pkg::F_SPLIT) ? d_rc : d_rb;
                r_prov1 <= ((d_fop == mantissa_pkg::F_SPLIT) && is_nan(rc))
                          ? mantissa_pkg::PROV_INHERIT : mantissa_pkg::PROV_CANON;
                r_src1 <= d_rc;
                state <= ST_FPU;
              end

              C_ST: begin
                st_fmt <= fb;
                st_cells <= {2'b0, 1'b1} << fb;
                st_sel <= st_i[1:0];                       // the cell of the word it starts at
                st_idx <= st_i[AW-1:0];
                st_val <= rb;
                r_ret <= RET_ST;
                state <= ST_FPU;
              end

              C_LDI: begin
                r_ret <= RET_ALU;
                r_skip <= d_skip;
                r_a <= d_ra;
                r_fmt <= fa;
                r_oret <= O_LDI;
                r_prov0 <= mantissa_pkg::PROV_CANON;
                r_wb2 <= 1'b0;
                r_setflags <= 1'b1;
                state <= ST_OPND;
              end

              C_LD: begin
                r_ret <= RET_ALU;
                r_skip <= d_skip;
                r_a <= d_ra;
                r_fmt <= fa;
                r_oret <= O_LD;
                r_prov0 <= mantissa_pkg::PROV_FRESH;       // a load makes a value of its own
                r_wb2 <= 1'b0;
                r_setflags <= 1'b0;
                st_sel <= ld_i[1:0];                       // which cell of the word it starts at
                state <= ST_OPND;
              end

              C_BR: begin
                r_oret <= O_TARGET;
                r_opnd_off <= 2'd0;
                state <= ST_OPND;
              end

              C_JAL: begin
                r_oret <= O_JAL;
                r_a <= d_ra;
                r_opnd_off <= 2'd0;
                state <= ST_OPND;
              end

              C_BSGN, C_BZ, C_BCMP: begin
                r_ret <= RET_BRANCH;
                r_skip <= d_skip;
                r_op <= d_op;
                state <= ST_FPU;
              end

              default: ;
            endcase
          end
        end

        ST_FPU: begin
          if (fpu_done) begin
            case (r_ret)
              RET_ALU: begin
                regs[r_a] <= fpu_y;
                if (r_wb2) regs[r_b] <= fpu_y2;
                if (r_setflags) flags <= flags | fpu_flags;
                ev_nwr <= r_wb2 ? 2'd2 : 2'd1;
                ev_wcode0 <= r_a;
                ev_wcode1 <= r_b;
                ev_wval0 <= fpu_y;
                ev_wval1 <= fpu_y2;
                ev_wprov0 <= r_prov0;
                ev_wprov1 <= r_prov1;
                ev_wsrc0 <= r_src0;
                ev_wsrc1 <= r_src1;
                step_count <= step_count + 32'd1;
              end
              RET_ST: state <= ST_POST;                    // the write went out this cycle
              RET_BRANCH: begin
                if (taken) begin
                  r_opnd_off <= ((r_op == OP_BSGN) && bsgn_pos) ? 2'd1 : 2'd0;
                  r_oret <= O_TARGET;
                  state <= ST_OPND;
                end else begin
                  step_count <= step_count + 32'd1;        // the tail advances past the operands
                end
              end
              default: ;
            endcase
          end
        end

        ST_OPND: begin
          if (r_oret == O_LDI) begin
            r_ret <= RET_ALU;
            state <= ST_FPU;
          end else if (r_oret == O_LD) begin
            r_ret <= RET_ALU;
            state <= ST_FPU;
          end else begin
            if (!target_now) begin
              // the operand ran off the end of memory, or the target is not a
              // place an instruction could be fetched from
              ev_end <= mantissa_pkg::EV_FAULT;
              ev_fault <= mantissa_pkg::FAULT_ADDRESS;
              fault_kind <= mantissa_pkg::FAULT_ADDRESS;
              step_count <= step_count + 32'd1;
              state <= ST_FAULT;
            end else begin
              if (r_oret == O_JAL) begin
                regs[r_a] <= link_d;                      // pc + 8, as it is: not rounded to a float
                ev_nwr <= 2'd1;
                ev_wcode0 <= r_a;
                ev_wval0 <= link_d;
                ev_wprov0 <= mantissa_pkg::PROV_CANON;
              end
              step_count <= step_count + 32'd1;
            end
          end
        end

        ST_POST: begin
          ev_store <= 1'b1;
          ev_saddr <= st_idx;
          ev_sfmt <= st_fmt;
          ev_sval <= st_val;
          step_count <= step_count + 32'd1;
        end

        default: ;                                       // ST_HALT, ST_FAULT, ST_LIMIT
      endcase

      // The next pc, when this cycle has settled it.  The fetch went out with
      // it (fetch_now) unless the pc is not a place an instruction could be.
      if (target_now) begin
        pc <= mem_rdata;
        state <= limit_new ? ST_LIMIT : ST_DECODE;
      end
      if (jmp_now) begin
        pc <= ra;
        state <= limit_new ? ST_LIMIT : ST_DECODE;
      end
      if (adv_now) begin
        pc <= pcn_d;
        state <= limit_new ? ST_LIMIT : (pcn_ok ? ST_DECODE : ST_FETCH_FAULT);
      end
    end
  end

`ifdef MANTISSA_DEBUG
  // A line per cycle, for simulation only: build the model with
  // -DMANTISSA_DEBUG to watch the state machine.
  always_ff @(posedge clk) begin
    if (rst_n)
      $display("t=%0t state=%0d pc=%h ins=%0d op=%0d a=%0d b=%0d c=%0d cls=%0d ok=%b go=%b fault=%0d fetch=%b memen=%b we=%b mask=%b addr=%h wdata=%h rdata=%h start=%b fop=%0d fmt=%0d f2=%0d fa=%h done=%b y=%h y2=%h ret=%0d oret=%0d skip=%0d adv=%b k=%0d tgt=%b taken=%b ev=%b",
               $time, state, pc, ins_n, d_op, d_a, d_b, d_c, d_cls, d_ok, dec_go, dec_fault, fetch_now,
               mem_en, mem_we, mem_wmask, mem_addr, mem_wdata, mem_rdata, fpu_start, fpu_op, fpu_fmt, fpu_fmt2,
               fpu_a, fpu_done, fpu_y, fpu_y2, r_ret, r_oret, r_skip, adv_now, adv_k, target_now, taken, ev_valid);
  end
`endif

endmodule

`default_nettype wire
