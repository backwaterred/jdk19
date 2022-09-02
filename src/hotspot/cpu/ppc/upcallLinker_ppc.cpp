/*
 * Copyright (c) 2020 SAP SE. All rights reserved.
 * Copyright (c) 2020, 2022, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

#include "precompiled.hpp"
#include "asm/macroAssembler.hpp"
#include "asm/macroAssembler.inline.hpp"
#include "code/vmreg.inline.hpp"
#include "logging/logStream.hpp"
#include "memory/resourceArea.hpp"
#include "prims/upcallLinker.hpp"
#include "register_ppc.hpp"
#include "runtime/sharedRuntime.hpp"
#include "runtime/signature.hpp"
#include "runtime/stubRoutines.hpp"
#include "utilities/formatBuffer.hpp"
#include "utilities/globalDefinitions.hpp"
#include "utilities/debug.hpp"

#define __ _masm->

// for callee saved regs, according to the caller's ABI
static int compute_reg_save_area_size(const ABIDescriptor& abi) {
  int size = 0;
  for (int i = 0; i < RegisterImpl::number_of_registers; i++) {
    Register reg = as_Register(i);
    if (!abi.is_volatile_reg(reg)) {
      size += 8; // bytes
    }
  }

  for (int i = 0; i < FloatRegisterImpl::number_of_registers; i++) {
    FloatRegister reg = as_FloatRegister(i);
    if (!abi.is_volatile_reg(reg)) {
      // TODO: Confirm (copied from AArch64) Only the lower 64 bits of vector registers need to be preserved.
      size += 8; // bytes
    }
  }

  return size;
}

static void preserve_callee_saved_registers(MacroAssembler* _masm, const ABIDescriptor& abi, int reg_save_area_offset) {
  // 1. iterate all registers in the architecture
  //     - check if they are volatile or not for the given abi
  //     - if NOT, we need to save it here

  int offset = reg_save_area_offset;

  __ block_comment("{ preserve_callee_saved_regs ");
  for (int i = 0; i < RegisterImpl::number_of_registers; i++) {
    Register reg = as_Register(i);
    if (!abi.is_volatile_reg(reg)) {
      __ std(reg, offset, R1_SP);
      offset += 8;
    }
  }

  for (int i = 0; i < FloatRegisterImpl::number_of_registers; i++) {
    FloatRegister reg = as_FloatRegister(i);
    if (!abi.is_volatile_reg(reg)) {
      __ stfd(reg, offset, R1_SP);
      offset += 8;
    }
  }

  __ block_comment("} preserve_callee_saved_regs ");
}

static void restore_callee_saved_registers(MacroAssembler* _masm, const ABIDescriptor& abi, int reg_save_area_offset) {
  // 1. iterate all registers in the architecture
  //     - check if they are volatile or not for the given abi
  //     - if NOT, we need to restore it here

  int offset = reg_save_area_offset;

  __ block_comment("{ restore_callee_saved_regs ");
  for (int i = 0; i < RegisterImpl::number_of_registers; i++) {
    Register reg = as_Register(i);
    if (!abi.is_volatile_reg(reg)) {
      __ ld(reg, offset, R1_SP);
      offset += 8;
    }
  }

  for (int i = 0; i < FloatRegisterImpl::number_of_registers; i++) {
    FloatRegister reg = as_FloatRegister(i);
    if (!abi.is_volatile_reg(reg)) {
      __ lfd(reg, offset, R1_SP);
      offset += 8;
    }
  }

  __ block_comment("} restore_callee_saved_regs ");
}

address UpcallLinker::make_upcall_stub(jobject receiver, Method* entry,
                                       BasicType* in_sig_bt, int total_in_args,
                                       BasicType* out_sig_bt, int total_out_args,
                                       BasicType ret_type,
                                       jobject jabi, jobject jconv,
                                       bool needs_return_buffer, int ret_buf_size) {
  ResourceMark rm;
  const ABIDescriptor abi = ForeignGlobals::parse_abi_descriptor(jabi);
  const CallRegs call_regs = ForeignGlobals::parse_call_regs(jconv);
  CodeBuffer buffer("upcall_stub", /* code_size = */ 2048, /* locs_size = */ 1024);

  Register rtmp1 = R5;
  Register rtmp2 = R6;
  Register rshuffle = R11; // TODO: Verify
  JavaCallingConvention out_conv;
  NativeCallingConvention in_conv(call_regs._arg_regs);
  ArgumentShuffle arg_shuffle(in_sig_bt, total_in_args, out_sig_bt, total_out_args, &in_conv, &out_conv, rshuffle->as_VMReg());
  int stack_slots = SharedRuntime::out_preserve_stack_slots() + arg_shuffle.out_arg_stack_slots();
  int out_arg_area = align_up(stack_slots * VMRegImpl::stack_slot_size, StackAlignmentInBytes);

#ifndef PRODUCT
  LogTarget(Trace, foreign, upcall) lt;
  if (lt.is_enabled()) {
    ResourceMark rm;
    LogStream ls(lt);
    arg_shuffle.print_on(&ls);
  }
#endif

  // out_arg_area (for stack arguments) doubles as shadow space for native calls.
  // make sure it is big enough.
  // if (out_arg_area < frame::arg_reg_save_area_bytes) {
  //   out_arg_area = frame::arg_reg_save_area_bytes;
  // }

  int reg_save_area_size = compute_reg_save_area_size(abi);
  RegSpiller arg_spilller(call_regs._arg_regs);
  RegSpiller result_spiller(call_regs._ret_regs);

  int shuffle_area_offset    = 0;
  int res_save_area_offset   = shuffle_area_offset    + out_arg_area;
  int arg_save_area_offset   = res_save_area_offset   + result_spiller.spill_size_bytes();
  int reg_save_area_offset   = arg_save_area_offset   + arg_spilller.spill_size_bytes();
  int frame_data_offset      = reg_save_area_offset   + reg_save_area_size;
  int frame_bottom_offset    = frame_data_offset      + sizeof(UpcallStub::FrameData);

  int ret_buf_offset = -1;
  if (needs_return_buffer) {
    ret_buf_offset = frame_bottom_offset;
    frame_bottom_offset += ret_buf_size;
  }

  int frame_size = frame_bottom_offset;
  frame_size = align_up(frame_size, StackAlignmentInBytes);

  // The space we have allocated will look like:
  // TODO: Is this correct on PPC64/LE?
  //
  //
  // FP-> |                     |
  //      |---------------------| = frame_bottom_offset = frame_size
  //      | (optional)          |
  //      | ret_buf             |
  //      |---------------------| = ret_buf_offset
  //      |                     |
  //      | FrameData           |
  //      |---------------------| = frame_data_offset
  //      |                     |
  //      | reg_save_area       |
  //      |---------------------| = reg_save_are_offset
  //      |                     |
  //      | arg_save_area       |
  //      |---------------------| = arg_save_are_offset
  //      |                     |
  //      | res_save_area       |
  //      |---------------------| = res_save_are_offset
  //      |                     |
  // SP-> | out_arg_area        |   needs to be at end for shadow space
  //
  //

  //////////////////////////////////////////////////////////////////////////////

  MacroAssembler* _masm = new MacroAssembler(&buffer);
  address start = __ pc();
  __ push_frame(frame_size, rtmp1);
  assert((abi._stack_alignment_bytes % 16) == 0, "must be 16 byte aligned");

  // we have to always spill args since we need to do a call to get the thread
  // (and maybe attach it).
  arg_spilller.generate_spill(_masm, arg_save_area_offset);
  preserve_callee_saved_registers(_masm, abi, reg_save_area_offset);

  __ block_comment("{ on_entry");
  //     On Entry:
  //     - Call UpcallLinker::on_entry to get thread pointer.
  //     - Store thread pointer in R16.
  __ andi(rtmp1, rtmp1, 0);                     // rtmp1 <- 0
  __ add_const_optimized(rtmp1, rtmp1,          // rtmp1 <- &UpcallLinker::on_entry
                         CAST_FROM_FN_PTR(uint64_t, UpcallLinker::on_entry), rtmp2, false);
  __ mtctr(rtmp1);                              // CTR <- &UpcallLinker::on_entry
  __ ld(R3, frame_data_offset, R1_SP);          // arg0 <- current-offset
  __ bctrl();                                   // UpcallLinker::on_entry(FrameData* context)
  __ mr(R16_thread, R3);                        // R16 <- thread_ptr
  __ block_comment("} on_entry");

  __ block_comment("{ argument shuffle");
  //     Argument Shuffle
  assert(NULL != NULL, "Entering arg shuffle");
  arg_spilller.generate_fill(_masm, arg_save_area_offset);
  if (needs_return_buffer) {
    assert(ret_buf_offset != -1, "no return buffer allocated");
    // TODO Implement on PPC
    __ should_not_reach_here();
  }
  arg_shuffle.generate(_masm, rshuffle->as_VMReg(), 0, abi._shadow_space_bytes);
  __ block_comment("} argument shuffle");

  __ block_comment("{ receiver ");
  //     Receiver
  //     - Load reciever (JObject) metadata
  assert(NULL != NULL, "Entering receiver");
  __ add_const_optimized(rshuffle, rshuffle,        // rshuffle <- &reciever
                         receiver, rtmp1, false);
  __ resolve_jobject(rshuffle, rtmp1,        // rshuffle <- resolved oop with base rshuffle
                     R14, MacroAssembler::PRESERVATION_FRAME_LR); // TODO: Set Preservation Level, R14 == nvtemp
  __ mr(R4, rshuffle);
  __ block_comment("} receiver ");

  __ block_comment("{ perform-upcall");
  //     Perform Upcall
  //     - Call the method stored in entry, after loading thread pointer
  //       and reciever-object data
  __ andi(rtmp1, rtmp1, 0);                     // rtmp1 <- 0
  __ add_const_optimized(rtmp1, rtmp1, (intptr_t)entry, rtmp2, false);
  __ ld(rtmp1, 0, rtmp1);
  __ add_const_optimized(rtmp1, rtmp1, in_bytes(Method::from_compiled_offset()),
                         rtmp2, false);
  __ ld(rtmp1, 0, rtmp1);
  __ mtctr(rtmp1);
  __ bctrl();
  // TODO: Handle rtn value
  __ block_comment("} perform-upcall");

  // return value shuffle
  if (!needs_return_buffer) {
#ifdef ASSERT
    if (call_regs._ret_regs.length() == 1) { // 0 or 1
      VMReg j_expected_result_reg;
      switch (ret_type) {
        case T_BOOLEAN:
        case T_BYTE:
        case T_SHORT:
        case T_CHAR:
        case T_INT:
        case T_LONG:
        // TODO PPC
        j_expected_result_reg = R3->as_VMReg();
        break;
        case T_FLOAT:
        case T_DOUBLE:
          // TODO PPC
          j_expected_result_reg = F1->as_VMReg();
          break;
        default:
          fatal("unexpected return type: %s", type2name(ret_type));
      }
      // No need to move for now, since CallArranger can pick a return type
      // that goes in the same reg for both CCs. But, at least assert they are the same
      assert(call_regs._ret_regs.at(0) == j_expected_result_reg,
      "unexpected result register: %s != %s", call_regs._ret_regs.at(0)->name(), j_expected_result_reg->name());
    }
#endif
  } else {
    assert(ret_buf_offset != -1, "no return buffer allocated");
    // TODO: PPC Return value in buffer
  }

  result_spiller.generate_spill(_masm, res_save_area_offset);

  __ block_comment("{ on_exit");
  //     On Exit
  //     Clean up stack.
  //     - Remove stack frame.
  //     - Restore !!!
  // TODO
  __ pop_frame();
  __ block_comment("} on_exit");

  restore_callee_saved_registers(_masm, abi, reg_save_area_offset);

  result_spiller.generate_fill(_masm, res_save_area_offset);

  // TODO PPC
  // __ leave();
  // __ ret(lr);

  //////////////////////////////////////////////////////////////////////////////

  __ block_comment("{ exception handler");

  intptr_t exception_handler_offset = __ pc() - start;

  // Native caller has no idea how to handle exceptions,
  // so we just crash here. Up to callee to catch exceptions.
  // __ verify_oop(???);
  // TODO: PPC
  __ should_not_reach_here();

  __ block_comment("} exception handler");

  _masm->flush();

#ifndef PRODUCT
  stringStream ss;
  ss.print("upcall_stub_%s", entry->signature()->as_C_string());
  const char* name = _masm->code_string(ss.as_string());
#else // PRODUCT
  const char* name = "upcall_stub";
#endif // PRODUCT

  UpcallStub* blob
    = UpcallStub::create(name,
                         &buffer,
                         exception_handler_offset,
                         receiver,
                         in_ByteSize(frame_data_offset));

  if (TraceOptimizedUpcallStubs) {
    blob->print_on(tty);
  }

  return blob->code_begin();
}
