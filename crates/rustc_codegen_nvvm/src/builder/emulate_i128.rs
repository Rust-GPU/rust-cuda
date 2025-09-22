use rustc_codegen_ssa::common::IntPredicate;
use rustc_codegen_ssa::traits::*;

use crate::llvm::{self, Value};

use super::{Builder, CountZerosKind, UNNAMED};

impl<'a, 'll, 'tcx> Builder<'a, 'll, 'tcx> {
    pub(super) fn is_i128(&self, val: &'ll Value) -> bool {
        let ty = self.val_ty(val);
        if ty == self.type_i128() {
            return true;
        }
        if unsafe { llvm::LLVMRustGetTypeKind(ty) == llvm::TypeKind::Integer } {
            unsafe { llvm::LLVMGetIntTypeWidth(ty) == 128 }
        } else {
            false
        }
    }

    // Helper to split i128 into low and high u64 parts
    fn split_i128(&mut self, val: &'ll Value) -> (&'ll Value, &'ll Value) {
        let vec_ty = self.type_vector(self.type_i64(), 2);
        let bitcast = unsafe { llvm::LLVMBuildBitCast(self.llbuilder, val, vec_ty, UNNAMED) };
        let lo = unsafe {
            llvm::LLVMBuildExtractElement(self.llbuilder, bitcast, self.cx.const_i32(0), UNNAMED)
        };
        let hi = unsafe {
            llvm::LLVMBuildExtractElement(self.llbuilder, bitcast, self.cx.const_i32(1), UNNAMED)
        };

        (lo, hi)
    }

    fn uadd_with_overflow_i64(
        &mut self,
        lhs: &'ll Value,
        rhs: &'ll Value,
    ) -> (&'ll Value, &'ll Value) {
        let call = self.call_intrinsic("llvm.uadd.with.overflow.i64", &[lhs, rhs]);
        (self.extract_value(call, 0), self.extract_value(call, 1))
    }

    fn usub_with_overflow_i64(
        &mut self,
        lhs: &'ll Value,
        rhs: &'ll Value,
    ) -> (&'ll Value, &'ll Value) {
        let call = self.call_intrinsic("llvm.usub.with.overflow.i64", &[lhs, rhs]);
        (self.extract_value(call, 0), self.extract_value(call, 1))
    }

    // Helper to combine two u64 values into i128
    fn combine_i128(&mut self, lo: &'ll Value, hi: &'ll Value) -> &'ll Value {
        let vec_ty = self.type_vector(self.type_i64(), 2);
        let mut vec = self.const_undef(vec_ty);
        vec = unsafe {
            llvm::LLVMBuildInsertElement(self.llbuilder, vec, lo, self.cx.const_i32(0), UNNAMED)
        };
        vec = unsafe {
            llvm::LLVMBuildInsertElement(self.llbuilder, vec, hi, self.cx.const_i32(1), UNNAMED)
        };
        unsafe { llvm::LLVMBuildBitCast(self.llbuilder, vec, self.type_i128(), UNNAMED) }
    }

    // Multiply two u64 values and return the full 128-bit product as (lo, hi)
    fn mul_u64_to_u128(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> (&'ll Value, &'ll Value) {
        let i64_ty = self.type_i64();
        let i32_ty = self.type_i32();
        let shift_32 = self.const_u64(32);
        let mask_32 = self.const_u64(0xFFFF_FFFF);

        // Split the operands into 32-bit halves
        let lhs_lo32 = self.trunc(lhs, i32_ty);
        let lhs_shifted = self.lshr(lhs, shift_32);
        let lhs_hi32 = self.trunc(lhs_shifted, i32_ty);
        let rhs_lo32 = self.trunc(rhs, i32_ty);
        let rhs_shifted = self.lshr(rhs, shift_32);
        let rhs_hi32 = self.trunc(rhs_shifted, i32_ty);

        // Extend halves to 64 bits for the partial products
        let lhs_lo64 = self.zext(lhs_lo32, i64_ty);
        let lhs_hi64 = self.zext(lhs_hi32, i64_ty);
        let rhs_lo64 = self.zext(rhs_lo32, i64_ty);
        let rhs_hi64 = self.zext(rhs_hi32, i64_ty);

        // Compute partial products (32-bit x 32-bit -> 64-bit)
        let p0 = unsafe { llvm::LLVMBuildMul(self.llbuilder, lhs_lo64, rhs_lo64, UNNAMED) };
        let p1 = unsafe { llvm::LLVMBuildMul(self.llbuilder, lhs_lo64, rhs_hi64, UNNAMED) };
        let p2 = unsafe { llvm::LLVMBuildMul(self.llbuilder, lhs_hi64, rhs_lo64, UNNAMED) };
        let p3 = unsafe { llvm::LLVMBuildMul(self.llbuilder, lhs_hi64, rhs_hi64, UNNAMED) };

        // Sum cross terms and track the carry that escapes the low 64 bits
        let (cross, cross_carry_bit) = self.uadd_with_overflow_i64(p1, p2);

        let cross_low = self.and(cross, mask_32);
        let cross_low_shifted = self.shl(cross_low, shift_32);
        let (lo, lo_carry_bit) = self.uadd_with_overflow_i64(p0, cross_low_shifted);

        let cross_high = self.lshr(cross, shift_32);
        let cross_carry_ext = self.zext(cross_carry_bit, i64_ty);
        let cross_carry_high = self.shl(cross_carry_ext, shift_32);
        let cross_total_high = self.add(cross_high, cross_carry_high);

        let (hi_temp, _) = self.uadd_with_overflow_i64(p3, cross_total_high);
        let lo_carry = self.zext(lo_carry_bit, i64_ty);
        let (hi, _) = self.uadd_with_overflow_i64(hi_temp, lo_carry);

        (lo, hi)
    }

    // Emulate 128-bit addition using two 64-bit additions with carry
    pub(super) fn emulate_i128_add(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let i64_ty = self.type_i64();
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        // Add low parts
        let sum_lo = unsafe { llvm::LLVMBuildAdd(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };

        // Check for carry from low addition
        let carry = self.icmp(IntPredicate::IntULT, sum_lo, lhs_lo);
        let carry_ext = self.zext(carry, i64_ty);

        // Add high parts with carry
        let sum_hi_temp = unsafe { llvm::LLVMBuildAdd(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };
        let sum_hi = unsafe { llvm::LLVMBuildAdd(self.llbuilder, sum_hi_temp, carry_ext, UNNAMED) };

        self.combine_i128(sum_lo, sum_hi)
    }

    // Emulate 128-bit subtraction
    pub(super) fn emulate_i128_sub(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let i64_ty = self.type_i64();
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        // Subtract low parts
        let diff_lo = unsafe { llvm::LLVMBuildSub(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };

        // Check for borrow
        let borrow = self.icmp(IntPredicate::IntUGT, rhs_lo, lhs_lo);
        let borrow_ext = self.zext(borrow, i64_ty);

        // Subtract high parts with borrow
        let diff_hi_temp = unsafe { llvm::LLVMBuildSub(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };
        let diff_hi =
            unsafe { llvm::LLVMBuildSub(self.llbuilder, diff_hi_temp, borrow_ext, UNNAMED) };

        self.combine_i128(diff_lo, diff_hi)
    }

    // Emulate 128-bit bitwise AND
    pub(super) fn emulate_i128_and(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        let and_lo = unsafe { llvm::LLVMBuildAnd(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };
        let and_hi = unsafe { llvm::LLVMBuildAnd(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };

        self.combine_i128(and_lo, and_hi)
    }

    // Emulate 128-bit bitwise OR
    pub(super) fn emulate_i128_or(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        let or_lo = unsafe { llvm::LLVMBuildOr(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };
        let or_hi = unsafe { llvm::LLVMBuildOr(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };

        self.combine_i128(or_lo, or_hi)
    }

    // Emulate 128-bit bitwise XOR
    pub(super) fn emulate_i128_xor(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        let xor_lo = unsafe { llvm::LLVMBuildXor(self.llbuilder, lhs_lo, rhs_lo, UNNAMED) };
        let xor_hi = unsafe { llvm::LLVMBuildXor(self.llbuilder, lhs_hi, rhs_hi, UNNAMED) };

        self.combine_i128(xor_lo, xor_hi)
    }

    // Emulate 128-bit multiplication using 64-bit partial products
    pub(super) fn emulate_i128_mul(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        // Compute partial products
        let (prod_ll_lo, prod_ll_hi) = self.mul_u64_to_u128(lhs_lo, rhs_lo);
        let partial_ll = self.combine_i128(prod_ll_lo, prod_ll_hi);

        let (prod_lh_lo, prod_lh_hi) = self.mul_u64_to_u128(lhs_lo, rhs_hi);
        let partial_lh = self.combine_i128(prod_lh_lo, prod_lh_hi);

        let (prod_hl_lo, prod_hl_hi) = self.mul_u64_to_u128(lhs_hi, rhs_lo);
        let partial_hl = self.combine_i128(prod_hl_lo, prod_hl_hi);

        let shift_64 = self.cx.const_i32(64);

        // Shift cross terms by 64 bits to align them within the 128-bit space
        let partial_lh_shifted = self.emulate_i128_shl(partial_lh, shift_64);
        let partial_hl_shifted = self.emulate_i128_shl(partial_hl, shift_64);

        // Sum the partial products modulo 2^128
        let acc = self.emulate_i128_add(partial_ll, partial_lh_shifted);
        self.emulate_i128_add(acc, partial_hl_shifted)
    }

    fn emulate_i128_udivrem(
        &mut self,
        num: &'ll Value,
        den: &'ll Value,
    ) -> (&'ll Value, &'ll Value) {
        let zero_i128 = self.const_u128(0);
        let one_i128 = self.const_u128(1);
        let one_i32 = self.cx.const_i32(1);
        let i32_ty = self.cx.type_i32();
        let i128_ty = self.type_i128();

        let trap_bb = self.append_sibling_block("i128_udiv_trap");
        let main_bb = self.append_sibling_block("i128_udiv_main");
        let lt_bb = self.append_sibling_block("i128_udiv_lt");
        let loop_header_bb = self.append_sibling_block("i128_udiv_loop_header");
        let loop_body_bb = self.append_sibling_block("i128_udiv_loop_body");
        let exit_bb = self.append_sibling_block("i128_udiv_exit");

        let mut exit_bx = Self::build(self.cx, exit_bb);
        let quot_res = exit_bx.phi(i128_ty, &[], &[]);
        let rem_res = exit_bx.phi(i128_ty, &[], &[]);

        let den_is_zero = self.icmp(IntPredicate::IntEQ, den, zero_i128);
        self.cond_br(den_is_zero, trap_bb, main_bb);

        let mut trap_bx = Self::build(self.cx, trap_bb);
        trap_bx.call_intrinsic("llvm.trap", &[]);
        trap_bx.unreachable();

        let mut main_bx = Self::build(self.cx, main_bb);
        let num_lt_den = main_bx.icmp(IntPredicate::IntULT, num, den);
        let bit_init = main_bx.cx.const_int(i32_ty, 128);
        main_bx.cond_br(num_lt_den, lt_bb, loop_header_bb);

        let mut lt_bx = Self::build(self.cx, lt_bb);
        lt_bx.br(exit_bb);
        exit_bx.add_incoming_to_phi(quot_res, zero_i128, lt_bb);
        exit_bx.add_incoming_to_phi(rem_res, num, lt_bb);

        let mut header_bx = Self::build(self.cx, loop_header_bb);
        let bit_phi = header_bx.phi(i32_ty, &[bit_init], &[main_bb]);
        let quot_phi = header_bx.phi(i128_ty, &[zero_i128], &[main_bb]);
        let rem_phi = header_bx.phi(i128_ty, &[zero_i128], &[main_bb]);
        let bit_zero = header_bx.icmp(IntPredicate::IntEQ, bit_phi, header_bx.cx.const_i32(0));
        header_bx.cond_br(bit_zero, exit_bb, loop_body_bb);
        exit_bx.add_incoming_to_phi(quot_res, quot_phi, loop_header_bb);
        exit_bx.add_incoming_to_phi(rem_res, rem_phi, loop_header_bb);

        let mut body_bx = Self::build(self.cx, loop_body_bb);
        let bit_next = body_bx.sub(bit_phi, one_i32);
        let num_shifted = body_bx.emulate_i128_lshr(num, bit_next);
        let bit_val = body_bx.and(num_shifted, one_i128);
        let rem_shift = body_bx.emulate_i128_shl(rem_phi, one_i32);
        let rem_candidate = body_bx.or(rem_shift, bit_val);
        let rem_ge = body_bx.icmp(IntPredicate::IntUGE, rem_candidate, den);
        let rem_sub = body_bx.emulate_i128_sub(rem_candidate, den);
        let rem_new = body_bx.select(rem_ge, rem_sub, rem_candidate);
        let one_shifted = body_bx.emulate_i128_shl(one_i128, bit_next);
        let quot_candidate = body_bx.or(quot_phi, one_shifted);
        let quot_new = body_bx.select(rem_ge, quot_candidate, quot_phi);
        let loop_body_exit_bb = body_bx.llbb();
        body_bx.br(loop_header_bb);
        header_bx.add_incoming_to_phi(bit_phi, bit_next, loop_body_exit_bb);
        header_bx.add_incoming_to_phi(quot_phi, quot_new, loop_body_exit_bb);
        header_bx.add_incoming_to_phi(rem_phi, rem_new, loop_body_exit_bb);

        *self = exit_bx;
        (quot_res, rem_res)
    }

    pub(super) fn emulate_i128_udiv(&mut self, num: &'ll Value, den: &'ll Value) -> &'ll Value {
        self.emulate_i128_udivrem(num, den).0
    }

    pub(super) fn emulate_i128_urem(&mut self, num: &'ll Value, den: &'ll Value) -> &'ll Value {
        self.emulate_i128_udivrem(num, den).1
    }

    fn emulate_i128_sdivrem(
        &mut self,
        num: &'ll Value,
        den: &'ll Value,
    ) -> (&'ll Value, &'ll Value) {
        let min_i128 = self.const_u128(1u128 << 127);
        let neg_one_i128 = self.const_u128(u128::MAX);
        let num_is_min = self.icmp(IntPredicate::IntEQ, num, min_i128);
        let den_is_neg_one = self.icmp(IntPredicate::IntEQ, den, neg_one_i128);
        let overflow_case = self.and(num_is_min, den_is_neg_one);

        let trap_bb = self.append_sibling_block("i128_sdiv_overflow_trap");
        let cont_bb = self.append_sibling_block("i128_sdiv_overflow_cont");

        self.cond_br(overflow_case, trap_bb, cont_bb);

        let mut trap_bx = Self::build(self.cx, trap_bb);
        trap_bx.call_intrinsic("llvm.trap", &[]);
        trap_bx.unreachable();

        let mut bx = Self::build(self.cx, cont_bb);

        let zero_u64 = bx.const_u64(0);
        let (_, num_hi) = bx.split_i128(num);
        let (_, den_hi) = bx.split_i128(den);
        let num_neg = bx.icmp(IntPredicate::IntSLT, num_hi, zero_u64);
        let den_neg = bx.icmp(IntPredicate::IntSLT, den_hi, zero_u64);
        let num_negated = bx.emulate_i128_neg(num);
        let num_abs = bx.select(num_neg, num_negated, num);
        let den_negated = bx.emulate_i128_neg(den);
        let den_abs = bx.select(den_neg, den_negated, den);
        let (quot_raw, rem_raw) = bx.emulate_i128_udivrem(num_abs, den_abs);
        let sign_diff = bx.xor(num_neg, den_neg);
        let quot_negated = bx.emulate_i128_neg(quot_raw);
        let quot = bx.select(sign_diff, quot_negated, quot_raw);
        let rem_negated = bx.emulate_i128_neg(rem_raw);
        let rem = bx.select(num_neg, rem_negated, rem_raw);

        *self = bx;
        (quot, rem)
    }

    pub(super) fn emulate_i128_sdiv(&mut self, num: &'ll Value, den: &'ll Value) -> &'ll Value {
        self.emulate_i128_sdivrem(num, den).0
    }

    pub(super) fn emulate_i128_srem(&mut self, num: &'ll Value, den: &'ll Value) -> &'ll Value {
        self.emulate_i128_sdivrem(num, den).1
    }

    pub(super) fn emulate_i128_add_with_overflow(
        &mut self,
        lhs: &'ll Value,
        rhs: &'ll Value,
        signed: bool,
    ) -> (&'ll Value, &'ll Value) {
        let i64_ty = self.type_i64();
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        let (sum_lo, carry_lo) = self.uadd_with_overflow_i64(lhs_lo, rhs_lo);
        let carry_lo_ext = self.zext(carry_lo, i64_ty);

        let (sum_hi_temp, carry_hi1) = self.uadd_with_overflow_i64(lhs_hi, rhs_hi);
        let (sum_hi, carry_hi2) = self.uadd_with_overflow_i64(sum_hi_temp, carry_lo_ext);

        let unsigned_overflow = self.or(carry_hi1, carry_hi2);

        let overflow_flag = if signed {
            let zero = self.const_u64(0);
            let lhs_neg = self.icmp(IntPredicate::IntSLT, lhs_hi, zero);
            let rhs_neg = self.icmp(IntPredicate::IntSLT, rhs_hi, zero);
            let res_neg = self.icmp(IntPredicate::IntSLT, sum_hi, zero);
            let same_sign = self.icmp(IntPredicate::IntEQ, lhs_neg, rhs_neg);
            let sign_diff = self.icmp(IntPredicate::IntNE, lhs_neg, res_neg);
            self.and(same_sign, sign_diff)
        } else {
            unsigned_overflow
        };

        let result = self.combine_i128(sum_lo, sum_hi);
        (result, overflow_flag)
    }

    pub(super) fn emulate_i128_sub_with_overflow(
        &mut self,
        lhs: &'ll Value,
        rhs: &'ll Value,
        signed: bool,
    ) -> (&'ll Value, &'ll Value) {
        let i64_ty = self.type_i64();
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        let (diff_lo, borrow_lo) = self.usub_with_overflow_i64(lhs_lo, rhs_lo);
        let borrow_lo_ext = self.zext(borrow_lo, i64_ty);

        let (diff_hi_temp, borrow_hi1) = self.usub_with_overflow_i64(lhs_hi, rhs_hi);
        let (diff_hi, borrow_hi2) = self.usub_with_overflow_i64(diff_hi_temp, borrow_lo_ext);

        let unsigned_overflow = self.or(borrow_hi1, borrow_hi2);

        let overflow_flag = if signed {
            let zero = self.const_u64(0);
            let lhs_neg = self.icmp(IntPredicate::IntSLT, lhs_hi, zero);
            let rhs_neg = self.icmp(IntPredicate::IntSLT, rhs_hi, zero);
            let res_neg = self.icmp(IntPredicate::IntSLT, diff_hi, zero);
            let lhs_diff_rhs = self.icmp(IntPredicate::IntNE, lhs_neg, rhs_neg);
            let res_diff_lhs = self.icmp(IntPredicate::IntNE, res_neg, lhs_neg);
            self.and(lhs_diff_rhs, res_diff_lhs)
        } else {
            unsigned_overflow
        };

        let result = self.combine_i128(diff_lo, diff_hi);
        (result, overflow_flag)
    }

    pub(super) fn emulate_i128_mul_with_overflow(
        &mut self,
        lhs: &'ll Value,
        rhs: &'ll Value,
        signed: bool,
    ) -> (&'ll Value, &'ll Value) {
        let i64_ty = self.type_i64();
        let (lhs_lo, lhs_hi) = self.split_i128(lhs);
        let (rhs_lo, rhs_hi) = self.split_i128(rhs);

        let (ll_lo, ll_hi) = self.mul_u64_to_u128(lhs_lo, rhs_lo);
        let (lh_lo, lh_hi) = self.mul_u64_to_u128(lhs_lo, rhs_hi);
        let (hl_lo, hl_hi) = self.mul_u64_to_u128(lhs_hi, rhs_lo);
        let (hh_lo, hh_hi) = self.mul_u64_to_u128(lhs_hi, rhs_hi);

        let (mid_temp, carry_mid1) = self.uadd_with_overflow_i64(ll_hi, lh_lo);
        let (mid_sum, carry_mid2) = self.uadd_with_overflow_i64(mid_temp, hl_lo);
        let carry_mid1_ext = self.zext(carry_mid1, i64_ty);
        let carry_mid2_ext = self.zext(carry_mid2, i64_ty);
        let mid_carry_sum = self.add(carry_mid1_ext, carry_mid2_ext);

        let result = self.combine_i128(ll_lo, mid_sum);

        let (upper_temp0, carry_upper1) = self.uadd_with_overflow_i64(lh_hi, hl_hi);
        let (upper_temp1, carry_upper2) = self.uadd_with_overflow_i64(upper_temp0, hh_lo);
        let (upper_low, carry_upper3) = self.uadd_with_overflow_i64(upper_temp1, mid_carry_sum);

        let carry_upper1_ext = self.zext(carry_upper1, i64_ty);
        let carry_upper2_ext = self.zext(carry_upper2, i64_ty);
        let carry_upper3_ext = self.zext(carry_upper3, i64_ty);
        let carries_sum = self.add(carry_upper1_ext, carry_upper2_ext);
        let carries_sum = self.add(carries_sum, carry_upper3_ext);

        let (upper_high, carry_upper4) = self.uadd_with_overflow_i64(hh_hi, carries_sum);

        let zero = self.const_u64(0);
        let upper_low_nonzero = self.icmp(IntPredicate::IntNE, upper_low, zero);
        let upper_high_nonzero = self.icmp(IntPredicate::IntNE, upper_high, zero);
        let upper_nonzero = self.or(upper_low_nonzero, upper_high_nonzero);
        let unsigned_overflow = self.or(upper_nonzero, carry_upper4);

        let overflow_flag = if signed {
            let max = self.const_u64(u64::MAX);
            let (_, res_hi) = self.split_i128(result);
            let res_neg = self.icmp(IntPredicate::IntSLT, res_hi, zero);

            let upper_low_zero = self.icmp(IntPredicate::IntEQ, upper_low, zero);
            let upper_high_zero = self.icmp(IntPredicate::IntEQ, upper_high, zero);
            let high_all_zero = self.and(upper_low_zero, upper_high_zero);

            let upper_low_ones = self.icmp(IntPredicate::IntEQ, upper_low, max);
            let upper_high_ones = self.icmp(IntPredicate::IntEQ, upper_high, max);
            let high_all_ones = self.and(upper_low_ones, upper_high_ones);

            let not_high_all_zero = self.not(high_all_zero);
            let overflow_pos = self.or(not_high_all_zero, carry_upper4);
            let not_high_all_ones = self.not(high_all_ones);
            let overflow_neg = self.or(not_high_all_ones, carry_upper4);

            self.select(res_neg, overflow_neg, overflow_pos)
        } else {
            unsigned_overflow
        };

        (result, overflow_flag)
    }

    pub(super) fn emulate_i128_shl(&mut self, val: &'ll Value, shift: &'ll Value) -> &'ll Value {
        let (lo, hi) = self.split_i128(val);
        let zero_i32 = self.cx.const_i32(0);
        let sixty_four = self.cx.const_i32(64);
        let one_twenty_eight = self.cx.const_i32(128);

        let shift_is_zero = self.icmp(IntPredicate::IntEQ, shift, zero_i32);
        let shift_ge64 = self.icmp(IntPredicate::IntUGE, shift, sixty_four);
        let shift_ge128 = self.icmp(IntPredicate::IntUGE, shift, one_twenty_eight);

        let ge128_bb = self.append_sibling_block("i128_shl_ge128");
        let lt128_bb = self.append_sibling_block("i128_shl_lt128");
        let ge64_bb = self.append_sibling_block("i128_shl_ge64");
        let lt64_bb = self.append_sibling_block("i128_shl_lt64");
        let zero_bb = self.append_sibling_block("i128_shl_zero");
        let nz_bb = self.append_sibling_block("i128_shl_lt64_nz");
        let merge_bb = self.append_sibling_block("i128_shl_merge");

        self.cond_br(shift_ge128, ge128_bb, lt128_bb);

        let mut ge128_bx = Self::build(self.cx, ge128_bb);
        let ge128_res = ge128_bx.const_u128(0);
        ge128_bx.br(merge_bb);

        let mut lt128_bx = Self::build(self.cx, lt128_bb);
        lt128_bx.cond_br(shift_ge64, ge64_bb, lt64_bb);

        let mut ge64_bx = Self::build(self.cx, ge64_bb);
        let shift_minus64 = ge64_bx.sub(shift, sixty_four);
        let shift_minus64_i64 = ge64_bx.zext(shift_minus64, ge64_bx.type_i64());
        let hi_from_lo = ge64_bx.shl(lo, shift_minus64_i64);
        let ge64_res = ge64_bx.combine_i128(ge64_bx.const_u64(0), hi_from_lo);
        ge64_bx.br(merge_bb);

        let mut lt64_bx = Self::build(self.cx, lt64_bb);
        lt64_bx.cond_br(shift_is_zero, zero_bb, nz_bb);

        let mut zero_bx = Self::build(self.cx, zero_bb);
        zero_bx.br(merge_bb);

        let mut nz_bx = Self::build(self.cx, nz_bb);
        let shift_i64 = nz_bx.zext(shift, nz_bx.type_i64());
        let new_lo = nz_bx.shl(lo, shift_i64);
        let hi_shifted = nz_bx.shl(hi, shift_i64);
        let inv = nz_bx.sub(sixty_four, shift);
        let inv_i64 = nz_bx.zext(inv, nz_bx.type_i64());
        let carry = nz_bx.lshr(lo, inv_i64);
        let new_hi = nz_bx.or(hi_shifted, carry);
        let nz_res = nz_bx.combine_i128(new_lo, new_hi);
        nz_bx.br(merge_bb);

        let mut merge_bx = Self::build(self.cx, merge_bb);
        let result = merge_bx.phi(self.type_i128(), &[], &[]);
        merge_bx.add_incoming_to_phi(result, ge128_res, ge128_bb);
        merge_bx.add_incoming_to_phi(result, ge64_res, ge64_bb);
        merge_bx.add_incoming_to_phi(result, val, zero_bb);
        merge_bx.add_incoming_to_phi(result, nz_res, nz_bb);

        *self = merge_bx;
        result
    }

    pub(super) fn emulate_i128_lshr(&mut self, val: &'ll Value, shift: &'ll Value) -> &'ll Value {
        let (lo, hi) = self.split_i128(val);
        let zero_i32 = self.cx.const_i32(0);
        let sixty_four = self.cx.const_i32(64);
        let one_twenty_eight = self.cx.const_i32(128);

        let shift_is_zero = self.icmp(IntPredicate::IntEQ, shift, zero_i32);
        let shift_ge64 = self.icmp(IntPredicate::IntUGE, shift, sixty_four);
        let shift_ge128 = self.icmp(IntPredicate::IntUGE, shift, one_twenty_eight);

        let ge128_bb = self.append_sibling_block("i128_lshr_ge128");
        let lt128_bb = self.append_sibling_block("i128_lshr_lt128");
        let ge64_bb = self.append_sibling_block("i128_lshr_ge64");
        let lt64_bb = self.append_sibling_block("i128_lshr_lt64");
        let zero_bb = self.append_sibling_block("i128_lshr_zero");
        let nz_bb = self.append_sibling_block("i128_lshr_lt64_nz");
        let merge_bb = self.append_sibling_block("i128_lshr_merge");

        self.cond_br(shift_ge128, ge128_bb, lt128_bb);

        let mut ge128_bx = Self::build(self.cx, ge128_bb);
        let ge128_res = ge128_bx.const_u128(0);
        ge128_bx.br(merge_bb);

        let mut lt128_bx = Self::build(self.cx, lt128_bb);
        lt128_bx.cond_br(shift_ge64, ge64_bb, lt64_bb);

        let mut ge64_bx = Self::build(self.cx, ge64_bb);
        let shift_minus64 = ge64_bx.sub(shift, sixty_four);
        let shift_minus64_i64 = ge64_bx.zext(shift_minus64, ge64_bx.type_i64());
        let new_lo = ge64_bx.lshr(hi, shift_minus64_i64);
        let ge64_res = ge64_bx.combine_i128(new_lo, ge64_bx.const_u64(0));
        ge64_bx.br(merge_bb);

        let mut lt64_bx = Self::build(self.cx, lt64_bb);
        lt64_bx.cond_br(shift_is_zero, zero_bb, nz_bb);

        let mut zero_bx = Self::build(self.cx, zero_bb);
        zero_bx.br(merge_bb);

        let mut nz_bx = Self::build(self.cx, nz_bb);
        let shift_i64 = nz_bx.zext(shift, nz_bx.type_i64());
        let new_lo = nz_bx.lshr(lo, shift_i64);
        let inv = nz_bx.sub(sixty_four, shift);
        let inv_i64 = nz_bx.zext(inv, nz_bx.type_i64());
        let carry = nz_bx.shl(hi, inv_i64);
        let new_lo = nz_bx.or(new_lo, carry);
        let new_hi = nz_bx.lshr(hi, shift_i64);
        let nz_res = nz_bx.combine_i128(new_lo, new_hi);
        nz_bx.br(merge_bb);

        let mut merge_bx = Self::build(self.cx, merge_bb);
        let result = merge_bx.phi(self.type_i128(), &[], &[]);
        merge_bx.add_incoming_to_phi(result, ge128_res, ge128_bb);
        merge_bx.add_incoming_to_phi(result, ge64_res, ge64_bb);
        merge_bx.add_incoming_to_phi(result, val, zero_bb);
        merge_bx.add_incoming_to_phi(result, nz_res, nz_bb);

        *self = merge_bx;
        result
    }

    pub(super) fn emulate_i128_ashr(&mut self, val: &'ll Value, shift: &'ll Value) -> &'ll Value {
        let (lo, hi) = self.split_i128(val);
        let zero_i32 = self.cx.const_i32(0);
        let sixty_four = self.cx.const_i32(64);
        let one_twenty_eight = self.cx.const_i32(128);
        let zero_u64 = self.const_u64(0);

        let shift_is_zero = self.icmp(IntPredicate::IntEQ, shift, zero_i32);
        let shift_ge64 = self.icmp(IntPredicate::IntUGE, shift, sixty_four);
        let shift_ge128 = self.icmp(IntPredicate::IntUGE, shift, one_twenty_eight);
        let is_negative = self.icmp(IntPredicate::IntSLT, hi, zero_u64);

        let ge128_bb = self.append_sibling_block("i128_ashr_ge128");
        let lt128_bb = self.append_sibling_block("i128_ashr_lt128");
        let ge64_bb = self.append_sibling_block("i128_ashr_ge64");
        let lt64_bb = self.append_sibling_block("i128_ashr_lt64");
        let zero_bb = self.append_sibling_block("i128_ashr_zero");
        let nz_bb = self.append_sibling_block("i128_ashr_lt64_nz");
        let merge_bb = self.append_sibling_block("i128_ashr_merge");

        self.cond_br(shift_ge128, ge128_bb, lt128_bb);

        let mut ge128_bx = Self::build(self.cx, ge128_bb);
        let all_ones = ge128_bx.const_u128(u128::MAX);
        let zero = ge128_bx.const_u128(0);
        let ge128_res = ge128_bx.select(is_negative, all_ones, zero);
        ge128_bx.br(merge_bb);

        let mut lt128_bx = Self::build(self.cx, lt128_bb);
        lt128_bx.cond_br(shift_ge64, ge64_bb, lt64_bb);

        let mut ge64_bx = Self::build(self.cx, ge64_bb);
        let shift_minus64 = ge64_bx.sub(shift, sixty_four);
        let shift_minus64_i64 = ge64_bx.zext(shift_minus64, ge64_bx.type_i64());
        let new_lo = ge64_bx.ashr(hi, shift_minus64_i64);
        let fill_hi = ge64_bx.select(
            is_negative,
            ge64_bx.const_u64(u64::MAX),
            ge64_bx.const_u64(0),
        );
        let ge64_res = ge64_bx.combine_i128(new_lo, fill_hi);
        ge64_bx.br(merge_bb);

        let mut lt64_bx = Self::build(self.cx, lt64_bb);
        lt64_bx.cond_br(shift_is_zero, zero_bb, nz_bb);

        let mut zero_bx = Self::build(self.cx, zero_bb);
        zero_bx.br(merge_bb);

        let mut nz_bx = Self::build(self.cx, nz_bb);
        let shift_i64 = nz_bx.zext(shift, nz_bx.type_i64());
        let new_lo_shift = nz_bx.lshr(lo, shift_i64);
        let inv = nz_bx.sub(sixty_four, shift);
        let inv_i64 = nz_bx.zext(inv, nz_bx.type_i64());
        let carry = nz_bx.shl(hi, inv_i64);
        let new_lo = nz_bx.or(new_lo_shift, carry);
        let new_hi = nz_bx.ashr(hi, shift_i64);
        let nz_res = nz_bx.combine_i128(new_lo, new_hi);
        nz_bx.br(merge_bb);

        let mut merge_bx = Self::build(self.cx, merge_bb);
        let result = merge_bx.phi(self.type_i128(), &[], &[]);
        merge_bx.add_incoming_to_phi(result, ge128_res, ge128_bb);
        merge_bx.add_incoming_to_phi(result, ge64_res, ge64_bb);
        merge_bx.add_incoming_to_phi(result, val, zero_bb);
        merge_bx.add_incoming_to_phi(result, nz_res, nz_bb);

        *self = merge_bx;
        result
    }

    // Emulate 128-bit bitwise NOT
    pub(super) fn emulate_i128_not(&mut self, val: &'ll Value) -> &'ll Value {
        let (lo, hi) = self.split_i128(val);

        let not_lo = unsafe { llvm::LLVMBuildNot(self.llbuilder, lo, UNNAMED) };
        let not_hi = unsafe { llvm::LLVMBuildNot(self.llbuilder, hi, UNNAMED) };

        self.combine_i128(not_lo, not_hi)
    }

    // Emulate 128-bit negation (two's complement)
    pub(super) fn emulate_i128_neg(&mut self, val: &'ll Value) -> &'ll Value {
        // Two's complement: ~val + 1
        let not_val = self.emulate_i128_not(val);
        let one = self.const_u128(1);
        self.emulate_i128_add(not_val, one)
    }

    pub(crate) fn emulate_i128_bswap(&mut self, val: &'ll Value) -> &'ll Value {
        // Split the 128-bit value into two 64-bit halves
        let (lo, hi) = self.split_i128(val);

        // Byte-swap each 64-bit half using the LLVM intrinsic (which exists in LLVM 7.1)
        let swapped_lo = self.call_intrinsic("llvm.bswap.i64", &[lo]);
        let swapped_hi = self.call_intrinsic("llvm.bswap.i64", &[hi]);

        // Swap the halves: the high part becomes low and vice versa
        self.combine_i128(swapped_hi, swapped_lo)
    }

    pub(crate) fn emulate_i128_count_zeros(
        &mut self,
        val: &'ll Value,
        kind: CountZerosKind,
        is_nonzero: bool,
    ) -> &'ll Value {
        // Split the 128-bit value into two 64-bit halves
        let (lo, hi) = self.split_i128(val);

        match kind {
            CountZerosKind::Leading => {
                // Count leading zeros: check high part first
                let hi_is_zero = self.icmp(IntPredicate::IntEQ, hi, self.const_u64(0));
                let hi_ctlz =
                    self.call_intrinsic("llvm.ctlz.i64", &[hi, self.const_bool(is_nonzero)]);
                let lo_ctlz =
                    self.call_intrinsic("llvm.ctlz.i64", &[lo, self.const_bool(is_nonzero)]);

                // If high part is zero, result is 64 + ctlz(lo), otherwise ctlz(hi)
                let lo_ctlz_plus_64 = self.add(lo_ctlz, self.const_u64(64));
                let result_64 = self.select(hi_is_zero, lo_ctlz_plus_64, hi_ctlz);

                // Zero-extend to i128
                self.zext(result_64, self.type_i128())
            }
            CountZerosKind::Trailing => {
                // Count trailing zeros: check low part first
                let lo_is_zero = self.icmp(IntPredicate::IntEQ, lo, self.const_u64(0));
                let lo_cttz =
                    self.call_intrinsic("llvm.cttz.i64", &[lo, self.const_bool(is_nonzero)]);
                let hi_cttz =
                    self.call_intrinsic("llvm.cttz.i64", &[hi, self.const_bool(is_nonzero)]);

                // If low part is zero, result is 64 + cttz(hi), otherwise cttz(lo)
                let hi_cttz_plus_64 = self.add(hi_cttz, self.const_u64(64));
                let result_64 = self.select(lo_is_zero, hi_cttz_plus_64, lo_cttz);

                // Zero-extend to i128
                self.zext(result_64, self.type_i128())
            }
        }
    }

    pub(crate) fn emulate_i128_ctpop(&mut self, val: &'ll Value) -> &'ll Value {
        // Split the 128-bit value into two 64-bit halves
        let (lo, hi) = self.split_i128(val);

        // Count population (number of 1 bits) in each half
        let lo_popcount = self.call_intrinsic("llvm.ctpop.i64", &[lo]);
        let hi_popcount = self.call_intrinsic("llvm.ctpop.i64", &[hi]);

        // Add the two counts
        let total_64 = self.add(lo_popcount, hi_popcount);

        // Zero-extend to i128
        self.zext(total_64, self.type_i128())
    }

    pub(crate) fn emulate_i128_rotate(
        &mut self,
        val: &'ll Value,
        shift: &'ll Value,
        is_left: bool,
    ) -> &'ll Value {
        // Rotate is implemented as: (val << shift) | (val >> (128 - shift))
        // For rotate right: (val >> shift) | (val << (128 - shift))

        // Ensure shift is i128
        let shift_128 = if self.val_ty(shift) == self.type_i128() {
            shift
        } else {
            self.zext(shift, self.type_i128())
        };

        // Calculate 128 - shift for the complementary shift
        let bits_128 = self.const_u128(128);
        let shift_complement = self.sub(bits_128, shift_128);

        // Perform the two shifts
        let (first_shift, second_shift) = if is_left {
            (self.shl(val, shift_128), self.lshr(val, shift_complement))
        } else {
            (self.lshr(val, shift_128), self.shl(val, shift_complement))
        };

        // Combine with OR
        self.or(first_shift, second_shift)
    }

    pub(crate) fn emulate_i128_bitreverse(&mut self, val: &'ll Value) -> &'ll Value {
        // Split the 128-bit value into two 64-bit halves
        let (lo, hi) = self.split_i128(val);

        // Reverse bits in each half using the 64-bit intrinsic
        let reversed_lo = self.call_intrinsic("llvm.bitreverse.i64", &[lo]);
        let reversed_hi = self.call_intrinsic("llvm.bitreverse.i64", &[hi]);

        // Swap the halves: reversed high becomes low and vice versa
        self.combine_i128(reversed_hi, reversed_lo)
    }
}
