// 公共API接口模块
// 导出给外部使用的矩阵运算函数

use crate::gemm::gemm_macro_generic;
use crate::gemm::naive_matmul;
use crate::kernel::{Avx2KernelF32, Avx2KernelI32, Avx512KernelI64, ScalarKernelI64};
use crate::Parallel;
use std::ops::{AddAssign, Mul};

pub unsafe fn gemm_f32(
    m: usize, n: usize, k: usize, // 输入进来的是一个m*k的矩阵A和一个k*n的矩阵B，输出一个m*n的矩阵C
    a_ptr: *const f32, step_a: usize, // A矩阵的行跨度，即原始矩阵有多少列
    b_ptr: *const f32, step_b: usize, // B矩阵的行跨度，即原始矩阵有多少列
    c_ptr: *mut f32, step_c: usize,  // C矩阵的行跨度，即原始矩阵有多少列
    parallel: bool
){
    gemm_macro_generic::<f32, Avx2KernelF32>(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
}

pub unsafe fn gemm_i32(
    m: usize, n: usize, k: usize,
    a_ptr: *const i32, step_a: usize,
    b_ptr: *const i32, step_b: usize,
    c_ptr: *mut i32, step_c: usize,
    parallel: bool
){
    gemm_macro_generic::<i32, Avx2KernelI32>(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
}

pub unsafe fn gemm_i64_avx512(
    m: usize, n: usize, k: usize,
    a_ptr: *const i64, step_a: usize,
    b_ptr: *const i64, step_b: usize,
    c_ptr: *mut i64, step_c: usize,
    parallel: bool
){
    gemm_macro_generic::<i64, Avx512KernelI64>(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
}

pub unsafe fn gemm_i64_scalar(
    m: usize, n: usize, k: usize,
    a_ptr: *const i64, step_a: usize,
    b_ptr: *const i64, step_b: usize,
    c_ptr: *mut i64, step_c: usize,
    parallel: bool
){
    gemm_macro_generic::<i64, ScalarKernelI64>(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
}

pub unsafe fn gemm_i64(
    m: usize, n: usize, k: usize,
    a_ptr: *const i64, step_a: usize,
    b_ptr: *const i64, step_b: usize,
    c_ptr: *mut i64, step_c: usize,
    parallel: bool
){
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq") {
            unsafe {
                gemm_i64_avx512(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
            }
            return;
        }
    }
    // 如果不支持AVX-512，就退回到标量实现
    unsafe {
        gemm_i64_scalar(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
    }
}

pub unsafe fn naive_gemm<T>(
    m: usize, n: usize, k: usize,
    a_ptr: *const T, step_a: usize,
    b_ptr: *const T, step_b: usize,
    c_ptr: *mut T, step_c: usize,
    parallel: bool
) where T: Default + Copy + AddAssign + Mul<Output = T>
{
    naive_matmul(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
}

pub trait GemmType: Default + Copy + AddAssign + Mul<Output = Self> {
    unsafe fn run_gemm(
        m: usize, n: usize, k: usize,
        a_ptr: *const Self, step_a: usize,
        b_ptr: *const Self, step_b: usize,
        c_ptr: *mut Self, step_c: usize,
        parallel: bool
    );
}

impl GemmType for f32 {
    unsafe fn run_gemm(
        m: usize, n: usize, k: usize,
        a_ptr: *const Self, step_a: usize,
        b_ptr: *const Self, step_b: usize,
        c_ptr: *mut Self, step_c: usize,
        parallel: bool
    ) {
        gemm_f32(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
    }
}

impl GemmType for i32 {
    unsafe fn run_gemm(
        m: usize, n: usize, k: usize,
        a_ptr: *const Self, step_a: usize,
        b_ptr: *const Self, step_b: usize,
        c_ptr: *mut Self, step_c: usize,
        parallel: bool
    ) {
        gemm_i32(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
    }
}

impl GemmType for i64 {
    unsafe fn run_gemm(
        m: usize, n: usize, k: usize,
        a_ptr: *const Self, step_a: usize,
        b_ptr: *const Self, step_b: usize,
        c_ptr: *mut Self, step_c: usize,
        parallel: bool
    ) {
        gemm_i64(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, parallel);
    }
}

pub unsafe fn gemm<T: GemmType>(
    m: usize, n: usize, k: usize,
    a_ptr: *const T, step_a: usize,
    b_ptr: *const T, step_b: usize,
    c_ptr: *mut T, step_c: usize,
    parallel: Parallel
){
    let use_parallel = parallel.resolve(m, n, k);
    if m <= 64 && n <= 64 && k <= 64 {
        naive_gemm(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, false);
        return;
    }
    unsafe {
        T::run_gemm(m, n, k, a_ptr, step_a, b_ptr, step_b, c_ptr, step_c, use_parallel);
    }
}

pub fn gemm_safe<T: GemmType>(
    m: usize, n: usize, k: usize,
    a: &[T], step_a: usize,
    b: &[T], step_b: usize,
    c: &mut [T], step_c: usize,
    parallel: Parallel
){
    assert!(a.len() >= m.saturating_sub(1) * step_a + k, "A矩阵的长度不足");
    assert!(b.len() >= k.saturating_sub(1) * step_b + n, "B矩阵的长度不足");
    assert!(c.len() >= m.saturating_sub(1) * step_c + n, "C矩阵的长度不足");

    unsafe {
        gemm(m, n, k,
            a.as_ptr(), step_a, 
            b.as_ptr(), step_b, 
            c.as_mut_ptr(), step_c,
            parallel
        );
    }
}
