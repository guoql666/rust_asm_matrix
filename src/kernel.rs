use std::arch::asm;
use crate::gemm::MicroKernel;

pub struct Avx2KernelF32;
pub struct Avx2KernelI32;
pub struct Avx512KernelI64;
pub struct ScalarKernelI64;

impl MicroKernel<f32> for Avx2KernelF32 {
    const MICRO_COL_SIZE: usize = 8;
    const MICRO_ROW_SIZE: usize = 8;

    unsafe fn package_a(row_size: usize, col_size: usize, step: usize, a_ptr: *const f32, buffer: &mut [f32]) {
        crate::memory::package_a(row_size, col_size, step, a_ptr, buffer, Self::MICRO_ROW_SIZE);
    }

    unsafe fn package_b(row_size: usize, col_size: usize, step: usize, b_ptr: *const f32, buffer: &mut [f32]) {
        crate::memory::package_b(row_size, col_size, step, b_ptr, buffer, Self::MICRO_COL_SIZE);
    }

    unsafe fn invoke(k: usize, a: *const f32, b: *const f32, c: *mut f32, step: usize) {
        micro_kernel_f32_8x8(k, a, b, c, step);
    }
}

impl MicroKernel<i32> for Avx2KernelI32 {
    const MICRO_COL_SIZE: usize = 8;
    const MICRO_ROW_SIZE: usize = 8;

    unsafe fn package_a(row_size: usize, col_size: usize, step: usize, a_ptr: *const i32, buffer: &mut [i32]) {
        crate::memory::package_a(row_size, col_size, step, a_ptr, buffer, Self::MICRO_ROW_SIZE);
    }

    unsafe fn package_b(row_size: usize, col_size: usize, step: usize, b_ptr: *const i32, buffer: &mut [i32]) {
        crate::memory::package_b(row_size, col_size, step, b_ptr, buffer, Self::MICRO_COL_SIZE);
    }

    unsafe fn invoke(k: usize, a: *const i32, b: *const i32, c: *mut i32, step: usize) {
        micro_kernel_i32_8x8(k, a, b, c, step);
    }
}

impl MicroKernel<i64> for Avx512KernelI64 {
    const MICRO_COL_SIZE: usize = 8;
    const MICRO_ROW_SIZE: usize = 8;

    unsafe fn package_a(row_size: usize, col_size: usize, step: usize, a_ptr: *const i64, buffer: &mut [i64]) {
        crate::memory::package_a(row_size, col_size, step, a_ptr, buffer, Self::MICRO_ROW_SIZE);
    }

    unsafe fn package_b(row_size: usize, col_size: usize, step: usize, b_ptr: *const i64, buffer: &mut [i64]) {
        crate::memory::package_b(row_size, col_size, step, b_ptr, buffer, Self::MICRO_COL_SIZE);
    }

    unsafe fn invoke(k: usize, a: *const i64, b: *const i64, c: *mut i64, step: usize) {
        micro_kernel_i64_8x8_avx512(k, a, b, c, step);
    }
}

impl MicroKernel<i64> for ScalarKernelI64 {
    const MICRO_COL_SIZE: usize = 4;
    const MICRO_ROW_SIZE: usize = 4;

    unsafe fn package_a(row_size: usize, col_size: usize, step: usize, a_ptr: *const i64, buffer: &mut [i64]) {
        crate::memory::package_a(row_size, col_size, step, a_ptr, buffer, Self::MICRO_ROW_SIZE);
    }

    unsafe fn package_b(row_size: usize, col_size: usize, step: usize, b_ptr: *const i64, buffer: &mut [i64]) {
        crate::memory::package_b(row_size, col_size, step, b_ptr, buffer, Self::MICRO_COL_SIZE);
    }

    unsafe fn invoke(k: usize, a: *const i64, b: *const i64, c: *mut i64, step: usize) {
        micro_kernel_i64_4x4_scalar(k, a, b, c, step);
    }
}


#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn micro_kernel_f32_8x8(
    k: usize,
    a_ptr: *const f32, // 8*256
    b_ptr: *const f32, // 256*8
    c_ptr: *mut f32,   // 8*8
    step: usize
){
    // f32是4B,所以将step从个数转为字节
    let step_bytes = step * std::mem::size_of::<f32>();
    unsafe {
        asm!(
            // 创建一个指针来记录c的偏移量
            "mov {temp_c}, {c}",
            // 读取数据
            "vmovups ymm0, ymmword ptr [{temp_c}]",
            "add {temp_c}, {step}",
            "vmovups ymm1, ymmword ptr [{temp_c}]",

            "add {temp_c}, {step}",
            "vmovups ymm2, ymmword ptr [{temp_c}]",
            
            "add {temp_c}, {step}",
            "vmovups ymm3, ymmword ptr [{temp_c}]",
            
            "add {temp_c}, {step}",
            "vmovups ymm4, ymmword ptr [{temp_c}]",
            
            "add {temp_c}, {step}",
            "vmovups ymm5, ymmword ptr [{temp_c}]",
            
            "add {temp_c}, {step}",
            "vmovups ymm6, ymmword ptr [{temp_c}]",
            
            "add {temp_c}, {step}",
            "vmovups ymm7, ymmword ptr [{temp_c}]",
            // 循环计算k次
            "2:",
            "test {k}, {k}",
            "jz 3f",

            "vmovaps ymm15, ymmword ptr [{b}]",

            "vbroadcastss ymm14, dword ptr [{a}]",
            "vfmadd231ps ymm0, ymm14, ymm15",

            "vbroadcastss ymm14, dword ptr [{a} + 4]",
            "vfmadd231ps ymm1, ymm14, ymm15",

            "vbroadcastss ymm14, dword ptr [{a} + 8]",
            "vfmadd231ps ymm2, ymm14, ymm15",

            "vbroadcastss ymm14, dword ptr [{a} + 12]",
            "vfmadd231ps ymm3, ymm14, ymm15",

            "vbroadcastss ymm14, dword ptr [{a} + 16]",
            "vfmadd231ps ymm4, ymm14, ymm15",

            "vbroadcastss ymm14, dword ptr [{a} + 20]",
            "vfmadd231ps ymm5, ymm14, ymm15",

            "vbroadcastss ymm14, dword ptr [{a} + 24]",
            "vfmadd231ps ymm6, ymm14, ymm15",

            "vbroadcastss ymm14, dword ptr [{a} + 28]",
            "vfmadd231ps ymm7, ymm14, ymm15",

            "add {a}, 32",
            "add {b}, 32",
            "dec {k}",
            "jmp 2b",

            "3:",
            "mov {temp_c}, {c}", // 再次把指针重置到左上角
            "vmovups ymmword ptr [{temp_c}], ymm0",
            
            "add {temp_c}, {step}",
            "vmovups ymmword ptr [{temp_c}], ymm1",
            
            "add {temp_c}, {step}",
            "vmovups ymmword ptr [{temp_c}], ymm2",
            
            "add {temp_c}, {step}",
            "vmovups ymmword ptr [{temp_c}], ymm3",
            
            "add {temp_c}, {step}",
            "vmovups ymmword ptr [{temp_c}], ymm4",
            
            "add {temp_c}, {step}",
            "vmovups ymmword ptr [{temp_c}], ymm5",
            
            "add {temp_c}, {step}",
            "vmovups ymmword ptr [{temp_c}], ymm6",
            
            "add {temp_c}, {step}",
            "vmovups ymmword ptr [{temp_c}], ymm7",

            k = inout(reg) k => _,
            a = inout(reg) a_ptr => _,
            b = inout(reg) b_ptr => _,
            c = in(reg) c_ptr,
            step = in(reg) step_bytes,
            temp_c = out(reg) _,

            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm14") _, out("ymm15") _,
        )
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn micro_kernel_i32_8x8(
    k: usize,
    a_ptr: *const i32, // 8*256
    b_ptr: *const i32, // 256*8
    c_ptr: *mut i32,   // 8*8
    step: usize
){
    let step_bytes = step * std::mem::size_of::<i32>(); // i32是4B,所以将step从个数转为字节
    unsafe {
        asm!(
            // 1. 读取 C 矩阵的数据 (使用整数优化vmovdqu指令)
            "mov {temp_c}, {c}",
            "vmovdqu ymm0, ymmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu ymm1, ymmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu ymm2, ymmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu ymm3, ymmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu ymm4, ymmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu ymm5, ymmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu ymm6, ymmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu ymm7, ymmword ptr [{temp_c}]",
            // 2. 循环计算 k 次
            "2:",
            "test {k}, {k}",
            "jz 3f",

            "vmovdqu ymm15, ymmword ptr [{b}]",

            "vpbroadcastd ymm14, dword ptr [{a}]",  
            "vpmulld ymm14, ymm14, ymm15",          
            "vpaddd ymm0, ymm0, ymm14",

            "vpbroadcastd ymm14, dword ptr [{a} + 4]",
            "vpmulld ymm14, ymm14, ymm15",
            "vpaddd ymm1, ymm1, ymm14",

            "vpbroadcastd ymm14, dword ptr [{a} + 8]",
            "vpmulld ymm14, ymm14, ymm15",
            "vpaddd ymm2, ymm2, ymm14",

            "vpbroadcastd ymm14, dword ptr [{a} + 12]",
            "vpmulld ymm14, ymm14, ymm15",
            "vpaddd ymm3, ymm3, ymm14",

            "vpbroadcastd ymm14, dword ptr [{a} + 16]",
            "vpmulld ymm14, ymm14, ymm15",
            "vpaddd ymm4, ymm4, ymm14",

            "vpbroadcastd ymm14, dword ptr [{a} + 20]",
            "vpmulld ymm14, ymm14, ymm15",
            "vpaddd ymm5, ymm5, ymm14",

            "vpbroadcastd ymm14, dword ptr [{a} + 24]",
            "vpmulld ymm14, ymm14, ymm15",
            "vpaddd ymm6, ymm6, ymm14",

            "vpbroadcastd ymm14, dword ptr [{a} + 28]",
            "vpmulld ymm14, ymm14, ymm15",
            "vpaddd ymm7, ymm7, ymm14",

            "add {a}, 32",
            "add {b}, 32",
            "dec {k}",
            "jmp 2b",

            "3:",
            // 写回C矩阵
            "mov {temp_c}, {c}",
            "vmovdqu ymmword ptr [{temp_c}], ymm0",

            "add {temp_c}, {ldc}",
            "vmovdqu ymmword ptr [{temp_c}], ymm1",

            "add {temp_c}, {ldc}",
            "vmovdqu ymmword ptr [{temp_c}], ymm2",

            "add {temp_c}, {ldc}",
            "vmovdqu ymmword ptr [{temp_c}], ymm3",

            "add {temp_c}, {ldc}",
            "vmovdqu ymmword ptr [{temp_c}], ymm4",

            "add {temp_c}, {ldc}",
            "vmovdqu ymmword ptr [{temp_c}], ymm5",

            "add {temp_c}, {ldc}",
            "vmovdqu ymmword ptr [{temp_c}], ymm6",

            "add {temp_c}, {ldc}",
            "vmovdqu ymmword ptr [{temp_c}], ymm7",

            k = inout(reg) k => _,
            a = inout(reg) a_ptr => _,
            b = inout(reg) b_ptr => _,
            c = in(reg) c_ptr,
            ldc = in(reg) step_bytes,
            temp_c = out(reg) _,
            
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
            out("ymm14") _, out("ymm15") _,
        );
    }
}

#[target_feature(enable = "avx512f,avx512dq")]
pub(crate) unsafe fn micro_kernel_i64_8x8_avx512(
    k: usize,
    a_ptr: *const i64, // 8*256
    b_ptr: *const i64, // 256*8
    c_ptr: *mut i64,   // 8*8
    step: usize
){
    let step_bytes = step * std::mem::size_of::<i64>(); // i64是8B,所以将step从个数转为8字节
    unsafe{
        asm!(
            "mov {temp_c}, {c}",
            "vmovdqu64 zmm0, zmmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmm1, zmmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmm2, zmmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmm3, zmmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmm4, zmmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmm5, zmmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmm6, zmmword ptr [{temp_c}]",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmm7, zmmword ptr [{temp_c}]",
            // 循环计算 k 次
            "2:",
            "test {k}, {k}",
            "jz 3f",

            "vmovdqu64 zmm15, zmmword ptr [{b}]",

            "vpbroadcastq zmm14, qword ptr [{a}]",
            "vpmullq zmm14, zmm14, zmm15",
            "vpaddq zmm0, zmm0, zmm14",

            "vpbroadcastq zmm14, qword ptr [{a} + 8]", // i64是8字节，所以偏移8字节
            "vpmullq zmm14, zmm14, zmm15",
            "vpaddq zmm1, zmm1, zmm14",

            "vpbroadcastq zmm14, qword ptr [{a} + 16]",
            "vpmullq zmm14, zmm14, zmm15",
            "vpaddq zmm2, zmm2, zmm14",

            "vpbroadcastq zmm14, qword ptr [{a} + 24]",
            "vpmullq zmm14, zmm14, zmm15",
            "vpaddq zmm3, zmm3, zmm14",

            "vpbroadcastq zmm14, qword ptr [{a} + 32]",
            "vpmullq zmm14, zmm14, zmm15",
            "vpaddq zmm4, zmm4, zmm14",

            "vpbroadcastq zmm14, qword ptr [{a} + 40]",
            "vpmullq zmm14, zmm14, zmm15",
            "vpaddq zmm5, zmm5, zmm14",

            "vpbroadcastq zmm14, qword ptr [{a} + 48]",
            "vpmullq zmm14, zmm14, zmm15",
            "vpaddq zmm6, zmm6, zmm14",

            "vpbroadcastq zmm14, qword ptr [{a} + 56]",
            "vpmullq zmm14, zmm14, zmm15",
            "vpaddq zmm7, zmm7, zmm14",

            "add {a}, 64",
            "add {b}, 64",
            "dec {k}",
            "jmp 2b",

            "3:",
            "mov {temp_c}, {c}",
            "vmovdqu64 zmmword ptr [{temp_c}], zmm0",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmmword ptr [{temp_c}], zmm1",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmmword ptr [{temp_c}], zmm2",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmmword ptr [{temp_c}], zmm3",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmmword ptr [{temp_c}], zmm4",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmmword ptr [{temp_c}], zmm5",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmmword ptr [{temp_c}], zmm6",
            "add {temp_c}, {ldc}",
            "vmovdqu64 zmmword ptr [{temp_c}], zmm7",

            k = inout(reg) k => _,
            a = inout(reg) a_ptr => _,
            b = inout(reg) b_ptr => _,
            c = in(reg) c_ptr,
            ldc = in(reg) step_bytes,
            temp_c = out(reg) _,
            
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            out("zmm14") _, out("zmm15") _,
        );
}
}

pub(crate) unsafe fn micro_kernel_i64_4x4_scalar(
    k: usize,
    a_ptr: *const i64,
    b_ptr: *const i64,
    c_ptr: *mut i64,
    step: usize,
) {
    // 1. 在栈上分配 16 个局部变量作为累加器 (LLVM 会把它们全部分配到寄存器里)
    let mut c_local = [0i64; 16];

    // 2. 加载 C 矩阵的半成品旧数据
    for i in 0..4 {
        for j in 0..4 {
            c_local[i * 4 + j] = *c_ptr.add(i * step + j);
        }
    }

    // 3. K 维度核心循环
    let mut a_idx = 0;
    let mut b_idx = 0;
    
    for _ in 0..k {
        // 一次性读取 4 个 A 和 4 个 B
        let a0 = *a_ptr.add(a_idx + 0);
        let a1 = *a_ptr.add(a_idx + 1);
        let a2 = *a_ptr.add(a_idx + 2);
        let a3 = *a_ptr.add(a_idx + 3);

        let b0 = *b_ptr.add(b_idx + 0);
        let b1 = *b_ptr.add(b_idx + 1);
        let b2 = *b_ptr.add(b_idx + 2);
        let b3 = *b_ptr.add(b_idx + 3);

        // 手动展开 4x4 的外积乘加 (完全消除 for 循环的分支预测开销)
        // 第 0 行
        c_local[0] += a0 * b0;
        c_local[1] += a0 * b1;
        c_local[2] += a0 * b2;
        c_local[3] += a0 * b3;

        // 第 1 行
        c_local[4] += a1 * b0;
        c_local[5] += a1 * b1;
        c_local[6] += a1 * b2;
        c_local[7] += a1 * b3;

        // 第 2 行
        c_local[8] += a2 * b0;
        c_local[9] += a2 * b1;
        c_local[10] += a2 * b2;
        c_local[11] += a2 * b3;

        // 第 3 行
        c_local[12] += a3 * b0;
        c_local[13] += a3 * b1;
        c_local[14] += a3 * b2;
        c_local[15] += a3 * b3;

        // 推进指针
        a_idx += 4;
        b_idx += 4;
    }

    // 4. 写回内存
    for i in 0..4 {
        for j in 0..4 {
            *c_ptr.add(i * step + j) = c_local[i * 4 + j];
        }
    }
}