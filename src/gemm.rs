use crate::memory::{AVXBuffer};
use rayon::prelude::*;

// const MICRO_ROW_SIZE: usize = 8; 
// const MICRO_COL_SIZE: usize = 8;
const M_SIZE : usize = 256;
const N_SIZE : usize = 256;
const K_SIZE : usize = 256; // 送入asm处理的矩阵大小

pub trait MicroKernel<T> {
    const MICRO_ROW_SIZE: usize;
    const MICRO_COL_SIZE: usize;

    unsafe fn package_a(row_size: usize, col_size: usize, step: usize, a_ptr: *const T, buffer: &mut [T]);
    unsafe fn package_b(row_size: usize, col_size: usize, step: usize, b_ptr: *const T, buffer: &mut [T]);
    unsafe fn invoke(k: usize, a: *const T, b: *const T, c: *mut T, step: usize);
}


#[derive(Copy, Clone)]
struct SyncMutPtr<T>(*mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}

impl<T> SyncMutPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T { self.0 }
}

#[derive(Copy, Clone)]
struct SyncConstPtr<T>(*const T);
unsafe impl<T> Send for SyncConstPtr<T> {}
unsafe impl<T> Sync for SyncConstPtr<T> {}

impl<T> SyncConstPtr<T> {
    #[inline(always)]
    fn get(self) -> *const T { self.0 }
}

pub(crate) unsafe fn gemm_macro_generic<T, K>(
    m: usize, n: usize, k: usize, // 输入进来的是一个m*k的矩阵A和一个k*n的矩阵B，输出一个m*n的矩阵C
    a_ptr: *const T, step_a: usize, // A矩阵的行跨度，即原始矩阵有多少列
    b_ptr: *const T, step_b: usize, // B矩阵的行跨度，即原始矩阵有多少列
    c_ptr: *mut T, step_c: usize,  // C矩阵的行跨度，即原始矩阵有多少列
    parallel: bool
)
where T: Default + Copy + Send + Sync,
 K: MicroKernel<T>
{
    let safe_a = SyncConstPtr(a_ptr);
    let safe_b = SyncConstPtr(b_ptr);
    let safe_c = SyncMutPtr(c_ptr);
    //多线程优化版本
    // 判断是先切割M还是先切割N，原则上是先切割较大的维度，这样可以更好地利用缓存
    // 由于N具有较好的内存局部性，所以我们优先切割N维度，只有M远大于N时才切割M维度，这样可以更好地利用缓存并减少线程间的同步开销
    // 所以定义远大于是一个经验值，这里我们暂定为2倍，如果M大于N的2倍，且开启多线程，我们就优先切割M维度，否则优先切割N维度
    if m < 2 * n || !parallel {
        let process_n_block = move |n_position: usize| {
            let current_n_block_size = std::cmp::min(N_SIZE, n - n_position);
            // 解包
            let a_ptr = safe_a.get(); let b_ptr = safe_b.get(); let c_ptr = safe_c.get();

            let mut packed_a = AVXBuffer::<T>::new(M_SIZE * K_SIZE);
            let mut packed_b = AVXBuffer::<T>::new(K_SIZE * N_SIZE);
            // 分割B矩阵K行，A矩阵K列
            for k_position in (0..k).step_by(K_SIZE) {
                let current_k_block_size = std::cmp::min(K_SIZE, k - k_position);
                // 将分割的B矩阵打包到packed_b中，打包成256*8的矩阵
                for col_micro_position in (0..current_n_block_size).step_by(K::MICRO_COL_SIZE){
                    let actual_cols = std::cmp::min(K::MICRO_COL_SIZE, current_n_block_size - col_micro_position);
                    unsafe {
                        // 从原矩阵中提取，将256*256矩阵转化为按行分割的，然后头尾相连存入内存
                        let b_matrix_src = b_ptr.add(k_position * step_b + (n_position + col_micro_position));
                        let b_buffer_dst = &mut packed_b.as_mut_slice()[col_micro_position * K_SIZE..];
                    
                        K::package_b(current_k_block_size, actual_cols, step_b, b_matrix_src, b_buffer_dst);
                    }
                }

                for m_position in (0..m).step_by(M_SIZE) {
                    // 分割A矩阵M行
                    let current_m_block_size = std::cmp::min(M_SIZE, m - m_position);
                    for row_micro_position in (0..current_m_block_size).step_by(K::MICRO_ROW_SIZE){
                        let actual_rows = std::cmp::min(K::MICRO_ROW_SIZE, current_m_block_size - row_micro_position);
                        unsafe {
                            // 从原矩阵中提取，将256*256矩阵转化为按列分割的，然后头尾相连存入内存
                            let a_matrix_src = a_ptr.add((m_position + row_micro_position) * step_a + k_position);
                            let a_buffer_dst = &mut packed_a.as_mut_slice()[row_micro_position * K_SIZE..];
                        
                            K::package_a(actual_rows, current_k_block_size, step_a, a_matrix_src, a_buffer_dst);
                        }
                    }
                    // 至此，我们把AB两个矩阵都先分割成了256*256的矩阵，然后把A变形成了多个8*256的矩阵，然后头尾相连拼接到a_buffer中，把B变形成了多个256*8的矩阵
                    // 然后头尾相连拼接到b_buffer中，接下来就可以调用asm进行计算
                    for col_micro_start in (0..current_n_block_size).step_by(K::MICRO_COL_SIZE){
                        let actual_cols = std::cmp::min(K::MICRO_COL_SIZE, current_n_block_size - col_micro_start);
                        for row_micro_start in (0..current_m_block_size).step_by(K::MICRO_ROW_SIZE){
                            let actual_rows = std::cmp::min(K::MICRO_ROW_SIZE, current_m_block_size - row_micro_start);
                            let (c_matrix_dst, a_matrix_dst, b_matrix_dst) = unsafe {
                                (
                                    // 计算C矩阵的地址
                                    c_ptr.add((m_position + row_micro_start) * step_c + (n_position + col_micro_start)),
                                    // 计算A'和B'矩阵的地址
                                    packed_a.as_ptr().add(row_micro_start * K_SIZE),
                                    packed_b.as_ptr().add(col_micro_start * K_SIZE)
                                )
                            };
                            // 边界检查，若传入的不是8*8，有可能造成C的越界，因此需要检查是否为完整区块
                            if actual_rows == K::MICRO_ROW_SIZE && actual_cols == K::MICRO_COL_SIZE {
                                unsafe {
                                    K::invoke(current_k_block_size, a_matrix_dst, b_matrix_dst, c_matrix_dst, step_c);
                                }
                            }
                            else{
                                let mut temp_c = vec![T::default(); K::MICRO_ROW_SIZE * K::MICRO_COL_SIZE];
                                unsafe {
                                    for i in 0..actual_rows {
                                        for j in 0..actual_cols {
                                            temp_c[i * K::MICRO_COL_SIZE + j] = *c_matrix_dst.add(i * step_c + j);
                                        }
                                    }
                                    // 计算得到的结果先存到temp_c中，再将temp_c中的结果写回到C矩阵中
                                    K::invoke(current_k_block_size, a_matrix_dst, b_matrix_dst, temp_c.as_mut_ptr(), K::MICRO_COL_SIZE);
                                    for i in 0..actual_rows {
                                        for j in 0..actual_cols {
                                            *c_matrix_dst.add(i * step_c + j) = temp_c[i * K::MICRO_COL_SIZE + j];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        if parallel {
            (0..n).into_par_iter().step_by(N_SIZE).for_each(process_n_block);
        }else {
            (0..n).step_by(N_SIZE).for_each(process_n_block);
        }
    }
    // 只有开启多线程且M远大于N的2倍时才切割M维度，否则优先切割N维度
    else {
        let process_m_block = move |m_position: usize| {
            // 先分割M维度
            let current_m_block_size = std::cmp::min(M_SIZE, m - m_position);
            // 解包
            let a_ptr = safe_a.get(); let b_ptr = safe_b.get(); let c_ptr = safe_c.get();

            let mut packed_a = AVXBuffer::<T>::new(M_SIZE * K_SIZE);
            let mut packed_b = AVXBuffer::<T>::new(K_SIZE * N_SIZE);
            // 分割B矩阵K行，A矩阵K列
            for n_position in (0..n).step_by(N_SIZE) {
                let current_n_block_size = std::cmp::min(N_SIZE, n - n_position);
                for k_position in (0..k).step_by(K_SIZE) {
                    let current_k_block_size = std::cmp::min(K_SIZE, k - k_position);
                    for col_micro_position in (0..current_n_block_size).step_by(K::MICRO_COL_SIZE){
                        let actual_cols = std::cmp::min(K::MICRO_COL_SIZE, current_n_block_size - col_micro_position);
                        unsafe {
                            // 从原矩阵中提取，将256*256矩阵转化为按行分割的，然后头尾相连存入内存
                            let b_matrix_src = b_ptr.add(k_position * step_b + (n_position + col_micro_position));
                            let b_buffer_dst = &mut packed_b.as_mut_slice()[col_micro_position * K_SIZE..];
                            // 将分割的B矩阵打包到packed_b中，打包成256*8的矩阵
                            K::package_b(current_k_block_size, actual_cols, step_b, b_matrix_src, b_buffer_dst);
                        }
                    }
                    // 分割A矩阵
                    for row_micro_position in (0..current_m_block_size).step_by(K::MICRO_ROW_SIZE){
                        let actual_rows = std::cmp::min(K::MICRO_ROW_SIZE, current_m_block_size - row_micro_position);
                        unsafe {
                            // 从原矩阵中提取，将256*256矩阵转化为按列分割的，然后头尾相连存入内存
                            let a_matrix_src = a_ptr.add((m_position + row_micro_position) * step_a + k_position);
                            let a_buffer_dst = &mut packed_a.as_mut_slice()[row_micro_position * K_SIZE..];
                        
                            K::package_a(actual_rows, current_k_block_size, step_a, a_matrix_src, a_buffer_dst);
                        }
                    }
                    // 至此，我们把AB两个矩阵都先分割成了256*256的矩阵，然后把A变形成了多个8*256的矩阵，然后头尾相连拼接到a_buffer中，把B变形成了多个256*8的矩阵
                    // 然后头尾相连拼接到b_buffer中，接下来就可以调用asm进行计算
                    for col_micro_start in (0..current_n_block_size).step_by(K::MICRO_COL_SIZE){
                        let actual_cols = std::cmp::min(K::MICRO_COL_SIZE, current_n_block_size - col_micro_start);
                        for row_micro_start in (0..current_m_block_size).step_by(K::MICRO_ROW_SIZE){
                            let actual_rows = std::cmp::min(K::MICRO_ROW_SIZE, current_m_block_size - row_micro_start);
                            let (c_matrix_dst, a_matrix_dst, b_matrix_dst) = unsafe {
                                (
                                    // 计算C矩阵的地址
                                    c_ptr.add((m_position + row_micro_start) * step_c + (n_position + col_micro_start)),
                                    // 计算A'和B'矩阵的地址
                                    packed_a.as_ptr().add(row_micro_start * K_SIZE),
                                    packed_b.as_ptr().add(col_micro_start * K_SIZE)
                                )
                            };
                            // 边界检查，若传入的不是8*8，有可能造成C的越界，因此需要检查是否为完整区块
                            if actual_rows == K::MICRO_ROW_SIZE && actual_cols == K::MICRO_COL_SIZE {
                                unsafe {
                                    K::invoke(current_k_block_size, a_matrix_dst, b_matrix_dst, c_matrix_dst, step_c);
                                }
                            }
                            else{
                                let mut temp_c = vec![T::default(); K::MICRO_ROW_SIZE * K::MICRO_COL_SIZE];
                                unsafe {
                                    for i in 0..actual_rows {
                                        for j in 0..actual_cols {
                                            temp_c[i * K::MICRO_COL_SIZE + j] = *c_matrix_dst.add(i * step_c + j);
                                        }
                                    }
                                    // 计算得到的结果先存到temp_c中，再将temp_c中的结果写回到C矩阵中
                                    K::invoke(current_k_block_size, a_matrix_dst, b_matrix_dst, temp_c.as_mut_ptr(), K::MICRO_COL_SIZE);
                                    for i in 0..actual_rows {
                                        for j in 0..actual_cols {
                                            *c_matrix_dst.add(i * step_c + j) = temp_c[i * K::MICRO_COL_SIZE + j];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        if parallel {
            (0..m).into_par_iter().step_by(M_SIZE).for_each(process_m_block);
        }else {
            (0..m).step_by(M_SIZE).for_each(process_m_block);
        }
    }
}

use std::ops::{AddAssign, Mul};

pub(crate) unsafe fn naive_matmul<T>(
    m: usize, n: usize, k: usize,
    a_ptr: *const T, step_a: usize,
    b_ptr: *const T, step_b: usize,
    c_ptr: *mut T, step_c: usize,
    _parallel: bool
)
where T: Copy + Default + AddAssign + Mul<Output = T>
{
    for i in 0..m {
        for p in 0..k {
            let a_ip = unsafe { *a_ptr.add(i * step_a + p) };
            for j in 0..n {
                *c_ptr.add(i * step_c + j) += a_ip * unsafe { *b_ptr.add(p * step_b + j) };
            }
        }
    }
}