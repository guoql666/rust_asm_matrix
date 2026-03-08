use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

pub struct AVXBuffer<T> {
    ptr : NonNull<T>,
    len : usize,
    layout : Layout
}


impl<T> AVXBuffer<T> {
    pub fn new(len: usize) -> Self {
        let size = len * std::mem::size_of::<T>();
        let align = 64; // AVX需求32字节对齐，AVX-512需求64字节对齐，这里统一使用64字节对齐以兼容AVX-512
        let layout = Layout::from_size_align(size, align).expect("Invalid size or alignment");
        let ptr = unsafe {
            let p =  alloc_zeroed(layout)  as *mut T;
            NonNull::new(p).expect("Failed to allocate memory")
        };
        Self { ptr, len, layout }
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[allow(dead_code)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for AVXBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}


pub(crate) unsafe fn package_a<T: Copy + Default>(
    row_size: usize, // 实际行数，行数不足8时补齐到8行，空余补0
    col_size: usize, // 实际列数, 列数无需补齐，有几列实际提取几列即可，剩下的通过AVXBuffer的alloc_zeroed分配来补齐
    step: usize,     // 实际矩阵的行跨度，即原始矩阵有多少列
    a_ptr: *const T, // 原始矩阵的首地址
    buffer: &mut [T],// 将256*256矩阵转为micro_row_size*256的矩阵，存到buffer中
    micro_row_size: usize
){
    let mut idx = 0;
    for col in 0..col_size {
        for row in 0..micro_row_size {
            if row < row_size {
                unsafe {
                    let value = *a_ptr.add(row * step + col);
                    buffer[idx] = value;
                }
            }
            else {
                buffer[idx] = T::default();
            }
            idx += 1;
        }
    }
}

pub(crate) unsafe fn package_b<T: Copy + Default>(
    row_size: usize, //
    col_size: usize,
    step: usize,     // 实际矩阵的行跨度，即原始矩阵有多少列
    b_ptr: *const T, // 原始矩阵的首地址
    buffer: &mut [T], // 将256*256矩阵转为256*micro_col_size的矩阵，存到buffer中
    micro_col_size: usize 
){
    let mut idx = 0;
    for row in 0..row_size {
        for col in 0..micro_col_size {
            if col < col_size {
                unsafe {
                    let value = *b_ptr.add(row * step + col);
                    buffer[idx] = value;
                }
            }
            else {
                buffer[idx] = T::default();
            }
            idx += 1;
        }
    }
}