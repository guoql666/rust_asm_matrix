mod memory;
mod kernel;
mod gemm;
pub mod solver;

// 是否使用多线程的阈值，可以根据实际情况调整
const MULTITHREADING_THRESHOLD: u64 = 50_000_000; // 5千万次浮点运算
const MINIUM_SIZE_FOR_MULTITHREADING: usize = 32; // 最小矩阵维度，过小的矩阵不适合多线程

pub fn check_should_use_multithreading(m: usize, n: usize, k: usize) -> bool {
    let total_flops = m as u64 * n as u64 * k as u64;
    if total_flops < MULTITHREADING_THRESHOLD {
        return false;
    }
    // 3个围堵中任意有两个维度小于MINIUM_SIZE_FOR_MULTITHREADING都不适合多线程
    if m < MINIUM_SIZE_FOR_MULTITHREADING && n < MINIUM_SIZE_FOR_MULTITHREADING
    || n < MINIUM_SIZE_FOR_MULTITHREADING && k < MINIUM_SIZE_FOR_MULTITHREADING
    || k < MINIUM_SIZE_FOR_MULTITHREADING && m < MINIUM_SIZE_FOR_MULTITHREADING {
        return false;
    }
    // 极端长条矩阵 (如 k 极大，但 m 和 n 极小) 不适合多线程，因为线程间的调度和同步开销可能超过计算收益
    let memory_footprint = m * k + k * n + m * n;
    let arithmetic_intensity = total_flops as f64 / memory_footprint as f64;
    // 如果算术强度过低，说明内存访问较多，可能会成为瓶颈，不适合多线程
    // 每个数据元素平均只参与不到 10 次计算, 不适合多线程
    if arithmetic_intensity < 10.0 {
        return false;
    }
    true
}

pub enum Parallel {
    True,
    False,
    Auto,
}

impl Parallel {
    pub fn resolve(&self, m: usize, n: usize, k: usize) -> bool {
        match self {
            Parallel::True => true,
            Parallel::False => false,
            Parallel::Auto => check_should_use_multithreading(m, n, k),
        }
    }
}