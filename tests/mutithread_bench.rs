use rust_asm_matrix::solver::*;
use std::time::Instant;

// =============================================================================
// 多线程性能对比测试
// =============================================================================

#[test]
fn bench_parallel_f32_large() {
    println!("\n🚀 f32 大矩阵多线程性能测试");
    
    let sizes = vec![
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];
    
    for (m, k, n) in sizes {
        println!("\n  矩阵尺寸: {}x{}x{}", m, k, n);
        println!("  计算量: {:.2} GFLOPS", (2.0 * m as f64 * n as f64 * k as f64) / 1e9);
        
        let lda = k;
        let ldb = n;
        let ldc = n;
        
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        let mut c_single = vec![0.0f32; m * n];
        let mut c_parallel = vec![0.0f32; m * n];
        
        // 初始化数据
        for i in 0..m * k {
            a[i] = (i % 100) as f32 * 0.01;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as f32 * 0.01;
        }
        
        // 单线程测试
        let start = Instant::now();
        unsafe {
            gemm_f32(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c_single.as_mut_ptr(), ldc,
                false  // 不使用多线程
            );
        }
        let time_single = start.elapsed();
        let gflops_single = (2.0 * m as f64 * n as f64 * k as f64) / 1e9 / time_single.as_secs_f64();
        
        // 多线程测试
        let start = Instant::now();
        unsafe {
            gemm_f32(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c_parallel.as_mut_ptr(), ldc,
                true  // 使用多线程
            );
        }
        let time_parallel = start.elapsed();
        let gflops_parallel = (2.0 * m as f64 * n as f64 * k as f64) / 1e9 / time_parallel.as_secs_f64();
        
        // 验证结果一致性（采样检查）
        let sample_points = 100.min(m * n);
        for _ in 0..sample_points {
            let idx = (rand() * (m * n) as f64) as usize % (m * n);
            assert!(
                (c_single[idx] - c_parallel[idx]).abs() < 1e-3,
                "结果不一致在位置 {}: 单线程 {}, 多线程 {}",
                idx, c_single[idx], c_parallel[idx]
            );
        }
        
        println!("  ├─ 单线程: {:>8.3}s ({:>6.2} GFLOPS)", time_single.as_secs_f64(), gflops_single);
        println!("  ├─ 多线程: {:>8.3}s ({:>6.2} GFLOPS)", time_parallel.as_secs_f64(), gflops_parallel);
        println!("  └─ 加速比: {:.2}x", time_single.as_secs_f64() / time_parallel.as_secs_f64());
    }
}

// 简单的伪随机数生成器
fn rand() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos % 1000000) as f64 / 1000000.0
}

#[test]
fn bench_parallel_i32_large() {
    println!("\n🚀 i32 大矩阵多线程性能测试");
    
    let sizes = vec![
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];
    
    for (m, k, n) in sizes {
        println!("\n  矩阵尺寸: {}x{}x{}", m, k, n);
        
        let lda = k;
        let ldb = n;
        let ldc = n;
        
        let mut a = vec![0i32; m * k];
        let mut b = vec![0i32; k * n];
        let mut c_single = vec![0i32; m * n];
        let mut c_parallel = vec![0i32; m * n];
        
        // 初始化数据
        for i in 0..m * k {
            a[i] = (i % 100) as i32;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as i32;
        }
        
        // 单线程测试
        let start = Instant::now();
        unsafe {
            gemm_i32(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c_single.as_mut_ptr(), ldc,
                false
            );
        }
        let time_single = start.elapsed();
        
        // 多线程测试
        let start = Instant::now();
        unsafe {
            gemm_i32(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c_parallel.as_mut_ptr(), ldc,
                true
            );
        }
        let time_parallel = start.elapsed();
        
        // 验证结果一致性（采样检查）
        let sample_points = 100.min(m * n);
        for _ in 0..sample_points {
            let idx = (rand() * (m * n) as f64) as usize % (m * n);
            assert_eq!(
                c_single[idx], c_parallel[idx],
                "i32 结果不一致在位置 {}",
                idx
            );
        }
        
        println!("  ├─ 单线程: {:>8.3}s", time_single.as_secs_f64());
        println!("  ├─ 多线程: {:>8.3}s", time_parallel.as_secs_f64());
        println!("  └─ 加速比: {:.2}x", time_single.as_secs_f64() / time_parallel.as_secs_f64());
    }
}

#[test]
fn bench_parallel_i64_scalar_large() {
    println!("\n🚀 i64_scalar 大矩阵多线程性能测试");
    
    let sizes = vec![
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ];
    
    for (m, k, n) in sizes {
        println!("\n  矩阵尺寸: {}x{}x{}", m, k, n);
        
        let lda = k;
        let ldb = n;
        let ldc = n;
        
        let mut a = vec![0i64; m * k];
        let mut b = vec![0i64; k * n];
        let mut c_single = vec![0i64; m * n];
        let mut c_parallel = vec![0i64; m * n];
        
        // 初始化数据
        for i in 0..m * k {
            a[i] = (i % 100) as i64;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as i64;
        }
        
        // 单线程测试
        let start = Instant::now();
        unsafe {
            gemm_i64_scalar(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c_single.as_mut_ptr(), ldc,
                false
            );
        }
        let time_single = start.elapsed();
        
        // 多线程测试
        let start = Instant::now();
        unsafe {
            gemm_i64_scalar(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c_parallel.as_mut_ptr(), ldc,
                true
            );
        }
        let time_parallel = start.elapsed();
        
        // 验证结果
        let sample_points = 100.min(m * n);
        for _ in 0..sample_points {
            let idx = (rand() * (m * n) as f64) as usize % (m * n);
            assert_eq!(
                c_single[idx], c_parallel[idx],
                "i64_scalar 结果不一致在位置 {}",
                idx
            );
        }
        
        println!("  ├─ 单线程: {:>8.3}s", time_single.as_secs_f64());
        println!("  ├─ 多线程: {:>8.3}s", time_parallel.as_secs_f64());
        println!("  └─ 加速比: {:.2}x", time_single.as_secs_f64() / time_parallel.as_secs_f64());
    }
}

#[test]
fn bench_parallel_i64_avx512_large() {
    // 检查是否支持 AVX512
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("avx512dq") {
            println!("⚠️  CPU 不支持 AVX512，跳过测试");
            return;
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        println!("⚠️  非 x86 架构，跳过测试");
        return;
    }
    
    println!("\n🚀 i64_avx512 大矩阵多线程性能测试");
    
    let sizes = vec![
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];
    
    for (m, k, n) in sizes {
        println!("\n  矩阵尺寸: {}x{}x{}", m, k, n);
        
        let lda = k;
        let ldb = n;
        let ldc = n;
        
        let mut a = vec![0i64; m * k];
        let mut b = vec![0i64; k * n];
        let mut c_single = vec![0i64; m * n];
        let mut c_parallel = vec![0i64; m * n];
        
        // 初始化数据
        for i in 0..m * k {
            a[i] = (i % 100) as i64;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as i64;
        }
        
        // 单线程测试
        let start = Instant::now();
        unsafe {
            gemm_i64_avx512(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c_single.as_mut_ptr(), ldc,
                false
            );
        }
        let time_single = start.elapsed();
        
        // 多线程测试
        let start = Instant::now();
        unsafe {
            gemm_i64_avx512(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c_parallel.as_mut_ptr(), ldc,
                true
            );
        }
        let time_parallel = start.elapsed();
        
        // 验证结果
        let sample_points = 100.min(m * n);
        for _ in 0..sample_points {
            let idx = (rand() * (m * n) as f64) as usize % (m * n);
            assert_eq!(
                c_single[idx], c_parallel[idx],
                "i64_avx512 结果不一致在位置 {}",
                idx
            );
        }
        
        println!("  ├─ 单线程: {:>8.3}s", time_single.as_secs_f64());
        println!("  ├─ 多线程: {:>8.3}s", time_parallel.as_secs_f64());
        println!("  └─ 加速比: {:.2}x", time_single.as_secs_f64() / time_parallel.as_secs_f64());
    }
}

#[test]
fn bench_parallel_comprehensive() {
    println!("\n🎯 综合多线程性能对比测试");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let m = 4096;
    let k = 4096;
    let n = 4096;
    
    println!("\n矩阵尺寸: {}x{}x{}", m, k, n);
    println!("计算量: {:.2} GFLOPS\n", (2.0 * m as f64 * n as f64 * k as f64) / 1e9);
    
    // f32 测试
    {
        println!("📌 f32 类型:");
        let lda = k;
        let ldb = n;
        let ldc = n;
        
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        let mut c = vec![0.0f32; m * n];
        
        for i in 0..m * k {
            a[i] = (i % 100) as f32 * 0.01;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as f32 * 0.01;
        }
        
        // 单线程
        c.fill(0.0);
        let start = Instant::now();
        unsafe {
            gemm_f32(m, n, k, a.as_ptr(), lda, b.as_ptr(), ldb, c.as_mut_ptr(), ldc, false);
        }
        let time_single = start.elapsed();
        let gflops_single = (2.0 * m as f64 * n as f64 * k as f64) / 1e9 / time_single.as_secs_f64();
        
        // 多线程
        c.fill(0.0);
        let start = Instant::now();
        unsafe {
            gemm_f32(m, n, k, a.as_ptr(), lda, b.as_ptr(), ldb, c.as_mut_ptr(), ldc, true);
        }
        let time_parallel = start.elapsed();
        let gflops_parallel = (2.0 * m as f64 * n as f64 * k as f64) / 1e9 / time_parallel.as_secs_f64();
        
        println!("  单线程: {:.3}s ({:.2} GFLOPS)", time_single.as_secs_f64(), gflops_single);
        println!("  多线程: {:.3}s ({:.2} GFLOPS)", time_parallel.as_secs_f64(), gflops_parallel);
        println!("  加速比: {:.2}x\n", time_single.as_secs_f64() / time_parallel.as_secs_f64());
    }
    
    // i32 测试
    {
        println!("📌 i32 类型:");
        let lda = k;
        let ldb = n;
        let ldc = n;
        
        let mut a = vec![0i32; m * k];
        let mut b = vec![0i32; k * n];
        let mut c = vec![0i32; m * n];
        
        for i in 0..m * k {
            a[i] = (i % 100) as i32;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as i32;
        }
        
        // 单线程
        c.fill(0);
        let start = Instant::now();
        unsafe {
            gemm_i32(m, n, k, a.as_ptr(), lda, b.as_ptr(), ldb, c.as_mut_ptr(), ldc, false);
        }
        let time_single = start.elapsed();
        
        // 多线程
        c.fill(0);
        let start = Instant::now();
        unsafe {
            gemm_i32(m, n, k, a.as_ptr(), lda, b.as_ptr(), ldb, c.as_mut_ptr(), ldc, true);
        }
        let time_parallel = start.elapsed();
        
        println!("  单线程: {:.3}s", time_single.as_secs_f64());
        println!("  多线程: {:.3}s", time_parallel.as_secs_f64());
        println!("  加速比: {:.2}x\n", time_single.as_secs_f64() / time_parallel.as_secs_f64());
    }
    
    // i64 测试
    {
        println!("📌 i64 类型 (Scalar):");
        let lda = k;
        let ldb = n;
        let ldc = n;
        
        let mut a = vec![0i64; m * k];
        let mut b = vec![0i64; k * n];
        let mut c = vec![0i64; m * n];
        
        for i in 0..m * k {
            a[i] = (i % 100) as i64;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as i64;
        }
        
        // 单线程
        c.fill(0);
        let start = Instant::now();
        unsafe {
            gemm_i64_scalar(m, n, k, a.as_ptr(), lda, b.as_ptr(), ldb, c.as_mut_ptr(), ldc, false);
        }
        let time_single = start.elapsed();
        
        // 多线程
        c.fill(0);
        let start = Instant::now();
        unsafe {
            gemm_i64_scalar(m, n, k, a.as_ptr(), lda, b.as_ptr(), ldb, c.as_mut_ptr(), ldc, true);
        }
        let time_parallel = start.elapsed();
        
        println!("  单线程: {:.3}s", time_single.as_secs_f64());
        println!("  多线程: {:.3}s", time_parallel.as_secs_f64());
        println!("  加速比: {:.2}x\n", time_single.as_secs_f64() / time_parallel.as_secs_f64());
    }
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
