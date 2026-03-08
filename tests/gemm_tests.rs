use rust_asm_matrix::solver::*;
use std::time::Instant;

// =============================================================================
// 辅助函数：朴素三重循环矩阵乘法（用于小矩阵对比）
// =============================================================================

fn naive_matmul_f32(
    m: usize, n: usize, k: usize,
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] = sum;
        }
    }
}

fn naive_matmul_i32(
    m: usize, n: usize, k: usize,
    a: &[i32], lda: usize,
    b: &[i32], ldb: usize,
    c: &mut [i32], ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for p in 0..k {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] = sum;
        }
    }
}

fn naive_matmul_i64(
    m: usize, n: usize, k: usize,
    a: &[i64], lda: usize,
    b: &[i64], ldb: usize,
    c: &mut [i64], ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for p in 0..k {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] = sum;
        }
    }
}

// =============================================================================
// 测试 gemm_f32
// =============================================================================

#[test]
fn test_gemm_f32_identity() {
    // 设定一个极其刁钻的尺寸：257，用来测试防爆箱逻辑！
    let m = 4097;
    let k = 4097;
    let n = 4097;

    // 步长 (Leading Dimension) 就是矩阵的物理列数
    let lda = k;
    let ldb = n;
    let ldc = n;

    // 在堆上分配一维数组，模拟行主序 (Row-major) 矩阵
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    // ==========================================
    // 1. 初始化矩阵 A (填入有规律的测试数据)
    // ==========================================
    for i in 0..m {
        for j in 0..k {
            // 放入一些容易比对的数字，比如 0.0, 1.0, 2.0...
            a[i * lda + j] = (i * k + j) as f32;
        }
    }

    // ==========================================
    // 2. 初始化矩阵 B 为单位矩阵 (Identity Matrix)
    // ==========================================
    for i in 0..k {
        for j in 0..n {
            if i == j {
                b[i * ldb + j] = 2.0; // 主对角线为 2.0
            } else {
                b[i * ldb + j] = 0.0; // 其他全部为 0.0
            }
        }
    }

    // ==========================================
    // 3. 引擎点火！执行我们手写的 GEMM
    // ==========================================
    unsafe {
        gemm_f32(
            m, n, k, 
            a.as_ptr(), lda, 
            b.as_ptr(), ldb, 
            c.as_mut_ptr(), ldc,
            false // 不启用多线程，专注于正确性测试
        );
    }

    // ==========================================
    // 4. 断言验证：C 必须与 A*2 完全一致！
    // ==========================================
    for i in 0..m {
        for j in 0..n {
            let expected = 2.0 * a[i * lda + j];
            let actual = c[i * ldc + j];
            
            // 浮点数比较的最佳实践：允许极小的 epsilon 误差
            assert!(
                (expected - actual).abs() < 1e-5,
                "🚨 发现错误！坐标 ({}, {}): 期望值 {}, 实际值 {}",
                i, j, expected, actual
            );
        }
    }

    println!("✅ 测试完美通过！A * 2I = 2A (矩阵大小: {}x{})", m, n);
    println!("宏观分块、微观内核、以及边缘防爆箱逻辑全部运转正常！");
}

#[test]
fn test_gemm_f32_small_matrix() {
    // 小矩阵测试：验证基本正确性
    let m = 4;
    let k = 4;
    let n = 4;

    let lda = k;
    let ldb = n;
    let ldc = n;

    let a = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];

    let b = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    let mut c = vec![0.0f32; m * n];

    unsafe {
        gemm_f32(
            m, n, k,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            c.as_mut_ptr(), ldc,
            false // 小矩阵不启用多线程
        );
    }

    // 验证结果应该等于 A (因为 B 是单位矩阵)
    for i in 0..m * n {
        assert!(
            (a[i] - c[i]).abs() < 1e-5,
            "不匹配在位置 {}: 期望 {}, 得到 {}",
            i, a[i], c[i]
        );
    }

    println!("✅ 小矩阵测试通过！");
}

#[test]
fn test_gemm_f32_irregular_size() {
    // 非规则大小测试：257x257x257
    let m = 257;
    let k = 257;
    let n = 257;

    let lda = k;
    let ldb = n;
    let ldc = n;

    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    // 初始化为简单的值
    for i in 0..m {
        for j in 0..k {
            a[i * lda + j] = ((i + j) % 10) as f32;
        }
    }

    // B 设为单位矩阵
    for i in 0..k.min(n) {
        b[i * ldb + i] = 1.0;
    }

    unsafe {
        gemm_f32(
            m, n, k,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            c.as_mut_ptr(), ldc,
            false // 不启用多线程，专注于正确性测试
        );
    }

    // 验证 C 应该等于 A
    for i in 0..m {
        for j in 0..n {
            let expected = a[i * lda + j];
            let actual = c[i * ldc + j];
            assert!(
                (expected - actual).abs() < 1e-4,
                "位置 ({}, {}) 不匹配: 期望 {}, 得到 {}",
                i, j, expected, actual
            );
        }
    }

    println!("✅ 非规则尺寸测试通过 ({}x{}x{})！", m, k, n);
}

// =============================================================================
// 测试 gemm_i32
// =============================================================================

#[test]
fn test_gemm_i32_identity() {
    let m = 512;
    let k = 512;
    let n = 512;

    let lda = k;
    let ldb = n;
    let ldc = n;

    let mut a = vec![0i32; m * k];
    let mut b = vec![0i32; k * n];
    let mut c = vec![0i32; m * n];

    // 初始化矩阵 A
    for i in 0..m {
        for j in 0..k {
            a[i * lda + j] = ((i + j) % 100) as i32;
        }
    }

    // 初始化矩阵 B 为单位矩阵
    for i in 0..k {
        for j in 0..n {
            if i == j {
                b[i * ldb + j] = 1;
            } else {
                b[i * ldb + j] = 0;
            }
        }
    }

    unsafe {
        gemm_i32(
            m, n, k,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            c.as_mut_ptr(), ldc,
            false // 不启用多线程，专注于正确性测试
        );
    }

    // 验证 C == A
    for i in 0..m {
        for j in 0..n {
            let expected = a[i * lda + j];
            let actual = c[i * ldc + j];
            assert_eq!(
                expected, actual,
                "i32 测试失败在位置 ({}, {}): 期望 {}, 得到 {}",
                i, j, expected, actual
            );
        }
    }

    println!("✅ gemm_i32 单位矩阵测试通过！({}x{})", m, n);
}

#[test]
fn test_gemm_i32_small_matrix() {
    let m = 3;
    let k = 3;
    let n = 3;

    let lda = k;
    let ldb = n;
    let ldc = n;

    let a = vec![
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    ];

    let b = vec![
        9, 8, 7,
        6, 5, 4,
        3, 2, 1,
    ];

    let mut c = vec![0i32; m * n];
    let mut c_naive = vec![0i32; m * n];

    // 用优化版本计算
    unsafe {
        gemm_i32(
            m, n, k,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            c.as_mut_ptr(), ldc,
            false // 小矩阵不启用多线程
        );
    }

    // 用朴素版本计算
    naive_matmul_i32(m, n, k, &a, lda, &b, ldb, &mut c_naive, ldc);

    // 对比结果
    for i in 0..m * n {
        assert_eq!(
            c[i], c_naive[i],
            "i32 小矩阵测试失败在位置 {}: 优化版 {}, 朴素版 {}",
            i, c[i], c_naive[i]
        );
    }

    println!("✅ gemm_i32 小矩阵测试通过！");
    println!("   结果矩阵:");
    for i in 0..m {
        print!("   ");
        for j in 0..n {
            print!("{:4} ", c[i * ldc + j]);
        }
        println!();
    }
}

// =============================================================================
// 测试 gemm_i64_scalar
// =============================================================================

#[test]
fn test_gemm_i64_scalar_correctness() {
    let m = 256;
    let k = 256;
    let n = 256;

    let lda = k;
    let ldb = n;
    let ldc = n;

    let mut a = vec![0i64; m * k];
    let mut b = vec![0i64; k * n];
    let mut c = vec![0i64; m * n];

    // 初始化数据
    for i in 0..m {
        for j in 0..k {
            a[i * lda + j] = ((i + j) % 50) as i64;
        }
    }

    for i in 0..k {
        for j in 0..n {
            if i == j {
                b[i * ldb + j] = 1;
            } else {
                b[i * ldb + j] = 0;
            }
        }
    }

    unsafe {
        gemm_i64_scalar(
            m, n, k,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            c.as_mut_ptr(), ldc,
            false // 不启用多线程，专注于正确性测试
        );
    }

    // 验证结果
    for i in 0..m {
        for j in 0..n {
            let expected = a[i * lda + j];
            let actual = c[i * ldc + j];
            assert_eq!(
                expected, actual,
                "i64_scalar 测试失败在位置 ({}, {})",
                i, j
            );
        }
    }

    println!("✅ gemm_i64_scalar 正确性测试通过！({}x{})", m, n);
}

// =============================================================================
// 测试 gemm_i64_avx512
// =============================================================================

#[test]
fn test_gemm_i64_avx512_correctness() {
    // 检查是否支持 AVX512
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("avx512dq") {
            println!("⚠️  CPU 不支持 AVX512，跳过 gemm_i64_avx512 测试");
            return;
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        println!("⚠️  非 x86 架构，跳过 gemm_i64_avx512 测试");
        return;
    }

    let m = 512;
    let k = 512;
    let n = 512;

    let lda = k;
    let ldb = n;
    let ldc = n;

    let mut a = vec![0i64; m * k];
    let mut b = vec![0i64; k * n];
    let mut c = vec![0i64; m * n];

    // 初始化数据
    for i in 0..m {
        for j in 0..k {
            a[i * lda + j] = ((i + j) % 100) as i64;
        }
    }

    for i in 0..k {
        for j in 0..n {
            if i == j {
                b[i * ldb + j] = 1;
            } else {
                b[i * ldb + j] = 0;
            }
        }
    }

    unsafe {
        gemm_i64_avx512(
            m, n, k,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            c.as_mut_ptr(), ldc,
            false // 不启用多线程，专注于正确性测试
        );
    }

    // 验证结果
    for i in 0..m {
        for j in 0..n {
            let expected = a[i * lda + j];
            let actual = c[i * ldc + j];
            assert_eq!(
                expected, actual,
                "i64_avx512 测试失败在位置 ({}, {})",
                i, j
            );
        }
    }

    println!("✅ gemm_i64_avx512 正确性测试通过！({}x{})", m, n);
}

// =============================================================================
// 性能对比测试：小矩阵
// =============================================================================

#[test]
fn bench_small_matrix_f32() {
    let m = 64;
    let k = 64;
    let n = 64;

    let lda = k;
    let ldb = n;
    let ldc = n;

    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    let mut c_naive = vec![0.0f32; m * n];
    let mut c_opt = vec![0.0f32; m * n];

    // 初始化随机数据
    for i in 0..m {
        for j in 0..k {
            a[i * lda + j] = ((i * k + j) % 100) as f32 * 0.1;
        }
    }
    for i in 0..k {
        for j in 0..n {
            b[i * ldb + j] = ((i * n + j) % 100) as f32 * 0.1;
        }
    }

    // 朴素实现计时
    let start_naive = Instant::now();
    naive_matmul_f32(m, n, k, &a, lda, &b, ldb, &mut c_naive, ldc);
    let time_naive = start_naive.elapsed();

    // 优化实现计时
    let start_opt = Instant::now();
    unsafe {
        gemm_f32(
            m, n, k,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            c_opt.as_mut_ptr(), ldc,
            false // 不启用多线程，专注于正确性测试
        );
    }
    let time_opt = start_opt.elapsed();

    // 验证结果一致性
    for i in 0..m * n {
        assert!(
            (c_naive[i] - c_opt[i]).abs() < 1e-3,
            "结果不一致在位置 {}: 朴素 {}, 优化 {}",
            i, c_naive[i], c_opt[i]
        );
    }

    println!("\n📊 f32 小矩阵性能对比 ({}x{}x{}):", m, k, n);
    println!("   朴素三重循环: {:?}", time_naive);
    println!("   优化 GEMM:    {:?}", time_opt);
    println!("   加速比:       {:.2}x", time_naive.as_secs_f64() / time_opt.as_secs_f64());
}

#[test]
fn bench_small_matrix_i32() {
    let m = 64;
    let k = 64;
    let n = 64;

    let lda = k;
    let ldb = n;
    let ldc = n;

    let mut a = vec![0i32; m * k];
    let mut b = vec![0i32; k * n];
    let mut c_naive = vec![0i32; m * n];
    let mut c_opt = vec![0i32; m * n];

    // 初始化数据
    for i in 0..m {
        for j in 0..k {
            a[i * lda + j] = ((i * k + j) % 100) as i32;
        }
    }
    for i in 0..k {
        for j in 0..n {
            b[i * ldb + j] = ((i * n + j) % 100) as i32;
        }
    }

    // 朴素实现计时
    let start_naive = Instant::now();
    naive_matmul_i32(m, n, k, &a, lda, &b, ldb, &mut c_naive, ldc);
    let time_naive = start_naive.elapsed();

    // 优化实现计时
    let start_opt = Instant::now();
    unsafe {
        gemm_i32(
            m, n, k,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            c_opt.as_mut_ptr(), ldc,
            false // 不启用多线程，专注于正确性测试
        );
    }
    let time_opt = start_opt.elapsed();

    // 验证结果一致性
    for i in 0..m * n {
        assert_eq!(
            c_naive[i], c_opt[i],
            "i32 结果不一致在位置 {}",
            i
        );
    }

    println!("\n📊 i32 小矩阵性能对比 ({}x{}x{}):", m, k, n);
    println!("   朴素三重循环: {:?}", time_naive);
    println!("   优化 GEMM:    {:?}", time_opt);
    println!("   加速比:       {:.2}x", time_naive.as_secs_f64() / time_opt.as_secs_f64());
}

#[test]
fn bench_small_matrix_i64() {
    let m = 64;
    let k = 64;
    let n = 64;

    let lda = k;
    let ldb = n;
    let ldc = n;

    let mut a = vec![0i64; m * k];
    let mut b = vec![0i64; k * n];
    let mut c_naive = vec![0i64; m * n];
    let mut c_scalar = vec![0i64; m * n];
    let mut c_avx512 = vec![0i64; m * n];

    // 初始化数据
    for i in 0..m {
        for j in 0..k {
            a[i * lda + j] = ((i * k + j) % 100) as i64;
        }
    }
    for i in 0..k {
        for j in 0..n {
            b[i * ldb + j] = ((i * n + j) % 100) as i64;
        }
    }

    // 朴素实现计时
    let start_naive = Instant::now();
    naive_matmul_i64(m, n, k, &a, lda, &b, ldb, &mut c_naive, ldc);
    let time_naive = start_naive.elapsed();

    // Scalar 实现计时
    let start_scalar = Instant::now();
    unsafe {
        gemm_i64_scalar(
            m, n, k,
            a.as_ptr(), lda,
            b.as_ptr(), ldb,
            c_scalar.as_mut_ptr(), ldc,
            false
        );
    }
    let time_scalar = start_scalar.elapsed();

    // 验证 scalar 结果
    for i in 0..m * n {
        assert_eq!(
            c_naive[i], c_scalar[i],
            "i64_scalar 结果不一致在位置 {}",
            i
        );
    }

    println!("\n📊 i64 小矩阵性能对比 ({}x{}x{}):", m, k, n);
    println!("   朴素三重循环:  {:?}", time_naive);
    println!("   Scalar GEMM:  {:?}", time_scalar);
    println!("   Scalar 加速比: {:.2}x", time_naive.as_secs_f64() / time_scalar.as_secs_f64());

    // AVX512 实现计时（如果支持）
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq") {
            let start_avx512 = Instant::now();
            unsafe {
                gemm_i64_avx512(
                    m, n, k,
                    a.as_ptr(), lda,
                    b.as_ptr(), ldb,
                    c_avx512.as_mut_ptr(), ldc,
                    false // 不启用多线程，专注于正确性测试
                );
            }
            let time_avx512 = start_avx512.elapsed();

            // 验证 AVX512 结果
            for i in 0..m * n {
                assert_eq!(
                    c_naive[i], c_avx512[i],
                    "i64_avx512 结果不一致在位置 {}",
                    i
                );
            }

            println!("   AVX512 GEMM:  {:?}", time_avx512);
            println!("   AVX512 加速比: {:.2}x", time_naive.as_secs_f64() / time_avx512.as_secs_f64());
            println!("   AVX512 vs Scalar: {:.2}x", time_scalar.as_secs_f64() / time_avx512.as_secs_f64());
        } else {
            println!("   ⚠️  CPU 不支持 AVX512，跳过 AVX512 性能测试");
        }
    }
}

// =============================================================================
// 中等矩阵性能对比测试
// =============================================================================

#[test]
fn bench_medium_matrix_comparison() {
    let m = 256;
    let k = 256;
    let n = 256;

    let lda = k;
    let ldb = n;
    let ldc = n;

    println!("\n🚀 中等矩阵性能对比测试 ({}x{}x{}):", m, k, n);

    // =============================================================================
    // f32 测试
    // =============================================================================
    {
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        for i in 0..m * k {
            a[i] = (i % 100) as f32 * 0.1;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as f32 * 0.1;
        }

        let start = Instant::now();
        unsafe {
            gemm_f32(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c.as_mut_ptr(), ldc,
                false // 不启用多线程，专注于正确性测试
            );
        }
        let time_f32 = start.elapsed();
        println!("   gemm_f32:     {:?}", time_f32);
    }

    // =============================================================================
    // i32 测试
    // =============================================================================
    {
        let mut a = vec![0i32; m * k];
        let mut b = vec![0i32; k * n];
        let mut c = vec![0i32; m * n];

        for i in 0..m * k {
            a[i] = (i % 100) as i32;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as i32;
        }

        let start = Instant::now();
        unsafe {
            gemm_i32(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c.as_mut_ptr(), ldc,
                false // 不启用多线程，专注于正确性测试
            );
        }
        let time_i32 = start.elapsed();
        println!("   gemm_i32:     {:?}", time_i32);
    }

    // =============================================================================
    // i64 scalar 测试
    // =============================================================================
    {
        let mut a = vec![0i64; m * k];
        let mut b = vec![0i64; k * n];
        let mut c = vec![0i64; m * n];

        for i in 0..m * k {
            a[i] = (i % 100) as i64;
        }
        for i in 0..k * n {
            b[i] = (i % 100) as i64;
        }

        let start = Instant::now();
        unsafe {
            gemm_i64_scalar(
                m, n, k,
                a.as_ptr(), lda,
                b.as_ptr(), ldb,
                c.as_mut_ptr(), ldc,
                false // 不启用多线程，专注于正确性测试
            );
        }
        let time_i64_scalar = start.elapsed();
        println!("   gemm_i64_scalar: {:?}", time_i64_scalar);

        // AVX512 测试
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq") {
                c.fill(0);
                let start = Instant::now();
                unsafe {
                    gemm_i64_avx512(
                        m, n, k,
                        a.as_ptr(), lda,
                        b.as_ptr(), ldb,
                        c.as_mut_ptr(), ldc,
                        false // 不启用多线程，专注于正确性测试
                    );
                }
                let time_i64_avx512 = start.elapsed();
                println!("   gemm_i64_avx512: {:?}", time_i64_avx512);
                println!("   AVX512 vs Scalar: {:.2}x", 
                    time_i64_scalar.as_secs_f64() / time_i64_avx512.as_secs_f64());
            } else {
                println!("   ⚠️  CPU 不支持 AVX512");
            }
        }
    }
}
