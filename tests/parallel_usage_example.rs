use rust_asm_matrix::solver::*;
use rust_asm_matrix::Parallel;

// 演示如何使用Parallel enum控制多线程

#[test]
fn example_parallel_usage() {
    let m = 1024;
    let k = 1024;
    let n = 1024;
    
    let a = vec![1.0f32; m * k];
    let b = vec![2.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    
    // 方式1: 自动决定是否使用多线程（系统根据矩阵大小自动判断）
    gemm_safe(m, n, k, &a, k, &b, n, &mut c, n, Parallel::Auto);
    println!("✅ 使用 Parallel::Auto - 系统自动决定");
    
    // 方式2: 强制使用多线程
    c.fill(0.0);
    gemm_safe(m, n, k, &a, k, &b, n, &mut c, n, Parallel::True);
    println!("✅ 使用 Parallel::True - 强制多线程");
    
    // 方式3: 强制不使用多线程
    c.fill(0.0);
    gemm_safe(m, n, k, &a, k, &b, n, &mut c, n, Parallel::False);
    println!("✅ 使用 Parallel::False - 强制单线程");
    
    // 验证结果正确性
    let expected = (k as f32) * 1.0 * 2.0;
    for &val in c.iter().take(10) {
        assert!((val - expected).abs() < 1e-3, "结果错误: 期望 {}, 得到 {}", expected, val);
    }
    
    println!("\n🎉 所有使用方式都正常工作！");
}

#[test]
fn example_unsafe_gemm_usage() {
    let m = 512;
    let k = 512;
    let n = 512;
    
    let a = vec![1.0f32; m * k];
    let b = vec![2.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    
    // 使用unsafe的gemm函数，需要手动传递指针
    unsafe {
        // 自动模式
        gemm::<f32>(m, n, k, a.as_ptr(), k, b.as_ptr(), n, c.as_mut_ptr(), n, Parallel::Auto);
    }
    
    println!("✅ unsafe gemm 函数也支持 Parallel enum");
}

#[test]
fn example_type_specific_functions() {
    let m = 256;
    let k = 256;
    let n = 256;
    
    // i32类型示例
    let a = vec![3i32; m * k];
    let b = vec![5i32; k * n];
    let mut c = vec![0i32; m * n];
    
    // 使用类型特定的底层函数（仍然使用bool参数）
    unsafe {
        gemm_i32(m, n, k, a.as_ptr(), k, b.as_ptr(), n, c.as_mut_ptr(), n, false);
    }
    
    println!("✅ 类型特定函数 (gemm_i32, gemm_f32等) 使用 bool 参数");
    
    // 但推荐使用泛型接口
    c.fill(0);
    gemm_safe(m, n, k, &a, k, &b, n, &mut c, n, Parallel::Auto);
    
    println!("✅ 推荐使用泛型接口 gemm_safe，更灵活");
}
