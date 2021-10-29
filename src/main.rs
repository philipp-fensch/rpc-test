extern crate rpc_lib;
use rpc_lib::include_rpcl;
use std::mem::size_of;
use std::str::FromStr;
use std::fs;
use std::convert::TryFrom;

use std::time::*;

#[include_rpcl("rpc_cuda.x")]
struct RPCConnection;

fn read_matrix_from_file(file_name: &str) -> Vec<f64> {
    let vector = fs::read_to_string(file_name).unwrap();
    let splitted = vector.split('\n');
    let mut vec = Vec::new();
    for num_line in splitted {
        if num_line.len() > 0 {
            for num_str in num_line.split(' ') {
                if num_str.len() > 0 {
                    let num = f64::from_str(num_str).unwrap();
                    vec.push(num);
                }
            }
        }
    }
    vec
}

fn main() {
    let rpc_connection = RPCConnection::new("137.226.133.199");

    const ITERATIONS: u32 = 10;

    // memcpy_test(&rpc_connection, ITERATIONS);
    // latency_test(&rpc_connection, ITERATIONS);
    // linear_solver_test(&rpc_connection, ITERATIONS);
    mat_mul_test(&rpc_connection, ITERATIONS);
}

fn memcpy_test(rpc_connection: &RPCConnection, iterations: u32) {
    const N_BYTES: u64 = 2 << 29; // 512 MiB
    
    let mut memory_host = Vec::with_capacity(N_BYTES as usize);
    unsafe { memory_host.set_len(N_BYTES as usize); }

    let memory_dev = rpc_connection.cuda_malloc(N_BYTES).unwrap();

    let mut duration = Duration::new(0, 0);

    for _i in 0..iterations {
        let memory_host_copy = memory_host.clone();
        let begin = Instant::now();
        rpc_connection.cuda_memcpy_htod(memory_dev, memory_host_copy, N_BYTES);
        let end = Instant::now();
        duration += end - begin;
        println!("cudaMemcpy: {:?}", end - begin);
    }

    println!("cudaMemcpy Average: {:?}", duration / iterations);
    rpc_connection.cuda_free(memory_dev);
}

fn latency_test(rpc_connection: &RPCConnection, iterations: u32) {

    let mut duration = Duration::new(0, 0);

    for _i in 0..iterations {
        let begin = Instant::now();
        rpc_connection.cuda_get_device_count();
        let end = Instant::now();
        duration += end - begin;
        println!("cudaGetDeviceCount: {:?}", end - begin);
    }

    println!("cudaGetDeviceCount Average: {:?}", duration / iterations);
}

fn convert_f64_to_u8(vec: &Vec<f64>) -> Vec<u8> {
    let dim = vec.len();
    let mut matrix_host_vec: Vec<u8> = Vec::with_capacity(dim * size_of::<f64>());
    for f in vec {
        for byte in f.to_ne_bytes() {
            matrix_host_vec.push(byte);
        }
    }
    matrix_host_vec
}

fn convert_u8_to_f64(vec: &Vec<u8>) -> Vec<f64> {
    let dim = vec.len();
    let mut matrix_host_vec: Vec<f64> = Vec::with_capacity(dim / size_of::<f64>());
    let bytes = vec.as_slice();
    for i in 0..dim {
        let x = <&[u8; 8]>::try_from(&bytes[i * 8..i * 8 + 8]).unwrap();
        matrix_host_vec.push(f64::from_be_bytes(*x));
    }
    matrix_host_vec
}

fn linear_solver_test(rpc_connection: &RPCConnection, iterations: u32) {
    let right_side_host = read_matrix_from_file("vector.txt");
    let matrix_host = read_matrix_from_file("matrix.txt");
    const DIM: usize = 5000;

    // Cast f64 to u8 for cudamemcpy
    let matrix_host_vec = convert_f64_to_u8(&matrix_host);
    let matrix_host_cast = matrix_host_vec.as_slice();

    // Init Connection and CUDA
    let solver = rpc_connection.rpc_cusolverdncreate().unwrap();

    // Allocate Memory
    let (vector_device, matrix_device, piv_seq_device, err, workspace) = allocate_memory(&rpc_connection, solver, DIM);

    // Copy Matrix to Device
    rpc_connection.cuda_memcpy_htod(matrix_device, matrix_host_cast.to_vec(), matrix_host_cast.len() as u64);

    // LU
    lu_factorization(&rpc_connection, solver, matrix_device, workspace, piv_seq_device, err, DIM);
    rpc_connection.cuda_device_synchronize();

    // Solve System
    // Copy rhs to Device
    let right_side_host_vec = convert_f64_to_u8(&right_side_host);
    let right_side_host_cast = right_side_host_vec.as_slice();
    rpc_connection.cuda_memcpy_htod(vector_device, right_side_host_cast.to_vec(), right_side_host_cast.len() as u64);

    let mut duration = Duration::new(0, 0);
    for _i in 0..iterations {
        // Solve
        let begin = Instant::now();
        let res = rpc_connection.rpc_cusolverdndgetrs(
            solver,
            1, // CUBLAS_OP_T
            DIM as i32,
            1, //#right-hand-sides
            matrix_device,
            DIM as i32,
            piv_seq_device,
            vector_device,
            DIM as i32,
            err
        );
        assert!(res == 0, "Solving System failed (cusolverDnDgetrs)");
        rpc_connection.cuda_device_synchronize();
        let end = Instant::now();
        duration += end - begin;
        println!("linear solver: {:?}", end - begin);
    }
    println!("linear solver Average: {:?}", duration / iterations);

    // Copy left-hand-side back
    // let res = rpc_connection.cuda_memcpy_dtoh(vector_device, (size_of::<f64>() * DIM) as u64);
    // let res2 = res.as_ref().unwrap();

    // Cast mem_result from generic u8 to the actual f64
    // let solution_vec = convert_u8_to_f64(&res2);
    // let solution = solution_vec.as_slice();
    
    // Free Memory
    rpc_connection.cuda_free(vector_device);
    rpc_connection.cuda_free(matrix_device);
    rpc_connection.cuda_free(piv_seq_device);
    rpc_connection.cuda_free(err);
    rpc_connection.cuda_free(workspace);

    assert!(rpc_connection.rpc_cusolverdndestroy(solver) == 0, "rpc_cusolverdndestroy failed");
}

fn allocate_memory(rpc_connection: &RPCConnection, solver: u64, dim: usize) -> (u64, u64, u64, u64, u64) {
    // Vector
    let rhs_vector = rpc_connection.cuda_malloc((dim * size_of::<f64>()) as u64).unwrap();
    // Matrix
    let mat = rpc_connection.cuda_malloc((dim * dim * size_of::<f64>()) as u64).unwrap();
    // Pivot
    let piv = rpc_connection.cuda_malloc((dim * size_of::<f64>()) as u64).unwrap();
    // Error-Code
    let err = rpc_connection.cuda_malloc(size_of::<i32>() as u64).unwrap();

    let workspace_size = rpc_connection.rpc_cusolverdndgetrf_buffersize(
        solver,
        dim as i32,
        dim as i32,
        mat,
        dim as i32
    ).unwrap();
    // assert!(workspace_size == 0, "cusolverdndgetrf_buffersize failed");

    let workspace = rpc_connection.cuda_malloc(workspace_size as u64).unwrap();

    (rhs_vector, mat, piv, err, workspace)
}

fn lu_factorization(rpc_connection: &RPCConnection, solver: u64, matrix: u64, workspace: u64, piv: u64, err: u64, dim: usize) {
    let res = rpc_connection.rpc_cusolverdndgetrf(
        solver,
        dim as i32,
        dim as i32,
        matrix,
        dim as i32,
        workspace,
        piv,
        err
    );
    assert!(res == 0, "LU-Factorization failed (cusolverDnDgetrf)");
}

fn mat_mul_test(rpc_connection: &RPCConnection, iterations: u32) {
    let matrix_host = read_matrix_from_file("matrix.txt");
    const DIM: usize = 5000;

    let device_mat_a = rpc_connection.cuda_malloc((DIM * DIM * size_of::<f64>()) as u64).unwrap();
    let device_mat_b = rpc_connection.cuda_malloc((DIM * DIM * size_of::<f64>()) as u64).unwrap();

    rpc_connection.cuda_device_synchronize();
    // Cast f64 to u8 for cudamemcpy
    let matrix_host_vec = convert_f64_to_u8(&matrix_host);
    let len = matrix_host_vec.len();
    rpc_connection.cuda_memcpy_htod(device_mat_a, matrix_host_vec, len as u64);

    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;

    let handle = rpc_connection.rpc_cublascreate().unwrap();

    // "Warming up"
    rpc_connection.rpc_cublasdgemm(handle, 0, 0, DIM as i32, DIM as i32, DIM as i32, alpha,
        device_mat_a, DIM as i32,
        device_mat_a, DIM as i32, beta,
        device_mat_b, DIM as i32);
    rpc_connection.cuda_device_synchronize();

    let mut duration = Duration::new(0, 0);
    for _i in 0..iterations {
        // Solve
        let begin = Instant::now();
        rpc_connection.rpc_cublasdgemm(handle, 0, 0, DIM as i32, DIM as i32, DIM as i32, alpha,
            device_mat_a, DIM as i32,
            device_mat_a, DIM as i32, beta,
            device_mat_b, DIM as i32);
        rpc_connection.cuda_device_synchronize();
        let end = Instant::now();
        duration += end - begin;
        println!("Matmul: {:?}", end - begin);
    }
    println!("Matmul Average: {:?}", duration / iterations);

    // cudaMemcpy(solution, device_matrix_B, DIM * DIM * sizeof(double), cudaMemcpyDeviceToHost);

    rpc_connection.cuda_free(device_mat_a);
    rpc_connection.cuda_free(device_mat_b);

    rpc_connection.rpc_cublasdestroy(handle);
}
