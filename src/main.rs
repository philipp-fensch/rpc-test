extern crate rpc_lib;
use rpc_lib::include_rpcl;
use std::mem::size_of;

use std::time::*;

#[include_rpcl("rpc_cuda.x")]
struct RPCConnection;

fn main() {
    let rpc_connection = RPCConnection::new("137.226.133.199");

    const ITERATIONS: u32 = 10;

    memcpy_test(&rpc_connection, ITERATIONS);
    latency_test(&rpc_connection, ITERATIONS);
    // linear_solver_test();
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

fn linear_solver_test() {
        const DIM: usize = 3;
        // System to Solve: (transposed)
        // | 2 2 0 |       | 2 |               |-1 |
        // | 0 2 0 | * X = | 4 | Solution: X = | 2 |
        // | 0 1 1 |       | 5 |               | 3 |
        let matrix_host: [f64; DIM * DIM] = [
            2.0, 0.0, 0.0, // transposed
            2.0, 2.0, 1.0,
            0.0, 0.0, 1.0
        ];
        let right_side_host: [f64; DIM] = [
            2.0,
            4.0,
            5.0
        ];

        // Cast f64 to u8 for cudamemcpy
        let matrix_host_cast = unsafe {
            std::mem::transmute::<[f64; DIM * DIM], [u8; size_of::<f64>() * DIM * DIM]>(matrix_host)
        };

        // Init Connection and CUDA
        let rpc_connection = RPCConnection::new("137.226.133.199");
        let solver = rpc_connection.rpc_cusolverdncreate().unwrap();

        // Allocate Memory
        let (vector_device, matrix_device, piv_seq_device, err, workspace) = allocate_memory(&rpc_connection, solver);

        // Copy Matrix to Device
        rpc_connection.cuda_memcpy_htod(matrix_device, matrix_host_cast.to_vec(), matrix_host_cast.len() as u64);

        // LU
        lu_factorization(&rpc_connection, solver, matrix_device, workspace, piv_seq_device, err);
        rpc_connection.cuda_device_synchronize();

        // Solve System
        // Copy rhs to Device
        let right_side_host_cast = unsafe {
            std::mem::transmute::<[f64; DIM], [u8; size_of::<f64>() * DIM]>(right_side_host)
        };
        rpc_connection.cuda_memcpy_htod(vector_device, right_side_host_cast.to_vec(), right_side_host_cast.len() as u64);

        // Solve
        let res = rpc_connection.rpc_cusolverdndgetrs(
            solver,
            0, // CUBLAS_OP_N
            3,
            1, //#right-hand-sides
            matrix_device,
            3,
            piv_seq_device,
            vector_device,
            3,
            err
        );
        rpc_connection.cuda_device_synchronize();
        assert!(res == 0, "Solving System failed (cusolverDnDgetrs)");

        // Copy left-hand-side back
        let res = rpc_connection.cuda_memcpy_dtoh(vector_device, (size_of::<f64>() * DIM) as u64);
        let res2 = res.as_ref().unwrap();

        // Cast mem_result from generic u8 to the actual f64
        let solution = unsafe {
            std::mem::transmute::<&[u8], &[f64]>(&res2)
        };
        
        // Check Result
        assert!(
            (solution[0] + 1.0).abs() < 0.001 ||
            (solution[1] - 2.0).abs() < 0.001 ||
            (solution[2] - 3.0).abs() < 0.001,
            "Solution wrong"
        );

        // Free Memory
        rpc_connection.cuda_free(vector_device);
        rpc_connection.cuda_free(matrix_device);
        rpc_connection.cuda_free(piv_seq_device);
        rpc_connection.cuda_free(err);
        rpc_connection.cuda_free(workspace);

        assert!(rpc_connection.rpc_cusolverdndestroy(solver) == 0, "rpc_cusolverdndestroy failed");
    }

    fn allocate_memory(rpc_connection: &RPCConnection, solver: u64) -> (u64, u64, u64, u64, u64) {
        // Vector
        let rhs_vector = rpc_connection.cuda_malloc(3 * size_of::<f64>() as u64).unwrap();
        // Matrix
        let mat = rpc_connection.cuda_malloc(3 * 3 * size_of::<f64>() as u64).unwrap();
        // Pivot
        let piv = rpc_connection.cuda_malloc(3 * size_of::<f64>() as u64).unwrap();
        // Error-Code
        let err = rpc_connection.cuda_malloc(size_of::<i32>() as u64).unwrap();

        let workspace_size = rpc_connection.rpc_cusolverdndgetrf_buffersize(
            solver,
            3,
            3,
            mat,
            3
        ).unwrap();
        // assert!(workspace_size == 0, "cusolverdndgetrf_buffersize failed");

        let workspace = rpc_connection.cuda_malloc(workspace_size as u64).unwrap();

        (rhs_vector, mat, piv, err, workspace)
    }

    fn lu_factorization(rpc_connection: &RPCConnection, solver: u64, matrix: u64, workspace: u64, piv: u64, err: u64) {
        let res = rpc_connection.rpc_cusolverdndgetrf(
            solver,
            3,
            3,
            matrix,
            3,
            workspace,
            piv,
            err
        );
        assert!(res == 0, "LU-Factorization failed (cusolverDnDgetrf)");
    }
