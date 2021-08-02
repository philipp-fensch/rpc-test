extern crate rpc_lib;
use rpc_lib::include_rpcl;

#[include_rpcl("rpc_cuda.x")]
struct RPCConnection;

fn main() {
}

#[cfg(test)]
mod test {

    use super::*;
    use std::mem::size_of;

    #[test]
    fn init_test() {
        let _rpc_connection = RPCConnection::new("127.0.0.1");
    }

    // TODO: Integration-Test!
    #[test]
    fn cuda_malloc_free() {
        let rpc_connection = RPCConnection::new("127.0.0.1");
        let id = rpc_connection.cuda_malloc(8);
        assert!(id.err == 0, "cuda_malloc failed");
        rpc_connection.cuda_free(id.ptr_result_u);
    }

    #[test]
    fn cuda_solve_linear_system() {
        // System to Solve: (transposed)
        // | 2 2 0 |       | 2 |               |-1 |
        // | 0 2 0 | * X = | 4 | Solution: X = | 2 |
        // | 0 1 1 |       | 5 |               | 3 |
        let mut matrix_host: [f64; 3 * 3] = [
            2.0, 0.0, 0.0, // transposed
            2.0, 2.0, 1.0,
            0.0, 0.0, 1.0 
        ];
        let mut right_side_host: [f64; 3] = [
            2.0,
            4.0,
            5.0
        ];

        // Init Connection and CUDA
        let rpc_connection = RPCConnection::new("127.0.0.1");
        let solver = rpc_connection.rpc_cusolverdncreate();

        // Allocate Memory
        let (vector_device, matrix_device, piv_seq_device, err, workspace) = allocate_memory(&rpc_connection, &solver);

        // Copy Matrix to Device
        let matrix_ptr_host = mem_data {
            len: 3 * 3 * size_of::<f64>() as u32,
            data: matrix_host.as_mut_ptr() as *mut c_void
        };
        rpc_connection.cuda_memcpy_htod(matrix_device, matrix_ptr_host, 3 * 3 * size_of::<f64>() as u64);

        // LU
        lu_factorization(&rpc_connection, &solver, matrix_device, workspace, piv_seq_device, err);
        rpc_connection.cuda_device_synchronize();

        // Solve System
        // Copy rhs to Device
        let right_side_ptr_host = mem_data {
            len: 3 * size_of::<f64>() as u32,
            data: right_side_host.as_mut_ptr() as *mut c_void
        };
        rpc_connection.cuda_memcpy_htod(vector_device, right_side_ptr_host, 3 * size_of::<f64>() as u64);

        // Solve
        let res = rpc_connection.rpc_cusolverdndgetrs(
            solver.ptr_result_u,
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
        assert!(*res == 0, "Solving System failed (cusolverDnDgetrs)");

        // Copy left-hand-side back
        let res2 = rpc_connection.cuda_memcpy_dtoh(vector_device, 3 * size_of::<f64>() as u64);
        let solution = unsafe { std::slice::from_raw_parts(res2.mem_result_u.data as *mut f64, 3) };

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

        rpc_connection.rpc_cusolverdndestroy(solver.ptr_result_u);
    }

    fn allocate_memory(rpc_connection: &RPCConnection, solver: &ptr_result) -> (u64, u64, u64, u64, u64) {
        // Vector
        let rhs_vector = rpc_connection.cuda_malloc(3 * size_of::<f64>() as u64).ptr_result_u;
        // Matrix
        let mat = rpc_connection.cuda_malloc(3 * 3 * size_of::<f64>() as u64).ptr_result_u;
        // Pivot
        let piv = rpc_connection.cuda_malloc(3 * size_of::<f64>() as u64).ptr_result_u;
        // Error-Code
        let err = rpc_connection.cuda_malloc(size_of::<i32>() as u64).ptr_result_u;

        let workspace_size = rpc_connection.rpc_cusolverdndgetrf_buffersize(
            solver.ptr_result_u,
            3,
            3,
            mat,
            3
        );
        assert!(workspace_size.err == 0, "cusolverdndgetrf_buffersize failed");

        let workspace = rpc_connection.cuda_malloc(workspace_size.int_result_u as u64).ptr_result_u;

        (rhs_vector, mat, piv, err, workspace)
    }

    fn lu_factorization(rpc_connection: &RPCConnection, solver: &ptr_result, matrix: u64, workspace: u64, piv: u64, err: u64) {
        let res = rpc_connection.rpc_cusolverdndgetrf(
            solver.ptr_result_u,
            3,
            3,
            matrix,
            3,
            workspace,
            piv,
            err
        );
        assert!(*res == 0, "LU-Factorization failed (cusolverDnDgetrf)");
    }
}
