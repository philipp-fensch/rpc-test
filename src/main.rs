extern crate rpc_lib;
use rpc_lib::include_rpcl;

#[include_rpcl("rpc_cuda.x")]
struct RPCConnection;

fn main() {
    println!("Hello");
}

#[cfg(test)]
mod test {

    use super::*;

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
        let rpc_connection = RPCConnection::new("127.0.0.1");
        let solver = rpc_connection.rpc_cusolverdncreate();
        rpc_connection.rpc_cusolverdndestroy(solver.ptr_result_u);
    }
}
