use std::env;

fn main() {
    let driver_version = env::var("DEP_CUDA_DRIVER_VERSION")
        .expect("Cannot find transitive metadata 'driver_version' from cust_raw package.")
        .parse::<u32>()
        .expect("Failed to parse CUDA driver version");

    println!("cargo::rustc-check-cfg=cfg(conditional_node)");
    if driver_version >= 12030 {
        println!("cargo::rustc-cfg=conditional_node");
    }
    // In CUDA 13.0 several pairs/trios of functions were merged:
    // ```
    // CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);
    // CUresult cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location);
    //
    // CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
    // CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream);
    //
    // CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges);
    // CUresult cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, CUgraphEdgeData* edgeData, size_t* numEdges);
    //
    // CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
    // CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev);
    // CUresult cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev);
    // ```
    // In each case, the resulting single function has the name of the first function and the type
    // signature of the last.
    //
    // These cfgs let you call these functions and make it work for both pre CUDA-13.0 and CUDA
    // 13.0. When support for CUDA 12.x is dropped, these cfgs can be removed.
    println!("cargo::rustc-check-cfg=cfg(cuMemAdvise_v2)");
    println!("cargo::rustc-check-cfg=cfg(cuMemPrefetchAsync_v2)");
    println!("cargo::rustc-check-cfg=cfg(cuGraphGetEdges_v2)");
    println!("cargo::rustc-check-cfg=cfg(cuCtxCreate_v4)");
    if driver_version >= 13000 {
        println!("cargo::rustc-cfg=cuMemAdvise_v2");
        println!("cargo::rustc-cfg=cuMemPrefetchAsync_v2");
        println!("cargo::rustc-cfg=cuGraphGetEdges_v2");
        println!("cargo::rustc-cfg=cuCtxCreate_v4");
    }
}
