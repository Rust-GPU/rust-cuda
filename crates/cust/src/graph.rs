//! CUDA Graph management.

use std::{
    ffi::c_void,
    mem::{ManuallyDrop, MaybeUninit},
    os::raw::{c_char, c_uint},
    path::Path,
    ptr,
};

use cust_raw::driver_sys;

use crate::{
    error::{CudaResult, ToResult},
    function::{BlockSize, GridSize},
};

/// Creates a kernel invocation using the same syntax as [`launch`] to be used to insert kernel launches inside graphs.
/// This returns a Result of a kernel invocation object you can then pass to a graph.
#[macro_export]
macro_rules! kernel_invocation {
    ($module:ident . $function:ident <<<$grid:expr, $block:expr, $shared:expr, $stream:ident>>>( $( $arg:expr),* )) => {
        {
            let name = std::ffi::CString::new(stringify!($function)).unwrap();
            let function = $module.get_function(&name);
            match function {
                Ok(f) => kernel_invocation!(f<<<$grid, $block, $shared, $stream>>>( $($arg),* ) ),
                Err(e) => Err(e),
            }
        }
    };
    ($function:ident <<<$grid:expr, $block:expr, $shared:expr, $stream:ident>>>( $( $arg:expr),* )) => {
        {
            fn assert_impl_devicecopy<T: $crate::memory::DeviceCopy>(_val: T) {}
            if false {
                $(
                    assert_impl_devicecopy($arg);
                )*
            };

            let boxed = vec![$(&$arg as *const _ as *mut ::std::ffi::c_void),*].into_boxed_slice();

            Ok($crate::graph::KernelInvocation::_new_internal(
                $crate::function::BlockSize::from($block),
                $crate::function::GridSize::from($grid),
                $shared,
                $function.to_raw(),
                vec![].into_boxed_slice(),
            ))
        }
    };
}

/// A prepared kernel invocation to be added to a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelInvocation {
    pub block_dim: BlockSize,
    pub grid_dim: GridSize,
    pub shared_mem_bytes: u32,
    func: driver_sys::CUfunction,
    params: Box<*mut c_void>,
    params_len: Option<usize>,
}

impl KernelInvocation {
    #[doc(hidden)]
    pub fn _new_internal(
        block_dim: BlockSize,
        grid_dim: GridSize,
        shared_mem_bytes: u32,
        func: driver_sys::CUfunction,
        params: Box<*mut c_void>,
        params_len: usize,
    ) -> Self {
        Self {
            block_dim,
            grid_dim,
            shared_mem_bytes,
            func,
            params,
            params_len: Some(params_len),
        }
    }

    pub fn to_raw(self) -> driver_sys::CUDA_KERNEL_NODE_PARAMS {
        driver_sys::CUDA_KERNEL_NODE_PARAMS {
            func: self.func,
            gridDimX: self.grid_dim.x,
            gridDimY: self.grid_dim.y,
            gridDimZ: self.grid_dim.z,
            blockDimX: self.block_dim.x,
            blockDimY: self.block_dim.y,
            blockDimZ: self.block_dim.z,
            kernelParams: Box::into_raw(self.params),
            sharedMemBytes: self.shared_mem_bytes,
            extra: ptr::null_mut(),
            kern: ptr::null_mut(),
            ctx: ptr::null_mut(),
        }
    }

    /// Makes a new invocation from its raw counterpart.
    ///
    /// # Safety
    ///
    /// The function pointer must be a valid CUfunction pointer and
    /// params' "ownership" must be able to be transferred to the invocation
    /// (it will be turned into a Box).
    pub unsafe fn from_raw(raw: driver_sys::CUDA_KERNEL_NODE_PARAMS) -> Self {
        Self {
            func: raw.func,
            grid_dim: GridSize::xyz(raw.gridDimX, raw.gridDimY, raw.gridDimZ),
            block_dim: BlockSize::xyz(raw.blockDimX, raw.gridDimY, raw.gridDimZ),
            params: Box::from_raw(raw.kernelParams),
            shared_mem_bytes: raw.sharedMemBytes,
            params_len: None,
        }
    }
}

/// An opaque handle to a node in a graph. There are no methods on [`GraphNode`], they
/// are just handles for identifying nodes to be used on [`Graph`] functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GraphNode {
    raw: driver_sys::CUgraphNode,
}

unsafe impl Send for GraphNode {}
unsafe impl Sync for GraphNode {}

impl GraphNode {
    /// Creates a new node from a raw handle. This is safe because node checks
    /// happen on the graph when functions are called.
    pub fn from_raw(raw: driver_sys::CUgraphNode) -> Self {
        Self { raw }
    }

    /// Converts this node into a raw handle.
    pub fn to_raw(self) -> driver_sys::CUgraphNode {
        self.raw
    }
}

/// The different types that a node can be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GraphNodeType {
    /// Invokes a GPU kernel.
    KernelInvocation,
    /// Copies memory from one location to another (CPU to GPU/GPU to CPU/GPU to GPU).
    Memcpy,
    /// Sets some memory to some value.
    Memset,
    /// Executes a function on the host (CPU).
    HostExecute,
    /// Executes a child graph.
    ChildGraph,
    /// Does nothing.
    Empty,
    /// Waits for an event.
    WaitEvent,
    /// Record an event.
    EventRecord,
    /// Performs a signal operation on external semaphore objects.
    SemaphoreSignal,
    /// Performs a wait operation on external semaphore objects.
    SemaphoreWait,
    /// Allocates some memory.
    MemoryAllocation,
    /// Frees some memory.
    MemoryFree,
    /// Batch memory operation.
    BatchMemoryOperation,
    /// Conditional node.
    #[cfg(conditional_node)]
    Conditional,
}

impl GraphNodeType {
    /// Converts a raw type to a [`GraphNodeType`].
    pub fn from_raw(raw: driver_sys::CUgraphNodeType) -> Self {
        use driver_sys::CUgraphNodeType::*;
        match raw {
            CU_GRAPH_NODE_TYPE_KERNEL => GraphNodeType::KernelInvocation,
            CU_GRAPH_NODE_TYPE_MEMCPY => GraphNodeType::Memcpy,
            CU_GRAPH_NODE_TYPE_MEMSET => GraphNodeType::Memset,
            CU_GRAPH_NODE_TYPE_HOST => GraphNodeType::HostExecute,
            CU_GRAPH_NODE_TYPE_GRAPH => GraphNodeType::ChildGraph,
            CU_GRAPH_NODE_TYPE_EMPTY => GraphNodeType::Empty,
            CU_GRAPH_NODE_TYPE_WAIT_EVENT => GraphNodeType::WaitEvent,
            CU_GRAPH_NODE_TYPE_EVENT_RECORD => GraphNodeType::EventRecord,
            CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL => GraphNodeType::SemaphoreSignal,
            CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT => GraphNodeType::SemaphoreWait,
            CU_GRAPH_NODE_TYPE_MEM_ALLOC => GraphNodeType::MemoryAllocation,
            CU_GRAPH_NODE_TYPE_MEM_FREE => GraphNodeType::MemoryFree,
            CU_GRAPH_NODE_TYPE_BATCH_MEM_OP => GraphNodeType::BatchMemoryOperation,
            #[cfg(conditional_node)]
            CU_GRAPH_NODE_TYPE_CONDITIONAL => GraphNodeType::Conditional,
        }
    }

    /// Converts this type to its raw counterpart.
    pub fn to_raw(self) -> driver_sys::CUgraphNodeType {
        use driver_sys::CUgraphNodeType::*;
        match self {
            Self::KernelInvocation => CU_GRAPH_NODE_TYPE_KERNEL,
            Self::Memcpy => CU_GRAPH_NODE_TYPE_MEMCPY,
            Self::Memset => CU_GRAPH_NODE_TYPE_MEMSET,
            Self::HostExecute => CU_GRAPH_NODE_TYPE_HOST,
            Self::ChildGraph => CU_GRAPH_NODE_TYPE_GRAPH,
            Self::Empty => CU_GRAPH_NODE_TYPE_EMPTY,
            Self::WaitEvent => CU_GRAPH_NODE_TYPE_WAIT_EVENT,
            Self::EventRecord => CU_GRAPH_NODE_TYPE_EVENT_RECORD,
            Self::SemaphoreSignal => CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL,
            Self::SemaphoreWait => CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT,
            Self::MemoryAllocation => CU_GRAPH_NODE_TYPE_MEM_ALLOC,
            Self::MemoryFree => CU_GRAPH_NODE_TYPE_MEM_FREE,
            Self::BatchMemoryOperation => CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
            #[cfg(conditional_node)]
            Self::Conditional => CU_GRAPH_NODE_TYPE_CONDITIONAL,
        }
    }
}

/// A graph object used for building a hierarchy of kernels to launch at once.
/// Graphs are used to control jobs that have multiple kernels that need to be launched back to back.
/// They reduce the overhead of launching kernels and cpu/gpu transfer by launching everything at once.
///
/// CUDA Graphs are inherently extremely unsafe, it is very easy to cause UB by passing a dropped node,
/// an invalid node, a node from another graph, etc. To mostly solve this we query the nodes inside the
/// graph every time a node is used to check if it is valid. This sounds expensive, but in practice graphs
/// are not large enough where checking makes a big difference. Additionally, internally we cache the nodes
/// that are known to be up-to-date
///
/// These safety measures should account for most safety pitfalls, if you encounter a way to bypass them
/// please file an issue and we will try to fix it ASAP.
///
/// However, it is inherently impossible for us to validate graph usage, just like launching kernels.
/// Therefore, launching graphs is unsafe and always will be, the user must validate that:
/// - All kernel launches are safe (same invariants as launching a normal kernel)
/// - Memory structures used inside the graph must not be dropped before the graph is executed (this will likely
/// throw an error if you try doing it).
///
/// These problems can easily be avoided by launching the graph as soon as or right after it is instantiated,
/// instead of holding onto it long-term, which can cause problems if data is dropped before the graph is executed.
///
/// Graphs are **not** threadsafe, therefore it is not possible to modify them from multiple threads at the
/// same time. This is statically prevented by taking mutable references for all functions. You can however
/// send graphs between threads.
#[derive(Debug)]
pub struct Graph {
    raw: driver_sys::CUgraph,
    // a cache of nodes, this cache is None when the node cache is out of date,
    // it will get refreshed when get_nodes is called.
    node_cache: Option<Vec<GraphNode>>,
}

// SAFETY: the cuda driver API docs say that any operations on the same graph object are not
// thredsafe and must be serialized, but passing graphs to and from threads should be fine.
// The fact that methods on Graph take `&mut self` statically prevents this from happening (thanks rustc <3)
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

bitflags::bitflags! {
    /// Flags for creating a graph. This is currently empty but reserved for
    /// any flags which may be added in the future.
    #[derive(Default)]
    pub struct GraphCreationFlags: u32 {
        /// No flags, currently the only option available.
        const NONE = 0b00000000;
    }
}

impl Graph {
    fn check_deps_are_valid(&mut self, func_name: &str, nodes: &[GraphNode]) -> CudaResult<()> {
        // per the docs, nodes must be valid AND not duplicate.
        for (idx, node) in nodes.iter().enumerate() {
            if let Some(pos) = nodes
                .iter()
                .enumerate()
                .position(|(cur_idx, x)| x == node && cur_idx != idx)
            {
                panic!(
                    "Duplicate dependency found in call to `{}`, the first instance is at index {}, the second instance is at index {}",
                    func_name, idx, pos
                );
            }

            assert!(
                self.is_valid_node(*node)?,
                "Invalid (dropped or from another graph) node was given to `{}`",
                func_name
            );
        }
        Ok(())
    }

    /// Check if a node is valid in this graph.
    pub fn is_valid_node(&mut self, node: GraphNode) -> CudaResult<bool> {
        let nodes = self.nodes()?;
        Ok(nodes.contains(&node))
    }

    /// Get the number of nodes in this graph.
    pub fn num_nodes(&mut self) -> CudaResult<usize> {
        unsafe {
            let mut len = MaybeUninit::uninit();
            driver_sys::cuGraphGetNodes(self.raw, ptr::null_mut(), len.as_mut_ptr()).to_result()?;
            Ok(len.assume_init())
        }
    }

    /// Get all of the nodes in this graph.
    pub fn nodes(&mut self) -> CudaResult<&[GraphNode]> {
        if self.node_cache.is_none() {
            unsafe {
                let mut len = self.num_nodes()?;
                let mut vec = Vec::with_capacity(len);
                driver_sys::cuGraphGetNodes(
                    self.raw,
                    vec.as_mut_ptr() as *mut driver_sys::CUgraphNode,
                    &mut len as *mut usize,
                )
                .to_result()?;
                vec.set_len(len);
                self.node_cache = Some(vec);
            }
        }
        Ok(self.node_cache.as_ref().unwrap())
    }

    /// Creates a new graph from some flags.
    pub fn new(flags: GraphCreationFlags) -> CudaResult<Self> {
        let mut raw = MaybeUninit::uninit();

        unsafe {
            driver_sys::cuGraphCreate(raw.as_mut_ptr(), flags.bits()).to_result()?;

            Ok(Self {
                raw: raw.assume_init(),
                node_cache: Some(vec![]),
            })
        }
    }

    /// Dumps a dotfile to a path which contains a visual representation of the graph for debugging.
    /// This dotfile can be turned into an image with graphviz.
    #[cfg(any(windows, unix))]
    pub fn dump_debug_dotfile<P: AsRef<Path>>(&mut self, path: P) -> CudaResult<()> {
        let path = path.as_ref();
        let mut buf = Vec::new();
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            buf.extend(path.as_os_str().as_bytes());
            buf.push(0);
        }

        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            buf.extend(
                path.as_os_str()
                    .encode_wide()
                    .chain(Some(0))
                    .map(|b| {
                        let b = b.to_ne_bytes();
                        b.get(0).copied().into_iter().chain(b.get(1).copied())
                    })
                    .flatten(),
            );
        }

        unsafe {
            driver_sys::cuGraphDebugDotPrint(
                self.raw,
                c"./out.dot".as_ptr(),
                driver_sys::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE as u32,
            )
            .to_result()
        }
    }

    /// Adds a kernel invocation node to this graph, [`KernelInvocation`] can be created using
    /// [`kernel_invocation`] which uses the same syntax as [`launch`](crate::launch). This will
    /// place the node after its dependencies (which will execute before it).
    pub fn add_kernel_node(
        &mut self,
        invocation: KernelInvocation,
        dependencies: impl AsRef<[GraphNode]>,
    ) -> CudaResult<GraphNode> {
        let deps = dependencies.as_ref();
        self.check_deps_are_valid("add_kernel_node", deps)?;
        // invalidate cache because it will change.
        self.node_cache = None;
        unsafe {
            let deps_ptr = deps.as_ptr().cast();
            let mut node = MaybeUninit::<GraphNode>::uninit();
            let params = invocation.to_raw();
            driver_sys::cuGraphAddKernelNode(
                node.as_mut_ptr().cast(),
                self.raw,
                deps_ptr,
                deps.len(),
                &params as *const _,
            )
            .to_result()?;
            Ok(node.assume_init())
        }
    }

    /// The number of edges (dependency edges) inside this graph.
    pub fn num_edges(&mut self) -> CudaResult<usize> {
        unsafe {
            let mut size = MaybeUninit::uninit();
            driver_sys::cuGraphGetEdges(
                self.raw,
                ptr::null_mut(),
                ptr::null_mut(),
                #[cfg(cuGraphGetEdges_v2)]
                ptr::null_mut(),
                size.as_mut_ptr(),
            )
            .to_result()?;
            Ok(size.assume_init())
        }
    }

    /// Returns a list of the dependency edges of this graph.
    ///
    /// # Returns
    ///
    /// Returns a vector of the edge from one node to another. There may be multiples
    /// of the same node in the vector, since a node can have multiple edges coming out of it.
    /// `(A, B)` means that `B` has a dependency on `A`, that is, `A` will execute before `B`.
    pub fn edges(&mut self) -> CudaResult<Vec<(GraphNode, GraphNode)>> {
        unsafe {
            let num_edges = self.num_edges()?;
            let mut from = vec![ptr::null_mut(); num_edges].into_boxed_slice();
            let mut to = vec![ptr::null_mut(); num_edges].into_boxed_slice();

            driver_sys::cuGraphGetEdges(
                self.raw,
                from.as_mut_ptr(),
                to.as_mut_ptr(),
                #[cfg(cuGraphGetEdges_v2)]
                ptr::null_mut(),
                &num_edges as *const _ as *mut usize,
            )
            .to_result()?;

            let mut out = Vec::with_capacity(num_edges);
            for (from, to) in from.iter().zip(to.iter()) {
                out.push((GraphNode::from_raw(*from), GraphNode::from_raw(*to)))
            }
            Ok(out)
        }
    }

    /// Retrieves the type of a node.
    pub fn node_type(&mut self, node: GraphNode) -> CudaResult<GraphNodeType> {
        self.check_deps_are_valid("node_type", &[node])?;
        unsafe {
            let mut ty = MaybeUninit::uninit();
            driver_sys::cuGraphNodeGetType(node.to_raw(), ty.as_mut_ptr()).to_result()?;
            let raw = ty.assume_init();
            Ok(GraphNodeType::from_raw(raw))
        }
    }

    /// Retrieves the invocation parameters for a kernel invocation node.
    ///
    /// # Panics
    ///
    /// Panics if the node is invalid or if the node is not a kernel invocation node.
    pub fn kernel_node_params(&mut self, node: GraphNode) -> CudaResult<KernelInvocation> {
        self.check_deps_are_valid("kernel_node_params", &[node])?;
        assert_eq!(
            self.node_type(node)?,
            GraphNodeType::KernelInvocation,
            "Node given to `kernel_node_params` was not a kernel invocation node"
        );
        unsafe {
            let mut params = MaybeUninit::uninit();
            driver_sys::cuGraphKernelNodeGetParams(node.to_raw(), params.as_mut_ptr());
            Ok(KernelInvocation::from_raw(params.assume_init()))
        }
    }

    /// Creates a new [`Graph`] from a raw handle.
    ///
    /// # Safety
    ///
    /// This assumes a couple of things:
    /// - This handle is exclusive, nothing else can use it in any way, including trying to drop it.
    /// - It must be a valid handle. This invariant must be upheld, the library is allowed to rely on
    /// the fact that the handle is valid in terms of safety, therefore failure to uphold this invariant is UB.
    pub unsafe fn from_raw(raw: driver_sys::CUgraph) -> Self {
        Self {
            raw,
            node_cache: None,
        }
    }

    /// Consumes this [`Graph`], turning it into a raw handle. The handle will not be dropped,
    /// it is up to the caller to ensure the graph is destroyed.
    pub fn into_raw(self) -> driver_sys::CUgraph {
        let me = ManuallyDrop::new(self);
        me.raw
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            driver_sys::cuGraphDestroy(self.raw);
        }
    }
}
