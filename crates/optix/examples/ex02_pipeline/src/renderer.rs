use anyhow::{Context, Result};
use cust::context::Context as CuContext;
use cust::device::{Device, DeviceAttribute};
use cust::memory::{CopyDestination, DeviceBox, DeviceBuffer, DevicePointer, DeviceVariable};
use cust::stream::{Stream, StreamFlags};
use cust::CudaFlags;
use cust::DeviceCopy;
use optix::{
    context::DeviceContext,
    pipeline::{
        CompileDebugLevel, CompileOptimizationLevel, ExceptionFlags, Module, ModuleCompileOptions,
        Pipeline, PipelineCompileOptions, PipelineLinkOptions, ProgramGroup, ProgramGroupDesc,
        TraversableGraphFlags,
    },
    shader_binding_table::{SbtRecord, ShaderBindingTable},
};

pub struct Renderer {
    launch_params: DeviceVariable<LaunchParams>,
    sbt: ShaderBindingTable,
    pipeline: Pipeline,
    buf_raygen: DeviceBuffer<RaygenRecord>,
    buf_hitgroup: DeviceBuffer<HitgroupRecord>,
    buf_miss: DeviceBuffer<MissRecord>,
    color_buffer: DeviceBuffer<u32>,
    ctx: DeviceContext,
    stream: Stream,
    cuda_context: CuContext,
}

use kernels::LaunchParams;

impl Renderer {
    pub fn new(width: usize, height: usize) -> Result<Renderer, Box<dyn std::error::Error>> {
        init_optix()?;

        // create CUDA and OptiX contexts
        let device = Device::get_device(0)?;
        let tex_align = device.get_attribute(DeviceAttribute::TextureAlignment)?;
        let srf_align = device.get_attribute(DeviceAttribute::SurfaceAlignment)?;
        println!("tex align: {}\nsrf align: {}", tex_align, srf_align);

        let cuda_context = CuContext::new(device)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;

        let mut ctx = DeviceContext::new(&cuda_context, false)?;
        ctx.set_log_callback(|_level, tag, msg| println!("[{}]: {}", tag, msg), 4)?;

        // create module
        let module_compile_options = ModuleCompileOptions {
            max_register_count: 50,
            opt_level: CompileOptimizationLevel::Default,
            debug_level: CompileDebugLevel::None,
        };

        let pipeline_compile_options = PipelineCompileOptions::new()
            .pipeline_launch_params_variable_name("PARAMS")
            .uses_motion_blur(false)
            .num_attribute_values(2)
            .num_payload_values(2)
            .traversable_graph_flags(TraversableGraphFlags::ALLOW_SINGLE_GAS)
            .exception_flags(ExceptionFlags::NONE);

        // let ptx = include_str!(concat!(env!("OUT_DIR"), "/src/ex02_pipeline.ptx"));
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

        let (module, _log) = Module::new(
            &mut ctx,
            &module_compile_options,
            &pipeline_compile_options,
            ptx,
        )
        .context("Create module")?;

        // create raygen program
        let pgdesc_raygen = ProgramGroupDesc::raygen(&module, "__raygen__renderFrame");

        let (pg_raygen, _log) = ProgramGroup::new(&mut ctx, &[pgdesc_raygen])?;

        // create miss program
        let pgdesc_miss = ProgramGroupDesc::miss(&module, "__miss__radiance");

        let (pg_miss, _log) = ProgramGroup::new(&mut ctx, &[pgdesc_miss])?;

        let pgdesc_hitgroup = ProgramGroupDesc::hitgroup(
            Some((&module, "__closesthit__radiance")),
            Some((&module, "__anyhit__radiance")),
            None,
        );

        // create hitgroup programs
        let (pg_hitgroup, _log) = ProgramGroup::new(&mut ctx, &[pgdesc_hitgroup])?;

        // create SBT
        let rec_raygen: Vec<_> = pg_raygen
            .iter()
            .map(|pg| RaygenRecord::pack(0, pg).expect("failed to pack raygen record"))
            .collect();

        let rec_miss: Vec<_> = pg_miss
            .iter()
            .map(|pg| MissRecord::pack(0, pg).expect("failed to pack miss record"))
            .collect();

        let num_objects = 1;
        let rec_hitgroup: Vec<_> = (0..num_objects)
            .map(|i| {
                let object_type = 0;
                let rec = HitgroupRecord::pack(
                    HitgroupSbtData { object_id: i },
                    &pg_hitgroup[object_type],
                )
                .expect("failed to pack hitgroup record");
                rec
            })
            .collect();

        let mut buf_raygen = DeviceBuffer::from_slice(&rec_raygen)?;
        let mut buf_miss = DeviceBuffer::from_slice(&rec_miss)?;
        let mut buf_hitgroup = DeviceBuffer::from_slice(&rec_hitgroup)?;

        let sbt = ShaderBindingTable::new(&mut buf_raygen)
            .miss(&mut buf_miss)
            .hitgroup(&mut buf_hitgroup);

        // create pipeline
        let mut program_groups = Vec::new();
        program_groups.extend(pg_raygen.into_iter());
        program_groups.extend(pg_miss.into_iter());
        program_groups.extend(pg_hitgroup.into_iter());

        let pipeline_link_options = PipelineLinkOptions {
            max_trace_depth: 2,
            debug_level: CompileDebugLevel::LineInfo,
        };

        let (pipeline, _log) = Pipeline::new(
            &mut ctx,
            &pipeline_compile_options,
            pipeline_link_options,
            &program_groups,
        )?;

        pipeline.set_stack_size(2 * 1024, 2 * 1024, 2 * 1024, 1)?;

        let color_buffer = unsafe { DeviceBuffer::uninitialized(width * height)? };

        let launch_params = DeviceVariable::new(LaunchParams {
            frame_id: 17,
            fb_size: [width as u32, height as u32],
            color_buffer: color_buffer.as_device_ptr().as_raw(),
        })?;

        Ok(Renderer {
            ctx,
            cuda_context,
            stream,
            launch_params,
            buf_raygen,
            buf_hitgroup,
            buf_miss,
            sbt,
            pipeline,
            color_buffer,
        })
    }

    pub fn resize(
        &mut self,
        width: usize,
        height: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.color_buffer = unsafe { DeviceBuffer::uninitialized(width * height)? };
        self.launch_params.fb_size[0] = width as u32;
        self.launch_params.fb_size[1] = height as u32;
        self.launch_params.color_buffer = self.color_buffer.as_device_ptr().as_raw();
        Ok(())
    }

    pub fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.launch_params.frame_id = 555;
        self.launch_params.copy_htod()?;
        self.launch_params.frame_id = 777;

        unsafe {
            optix::launch(
                &self.pipeline,
                &self.stream,
                &self.launch_params,
                &self.sbt,
                self.launch_params.fb_size[0],
                self.launch_params.fb_size[1],
                1,
            )?;
        }

        self.stream.synchronize()?;

        Ok(())
    }
}

type RaygenRecord = SbtRecord<i32>;
type MissRecord = SbtRecord<i32>;

#[derive(Copy, Clone, Default, DeviceCopy)]
struct HitgroupSbtData {
    object_id: u32,
}
type HitgroupRecord = SbtRecord<HitgroupSbtData>;

fn init_optix() -> Result<(), Box<dyn std::error::Error>> {
    cust::init(CudaFlags::empty())?;
    let device_count = Device::num_devices()?;
    if device_count == 0 {
        panic!("No CUDA devices found!");
    }

    optix::init()?;
    Ok(())
}
