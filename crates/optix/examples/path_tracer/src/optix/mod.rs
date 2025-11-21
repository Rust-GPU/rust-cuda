use crate::cuda::CudaRendererBuffers;
use anyhow::Result;
use cust::{
    memory::{DeviceBox, DeviceBuffer},
    prelude::Stream,
};
use optix::{
    acceleration::{
        Aabb, Accel, AccelBuildOptions, BuildFlags, CustomPrimitiveArray, GeometryFlags,
        Traversable,
    },
    context::DeviceContext,
    prelude::{
        CompileDebugLevel, ExceptionFlags, Module as OptixModule, ModuleCompileOptions, Pipeline,
        PipelineCompileOptions, PipelineLinkOptions, ProgramGroup, ProgramGroupDesc,
        TraversableGraphFlags,
    },
    shader_binding_table::{SbtRecord, ShaderBindingTable},
};
use path_tracer_kernels::{optix::LaunchParams, scene::Scene, sphere::Sphere, Object};

pub type RaygenRecord = SbtRecord<i32>;
pub type MissRecord = SbtRecord<i32>;
pub type SphereHitgroupRecord = SbtRecord<Sphere>;

pub(crate) static OPTIX_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels_optix.ptx"));

/// A subset of the CUDA renderer that uses hardware raytracing with OptiX
pub struct OptixRenderer {
    sbt: ShaderBindingTable,
    gas: Accel,
    pipeline: Pipeline,
}

impl OptixRenderer {
    pub fn new(ctx: &mut DeviceContext, stream: &Stream, scene: &Scene) -> Result<Self> {
        let module_compile_options = ModuleCompileOptions {
            max_register_count: 100,
            debug_level: Default::default(),
            opt_level: Default::default(),
        };

        let pipeline_compile_options = PipelineCompileOptions::new()
            .pipeline_launch_params_variable_name("PARAMS")
            .uses_motion_blur(false)
            .num_attribute_values(7)
            .num_payload_values(2)
            .traversable_graph_flags(TraversableGraphFlags::ALLOW_SINGLE_GAS)
            .exception_flags(ExceptionFlags::NONE);

        let (module, _log) = OptixModule::new(
            ctx,
            &module_compile_options,
            &pipeline_compile_options,
            OPTIX_PTX,
        )?;

        let pgdesc_raygen = ProgramGroupDesc::raygen(&module, "__raygen__render");
        let (pg_raygen, _log) = ProgramGroup::new(ctx, &[pgdesc_raygen])?;

        let pgdesc_miss = ProgramGroupDesc::miss(&module, "__miss__miss");
        let (pg_miss, _log) = ProgramGroup::new(ctx, &[pgdesc_miss])?;

        let pgdesc_sphere_hitgroup = ProgramGroupDesc::hitgroup(
            Some((&module, "__closesthit__sphere")),
            None,
            Some((&module, "__intersection__sphere")),
        );
        let (pg_sphere_hitgroup, _log) = ProgramGroup::new(ctx, &[pgdesc_sphere_hitgroup])?;

        let accel = Self::build_accel_from_scene(ctx, stream, scene)?;

        let rec_raygen: Vec<_> = pg_raygen
            .iter()
            .map(|pg| RaygenRecord::pack(0, pg).expect("failed to pack raygen record"))
            .collect();

        let rec_miss: Vec<_> = pg_miss
            .iter()
            .map(|pg| MissRecord::pack(0, pg).expect("failed to pack miss record"))
            .collect();
        let rec_sphere_hitgroup =
            Self::build_scene_hitgroup_records(scene, &pg_sphere_hitgroup[0])?;

        let buf_raygen = DeviceBuffer::from_slice(&rec_raygen)?;
        let buf_miss = DeviceBuffer::from_slice(&rec_miss)?;
        let buf_sphere_hitgroup = DeviceBuffer::from_slice(&rec_sphere_hitgroup)?;

        let sbt = ShaderBindingTable::new(&buf_raygen)
            .miss(&buf_miss)
            .hitgroup(&buf_sphere_hitgroup);

        let mut program_groups = Vec::new();
        program_groups.extend(pg_raygen.into_iter());
        program_groups.extend(pg_miss.into_iter());
        program_groups.extend(pg_sphere_hitgroup.into_iter());

        let pipeline_link_options = PipelineLinkOptions {
            max_trace_depth: 10,
            debug_level: CompileDebugLevel::None,
        };

        let (pipeline, _log) = Pipeline::new(
            ctx,
            &pipeline_compile_options,
            pipeline_link_options,
            &program_groups,
        )?;
        pipeline.set_stack_size(2 * 1024, 2 * 1024, 2 * 1024, 1)?;
        Ok(Self {
            sbt,
            gas: accel,
            pipeline,
        })
    }

    fn build_accel_from_scene(
        ctx: &mut DeviceContext,
        stream: &Stream,
        scene: &Scene,
    ) -> Result<Accel> {
        let mut aabbs = Vec::with_capacity(scene.objects.len());
        for obj in scene.objects.iter() {
            match obj {
                Object::Sphere(sphere) => {
                    let center = sphere.center;
                    let radius = sphere.radius;
                    let aabb = Aabb::new(center - radius, center + radius);
                    aabbs.push(aabb);
                }
            }
        }
        let buf = DeviceBuffer::from_slice(&aabbs)?;

        let mut build_inputs = Vec::with_capacity(buf.len());
        let mut aabb_slices = Vec::with_capacity(buf.len());

        let dev_slice = buf.as_slice();

        for i in 0..buf.len() {
            aabb_slices.push(&dev_slice[i]);
        }
        for i in 0..buf.len() {
            build_inputs.push(CustomPrimitiveArray::new(
                &[&aabb_slices[i]],
                &[GeometryFlags::None],
            )?);
        }

        let accel_options =
            AccelBuildOptions::new(BuildFlags::PREFER_FAST_TRACE | BuildFlags::ALLOW_COMPACTION);
        let gas = Accel::build(ctx, stream, &[accel_options], &build_inputs, true)?;
        // dont need to synchronize, we enqueue the optix launch on the same stream so it will be ordered
        // correctly
        Ok(gas)
    }

    fn build_scene_hitgroup_records(
        scene: &Scene,
        pg_sphere_hitgroup: &ProgramGroup,
    ) -> Result<Vec<SphereHitgroupRecord>> {
        let mut sphere_records_len = 0;
        for obj in scene.objects.iter() {
            match obj {
                Object::Sphere(_) => {
                    sphere_records_len += 1;
                }
            }
        }
        let mut sphere_records = Vec::with_capacity(sphere_records_len);
        for obj in scene.objects.iter() {
            match obj {
                Object::Sphere(sphere) => {
                    sphere_records.push(SphereHitgroupRecord::pack(*sphere, pg_sphere_hitgroup)?);
                }
            }
        }
        Ok(sphere_records)
    }

    pub fn render(&mut self, stream: &Stream, buffers: &mut CudaRendererBuffers) -> Result<()> {
        let dims = buffers.viewport.bounds.as_uvec2();

        let launch_params = LaunchParams {
            image_buf: buffers.accumulated_buffer.as_device_ptr().as_mut_ptr(),
            size: dims,
            scene: Scene {
                objects: &buffers.objects,
                materials: &buffers.materials,
            },
            viewport: buffers.viewport,
            rand_states: buffers.rand_states.as_mut_ptr(),
            handle: unsafe { std::mem::transmute(self.gas.handle()) },
        };

        unsafe {
            let d_launch_params = DeviceBox::new_async(&launch_params, &stream)?;

            optix::launch(
                &self.pipeline,
                stream,
                &d_launch_params,
                &self.sbt,
                dims.x,
                dims.y,
                1,
            )?;

            d_launch_params.drop_async(stream)?;
        }

        Ok(())
    }
}
