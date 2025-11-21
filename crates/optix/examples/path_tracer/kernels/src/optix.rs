#![allow(dead_code, unused_imports)]

use crate::{
    hittable::{HitRecord, Hittable},
    material::Material,
    render::{self, generate_ray},
    scene::{Scene, MAX_BOUNCES},
    sphere::Sphere,
    Ray, Vec2, Vec3, Viewport,
};
use cuda_std::kernel;
use cust_core::DeviceCopy;
use gpu_rand::{DefaultRand, GpuRand};
use optix_device::{
    closesthit, get_launch_index,
    glam::{UVec2, Vec3Swizzles},
    intersection, payload, raygen, sbt_data,
    trace::{RayFlags, TraversableHandle},
    util::{get_vec3_attributes, pack_pointer, unpack_pointer},
};

extern "C" {
    #[cfg(target_os = "cuda")]
    #[cfg_attr(
        target_os = "cuda",
        nvvm_internal::addrspace(4),
        allow(improper_ctypes)
    )]
    static PARAMS: LaunchParams<'static>;
}

#[derive(Clone, Copy)]
pub struct LaunchParams<'a> {
    pub image_buf: *mut Vec3,
    pub size: UVec2,
    pub scene: Scene<'a>,
    pub viewport: Viewport,
    pub rand_states: *mut DefaultRand,
    pub handle: TraversableHandle,
}

unsafe impl DeviceCopy for LaunchParams<'_> {}

struct PerRayData {
    pub hit: bool,
    pub attenuation: Vec3,
    pub scattered: Option<Ray>,
    pub rand: *mut DefaultRand,
}

fn get_prd() -> *mut PerRayData {
    unsafe { unpack_pointer(payload::get_payload(0), payload::get_payload(1)) }
}

#[cfg(feature = "optix")]
#[kernel]
pub unsafe fn __miss__miss() {
    // nothing to do
}

#[cfg(feature = "optix")]
#[kernel]
pub unsafe fn __intersection__sphere() {
    let sphere = sbt_data::<Sphere>();
    let ray = Ray::from_optix();
    let tmin = intersection::ray_tmin();
    let tmax = intersection::ray_tmax();

    if let Some(hit) = sphere.hit(ray, tmin, tmax) {
        // you could also recompute these values in the closesthit pretty easily. But optix provides us
        // 7 32-bit attribute regs which are perfect for passing these values.
        let n = hit.normal.to_array().map(|x| x.to_bits());
        let p = hit.point.to_array().map(|x| x.to_bits());
        let mat = hit.material_handle as u32;
        intersection::report_intersection(hit.t, 0, [n[0], n[1], n[2], p[0], p[1], p[2], mat]);
    }
}

#[cfg(feature = "optix")]
#[kernel]
pub unsafe fn __closesthit__sphere() {
    let hit_t = closesthit::ray_tmax();
    let normal = Vec3::from(get_vec3_attributes(0).to_array());
    let hit_point = Vec3::from(get_vec3_attributes(3).to_array());
    let material_id = closesthit::get_attribute(6);

    let material = PARAMS.scene.materials[material_id as usize];
    let prd = get_prd();
    let incoming = Ray::from_optix();

    let hit_record = HitRecord {
        t: hit_t,
        point: hit_point,
        normal,
        material_handle: material_id as usize,
    };

    let (attenuation, scattered) = material.scatter(incoming, hit_record, &mut *(*prd).rand);
    (*prd).attenuation = attenuation;
    (*prd).scattered = scattered;
    (*prd).hit = true;
}

#[cfg(feature = "optix")]
#[kernel]
pub unsafe fn __raygen__render() {
    let i = get_launch_index().xy();
    let size = PARAMS.size;

    let idx = i.x + i.y * size.x;

    let rng = PARAMS.rand_states.add(idx as usize);
    let offset = (*rng).normal_f32_2();
    let mut cur_ray = generate_ray(i, &PARAMS.viewport, offset.into());

    let mut attenuation = Vec3::ONE;
    let mut color = Vec3::ZERO;

    for _ in 0..MAX_BOUNCES {
        let mut prd = PerRayData {
            hit: false,
            attenuation,
            rand: rng,
            scattered: None,
        };

        let (mut p0, mut p1) = pack_pointer(core::ptr::addr_of_mut!(prd));

        raygen::trace(
            PARAMS.handle,
            cur_ray.origin,
            cur_ray.dir,
            0.001,
            1e20,
            0.0,
            255,
            RayFlags::empty(),
            0,
            1,
            0,
            [&mut p0, &mut p1],
        );

        if !prd.hit {
            let miss_color = render::color(cur_ray);
            color = attenuation * miss_color;
            break;
        }

        if let Some(scatter) = prd.scattered {
            cur_ray = scatter;
            attenuation *= prd.attenuation;
        } else {
            break;
        }
    }

    *PARAMS.image_buf.add(idx as usize) += color;
}
