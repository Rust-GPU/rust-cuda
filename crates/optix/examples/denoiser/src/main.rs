use cust::memory::DeviceBuffer;
use cust::prelude::{Stream, StreamFlags};
use cust::util::SliceExt;
use glam::Vec3;
use image::ImageReader;
use optix::context::DeviceContext;
use optix::denoiser::{Denoiser, DenoiserModelKind, DenoiserParams, Image, ImageFormat};
use std::error::Error;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(
    name = "denoiser",
    about = "Denoises an input image using the OptiX AI Denoiser"
)]
struct Opt {
    /// Input image to denoise.
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    let name = opt
        .input
        .file_name()
        .expect("input was not a file")
        .to_string_lossy()
        .to_string();
    let img = ImageReader::open(opt.input)?.decode()?;

    let mut rgb = img.into_rgb8();
    let mut linear = vec![Vec3::ZERO; rgb.as_raw().len() / 3];

    let width = rgb.width();
    let height = rgb.height();

    rgb.pixels()
        .zip(linear.iter_mut())
        .for_each(|(rgb, linear)| {
            *linear = Vec3::new(
                rgb.0[0] as f32 / 255.0,
                rgb.0[1] as f32 / 255.0,
                rgb.0[2] as f32 / 255.0,
            );
        });

    // set up CUDA and OptiX then make the needed structs/contexts.
    let cuda_ctx = cust::quick_init()?;
    optix::init()?;
    let optix_ctx = DeviceContext::new(&cuda_ctx, false)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // set up the denoiser, choosing Ldr as our model because our colors are in
    // the 0.0 - 1.0 range.
    let mut denoiser = Denoiser::new(&optix_ctx, DenoiserModelKind::Ldr, Default::default())?;

    // setup the optix state for our required image dimensions. this allocates the required
    // state and scratch memory for further invocations.
    denoiser.setup_state(&stream, width, height, false)?;

    // allocate the buffer for the noisy image and copy the data to the GPU.
    let in_buf = linear.as_slice().as_dbuf()?;

    let mut out_buf = DeviceBuffer::<Vec3>::zeroed((width * height) as usize)?;

    // make an image to tell OptiX about how our image buffer is represented
    let input_image = Image::new(&in_buf, ImageFormat::Float3, width, height);

    // Invoke the denoiser on the image. OptiX will queue up the work on the
    // CUDA stream.
    denoiser.invoke(
        &stream,
        Default::default(),
        input_image,
        DenoiserParams::default(),
        &mut out_buf,
    )?;

    // Finally, synchronize the stream to wait until the denoiser is finished doing its work.
    stream.synchronize()?;

    // copy back the data from the gpu.
    let denoised = out_buf.as_host_vec()?;

    // finally, reuse the existing (noisy) rgb buffer we have and write to it.
    denoised
        .into_iter()
        .zip(rgb.pixels_mut())
        .for_each(|(linear, rgb)| {
            let transformed = (linear * 255.0).clamp(Vec3::ZERO, Vec3::splat(255.0));
            rgb.0 = [
                transformed.x as u8,
                transformed.y as u8,
                transformed.z as u8,
            ];
        });

    // ...and then save the image
    rgb.save(format!("./{}_denoised.png", name))?;

    Ok(())
}
