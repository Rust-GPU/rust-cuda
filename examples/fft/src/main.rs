use std::f32::consts::PI;

use cufft::{Direction, FftPlan, FftType};
use cufft_raw::float2;
use cust::memory::{CopyDestination, DeviceBuffer};

const FFT_SIZE: usize = 1024;
const FREQUENCY_BIN: usize = 13;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;

    // Build a complex sinusoid `x[n] = exp(j * 2 * pi * FREQUENCY_BIN * n / FFT_SIZE)`

    let host_signal: Vec<float2> = (0..FFT_SIZE)
        .map(|n| {
            let phase = 2.0 * PI * FREQUENCY_BIN as f32 * n as f32 / FFT_SIZE as f32;

            float2 {
                x: phase.cos(),
                y: phase.sin(),
            }
        })
        .collect();

    let device_signal = DeviceBuffer::from_slice(&host_signal)?;
    let mut device_spectrum = unsafe { DeviceBuffer::<float2>::uninitialized(FFT_SIZE)? };

    let plan = FftPlan::plan_1d(FFT_SIZE as i32, FftType::C2C, 1)?;

    // Forward FFT.

    plan.exec_c2c(&device_signal, &mut device_spectrum, Direction::Forward)?;

    let mut host_spectrum = vec![float2 { x: 0.0, y: 0.0 }; FFT_SIZE];
    device_spectrum.copy_to(&mut host_spectrum)?;

    // The energy should be concentrated at FREQUENCY_BIN.

    let peak = host_spectrum
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| {
            let norm = |vector: &float2| vector.x * vector.x + vector.y * vector.y;

            norm(lhs).partial_cmp(&norm(rhs)).unwrap()
        })
        .map(|(bin, _)| bin)
        .unwrap();

    println!("input frequency bin   : {FREQUENCY_BIN}");
    println!("peak frequency bin    : {peak}");

    assert_eq!(peak, FREQUENCY_BIN, "Input frequency bin does not equal peak frequency bin.");

    // Inverse FFT then normalize by FFT_SIZE to recover the original signal.

    let mut device_recovered = unsafe { DeviceBuffer::<float2>::uninitialized(FFT_SIZE)? };

    plan.exec_c2c(&device_spectrum, &mut device_recovered, Direction::Inverse)?;

    let mut host_recovered = vec![float2 { x: 0.0, y: 0.0 }; FFT_SIZE];

    device_recovered.copy_to(&mut host_recovered)?;

    let max_err = host_signal
        .iter()
        .zip(host_recovered.iter())
        .map(|(lhs, rhs)| {
            let dx = rhs.x / FFT_SIZE as f32 - lhs.x;
            let dy = rhs.y / FFT_SIZE as f32 - lhs.y;

            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0f32, f32::max);

    println!("Round-trip max error: {max_err}");

    assert!(max_err < 1e-5, "Round-trip error too large: {max_err}");

    Ok(())
}
