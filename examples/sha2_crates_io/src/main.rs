use cust::memory::DeviceBox;
use cust::prelude::*;
use sha2::{Digest, Sha256, Sha512};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn hex_string(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = cust::quick_init()?;

    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut test_failed = false;

    println!("SHA256 One-Shot API");
    {
        let kernel = module.get_function("sha256_oneshot")?;
        let input_data = b"hello world";
        let input_gpu = DeviceBuffer::from_slice(input_data)?;

        let mut output_data = [0u8; 32];
        let output_gpu = DeviceBox::new(&output_data)?;

        unsafe {
            launch!(
                kernel<<<1, 32, 0, stream>>>(
                    input_gpu.as_device_ptr(),
                    input_gpu.len(),
                    output_gpu.as_device_ptr()
                )
            )?;
        }

        stream.synchronize()?;
        output_gpu.copy_to(&mut output_data)?;

        println!("   GPU: {}", hex_string(&output_data));

        // CPU verification
        let cpu_hash = Sha256::digest(input_data);
        println!("   CPU: {}", hex_string(&cpu_hash));

        if output_data[..] == cpu_hash[..] {
            println!("   Results match");
        } else {
            println!("   ERROR: Results differ");
            test_failed = true;
        }
    }

    println!("\nSHA256 Incremental API");
    {
        let kernel = module.get_function("sha256_incremental")?;
        let input1 = b"hello ";
        let input2 = b"world";
        let input1_gpu = DeviceBuffer::from_slice(input1)?;
        let input2_gpu = DeviceBuffer::from_slice(input2)?;

        let mut output_data = [0u8; 32];
        let output_gpu = DeviceBox::new(&output_data)?;

        unsafe {
            launch!(
                kernel<<<1, 32, 0, stream>>>(
                    input1_gpu.as_device_ptr(),
                    input1_gpu.len(),
                    input2_gpu.as_device_ptr(),
                    input2_gpu.len(),
                    output_gpu.as_device_ptr()
                )
            )?;
        }

        stream.synchronize()?;
        output_gpu.copy_to(&mut output_data)?;

        println!("   GPU: {}", hex_string(&output_data));

        // CPU verification
        let mut hasher = Sha256::new();
        hasher.update(input1);
        hasher.update(input2);
        let cpu_hash = hasher.finalize();

        println!("   CPU: {}", hex_string(&cpu_hash));

        if output_data[..] == cpu_hash[..] {
            println!("   Results match");
        } else {
            println!("   ERROR: Results differ");
            test_failed = true;
        }
    }

    println!("\nSHA512 One-Shot API");
    {
        let kernel = module.get_function("sha512_oneshot")?;
        let input_data = b"hello world";
        let input_gpu = DeviceBuffer::from_slice(input_data)?;

        let mut output_data = [0u8; 64];
        let output_gpu = DeviceBox::new(&output_data)?;

        unsafe {
            launch!(
                kernel<<<1, 32, 0, stream>>>(
                    input_gpu.as_device_ptr(),
                    input_gpu.len(),
                    output_gpu.as_device_ptr()
                )
            )?;
        }

        stream.synchronize()?;
        output_gpu.copy_to(&mut output_data)?;

        println!("   GPU: {}", hex_string(&output_data));

        // CPU verification
        let cpu_hash = Sha512::digest(input_data);
        println!("   CPU: {}", hex_string(&cpu_hash));

        if output_data[..] == cpu_hash[..] {
            println!("   Results match");
        } else {
            println!("   ERROR: Results differ");
            test_failed = true;
        }
    }

    println!("\nSHA512 Incremental API");
    {
        let kernel = module.get_function("sha512_incremental")?;
        let input_data = b"hello world";
        let input_gpu = DeviceBuffer::from_slice(input_data)?;

        let mut output_data = [0u8; 64];
        let output_gpu = DeviceBox::new(&output_data)?;

        unsafe {
            launch!(
                kernel<<<1, 32, 0, stream>>>(
                    input_gpu.as_device_ptr(),
                    input_gpu.len(),
                    output_gpu.as_device_ptr()
                )
            )?;
        }

        stream.synchronize()?;
        output_gpu.copy_to(&mut output_data)?;

        println!("   GPU: {}", hex_string(&output_data));

        // CPU verification
        let mut hasher = Sha512::new();
        hasher.update(input_data);
        let cpu_hash = hasher.finalize();

        println!("   CPU: {}", hex_string(&cpu_hash));

        if output_data[..] == cpu_hash[..] {
            println!("   Results match");
        } else {
            println!("   ERROR: Results differ");
            test_failed = true;
        }
    }

    if test_failed {
        Err("One or more example failed".into())
    } else {
        Ok(())
    }
}
