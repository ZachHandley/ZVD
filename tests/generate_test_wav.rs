//! Generate a test WAV file for testing

use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    // Generate a 1-second 440Hz sine wave (A4 note)
    let sample_rate = 44100u32;
    let duration = 1.0; // seconds
    let frequency = 440.0; // Hz
    let amplitude = 0.5; // 50% volume

    let num_samples = (sample_rate as f32 * duration) as usize;

    // Generate samples
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (amplitude * (2.0 * PI * frequency * t).sin() * 32767.0) as i16;
        samples.push(sample);
    }

    // Write WAV file
    let file = File::create("test_440hz.wav")?;
    let mut writer = std::io::BufWriter::new(file);

    // RIFF header
    writer.write_all(b"RIFF")?;
    let file_size = 36 + (samples.len() * 2) as u32;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // fmt chunk size
    writer.write_all(&1u16.to_le_bytes())?; // PCM format
    writer.write_all(&1u16.to_le_bytes())?; // mono
    writer.write_all(&sample_rate.to_le_bytes())?;
    let byte_rate = sample_rate * 2; // 16-bit mono
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&2u16.to_le_bytes())?; // block align
    writer.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk
    writer.write_all(b"data")?;
    let data_size = (samples.len() * 2) as u32;
    writer.write_all(&data_size.to_le_bytes())?;

    // Write sample data
    for sample in samples {
        writer.write_all(&sample.to_le_bytes())?;
    }

    writer.flush()?;

    println!("Generated test_440hz.wav:");
    println!("  Duration: {} second", duration);
    println!("  Sample Rate: {} Hz", sample_rate);
    println!("  Frequency: {} Hz", frequency);
    println!("  Format: 16-bit PCM mono");
    println!("  File Size: {} bytes", file_size + 8);

    // Also generate a stereo version
    generate_stereo_wav()?;

    Ok(())
}

fn generate_stereo_wav() -> std::io::Result<()> {
    let sample_rate = 44100u32;
    let duration = 2.0; // seconds
    let freq_left = 440.0; // A4
    let freq_right = 554.37; // C#5
    let amplitude = 0.5;

    let num_samples = (sample_rate as f32 * duration) as usize;

    let file = File::create("test_stereo.wav")?;
    let mut writer = std::io::BufWriter::new(file);

    // RIFF header
    writer.write_all(b"RIFF")?;
    let file_size = 36 + (num_samples * 4) as u32; // 4 bytes per frame (2 channels * 2 bytes)
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?; // PCM
    writer.write_all(&2u16.to_le_bytes())?; // stereo
    writer.write_all(&sample_rate.to_le_bytes())?;
    let byte_rate = sample_rate * 4; // 16-bit stereo
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&4u16.to_le_bytes())?; // block align
    writer.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk
    writer.write_all(b"data")?;
    let data_size = (num_samples * 4) as u32;
    writer.write_all(&data_size.to_le_bytes())?;

    // Write interleaved stereo samples
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        let left = (amplitude * (2.0 * PI * freq_left * t).sin() * 32767.0) as i16;
        let right = (amplitude * (2.0 * PI * freq_right * t).sin() * 32767.0) as i16;

        writer.write_all(&left.to_le_bytes())?;
        writer.write_all(&right.to_le_bytes())?;
    }

    writer.flush()?;

    println!("\nGenerated test_stereo.wav:");
    println!("  Duration: {} seconds", duration);
    println!("  Sample Rate: {} Hz", sample_rate);
    println!("  Left Channel: {} Hz", freq_left);
    println!("  Right Channel: {} Hz", freq_right);
    println!("  Format: 16-bit PCM stereo");
    println!("  File Size: {} bytes", file_size + 8);

    Ok(())
}
