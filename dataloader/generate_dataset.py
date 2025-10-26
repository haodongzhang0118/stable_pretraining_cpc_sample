import torch
import numpy as np
import os
from tqdm import tqdm


# --- Base function you already provided ---
def synthesize_random_burst_audio(
    sample_rate=16000,
    duration=60.0,
    burst_len_range=(0.5, 0.65),
    burst_amp_range=(10.0, 20.0),
    burst_freq_hz=800,
    noise_std=0.01,
    have_signal=1,
    seed=42
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_samples = int(sample_rate * duration)
    t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
    noise = noise_std * torch.randn(n_samples)

    if not have_signal:
        return noise, {}

    # Random burst parameters
    burst_duration_sec = float(torch.empty(1).uniform_(*burst_len_range))
    burst_amp = float(torch.empty(1).uniform_(*burst_amp_range))
    max_start_time = duration - burst_duration_sec
    burst_start_sec = float(torch.empty(1).uniform_(0, max_start_time))

    # Insert burst signal
    start_idx = int(burst_start_sec * sample_rate)
    end_idx = start_idx + int(burst_duration_sec * sample_rate)
    burst = torch.zeros(n_samples, dtype=torch.float32)
    if end_idx > start_idx:
        win = torch.hann_window(end_idx - start_idx, periodic=False)
        burst_segment = (
            burst_amp
            * torch.cos(2 * np.pi * burst_freq_hz * t[start_idx:end_idx])
            * win
        )
        burst[start_idx:end_idx] = burst_segment

    audio = noise + burst
    burst_info = {
        "burst_start_sec": burst_start_sec,
        "burst_duration_sec": burst_duration_sec,
        "burst_amp": burst_amp,
    }
    return audio, burst_info


# --- Dataset generator ---
def generate_sparse_burst_dataset(
    output_dir,
    num_samples=10000,
    duration=60.0,
    sample_rate=16000,
    amp_range=(10.0, 15.0),
    noise_std=1.0,
    signal_fraction_range=(0.005, 0.01),
    burst_freq_hz=800,
    base_seed=42,
    save_as="pt"
):
    """
    Generate a dataset of synthetic sparse burst signals.

    Each sample contains a short sinusoidal burst (0.5–1% of total duration)
    embedded in Gaussian noise.

    Args:
        output_dir (str): Directory to save the generated samples.
        num_samples (int): Number of samples to generate.
        duration (float): Length of each sample in seconds.
        sample_rate (int): Sampling rate in Hz.
        amp_range (tuple): Amplitude range for the burst (min, max).
        noise_std (float): Standard deviation of Gaussian noise.
        signal_fraction_range (tuple): Fraction of total duration occupied by the signal (e.g., (0.005, 0.01)).
        burst_freq_hz (float): Frequency of the sinusoidal burst in Hz.
        base_seed (int): Random seed for reproducibility.
        save_as (str): File format: "pt" (PyTorch tensor) or "npy" (NumPy array).
    """
    os.makedirs(output_dir, exist_ok=True)
    infos = []

    for i in tqdm(range(num_samples), desc="Generating dataset"):
        # Compute the allowed burst duration range (in seconds)
        min_burst = signal_fraction_range[0] * duration
        max_burst = signal_fraction_range[1] * duration

        # Generate one sample with a random burst
        audio, info = synthesize_random_burst_audio(
            sample_rate=sample_rate,
            duration=duration,
            burst_len_range=(min_burst, max_burst),
            burst_amp_range=amp_range,
            burst_freq_hz=burst_freq_hz,
            noise_std=noise_std,
            have_signal=1,
            seed=base_seed + i
        )

        # Save the waveform
        filename = f"sample_{i:05d}.{save_as}"
        path = os.path.join(output_dir, filename)
        if save_as == "pt":
            torch.save(audio, path)
        elif save_as == "npy":
            np.save(path, audio.numpy())

        # Store metadata
        info.update({
            "index": i,
            "file": filename,
            "duration": duration,
            "sample_rate": sample_rate
        })
        infos.append(info)

    # Save metadata for the whole dataset
    torch.save(infos, os.path.join(output_dir, "metadata.pt"))
    print(f"\n✅ Dataset generated: {num_samples} samples saved to {output_dir}")


if __name__ == "__main__":
# --- Example usage ---
    generate_sparse_burst_dataset(
        output_dir="./synthetic_dataset",
        num_samples=10000,
        duration=60.0,
        sample_rate=16000,
        amp_range=(2.0, 7.0),
        noise_std=1.0,
        signal_fraction_range=(0.005, 0.01),
        burst_freq_hz=1000,
        base_seed=42,
        save_as="pt"
    )
