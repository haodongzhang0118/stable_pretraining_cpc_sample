import os
import math
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SparseBurstChunkDataset(Dataset):
    """
    PyTorch Dataset that splits 60s synthetic signals into overlapping 3s chunks.

    Each 3s chunk has a label:
        1 → more than 10% of the chunk overlaps with the signal burst
        0 → otherwise
    """

    def __init__(
        self,
        dataset_dir,
        chunk_sec=3.0,
        overlap=0.5,
        signal_threshold=0.05,
        sample_rate=16000,
        transform=None,
    ):
        """
        Args:
            dataset_dir (str): Directory containing the generated dataset (.pt files and metadata.pt)
            chunk_sec (float): Duration of each chunk in seconds
            overlap (float): Fractional overlap between consecutive chunks (e.g., 0.5 = 50%)
            signal_threshold (float): Label threshold (fraction of chunk covered by signal)
            sample_rate (int): Sampling rate of the audio
            transform (callable, optional): Optional transform to apply to each chunk
        """
        self.dataset_dir = dataset_dir
        self.chunk_sec = chunk_sec
        self.overlap = overlap
        self.signal_threshold = signal_threshold
        self.sample_rate = sample_rate
        self.transform = transform

        # Load metadata
        self.metadata = torch.load(os.path.join(dataset_dir, "metadata.pt"))

        # Precompute index mapping: (file_index, chunk_start_sec)
        self.index_map = []
        for file_idx, info in tqdm(enumerate(self.metadata), total=len(self.metadata), desc="Loading dataset"):
            total_sec = info["duration"]
            stride_sec = self.chunk_sec * (1 - self.overlap)
            num_chunks = math.floor((total_sec - self.chunk_sec) / stride_sec) + 1
            for n in range(num_chunks):
                start_t = n * stride_sec
                end_t = start_t + self.chunk_sec
                self.index_map.append((file_idx, start_t, end_t))

        print(f"✅ Loaded {len(self.index_map)} chunks from {len(self.metadata)} long samples.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, start_t, end_t = self.index_map[idx]
        info = self.metadata[file_idx]
        filename = info["file"]
        filepath = os.path.join(self.dataset_dir, filename)
        audio = torch.load(filepath)

        # Extract waveform segment
        start_samp = int(start_t * self.sample_rate)
        end_samp = int(end_t * self.sample_rate)
        chunk = audio[start_samp:end_samp]

        # Compute overlap fraction with burst
        burst_start = info["burst_start_sec"]
        burst_end = burst_start + info["burst_duration_sec"]

        overlap_start = max(start_t, burst_start)
        overlap_end = min(end_t, burst_end)
        overlap_dur = max(0.0, overlap_end - overlap_start)
        overlap_frac = overlap_dur / self.chunk_sec

        # Assign label
        label = 1 if overlap_frac > self.signal_threshold else 0

        if self.transform:
            chunk = self.transform(chunk)

        # Return dictionary format for stable-pretraining framework
        return {
            "raw_audio": chunk,
            "label": torch.tensor(label, dtype=torch.long)  # MulticlassAccuracy needs long, will be converted to float for BCE loss
        }
