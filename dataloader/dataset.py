import os
import pandas as pd  # Required to read .csv files
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split

class ESC50LocalDataset(Dataset):
    """
    A basic PyTorch Dataset for loading local ESC-50 files.
    It is only responsible for reading the CSV and returning audio paths and labels.
    """
    def __init__(self, meta_csv, audio_dir):
        self.df = pd.read_csv(meta_csv)
        self.audio_dir = audio_dir
        # Convert labels to tensors for convenience
        self.labels = torch.tensor(self.df['target'].values).long()
        self.files = self.df['filename'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get filename and label
        filename = self.files[idx]
        label = self.labels[idx]
        
        # Construct the full file path
        audio_path = os.path.join(self.audio_dir, filename)
        
        # Return a dictionary to maintain compatibility with your WindowedDataset
        return {"audio_path": audio_path, "target": label}


class ESC50WindowedDataset(Dataset):
    """
    ESC-50 PyTorch Dataset with deterministic sliding windows.
    (This class is almost identical to your original version, 
     only __getitem__ is modified to get the file path.)
    """

    def __init__(
        self,
        split_dataset,        # A subset from random_split(ESC50LocalDataset)
        target_sr=16000,
        window_sec=1.0,
        hop_sec=0.5,
    ):
        self.ds = split_dataset  # This is a torch.utils.data.Subset object
        self.target_sr = target_sr
        self.window_len = int(target_sr * window_sec)
        self.hop_len = int(target_sr * hop_sec)
        # Assume original SR is 44100 Hz (ESC-50 standard)
        self.resampler = torchaudio.transforms.Resample(44100, target_sr)

        # Pre-compute an index map: (clip_index, start_sample)
        self.index_map = []
        # self.ds is a Subset, len(self.ds) is the size of this subset
        for clip_idx in range(len(self.ds)):
            total_len = int(target_sr * 5.0) # All ESC-50 clips are 5s
            for start in range(0, total_len - self.window_len + 1, self.hop_len):
                # clip_idx is the index within the *subset*
                self.index_map.append((clip_idx, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # (clip_idx, start) corresponds to an index in the *subset*
        clip_idx, start = self.index_map[idx]
        
        # Get the original data item ({"audio_path": ..., "target": ...}) from the subset
        item = self.ds[clip_idx]

        # *** This is the only key change ***
        # Your original code: path = item["audio"]["path"]
        # New code:
        path = item["audio_path"]
        
        # --- The following logic is identical to your original code ---
        
        # Load waveform
        wav, sr = torchaudio.load(path)

        # Resample to target_sr if needed
        if sr != self.target_sr:
            wav = self.resampler(wav)

        # Pad to exactly 5 seconds if needed
        target_total = int(self.target_sr * 5.0)
        if wav.size(-1) < target_total:
            wav = torch.nn.functional.pad(wav, (0, target_total - wav.size(-1)))
        # Also truncate, just in case (though ESC-50 clips shouldn't exceed 5s)
        elif wav.size(-1) > target_total:
             wav = wav[..., :target_total]


        # Extract 1-s window
        window = wav[..., start:start + self.window_len]

        label = item["target"] # This is already a tensor
        return window.squeeze(0), label  # squeeze(0) -> [T] instead of [1, T]


def get_esc50_dataloaders(
    meta_csv_path,  
    audio_dir_path, 
    batch_size=512,
    num_workers=4,
    seed=42,
    target_sr=16000,
    window_sec=1.0,
    hop_sec=0.5,
):
    """
    Load local ESC-50 and create train/validation dataloaders (80/20 split).
    """
    # *** This is the key modification ***
    # 1. We no longer use load_dataset
    # full = load_dataset("ashraq/esc50", split="train") 
    
    # 2. Instantiate our local Dataset
    full_ds = ESC50LocalDataset(meta_csv=meta_csv_path, audio_dir=audio_dir_path)

    # Reproducible 80/20 split (this logic is unchanged)
    g = torch.Generator().manual_seed(seed)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_subset, val_subset = random_split(full_ds, [train_size, val_size], generator=g)

    # Instantiate the windowed Dataset (this logic is unchanged)
    train_ds = ESC50WindowedDataset(train_subset, target_sr, window_sec, hop_sec)
    val_ds = ESC50WindowedDataset(val_subset, target_sr, window_sec, hop_sec)

    # Create DataLoaders (this logic is unchanged)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# --- Example Usage ---
if __name__ == "__main__":
    
    # *** Fill in your local paths here ***
    BASE_PATH = "/Users/zhanghaodong/Desktop/ESC-50-master"
    META_CSV = os.path.join(BASE_PATH, "meta", "esc50.csv")
    AUDIO_DIR = os.path.join(BASE_PATH, "audio")

    # Check if paths exist
    if not os.path.exists(META_CSV) or not os.path.exists(AUDIO_DIR):
        print(f"Error: Paths not found. Please check your BASE_PATH.")
        print(f"Checking for META_CSV: {META_CSV}")
        print(f"Checking for AUDIO_DIR: {AUDIO_DIR}")
    else:
        print(f"Loading from path: {BASE_PATH}")
        
        train_loader, val_loader = get_esc50_dataloaders(
            meta_csv_path=META_CSV,    # <-- Pass the path
            audio_dir_path=AUDIO_DIR,  # <-- Pass the path
            batch_size=32,
            num_workers=4,
            target_sr=16000,
            window_sec=1.0,
            hop_sec=0.25,
        )

        # Original ESC-50 has 2000 5s clips
        # 80% train = 1600 clips
        # 20% val = 400 clips
        # 1s window, 0.5s hop => 9 windows per clip
        # Expected Train windows: 1600 * 9 = 14400
        # Expected Val windows: 400 * 9 = 3600
        print(f"Train windows: {len(train_loader.dataset)},  Val windows: {len(val_loader.dataset)}")
        
        print("\n--- Checking Train Loader ---")
        for x, y in train_loader:
            # Expected x.shape: [32, 16000], y.shape: [32]
            print(f"Batch shape: {x.shape}, Label shape: {y.shape}")
            print(f"Example labels: {y[:5]}")
            break
            
        print("\n--- Checking Val Loader ---")
        for x, y in val_loader:
            # Expected x.shape: [32, 16000], y.shape: [32]
            print(f"Batch shape: {x.shape}, Label shape: {y.shape}")
            print(f"Example labels: {y[:5]}")
            break