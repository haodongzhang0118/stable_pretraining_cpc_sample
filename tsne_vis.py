"""
t-SNE visualization for CPC features
Generates 2x2 plot: frame-level (top) and chunk-level (bottom) for encoder and GRU features
"""
import torch
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from model.backbone import cpc_backbone
from dataloader.dataset import SparseBurstChunkDataset


def load_model(ckpt_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = cpc_backbone()
    
    state_dict = checkpoint.get('state_dict', checkpoint)
    backbone_state_dict = {k.replace('backbone.', ''): v 
                          for k, v in state_dict.items() if k.startswith('backbone.')}
    
    model.load_state_dict(backbone_state_dict, strict=False)
    model.to(device).eval()
    return model


def select_balanced_samples(dataset, num_positive):
    """Select balanced positive and negative samples far from signal regions
    
    Args:
        dataset: SparseBurstChunkDataset instance
        num_positive: Number of positive samples to select
    
    Returns:
        List of indices for balanced positive and negative samples
    """
    positive_indices = []
    far_negative_indices = []  # Negative samples far from signal
    
    # Find positive and far negative samples
    for idx, (file_idx, start_t, end_t) in enumerate(dataset.index_map):
        info = dataset.metadata[file_idx]
        burst_start = info["burst_start_sec"]
        burst_end = burst_start + info["burst_duration_sec"]
        
        # Check overlap
        overlap_start = max(start_t, burst_start)
        overlap_end = min(end_t, burst_end)
        overlap_dur = max(0.0, overlap_end - overlap_start)
        overlap_frac = overlap_dur / dataset.chunk_sec
        
        if overlap_frac > dataset.signal_threshold:
            positive_indices.append(idx)
        elif overlap_frac == 0:
            # Only select negatives far from signal (>2 chunk lengths away)
            distance_to_signal = min(abs(start_t - burst_end), abs(end_t - burst_start))
            if distance_to_signal > 2 * dataset.chunk_sec:
                far_negative_indices.append(idx)
    
    # Random selection
    np.random.seed(42)
    selected_positive = np.random.choice(positive_indices, 
                                        size=min(num_positive, len(positive_indices)), 
                                        replace=False)
    selected_negative = np.random.choice(far_negative_indices, 
                                        size=min(num_positive, len(far_negative_indices)), 
                                        replace=False)
    
    indices = np.concatenate([selected_positive, selected_negative])
    labels = np.concatenate([np.ones(len(selected_positive)), np.zeros(len(selected_negative))])
    
    print(f"Selected {len(selected_positive)} positive + {len(selected_negative)} negative = {len(indices)} samples")
    return indices, labels


def extract_features_subset(model, dataset, indices, device='cuda'):
    """Extract features for selected samples with frame-level timestamps"""
    encoder_features = []
    gru_features = []
    frame_timestamps = []  # Store (file_idx, chunk_start_sec, frame_idx) for each frame
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[int(idx)]
            file_idx, chunk_start_sec, chunk_end_sec = dataset.index_map[int(idx)]
            
            audio = sample['raw_audio'].unsqueeze(0).unsqueeze(1).to(device)  # [1, 1, T]
            
            gru_feat, encoder_feat = model(audio)  # [1, T', C]
            num_frames = encoder_feat.shape[1]
            
            encoder_features.append(encoder_feat.squeeze(0).cpu())  # [T', 512]
            gru_features.append(gru_feat.squeeze(0).cpu())  # [T', 256]
            
            # Store timestamp info for each frame
            for frame_idx in range(num_frames):
                frame_timestamps.append((file_idx, chunk_start_sec, chunk_end_sec, frame_idx, num_frames))
    
    return torch.stack(encoder_features), torch.stack(gru_features), frame_timestamps


def compute_frame_labels(frame_timestamps, dataset):
    """Compute precise label for each frame based on signal overlap
    
    Args:
        frame_timestamps: List of (file_idx, chunk_start_sec, chunk_end_sec, frame_idx, num_frames)
        dataset: Dataset instance with metadata
    
    Returns:
        labels: [B*T] array with 1 for frames containing signal, 0 otherwise
    """
    labels = []
    
    for file_idx, chunk_start_sec, chunk_end_sec, frame_idx, num_frames in frame_timestamps:
        info = dataset.metadata[file_idx]
        burst_start = info["burst_start_sec"]
        burst_end = burst_start + info["burst_duration_sec"]
        
        # Calculate time range for this specific frame
        chunk_duration = chunk_end_sec - chunk_start_sec
        frame_duration = chunk_duration / num_frames
        frame_start = chunk_start_sec + frame_idx * frame_duration
        frame_end = frame_start + frame_duration
        
        # Check if this frame overlaps with signal
        overlap_start = max(frame_start, burst_start)
        overlap_end = min(frame_end, burst_end)
        overlap = max(0.0, overlap_end - overlap_start)
        
        # Frame is positive if it has any overlap with signal
        label = 1 if overlap > 0 else 0
        labels.append(label)
    
    return np.array(labels)


def compute_tsne(features, labels, perplexity=30):
    """Compute t-SNE embedding
    
    Args:
        features: [N, D] tensor
        labels: [N] array
        perplexity: t-SNE perplexity parameter
    
    Returns:
        embedded: [N, 2] array
        labels: [N] array
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embedded = tsne.fit_transform(features.numpy())
    return embedded, labels


def plot_tsne_grid(encoder_chunk, gru_chunk, encoder_frame, gru_frame, 
                   labels_chunk, labels_frame, save_path='tsne_visualization.png'):
    """Create 2x2 grid of t-SNE plots
    
    Layout:
        Row 1 (Frame-level): encoder | gru
        Row 2 (Chunk-level): encoder | gru
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Color mapping
    colors = ['blue', 'red']
    label_names = ['Negative', 'Positive']
    
    # Row 1: Frame-level
    for col, (data, title) in enumerate([(encoder_frame, 'Encoder'), (gru_frame, 'GRU')]):
        ax = axes[0, col]
        for label_idx in [0, 1]:
            mask = labels_frame == label_idx
            ax.scatter(data[mask, 0], data[mask, 1], 
                      c=colors[label_idx], label=label_names[label_idx],
                      alpha=0.6, s=20, edgecolors='none')
        ax.set_title(f'Frame-level: {title} Features', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: Chunk-level
    for col, (data, title) in enumerate([(encoder_chunk, 'Encoder'), (gru_chunk, 'GRU')]):
        ax = axes[1, col]
        for label_idx in [0, 1]:
            mask = labels_chunk == label_idx
            ax.scatter(data[mask, 0], data[mask, 1], 
                      c=colors[label_idx], label=label_names[label_idx],
                      alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
        ax.set_title(f'Chunk-level: {title} Features', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    # Configuration
    method = "BN"
    CHECKPOINT_PATH = f"/root/stable_pretraining_cpc_sample/checkpoints/{method}/best.ckpt"
    CONFIG_PATH = "configs/cpc.yaml"
    NUM_POSITIVE = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("t-SNE Visualization for CPC Features")
    print("=" * 60)
    
    # Load config and dataset
    cfg = OmegaConf.load(CONFIG_PATH)
    dataset = SparseBurstChunkDataset(
        dataset_dir=cfg.data.test_dataset_dir,
        chunk_sec=cfg.data.chunk_sec,
        overlap=cfg.data.overlap,
        signal_threshold=cfg.data.signal_threshold
    )
    
    # Select balanced samples
    indices, labels = select_balanced_samples(dataset, NUM_POSITIVE)
    
    # Load model and extract features
    model = load_model(CHECKPOINT_PATH, device=DEVICE)
    encoder_feats, gru_feats, frame_timestamps = extract_features_subset(model, dataset, indices, DEVICE)
    
    # Chunk-level: average over time [B, T, C] -> [B, C]
    encoder_chunk = encoder_feats.mean(dim=1)  # [B, 512]
    gru_chunk = gru_feats.mean(dim=1)  # [B, 256]
    
    # Frame-level: flatten [B, T, C] -> [B*T, C]
    B, T, _ = encoder_feats.shape
    encoder_frame = encoder_feats.reshape(-1, encoder_feats.shape[-1])  # [B*T, 512]
    gru_frame = gru_feats.reshape(-1, gru_feats.shape[-1])  # [B*T, 256]
    
    # Compute precise frame-level labels based on signal overlap
    labels_frame = compute_frame_labels(frame_timestamps, dataset)  # [B*T]
    
    print(f"\nChunk-level shapes: encoder {encoder_chunk.shape}, gru {gru_chunk.shape}")
    print(f"Chunk-level labels: {labels.sum():.0f} positive / {len(labels)} total")
    print(f"\nFrame-level shapes: encoder {encoder_frame.shape}, gru {gru_frame.shape}")
    print(f"Frame-level labels: {labels_frame.sum():.0f} positive / {len(labels_frame)} total")
    
    # Compute t-SNE embeddings
    print("\nComputing t-SNE for chunk-level features...")
    encoder_chunk_tsne, _ = compute_tsne(encoder_chunk, labels)
    gru_chunk_tsne, _ = compute_tsne(gru_chunk, labels)
    
    print("Computing t-SNE for frame-level features...")
    encoder_frame_tsne, _ = compute_tsne(encoder_frame, labels_frame, perplexity=50)
    gru_frame_tsne, _ = compute_tsne(gru_frame, labels_frame, perplexity=50)
    
    # Plot all results
    plot_tsne_grid(
        encoder_chunk_tsne, gru_chunk_tsne,
        encoder_frame_tsne, gru_frame_tsne,
        labels, labels_frame,
        save_path=f'vis/{method}_tsne_visualization.png'
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

