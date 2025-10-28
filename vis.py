"""
可视化和特征提取脚本
用于加载训练好的CPC模型，提取encoder和GRU的特征
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.backbone import cpc_backbone
from dataloader.dataset import SparseBurstChunkDataset


def load_model_from_checkpoint(ckpt_path, device='cuda'):
    """加载训练好的模型
    
    Args:
        ckpt_path: checkpoint文件路径
        device: 运行设备 ('cuda' 或 'cpu')
    
    Returns:
        model: 加载好的backbone模型
    """
    # 加载checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 创建backbone模型
    model = cpc_backbone()
    
    # 从checkpoint中提取backbone的权重
    # stable-pretraining会将模型保存在 'state_dict' 键中
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 过滤出backbone相关的权重（移除前缀如 'backbone.'）
    backbone_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            # 移除 'backbone.' 前缀
            new_key = key.replace('backbone.', '')
            backbone_state_dict[new_key] = value
    
    # 加载权重到模型
    model.load_state_dict(backbone_state_dict, strict=False)
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print(f"✅ 成功加载模型: {ckpt_path}")
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def extract_features(model, dataloader, num_samples=100, device='cuda'):
    """提取encoder和GRU特征
    
    Args:
        model: CPC backbone模型
        dataloader: 数据加载器
        num_samples: 要处理的样本数量
        device: 运行设备
    
    Returns:
        dict: 包含特征和标签的字典
            - 'encoder_features': List of encoder outputs (B, T, 512)
            - 'gru_features': List of GRU outputs (B, T, 256)
            - 'raw_audio': List of raw audio (B, T)
            - 'labels': List of labels
    """
    encoder_features = []
    gru_features = []
    raw_audios = []
    labels = []
    
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            if total_samples >= num_samples:
                break
            
            # 获取数据
            audio = batch['raw_audio'].to(device)  # [B, T]
            label = batch['label']  # [B]
            
            # 添加channel维度: [B, T] -> [B, 1, T]
            audio_input = audio.unsqueeze(1)
            
            # 前向传播获取特征
            # output: GRU特征 [B, T', 256]
            # z: Encoder特征 [B, T', 512]
            gru_output, encoder_output = model(audio_input)
            
            # 转移到CPU并保存
            encoder_features.append(encoder_output.cpu())
            gru_features.append(gru_output.cpu())
            raw_audios.append(audio.cpu())
            labels.append(label)
            
            total_samples += audio.size(0)
            
            if total_samples >= num_samples:
                # 截断最后一个batch
                excess = total_samples - num_samples
                if excess > 0:
                    encoder_features[-1] = encoder_features[-1][:-excess]
                    gru_features[-1] = gru_features[-1][:-excess]
                    raw_audios[-1] = raw_audios[-1][:-excess]
                    labels[-1] = labels[-1][:-excess]
                break
    
    # 合并所有batch
    results = {
        'encoder_features': torch.cat(encoder_features, dim=0),  # [N, T', 512]
        'gru_features': torch.cat(gru_features, dim=0),          # [N, T', 256]
        'raw_audio': torch.cat(raw_audios, dim=0),               # [N, T]
        'labels': torch.cat(labels, dim=0)                       # [N]
    }
    
    print(f"\n📊 特征提取完成:")
    print(f"   - Encoder features shape: {results['encoder_features'].shape}")
    print(f"   - GRU features shape: {results['gru_features'].shape}")
    print(f"   - Raw audio shape: {results['raw_audio'].shape}")
    print(f"   - Labels shape: {results['labels'].shape}")
    
    return results


def visualize_features(results, save_dir='visualizations', num_examples=5):
    """可视化特征
    
    Args:
        results: extract_features返回的字典
        save_dir: 保存可视化结果的目录
        num_examples: 要可视化的样本数量
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    encoder_feats = results['encoder_features'][:num_examples]  # [N, T, 512]
    gru_feats = results['gru_features'][:num_examples]          # [N, T, 256]
    raw_audio = results['raw_audio'][:num_examples]             # [N, T]
    labels = results['labels'][:num_examples]
    
    for idx in range(num_examples):
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # 1. 原始音频波形
        axes[0].plot(raw_audio[idx].numpy())
        axes[0].set_title(f'Sample {idx} - Raw Audio (Label: {labels[idx].item()})')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Encoder特征 (降维显示前64个通道)
        encoder_2d = encoder_feats[idx].numpy().T[:64]  # [64, T]
        im1 = axes[1].imshow(encoder_2d, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Encoder Features (first 64 channels)')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Feature Channels')
        plt.colorbar(im1, ax=axes[1])
        
        # 3. GRU特征 (显示所有256个通道)
        gru_2d = gru_feats[idx].numpy().T  # [256, T]
        im2 = axes[2].imshow(gru_2d, aspect='auto', origin='lower', cmap='plasma')
        axes[2].set_title('GRU Features (all 256 channels)')
        axes[2].set_xlabel('Time Steps')
        axes[2].set_ylabel('Feature Channels')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{idx}_features.png', dpi=150)
        plt.close()
        
        print(f"✅ 保存可视化: {save_dir / f'sample_{idx}_features.png'}")


def save_features(results, save_path='extracted_features.pt'):
    """保存提取的特征到文件
    
    Args:
        results: extract_features返回的字典
        save_path: 保存路径
    """
    torch.save(results, save_path)
    print(f"\n💾 特征已保存到: {save_path}")
    print(f"   文件大小: {Path(save_path).stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """主函数"""
    # ============ 配置参数 ============
    CHECKPOINT_PATH = "root/checkpoints/BN/last.ckpt"  # 你的checkpoint路径
    CONFIG_PATH = "configs/cpc.yaml"
    NUM_SAMPLES = 100
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("🚀 CPC特征提取工具")
    print("=" * 60)
    print(f"📦 Checkpoint: {CHECKPOINT_PATH}")
    print(f"🔧 Device: {DEVICE}")
    print(f"📊 Samples: {NUM_SAMPLES}")
    print("=" * 60)
    
    # ============ 加载配置 ============
    cfg = OmegaConf.load(CONFIG_PATH)
    
    # ============ 加载数据 ============
    print("\n📂 加载数据...")
    dataset = SparseBurstChunkDataset(
        dataset_dir=cfg.data.val_dataset_dir,  # 使用验证集
        chunk_sec=cfg.data.chunk_sec,
        overlap=cfg.data.overlap,
        signal_threshold=cfg.data.signal_threshold
    )
    
    # 创建dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # ============ 加载模型 ============
    print("\n🔧 加载模型...")
    model = load_model_from_checkpoint(CHECKPOINT_PATH, device=DEVICE)
    
    # ============ 提取特征 ============
    print(f"\n🔍 提取前 {NUM_SAMPLES} 个样本的特征...")
    results = extract_features(
        model=model,
        dataloader=dataloader,
        num_samples=NUM_SAMPLES,
        device=DEVICE
    )
    
    # ============ 保存特征 ============
    save_features(results, save_path='extracted_features.pt')
    
    # ============ 可视化 ============
    print("\n🎨 生成可视化...")
    visualize_features(results, save_dir='visualizations', num_examples=5)
    
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    
    # ============ 使用示例 ============
    print("\n💡 如何使用提取的特征:")
    print("```python")
    print("# 加载特征")
    print("features = torch.load('extracted_features.pt')")
    print()
    print("# 访问不同的特征")
    print("encoder_feats = features['encoder_features']  # [100, T, 512]")
    print("gru_feats = features['gru_features']          # [100, T, 256]")
    print("raw_audio = features['raw_audio']             # [100, T]")
    print("labels = features['labels']                   # [100]")
    print()
    print("# 例如：计算特征的统计信息")
    print("print('Encoder特征均值:', encoder_feats.mean())")
    print("print('GRU特征标准差:', gru_feats.std())")
    print("```")


if __name__ == "__main__":
    main()

