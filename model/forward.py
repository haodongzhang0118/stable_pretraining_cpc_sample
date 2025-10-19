import torch

def cpc_forward(self, batch, stage):
    """Forward function for CPC (Contrastive Predictive Coding) on audio.

    CPC learns representations by predicting future latent representations
    from past context using a contrastive loss. This implementation follows
    the CDCK2 architecture with a convolutional encoder and GRU context network.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: CPC backbone (encoder + GRU)
            - Wk: List of prediction heads for different time steps
            - cpc_loss: CPC contrastive loss function
            - timestep: Number of future steps to predict
        batch: Input batch dictionary containing:
            - 'raw_audio': Raw audio waveform tensor [B, T]
            - 'label': Optional labels for downstream tasks
        stage: Training stage ('fit', 'validate', 'test', or 'predict')

    Returns:
        Dictionary containing:
            - 'embedding': Pooled representation [B, 256] for downstream tasks
            - 'loss': CPC contrastive loss (during training only)
            - 'CPC_acc': CPC prediction accuracy (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        During training, the backbone outputs temporal features [B, T', 256] which
        are used for CPC loss computation. For consistency with OnlineProbe/KNN,
        we pool these to [B, 256] for the "embedding" output.
    """
    out = {}
    x = batch["raw_audio"].unsqueeze(1)  # [B, T] -> [B, 1, T]
    batch_size = x.size(0)

    # Always get the temporal features from backbone
    output, z = self.backbone(x)  # output: [B, T', 256], z: [B, T', 512]
    
    if self.training:
        seq_len = z.size(1)
        timestep = self.timestep

        # Choose random timestep t (ensure we have enough future steps)
        t_samples = torch.randint(
            seq_len - timestep, size=(1,), device=x.device
        ).long()

        # Extract target representations (future steps to predict)
        encode_samples = torch.empty((timestep, batch_size, 512), device=x.device)
        for i in range(1, timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch_size, 512)

        # Extract context vector at time t
        c_t = output[:, t_samples, :].view(batch_size, 256)

        # Generate predictions for future timesteps using learned projections
        preds = torch.empty((timestep, batch_size, 512), device=x.device)
        for i in range(timestep):
            preds[i] = self.Wk[i](c_t)

        # Compute CPC contrastive loss
        nce, accuracy = self.cpc_loss(preds, encode_samples)
        out["loss"] = nce
        
        # Log CPC accuracy for monitoring
        self.log(
            "train/CPC_acc", 
            accuracy, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            sync_dist=True
        )
        
        out["ori_embeddings"] = output
        # Pool temporal features for consistent embedding shape
        out["embedding"] = output.mean(dim=1)  # [B, T', 256] -> [B, 256]
    else:
        # Validation mode - use pooled embeddings
        # No gradient needed, features are already computed above
        with torch.no_grad():
            out["embedding"] = output.mean(dim=1).detach()  # [B, 256]

    # Pass through labels for callbacks (OnlineProbe/KNN need this)
    if "label" in batch:
        out["label"] = batch["label"]

    return out