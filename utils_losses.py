"""
Utility functions for training and evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import random


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def contrastive_loss(features_a: torch.Tensor, 
                     features_b: torch.Tensor, 
                     temperature: float = 0.07) -> torch.Tensor:
    """
    Compute bidirectional contrastive loss (InfoNCE)
    
    Args:
        features_a: Features from modality A [batch_size, dim]
        features_b: Features from modality B [batch_size, dim]
        temperature: Temperature parameter for scaling
    
    Returns:
        Contrastive loss value
    """
    batch_size = features_a.shape[0]
    
    # Normalize features
    features_a = F.normalize(features_a, dim=-1)
    features_b = F.normalize(features_b, dim=-1)
    
    # Compute similarity matrix
    logits = features_a @ features_b.T / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=features_a.device)
    
    # Bidirectional cross-entropy loss
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    
    return (loss_a + loss_b) / 2


def clip_score(images: torch.Tensor,
               texts: List[str],
               model,
               processor) -> torch.Tensor:
    """
    Compute CLIP similarity scores between images and texts
    
    Args:
        images: Image tensors [batch_size, C, H, W]
        texts: List of text strings
        model: CLIP model
        processor: CLIP processor
    
    Returns:
        Similarity scores [batch_size]
    """
    device = next(model.parameters()).device
    
    # Process inputs
    inputs = processor(
        text=texts,
        images=images,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)
    
    # Get features
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    # Normalize and compute similarity
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    similarity = (image_features * text_features).sum(dim=-1)
    
    return similarity


def compute_recall_at_k(similarity_matrix: np.ndarray,
                        k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute Recall@K for image-text retrieval
    
    Args:
        similarity_matrix: Similarity scores [num_images, num_texts]
        k_values: List of K values to compute recall for
    
    Returns:
        Dictionary of recall metrics
    """
    num_images, num_texts = similarity_matrix.shape
    
    # Image-to-text retrieval
    i2t_ranks = []
    for i in range(num_images):
        sims = similarity_matrix[i]
        rank = np.where(np.argsort(-sims) == i)[0][0] + 1
        i2t_ranks.append(rank)
    
    # Text-to-image retrieval
    t2i_ranks = []
    for i in range(num_texts):
        sims = similarity_matrix[:, i]
        rank = np.where(np.argsort(-sims) == i)[0][0] + 1
        t2i_ranks.append(rank)
    
    # Compute Recall@K
    metrics = {}
    for k in k_values:
        i2t_rk = sum(r <= k for r in i2t_ranks) / len(i2t_ranks) * 100
        t2i_rk = sum(r <= k for r in t2i_ranks) / len(t2i_ranks) * 100
        
        metrics[f'i2t_r{k}'] = i2t_rk
        metrics[f't2i_r{k}'] = t2i_rk
    
    # Mean recall
    all_recalls = [metrics[f'i2t_r{k}'] for k in k_values] + \
                  [metrics[f't2i_r{k}'] for k in k_values]
    metrics['mean_recall'] = np.mean(all_recalls)
    
    # Median rank
    metrics['i2t_median_rank'] = np.median(i2t_ranks)
    metrics['t2i_median_rank'] = np.median(t2i_ranks)
    
    return metrics


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: Dict,
                    filepath: str,
                    is_best: bool = False):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state and metadata
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    import os
    from pathlib import Path
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(state, filepath)
    
    # Save best model separately
    if is_best:
        best_path = filepath.replace('.pt', '_best.pt')
        torch.save(state, best_path)


def load_checkpoint(filepath: str,
                    model,
                    optimizer=None,
                    scheduler=None) -> Dict:
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
    
    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def cosine_schedule_with_warmup(optimizer,
                                num_warmup_steps: int,
                                num_training_steps: int,
                                min_lr_ratio: float = 0.0):
    """
    Create cosine learning rate schedule with warmup
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR
    
    Returns:
        Learning rate scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def compute_fid_score(real_features: np.ndarray,
                      generated_features: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance (FID)
    
    Args:
        real_features: Features from real images [N, D]
        generated_features: Features from generated images [M, D]
    
    Returns:
        FID score
    """
    from scipy import linalg
    
    # Compute mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    # Compute FID
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return float(fid)


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all GPUs in distributed training
    
    Args:
        tensor: Tensor to gather [local_batch_size, ...]
    
    Returns:
        Gathered tensor [world_size * local_batch_size, ...]
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensor
    
    # Gather tensors
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    
    # Concatenate
    gathered_tensor = torch.cat(tensor_list, dim=0)
    
    return gathered_tensor


def log_metrics(metrics: Dict[str, float],
                step: int,
                prefix: str = '',
                logger=None):
    """
    Log metrics to console and/or logging framework
    
    Args:
        metrics: Dictionary of metric names and values
        step: Current training step
        prefix: Prefix for metric names (e.g., 'train/' or 'val/')
        logger: Optional logger (wandb, tensorboard, etc.)
    """
    # Format metrics for printing
    metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    print(f"Step {step} | {prefix}{metric_str}")
    
    # Log to framework
    if logger is not None:
        logged_metrics = {f'{prefix}{k}': v for k, v in metrics.items()}
        logged_metrics['step'] = step
        
        if hasattr(logger, 'log'):  # wandb
            logger.log(logged_metrics)
        elif hasattr(logger, 'add_scalars'):  # tensorboard
            logger.add_scalars(prefix, metrics, step)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., "2h 34m 12s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


def print_model_summary(model, model_name: str = 'Model'):
    """
    Print model architecture summary
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"{'='*60}\n")
