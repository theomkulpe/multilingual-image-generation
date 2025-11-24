"""
Training script for mCLIP using Triangle Knowledge Distillation (TriKD)

This script implements the three-stage training procedure:
1. Enhance Multilingual Text Encoder (MTE)
2. Add Contrastive Learning to MTE
3. Triangle Knowledge Distillation with frozen CLIP
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    CLIPModel, CLIPProcessor,
    XLMRobertaModel, XLMRobertaTokenizer,
    get_cosine_schedule_with_warmup
)
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import json

from models.mclip_model import mCLIPModel
from data.datasets import CC3MDataset, ParallelTextDataset
from utils.losses import contrastive_loss


class TriangleKnowledgeDistillation:
    """Triangle Knowledge Distillation trainer for mCLIP"""
    
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation
        )
        
        # Initialize models
        self.setup_models()
        
        # Initialize data
        self.setup_data()
        
        # Initialize optimizer and scheduler
        self.setup_optimizer()
        
        # Initialize logging
        if self.accelerator.is_main_process and args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args)
            )
    
    def setup_models(self):
        """Initialize CLIP, XLM-RoBERTa, and projectors"""
        
        # Load frozen CLIP
        self.clip_model = CLIPModel.from_pretrained(self.args.clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(self.args.clip_model)
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Load XLM-RoBERTa (enhanced from Stage 2)
        self.xlm_model = XLMRobertaModel.from_pretrained(self.args.mte_checkpoint)
        self.xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.mte_checkpoint)
        
        # Freeze XLM-RoBERTa
        for param in self.xlm_model.parameters():
            param.requires_grad = False
        
        # Initialize trainable projectors
        hidden_dim = self.xlm_model.config.hidden_size
        clip_dim = self.clip_model.config.projection_dim
        
        # X-Projector: 2-layer Transformer (trainable)
        self.x_projector = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Final projection to CLIP space
        self.x_projection_head = nn.Linear(hidden_dim, clip_dim)
        
        # CLIP projector (shared for image and text, trainable)
        self.clip_projector = nn.Linear(clip_dim, clip_dim)
        
        # Move to device
        self.clip_model = self.clip_model.to(self.accelerator.device)
        self.xlm_model = self.xlm_model.to(self.accelerator.device)
        self.x_projector = self.x_projector.to(self.accelerator.device)
        self.x_projection_head = self.x_projection_head.to(self.accelerator.device)
        self.clip_projector = self.clip_projector.to(self.accelerator.device)
        
        print(f"Initialized mCLIP model with:")
        print(f"  - Frozen CLIP: {sum(p.numel() for p in self.clip_model.parameters())} params")
        print(f"  - Frozen XLM-R: {sum(p.numel() for p in self.xlm_model.parameters())} params")
        print(f"  - Trainable X-Projector: {sum(p.numel() for p in self.x_projector.parameters())} params")
        print(f"  - Trainable CLIP-Projector: {sum(p.numel() for p in self.clip_projector.parameters())} params")
    
    def setup_data(self):
        """Setup training and validation datasets"""
        
        # CC3M dataset for Triangle KD
        self.train_dataset = CC3MDataset(
            data_dir=self.args.data_dir,
            split='train',
            clip_processor=self.clip_processor,
            xlm_tokenizer=self.xlm_tokenizer,
            max_length=self.args.max_length
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Validation dataset
        self.val_dataset = CC3MDataset(
            data_dir=self.args.data_dir,
            split='val',
            clip_processor=self.clip_processor,
            xlm_tokenizer=self.xlm_tokenizer,
            max_length=self.args.max_length
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"Loaded datasets:")
        print(f"  - Train: {len(self.train_dataset)} samples")
        print(f"  - Val: {len(self.val_dataset)} samples")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        
        # Collect trainable parameters
        trainable_params = (
            list(self.x_projector.parameters()) +
            list(self.x_projection_head.parameters()) +
            list(self.clip_projector.parameters())
        )
        
        # LAMB optimizer for large batch contrastive learning
        if self.args.optimizer == 'lamb':
            from apex.optimizers import FusedLAMB
            self.optimizer = FusedLAMB(
                trainable_params,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.args.weight_decay
            )
        
        # Cosine schedule with warmup
        total_steps = len(self.train_loader) * self.args.epochs // self.args.gradient_accumulation
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare with accelerator
        (
            self.x_projector,
            self.x_projection_head,
            self.clip_projector,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler
        ) = self.accelerator.prepare(
            self.x_projector,
            self.x_projection_head,
            self.clip_projector,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler
        )
    
    def forward_pass(self, batch):
        """Forward pass through all encoders and projectors"""
        
        images = batch['images'].to(self.accelerator.device)
        english_texts = batch['english_texts']
        multilingual_texts = batch['multilingual_texts']
        
        # 1. CLIP Image Encoder (frozen)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=images)
        
        # Project image features through trainable projector
        h_I = self.clip_projector(image_features)
        h_I = F.normalize(h_I, dim=-1)
        
        # 2. CLIP Text Encoder (frozen)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**english_texts)
        
        # Project text features through trainable projector
        h_T = self.clip_projector(text_features)
        h_T = F.normalize(h_T, dim=-1)
        
        # 3. XLM-RoBERTa Encoder (frozen)
        with torch.no_grad():
            xlm_outputs = self.xlm_model(**multilingual_texts)
            xlm_features = xlm_outputs.last_hidden_state
        
        # Project through X-Projector (trainable)
        h_X = self.x_projector(xlm_features)
        h_X = h_X[:, 0, :]  # Take [CLS] token
        h_X = self.x_projection_head(h_X)
        h_X = F.normalize(h_X, dim=-1)
        
        return h_I, h_T, h_X
    
    def compute_loss(self, h_I, h_T, h_X):
        """Compute Triangle Knowledge Distillation loss"""
        
        temperature = self.args.temperature
        
        # Image-Text Contrastive (ITC) loss
        loss_itc = contrastive_loss(h_I, h_X, temperature)
        
        # Text-Text Contrastive (TTC) loss
        loss_ttc = contrastive_loss(h_T, h_X, temperature)
        
        # Combined Triangle KD loss
        loss = loss_itc + self.args.ttc_weight * loss_ttc
        
        return loss, loss_itc, loss_ttc
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        
        self.x_projector.train()
        self.x_projection_head.train()
        self.clip_projector.train()
        
        total_loss = 0
        total_itc = 0
        total_ttc = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(
                self.x_projector,
                self.x_projection_head,
                self.clip_projector
            ):
                # Forward pass
                h_I, h_T, h_X = self.forward_pass(batch)
                
                # Compute loss
                loss, loss_itc, loss_ttc = self.compute_loss(h_I, h_T, h_X)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        list(self.x_projector.parameters()) +
                        list(self.x_projection_head.parameters()) +
                        list(self.clip_projector.parameters()),
                        self.args.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                total_itc += loss_itc.item()
                total_ttc += loss_ttc.item()
                
                # Log progress
                if step % self.args.logging_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    avg_itc = total_itc / (step + 1)
                    avg_ttc = total_ttc / (step + 1)
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'itc': f'{avg_itc:.4f}',
                        'ttc': f'{avg_ttc:.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                    })
                    
                    if self.accelerator.is_main_process and self.args.use_wandb:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/loss_itc': avg_itc,
                            'train/loss_ttc': avg_ttc,
                            'train/lr': self.scheduler.get_last_lr()[0],
                            'epoch': epoch,
                            'step': step
                        })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validation with image-text retrieval"""
        
        self.x_projector.eval()
        self.x_projection_head.eval()
        self.clip_projector.eval()
        
        all_image_features = []
        all_text_features = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            h_I, _, h_X = self.forward_pass(batch)
            
            all_image_features.append(h_I.cpu())
            all_text_features.append(h_X.cpu())
        
        # Concatenate features
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        
        # Compute similarity matrix
        similarity = image_features @ text_features.T
        
        # Image-to-text retrieval
        i2t_ranks = []
        for i in range(len(image_features)):
            sims = similarity[i]
            rank = (sims > sims[i]).sum().item() + 1
            i2t_ranks.append(rank)
        
        # Text-to-image retrieval
        t2i_ranks = []
        for i in range(len(text_features)):
            sims = similarity[:, i]
            rank = (sims > sims[i]).sum().item() + 1
            t2i_ranks.append(rank)
        
        # Compute recall metrics
        i2t_r1 = sum(r <= 1 for r in i2t_ranks) / len(i2t_ranks) * 100
        i2t_r5 = sum(r <= 5 for r in i2t_ranks) / len(i2t_ranks) * 100
        i2t_r10 = sum(r <= 10 for r in i2t_ranks) / len(i2t_ranks) * 100
        
        t2i_r1 = sum(r <= 1 for r in t2i_ranks) / len(t2i_ranks) * 100
        t2i_r5 = sum(r <= 5 for r in t2i_ranks) / len(t2i_ranks) * 100
        t2i_r10 = sum(r <= 10 for r in t2i_ranks) / len(t2i_ranks) * 100
        
        mean_recall = (i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10) / 6
        
        metrics = {
            'val/i2t_r1': i2t_r1,
            'val/i2t_r5': i2t_r5,
            'val/i2t_r10': i2t_r10,
            'val/t2i_r1': t2i_r1,
            'val/t2i_r5': t2i_r5,
            'val/t2i_r10': t2i_r10,
            'val/mean_recall': mean_recall
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        
        if not self.accelerator.is_main_process:
            return
        
        save_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save projectors
        unwrapped_x_proj = self.accelerator.unwrap_model(self.x_projector)
        unwrapped_x_head = self.accelerator.unwrap_model(self.x_projection_head)
        unwrapped_clip_proj = self.accelerator.unwrap_model(self.clip_projector)
        
        torch.save({
            'epoch': epoch,
            'x_projector_state_dict': unwrapped_x_proj.state_dict(),
            'x_projection_head_state_dict': unwrapped_x_head.state_dict(),
            'clip_projector_state_dict': unwrapped_clip_proj.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'args': vars(self.args)
        }, os.path.join(save_dir, 'model.pt'))
        
        # Save config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        print(f"Checkpoint saved to {save_dir}")
    
    def train(self):
        """Main training loop"""
        
        best_mean_recall = 0
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.args.epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if epoch % self.args.eval_every == 0:
                val_metrics = self.validate()
                print(f"Validation Metrics:")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.2f}")
                
                if self.accelerator.is_main_process and self.args.use_wandb:
                    wandb.log(val_metrics)
                
                # Save best checkpoint
                if val_metrics['val/mean_recall'] > best_mean_recall:
                    best_mean_recall = val_metrics['val/mean_recall']
                    self.save_checkpoint(epoch, val_metrics)
            
            # Save periodic checkpoint
            if epoch % self.args.save_every == 0:
                self.save_checkpoint(epoch, {})
        
        print(f"\nTraining completed!")
        print(f"Best Mean Recall: {best_mean_recall:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train mCLIP with Triangle KD')
    
    # Model arguments
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--mte_checkpoint', type=str, required=True,
                        help='Path to enhanced MTE from Stage 2')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=77)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='lamb', choices=['lamb', 'adamw'])
    
    # Loss arguments
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--ttc_weight', type=float, default=0.1,
                        help='Weight for text-text contrastive loss (lambda)')
    
    # Logging arguments
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='cross_lingual_diffusion')
    parser.add_argument('--run_name', type=str, default='mclip_trikd')
    
    # System arguments
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create trainer and train
    trainer = TriangleKnowledgeDistillation(args)
    trainer.train()


if __name__ == '__main__':
    main()
