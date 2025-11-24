"""
Evaluation script for multilingual image-text retrieval

Evaluates models on Multi30K, MS-COCO, and other benchmarks
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from data.datasets import Multi30KDataset, MSCOCOMultilingualDataset
from models.mclip_model import mCLIPModel


class RetrievalEvaluator:
    """Evaluator for multilingual image-text retrieval"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self.load_model()
        self.model.eval()
        
        # Setup datasets
        self.datasets = self.setup_datasets()
    
    def load_model(self):
        """Load trained model"""
        
        if self.args.model_type == 'mclip':
            model = mCLIPModel.from_pretrained(self.args.model_path)
        elif self.args.model_type == 'clip':
            model = CLIPModel.from_pretrained(self.args.model_path)
        else:
            raise ValueError(f"Unknown model type: {self.args.model_type}")
        
        model = model.to(self.device)
        print(f"Loaded {self.args.model_type} model from {self.args.model_path}")
        
        return model
    
    def setup_datasets(self):
        """Setup evaluation datasets"""
        
        datasets = {}
        
        if self.args.dataset == 'multi30k' or self.args.dataset == 'all':
            for lang in self.args.languages:
                datasets[f'multi30k_{lang}'] = Multi30KDataset(
                    data_dir=self.args.data_dir,
                    language=lang,
                    split=self.args.split
                )
        
        if self.args.dataset == 'mscoco' or self.args.dataset == 'all':
            for lang in self.args.languages:
                datasets[f'mscoco_{lang}'] = MSCOCOMultilingualDataset(
                    data_dir=self.args.data_dir,
                    language=lang,
                    split=self.args.split
                )
        
        print(f"Loaded {len(datasets)} datasets for evaluation")
        return datasets
    
    @torch.no_grad()
    def extract_features(self, dataset):
        """Extract image and text features"""
        
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        image_features = []
        text_features = []
        
        for batch in tqdm(loader, desc="Extracting features"):
            images = batch['images'].to(self.device)
            texts = batch['texts']
            
            # Extract image features
            if self.args.model_type == 'mclip':
                img_feats = self.model.encode_image(images)
                txt_feats = self.model.encode_text(texts)
            else:
                img_feats = self.model.get_image_features(pixel_values=images)
                txt_feats = self.model.get_text_features(**texts)
            
            # Normalize
            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats = F.normalize(txt_feats, dim=-1)
            
            image_features.append(img_feats.cpu())
            text_features.append(txt_feats.cpu())
        
        # Concatenate
        image_features = torch.cat(image_features, dim=0)
        text_features = torch.cat(text_features, dim=0)
        
        return image_features, text_features
    
    def compute_retrieval_metrics(self, image_features, text_features):
        """Compute recall@K metrics"""
        
        # Compute similarity matrix
        similarity = image_features @ text_features.T
        
        # Image-to-text retrieval
        i2t_ranks = []
        for i in range(len(image_features)):
            sims = similarity[i]
            # Get ground truth similarity
            gt_sim = sims[i]
            # Count how many are greater
            rank = (sims > gt_sim).sum().item() + 1
            i2t_ranks.append(rank)
        
        # Text-to-image retrieval
        t2i_ranks = []
        for i in range(len(text_features)):
            sims = similarity[:, i]
            gt_sim = sims[i]
            rank = (sims > gt_sim).sum().item() + 1
            t2i_ranks.append(rank)
        
        # Compute Recall@K
        metrics = {}
        
        for k in [1, 5, 10]:
            i2t_rk = sum(r <= k for r in i2t_ranks) / len(i2t_ranks) * 100
            t2i_rk = sum(r <= k for r in t2i_ranks) / len(t2i_ranks) * 100
            
            metrics[f'i2t_r{k}'] = i2t_rk
            metrics[f't2i_r{k}'] = t2i_rk
        
        # Mean recall
        metrics['mean_recall'] = sum(metrics.values()) / len(metrics)
        
        # Average rank
        metrics['i2t_avg_rank'] = np.mean(i2t_ranks)
        metrics['t2i_avg_rank'] = np.mean(t2i_ranks)
        
        return metrics
    
    def evaluate_dataset(self, dataset_name, dataset):
        """Evaluate on a single dataset"""
        
        print(f"\nEvaluating on {dataset_name}...")
        
        # Extract features
        image_features, text_features = self.extract_features(dataset)
        
        # Compute metrics
        metrics = self.compute_retrieval_metrics(image_features, text_features)
        
        # Print results
        print(f"Results for {dataset_name}:")
        print(f"  Image-to-Text:")
        print(f"    R@1:  {metrics['i2t_r1']:.2f}%")
        print(f"    R@5:  {metrics['i2t_r5']:.2f}%")
        print(f"    R@10: {metrics['i2t_r10']:.2f}%")
        print(f"  Text-to-Image:")
        print(f"    R@1:  {metrics['t2i_r1']:.2f}%")
        print(f"    R@5:  {metrics['t2i_r5']:.2f}%")
        print(f"    R@10: {metrics['t2i_r10']:.2f}%")
        print(f"  Mean Recall: {metrics['mean_recall']:.2f}%")
        
        return metrics
    
    def evaluate_all(self):
        """Evaluate on all datasets"""
        
        all_results = {}
        
        for dataset_name, dataset in self.datasets.items():
            metrics = self.evaluate_dataset(dataset_name, dataset)
            all_results[dataset_name] = metrics
        
        # Compute average across languages
        if len(all_results) > 1:
            avg_metrics = {}
            for key in all_results[list(all_results.keys())[0]].keys():
                values = [results[key] for results in all_results.values()]
                avg_metrics[f'avg_{key}'] = np.mean(values)
            
            print(f"\n{'='*50}")
            print("Average across all languages:")
            print(f"  Mean Recall: {avg_metrics['avg_mean_recall']:.2f}%")
            print(f"  I2T R@1: {avg_metrics['avg_i2t_r1']:.2f}%")
            print(f"  T2I R@1: {avg_metrics['avg_t2i_r1']:.2f}%")
            
            all_results['average'] = avg_metrics
        
        # Save results
        if self.args.output_file:
            output_path = Path(self.args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\nResults saved to {output_path}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate multilingual retrieval')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['mclip', 'clip', 'altdiffusion'],
                        help='Type of model')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['multi30k', 'mscoco', 'all'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--languages', type=str, nargs='+', required=True,
                        help='Languages to evaluate (e.g., en de fr)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save results JSON')
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = RetrievalEvaluator(args)
    results = evaluator.evaluate_all()


if __name__ == '__main__':
    main()
