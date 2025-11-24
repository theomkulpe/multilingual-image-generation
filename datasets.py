"""
Dataset classes for multilingual image-text tasks
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CC3MDataset(Dataset):
    """Conceptual Captions 3M dataset for training mCLIP"""
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 clip_processor=None,
                 xlm_tokenizer=None,
                 max_length: int = 77,
                 languages: List[str] = ['en']):
        """
        Args:
            data_dir: Root directory containing CC3M data
            split: Dataset split ('train', 'val')
            clip_processor: CLIP processor for images and English text
            xlm_tokenizer: XLM-RoBERTa tokenizer for multilingual text
            max_length: Maximum sequence length
            languages: List of languages to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.clip_processor = clip_processor
        self.xlm_tokenizer = xlm_tokenizer
        self.max_length = max_length
        self.languages = languages
        
        # Load captions
        self.captions = self.load_captions()
        
        print(f"Loaded CC3M {split}: {len(self.captions)} samples")
    
    def load_captions(self) -> List[Dict]:
        """Load captions from JSON file"""
        caption_file = self.data_dir / f'{self.split}_captions.json'
        
        with open(caption_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def __len__(self) -> int:
        return len(self.captions)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single sample"""
        item = self.captions[idx]
        
        # Load image
        image_path = self.data_dir / 'images' / item['image_file']
        image = Image.open(image_path).convert('RGB')
        
        # Get English caption
        english_caption = item['caption']
        
        # Get multilingual caption (could be translated or same as English)
        if 'translations' in item and self.languages[0] in item['translations']:
            multilingual_caption = item['translations'][self.languages[0]]
        else:
            multilingual_caption = english_caption
        
        # Process with CLIP processor
        clip_inputs = self.clip_processor(
            text=[english_caption],
            images=image,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        
        # Process with XLM tokenizer
        xlm_inputs = self.xlm_tokenizer(
            multilingual_caption,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        
        return {
            'images': clip_inputs['pixel_values'].squeeze(0),
            'english_texts': {
                'input_ids': clip_inputs['input_ids'].squeeze(0),
                'attention_mask': clip_inputs['attention_mask'].squeeze(0)
            },
            'multilingual_texts': {
                'input_ids': xlm_inputs['input_ids'].squeeze(0),
                'attention_mask': xlm_inputs['attention_mask'].squeeze(0)
            }
        }


class Multi30KDataset(Dataset):
    """Multi30K dataset for multilingual evaluation"""
    
    def __init__(self,
                 data_dir: str,
                 language: str = 'en',
                 split: str = 'test',
                 processor=None,
                 tokenizer=None,
                 max_length: int = 77):
        """
        Args:
            data_dir: Root directory containing Multi30K data
            language: Language code (en, de, fr, cs)
            split: Dataset split ('train', 'val', 'test')
            processor: Image processor
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
        """
        self.data_dir = Path(data_dir)
        self.language = language
        self.split = split
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load annotations
        self.annotations = self.load_annotations()
        
        print(f"Loaded Multi30K {split} ({language}): {len(self.annotations)} samples")
    
    def load_annotations(self) -> List[Dict]:
        """Load image-caption pairs"""
        caption_file = self.data_dir / f'{self.split}.{self.language}'
        image_list_file = self.data_dir / f'{self.split}_images.txt'
        
        # Load captions
        with open(caption_file, 'r', encoding='utf-8') as f:
            captions = [line.strip() for line in f]
        
        # Load image filenames
        with open(image_list_file, 'r') as f:
            image_files = [line.strip() for line in f]
        
        # Create annotations
        annotations = []
        for image_file, caption in zip(image_files, captions):
            annotations.append({
                'image_file': image_file,
                'caption': caption
            })
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single sample"""
        item = self.annotations[idx]
        
        # Load image
        image_path = self.data_dir / 'images' / item['image_file']
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        if self.processor is not None:
            image_inputs = self.processor(images=image, return_tensors='pt')
            image = image_inputs['pixel_values'].squeeze(0)
        
        # Process text
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                item['caption'],
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_length,
                truncation=True
            )
            texts = {
                'input_ids': text_inputs['input_ids'].squeeze(0),
                'attention_mask': text_inputs['attention_mask'].squeeze(0)
            }
        else:
            texts = item['caption']
        
        return {
            'images': image,
            'texts': texts,
            'caption': item['caption'],
            'image_file': item['image_file']
        }


class MSCOCOMultilingualDataset(Dataset):
    """MS-COCO dataset with multilingual captions"""
    
    def __init__(self,
                 data_dir: str,
                 language: str = 'en',
                 split: str = 'val',
                 processor=None,
                 tokenizer=None,
                 max_length: int = 77):
        """
        Args:
            data_dir: Root directory containing COCO data
            language: Language code (en, ja, zh)
            split: Dataset split ('train', 'val')
            processor: Image processor
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
        """
        self.data_dir = Path(data_dir)
        self.language = language
        self.split = split
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load annotations
        self.annotations = self.load_annotations()
        
        print(f"Loaded COCO {split} ({language}): {len(self.annotations)} samples")
    
    def load_annotations(self) -> List[Dict]:
        """Load COCO annotations"""
        if self.language == 'en':
            ann_file = self.data_dir / 'annotations' / f'captions_{self.split}2014.json'
        else:
            ann_file = self.data_dir / 'annotations' / f'captions_{self.split}2014_{self.language}.json'
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Create image_id to filename mapping
        id_to_filename = {
            img['id']: img['file_name']
            for img in coco_data['images']
        }
        
        # Extract annotations
        annotations = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            annotations.append({
                'image_file': id_to_filename[image_id],
                'caption': ann['caption'],
                'image_id': image_id
            })
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single sample"""
        item = self.annotations[idx]
        
        # Load image
        image_path = self.data_dir / f'{self.split}2014' / item['image_file']
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        if self.processor is not None:
            image_inputs = self.processor(images=image, return_tensors='pt')
            image = image_inputs['pixel_values'].squeeze(0)
        
        # Process text
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                item['caption'],
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_length,
                truncation=True
            )
            texts = {
                'input_ids': text_inputs['input_ids'].squeeze(0),
                'attention_mask': text_inputs['attention_mask'].squeeze(0)
            }
        else:
            texts = item['caption']
        
        return {
            'images': image,
            'texts': texts,
            'caption': item['caption'],
            'image_file': item['image_file']
        }


class ParallelTextDataset(Dataset):
    """Parallel text dataset for training multilingual text encoder"""
    
    def __init__(self,
                 data_dir: str,
                 source_lang: str = 'en',
                 target_langs: List[str] = ['de', 'fr', 'es'],
                 tokenizer=None,
                 max_length: int = 128):
        """
        Args:
            data_dir: Directory containing parallel text files
            source_lang: Source language (typically English)
            target_langs: List of target languages
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
        """
        self.data_dir = Path(data_dir)
        self.source_lang = source_lang
        self.target_langs = target_langs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load parallel sentences
        self.parallel_data = self.load_parallel_data()
        
        print(f"Loaded parallel text: {len(self.parallel_data)} sentence pairs")
    
    def load_parallel_data(self) -> List[Dict]:
        """Load parallel sentences from files"""
        all_data = []
        
        for target_lang in self.target_langs:
            # Load source and target files
            src_file = self.data_dir / f'{self.source_lang}-{target_lang}.{self.source_lang}'
            tgt_file = self.data_dir / f'{self.source_lang}-{target_lang}.{target_lang}'
            
            if not src_file.exists() or not tgt_file.exists():
                print(f"Warning: Missing parallel data for {self.source_lang}-{target_lang}")
                continue
            
            with open(src_file, 'r', encoding='utf-8') as f:
                source_sents = [line.strip() for line in f]
            
            with open(tgt_file, 'r', encoding='utf-8') as f:
                target_sents = [line.strip() for line in f]
            
            # Create pairs
            for src, tgt in zip(source_sents, target_sents):
                if src and tgt:  # Skip empty lines
                    all_data.append({
                        'source': src,
                        'target': tgt,
                        'source_lang': self.source_lang,
                        'target_lang': target_lang
                    })
        
        return all_data
    
    def __len__(self) -> int:
        return len(self.parallel_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single sample"""
        item = self.parallel_data[idx]
        
        # Tokenize source
        source_inputs = self.tokenizer(
            item['source'],
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        
        # Tokenize target
        target_inputs = self.tokenizer(
            item['target'],
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        
        return {
            'source_input_ids': source_inputs['input_ids'].squeeze(0),
            'source_attention_mask': source_inputs['attention_mask'].squeeze(0),
            'target_input_ids': target_inputs['input_ids'].squeeze(0),
            'target_attention_mask': target_inputs['attention_mask'].squeeze(0),
            'source_lang': item['source_lang'],
            'target_lang': item['target_lang']
        }
