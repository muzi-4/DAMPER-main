#!/usr/bin/env python3
"""
Merge multiple Arrow datasets under Pri_DDXPlus_SLJA_dpo_rewrite_batch directory
Save to /datasets/Pri_DDXPlus_SLJA_dpo_rewrite for easy load_from_disk loading
"""

import os
import glob
from pathlib import Path
from datasets import Dataset, load_from_disk, concatenate_datasets
import logging
import numpy as np
import argparse

# Optional import tqdm, use simple progress display if not installed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        """Simple progress display replacement for tqdm"""
        if desc:
            print(f"{desc}...")
        return iterable

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_sequence_field(example, field_name):

    if field_name in example:
        value = example[field_name]
        if value is None or (isinstance(value, list) and len(value) == 0):
            # Convert None or empty list to empty string list
            example[field_name] = []
        elif isinstance(value, list):
            # Ensure all elements in the list are strings
            example[field_name] = [str(item) if item is not None else "" for item in value]
    return example

def normalize_dataset_features(dataset):

    logger.info("Normalizing dataset features...")
    
    def normalize_example(example):
        # Normalize sequence fields
        example = normalize_sequence_field(example, 'non_private_spans')
        example = normalize_sequence_field(example, 'private_spans')
        example = normalize_sequence_field(example, 'prefer_private')
        return example
    
    # Apply normalization
    normalized_dataset = dataset.map(normalize_example)
    logger.info("Feature normalization completed")
    return normalized_dataset

def align_dataset_features(datasets):

    if not datasets:
        return datasets
    
    logger.info("Aligning dataset features...")
    
    # Get features of the first dataset as reference
    reference_features = datasets[0].features
    logger.info(f"Reference features: {reference_features}")
    
    aligned_datasets = []
    
    for i, dataset in enumerate(datasets):
        try:
            # Check if features match
            if dataset.features == reference_features:
                aligned_datasets.append(dataset)
                logger.info(f"Dataset {i+1} features already matched")
            else:
                logger.info(f"Dataset {i+1} features do not match, attempting alignment...")
                logger.info(f"Current features: {dataset.features}")
                
                # Try simpler method: directly convert features
                try:
                    # Create a new dataset, force using reference features
                    aligned_dataset = dataset.map(
                        lambda x: x,  # Keep data unchanged
                        features=reference_features,
                        batched=False
                    )
                    aligned_datasets.append(aligned_dataset)
                    logger.info(f"Dataset {i+1} feature alignment successful")
                except Exception as e:
                    logger.warning(f"Dataset {i+1} feature alignment failed: {str(e)}")
                    # If alignment fails, skip this dataset
                    logger.warning(f"Skipping dataset {i+1}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing dataset {i+1}: {str(e)}")
            continue
    
    logger.info(f"Successfully aligned {len(aligned_datasets)} datasets")
    return aligned_datasets

def merge_arrow_datasets():

    # Source and target directories
    source_dir = f"./datasets/DPO/candidate/Pri_DDXPlus_SLJA_dpo_candidate_batch"
    target_dir = f"./datasets/DPO/candidate/Pri_DDXPlus_SLJA_dpo_candidate"

    if os.path.exists(target_dir):
        logger.info(f"Target dataset already exists: {target_dir}")
        return
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        logger.error(f"Source directory does not exist: {source_dir}")
        return
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all batch directories
    batch_dirs = sorted(glob.glob(os.path.join(source_dir, "batch_*")))
    logger.info(f"Found {len(batch_dirs)} batch directories")
    
    if not batch_dirs:
        logger.error("No batch directories found")
        return
    
    # Store all datasets
    datasets = []
    
    # Load batch datasets one by one
    for batch_dir in tqdm(batch_dirs, desc="Loading batch datasets"):
        try:
            # Check if directory contains required files
            required_files = ["data-00000-of-00001.arrow", "dataset_info.json"]
            if not all(os.path.exists(os.path.join(batch_dir, f)) for f in required_files):
                logger.warning(f"Skipping incomplete batch directory: {batch_dir}")
                continue
            
            # Load dataset
            dataset = load_from_disk(batch_dir)
            
            # Normalize dataset features
            normalized_dataset = normalize_dataset_features(dataset)
            datasets.append(normalized_dataset)
            logger.info(f"Successfully loaded and normalized {batch_dir}, sample count: {len(normalized_dataset)}")
            
        except Exception as e:
            logger.error(f"Error loading {batch_dir}: {str(e)}")
            continue
    
    if not datasets:
        logger.error("No datasets successfully loaded")
        return
    
    logger.info(f"Successfully loaded {len(datasets)} datasets")
    
    # Align dataset features
    aligned_datasets = align_dataset_features(datasets)
    
    if not aligned_datasets:
        logger.error("No datasets successfully aligned")
        return
    
    # Merge all datasets
    logger.info("Starting to merge datasets...")
    try:
        merged_dataset = concatenate_datasets(aligned_datasets)
        logger.info(f"Merge completed, total samples: {len(merged_dataset)}")
        
        # Save merged dataset
        logger.info(f"Saving merged dataset to: {target_dir}")
        merged_dataset.save_to_disk(target_dir)
        
        # Verify saved dataset can be loaded normally
        logger.info("Verifying saved dataset...")
        test_dataset = load_from_disk(target_dir)
        logger.info(f"Verification successful, loaded sample count: {len(test_dataset)}")
        
        # Display dataset information
        logger.info("Dataset information:")
        logger.info(f"  - Total samples: {len(test_dataset)}")
        logger.info(f"  - Feature columns: {list(test_dataset.features.keys())}")
        logger.info(f"  - Data types: {test_dataset.features}")
        
        logger.info("Dataset merge and save completed!")
        
    except Exception as e:
        logger.error(f"Error merging or saving dataset: {str(e)}")
        raise

def main():

    logger.info("Starting to merge Arrow datasets...")
    merge_arrow_datasets()
    logger.info("Merge completed!")

if __name__ == "__main__":
    main()
