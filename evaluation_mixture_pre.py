"""Generate eight-option mixed evaluation dataset"""

import argparse
import os
import random
from typing import List, Dict, Any

import numpy as np
import torch
from datasets import Dataset, load_from_disk, concatenate_datasets

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return load_from_disk(path)


def has_valid_options(sample: Dict[str, Any]) -> bool:

    options = sample.get("options")
    return isinstance(options, list) and len(options) == 4


def build_mixture_dataset(primary_dataset, secondary_dataset) -> Dataset:
    mixed_samples: List[Dict[str, Any]] = []
    secondary_size = len(secondary_dataset)
    if secondary_size == 0:
        raise ValueError("Secondary dataset is empty, cannot construct mixed options")
    valid_secondary_indices = [
        idx for idx, sec_sample in enumerate(secondary_dataset)
        if "options" in sec_sample and isinstance(sec_sample["options"], list) and len(sec_sample["options"]) == 4
    ]
    if not valid_secondary_indices:
        raise ValueError("No samples with 4 options found in secondary dataset, cannot expand")

    for sample in primary_dataset:
        if not has_valid_options(sample):
            # Directly skip primary samples that don't have 4 options
            continue
        secondary_idx = random.choice(valid_secondary_indices)
        secondary_sample = secondary_dataset[secondary_idx]

        combined_options = sample["options"] + secondary_sample["options"]

        new_sample = dict(sample)
        new_sample["options"] = combined_options
        mixed_samples.append(new_sample)

    return Dataset.from_list(mixed_samples)


def parse_args():
    parser = argparse.ArgumentParser(description="Mix two evaluation datasets to generate eight-option version")
    parser.add_argument("--primary_dir", type=str, default="./datasets/Evaluation/Pri_DDXPlus",
                        help="Path to primary dataset (to be expanded to eight options)")
    parser.add_argument("--secondary_dir", type=str, default="./datasets/Evaluation/Pri_SLJA",
                        help="Path to secondary dataset providing additional four options")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature parameter for InfoNCE loss')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta parameter for DPO training')
    parser.add_argument('--preference_alpha', type=float, default=0.5,
                        help='Weight for domain similarity in preference scoring')
    parser.add_argument("--similarity_threshold_medical", type=float, default=0.9,
                       help="Similarity threshold for medical domain privacy span detection")
    parser.add_argument("--similarity_threshold_legal", type=float, default=0.9,
                       help="Similarity threshold for legal domain privacy span detection")
    parser.add_argument("--epsilon", type=float, default=50,
                       help="Differential privacy epsilon parameter")
    parser.add_argument("--mixture_output_dir", type=str,
                        default="./datasets/Evaluation/Pri_Mixture",
                        help="Output directory path for merged mixed dataset (Arrow format)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    primary_path = os.path.join(args.primary_dir, f"temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold_medical}/epsilon_{int(args.epsilon)}/Pri_DDXPlus_eval_rewrite_ours")
    secondary_path = os.path.join(args.secondary_dir, f"temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold_legal}/epsilon_{int(args.epsilon)}/Pri_SLJA_eval_rewrite_ours")
    
    primary_dataset = load_dataset(primary_path)
    secondary_dataset = load_dataset(secondary_path)

    mixed_dataset_primary = build_mixture_dataset(primary_dataset=primary_dataset, secondary_dataset=secondary_dataset)

    mixed_dataset_secondary = build_mixture_dataset(primary_dataset=secondary_dataset, secondary_dataset=primary_dataset)

    combined_dataset = concatenate_datasets([mixed_dataset_primary, mixed_dataset_secondary]).shuffle(seed=args.seed)

    mixture_output_path = os.path.join(args.mixture_output_dir, f"temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold_medical}_{args.similarity_threshold_legal}/epsilon_{int(args.epsilon)}/Pri_Mixture_eval_rewrite_ours")


    os.makedirs(mixture_output_path, exist_ok=True)
    combined_dataset.save_to_disk(mixture_output_path)
    print(
        f"Generated merged mixed dataset: {mixture_output_path}, sample count: {len(combined_dataset)}"
    )


if __name__ == "__main__":
    main()

