#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset processing framework - DPO format
"""

import os
import re
import argparse
from typing import List, Tuple
from datasets import Dataset
import json
from pathlib import Path

def write_json(output_path: str, records: list) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def collect_non_overlapping_matches(text: str, spans: List[str], case_sensitive: bool = True,
                                    max_per_span: int = None) -> List[Tuple[int, int]]:
    flags = 0 if case_sensitive else re.I
    occupied: List[Tuple[int, int]] = []
    matches: List[Tuple[int, int]] = []

    def overlapped(a: int, b: int) -> bool:
        for (s, e) in occupied:
            if not (b <= s or a >= e):
                return True
        return False

    for span in spans:
        if not span or not isinstance(span, str):
            continue
        pat = re.compile(re.escape(span), flags)
        count = 0
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if overlapped(s, e):
                continue
            matches.append((s, e))
            occupied.append((s, e))
            count += 1
            if max_per_span is not None and count >= max_per_span:
                break

    matches = sorted(set(matches), key=lambda x: (x[0], x[1]))
    return matches


def add_rewrite_tags_all(text: str, spans: List[str], case_sensitive: bool = True,
                         max_per_span: int = None) -> str:
    matches = collect_non_overlapping_matches(text, spans, case_sensitive, max_per_span)
    if not matches:
        return text
    tagged = text
    for s, e in reversed(matches):
        tagged = tagged[:e] + "</REWRITE>" + tagged[e:]
        tagged = tagged[:s] + "<REWRITE>" + tagged[s:]
    return tagged

def load_dataset(args):
    
    # Dataset path
    dataset_path = f"./datasets/DPO/temperature_{args.temperature}/alpha_{args.preference_alpha}/Pri_DDXPlus_SLJA_dpo_preference"
    
    print(f"Loading dataset: {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        print("Please run batch_merge.py to merge the dataset first")
        return None
    
    try:
        # Load dataset
        dataset = Dataset.load_from_disk(dataset_path)
        print(f"Successfully loaded dataset, total {len(dataset)} records")
        
        # Check dataset structure
        if len(dataset) > 0:
            print(f"Dataset features: {list(dataset.features.keys())}")
            print(f"\nFirst record:")
            sample = dataset[0]
            print(sample)
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None





def process_dataset(dataset):
    """
    Process dataset using DPO format
    Process text by text
    
    Args:
        dataset: Dataset
    
    Returns:
        Processed results
    """
    print(f"\nStarting to process dataset, total {len(dataset)} records")
    
    results = []
    
    for i, sample in enumerate(dataset):
        
        original_text = sample['question_init']
        private_span = sample['private_spans']
        reject_private_span = sample['reject_private']
        prefer_private_span = sample['prefer_private']

        tagged_text = add_rewrite_tags_all(original_text, private_span, case_sensitive=True)
        n_spans = tagged_text.count("<REWRITE>")

        system_prompt = f"""You are a sophisticated privacy-focused text anonymizer. You need to rewrite ONLY the content between <REWRITE> and </REWRITE> tags.

CRITICAL INSTRUCTIONS:
1. Find ALL text between <REWRITE> and </REWRITE> tags
2. Replace each marked content with generic, non-identifying alternatives that are semantically similar to the original but use different wording (avoid repeating the same term, lemma, or obvious near-copy). Do not introduce domain-sensitive private attributes
3. Ensure replacements maintain semantic coherence and fluency with the ORIGINAL text
4. Return ONLY a JSON array of replacement spans in EXACT order
5. Do NOT include <REWRITE> or </REWRITE> tags in the replacement spans
6. STRICT FORMAT: ["span1","span2"] — MUST be a valid JSON array of strings
7. NO additional text, explanations, code fences, keys, or formatting outside the JSON array
8. Each replacement span should be a single string without internal quotes
9. If any rule is violated or no spans exist, return an empty array: []
10.There are exactly {n_spans} spans → return exactly {n_spans} items (1:1 in order).

OUTPUT HARD RULES (ENFORCE):
- The entire model output MUST start with '[' and end with ']'. After generating ']', you MUST STOP.
- Do NOT include newlines outside JSON. Do NOT include trailing commas.
- Use only standard JSON strings; no objects, no numbers, no booleans.
- The array length MUST equal the number of {n_spans}.

Example:
Original text: A patient has <REWRITE>fever</REWRITE> and <REWRITE>headache</REWRITE>.
Privacy spans: ["fever","headache"]
Replacement spans: ["sickness","discomfort"]
"""

        user_prompt = f"""
Original text: {tagged_text}
Privacy spans: {private_span}
Replacement spans:"""

        msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ]

        item = {
                "messages": msgs,
                "chosen": json.dumps(prefer_private_span, ensure_ascii=False),
                "rejected": json.dumps(reject_private_span, ensure_ascii=False)
            }
        
        results.append(item)
        
    
    print(f"Dataset processing completed, processed {len(results)} records")
    return results



def main():

    print("=== Dataset Processing Framework ===")

    parser = argparse.ArgumentParser(description='DPO dataset preprocessing')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature parameter for InfoNCE loss')
    parser.add_argument('--preference_alpha', type=float, default=0.5,
                        help='Weight for domain similarity in preference scoring')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args)
    
    if dataset is None:
        print("Dataset loading failed, exiting program")
        return
    
    # Process dataset
    results = process_dataset(dataset)
    print(f"\nProcessing results: {len(results)} records")
    
    # Save dataset (JSON)
    output_path = f"./data/Pri_DDXPlus_SLJA_dpo_factory_temperature_{args.temperature}_alpha_{args.preference_alpha}.json"
    print(f"\nSaving dataset (JSON) to: {output_path}")
    write_json(output_path, results)
    print(f"Dataset saved as JSON: {output_path}")
    
    # Display first few processing results
    print("\nProcessing result examples:")
    for i in range(min(2, len(results))):
        print(f"  Sample {i+1}:")
        sample = results[i]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"    {key}: {value[:100]}...")
            else:
                print(f"    {key}: {value}")
        print()
    
    print("\n=== Processing Completed ===")
    print(f"Dataset (DPO): {len(results)} records")

    # Modify configuration file
    config_path = Path("./data/dataset_info.json")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    key = f"privacy_dpo_temperature_{args.temperature}_alpha_{args.preference_alpha}_json"
    data[key] = {
        "file_name": f"Pri_DDXPlus_SLJA_dpo_factory_temperature_{args.temperature}_alpha_{args.preference_alpha}.json",
        "ranking": True,
        "columns": {
            "messages": "messages",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
