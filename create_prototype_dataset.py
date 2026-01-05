"""Create prototype_span dataset"""

import json
import os
import pyarrow as pa
import random
from typing import List, Dict, Any

def load_spans_stats(file_path: str) -> Dict[str, int]:

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('span_frequencies', {})
    except Exception as e:
        print(f"Error: Unable to load file {file_path}: {e}")
        return {}

def create_prototype_span_dataset():
    
    # Define file paths
    files = [
        {
            "path": "./datasets/Pri-DDXPlus_preprocess/Pri-DDXPlus_preprocess_private_spans_stats.json",
            "domain": "medical",
            "private": 1
        },
        {
            "path": "./datasets/Pri-DDXPlus_preprocess/Pri-DDXPlus_preprocess_non_private_spans_stats.json",
            "domain": "medical", 
            "private": 0
        },
        {
            "path": "./datasets/Pri-SLJA_preprocess/Pri-SLJA_preprocess_private_spans_stats.json",
            "domain": "legal",
            "private": 1
        },
        {
            "path": "./datasets/Pri-SLJA_preprocess/Pri-SLJA_preprocess_non_private_spans_stats.json",
            "domain": "legal",
            "private": 0
        }
    ]
    
    prototype_data = []
    
    for file_info in files:
        file_path = file_info["path"]
        domain = file_info["domain"]
        private = file_info["private"]
        
        print(f"Processing: {file_path}")
        
        # Load spans statistics
        spans_freq = load_spans_stats(file_path)
        
        if not spans_freq:
            print(f"Skipping file: {file_path}")
            continue
        
        # For Pri-SLJA non_private_spans, only consider frequency > 10
        if "Pri-SLJA_preprocess_non_private_spans_stats.json" in file_path:
            filtered_spans = {span: freq for span, freq in spans_freq.items() if freq > 10}
            print(f"  Pri-SLJA non-private spans: original {len(spans_freq)}, filtered {len(filtered_spans)} (frequency > 10)")
            spans_freq = filtered_spans
        
        # Add to prototype data (only consider unique spans, ignore frequency)
        for span in spans_freq.keys():
            prototype_data.append({
                "span": span,
                "domain": domain,
                "private": private
            })
        
        print(f"  Added {len(spans_freq)} spans")
    
    print(f"\nTotal collected: {len(prototype_data)} spans")
    
    # Create output directory
    output_dir = "./datasets/prototype_span"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as Arrow format
    table = pa.Table.from_pylist(prototype_data)
    
    # Save as Arrow file
    output_path = os.path.join(output_dir, "prototype_spans.arrow")
    with pa.OSFile(output_path, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    
    print(f"Dataset saved to: {output_path}")
    
    # Display statistics
    print(f"\n=== Dataset Statistics ===")
    medical_private = len([d for d in prototype_data if d["domain"] == "medical" and d["private"] == 1])
    medical_non_private = len([d for d in prototype_data if d["domain"] == "medical" and d["private"] == 0])
    legal_private = len([d for d in prototype_data if d["domain"] == "legal" and d["private"] == 1])
    legal_non_private = len([d for d in prototype_data if d["domain"] == "legal" and d["private"] == 0])
    
    print(f"Medical domain:")
    print(f"  Private spans: {medical_private}")
    print(f"  Non-private spans: {medical_non_private}")
    print(f"Legal domain:")
    print(f"  Private spans: {legal_private}")
    print(f"  Non-private spans: {legal_non_private}")
    
    print(f"\nTotal:")
    print(f"  Private spans: {medical_private + legal_private}")
    print(f"  Non-private spans: {medical_non_private + legal_non_private}")
    
    # Randomly display some sample data
    print(f"\n=== Random Sample Data ===")
    random_samples = random.sample(prototype_data, min(20, len(prototype_data)))
    for i, item in enumerate(random_samples):
        print(f"{i+1:2d}. span: '{item['span']}', domain: {item['domain']}, private: {item['private']}")
    
    # Randomly display by domain and privacy type
    print(f"\n=== Random Samples by Type ===")
    
    # Medical private
    medical_private_samples = [d for d in prototype_data if d["domain"] == "medical" and d["private"] == 1]
    if medical_private_samples:
        print("Medical Private spans:")
        for item in random.sample(medical_private_samples, min(5, len(medical_private_samples))):
            print(f"  - '{item['span']}'")
    
    # Medical non-private
    medical_non_private_samples = [d for d in prototype_data if d["domain"] == "medical" and d["private"] == 0]
    if medical_non_private_samples:
        print("Medical Non-private spans:")
        for item in random.sample(medical_non_private_samples, min(5, len(medical_non_private_samples))):
            print(f"  - '{item['span']}'")
    
    # Legal private
    legal_private_samples = [d for d in prototype_data if d["domain"] == "legal" and d["private"] == 1]
    if legal_private_samples:
        print("Legal Private spans:")
        for item in random.sample(legal_private_samples, min(5, len(legal_private_samples))):
            print(f"  - '{item['span']}'")
    
    # Legal non-private
    legal_non_private_samples = [d for d in prototype_data if d["domain"] == "legal" and d["private"] == 0]
    if legal_non_private_samples:
        print("Legal Non-private spans:")
        for item in random.sample(legal_non_private_samples, min(5, len(legal_non_private_samples))):
            print(f"  - '{item['span']}'")
    
    return prototype_data

if __name__ == "__main__":
    create_prototype_span_dataset()
