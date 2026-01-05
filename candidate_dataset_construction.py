#!/usr/bin/env python3
"""
Privacy rewriting system - subtask implementation
Selective rewriting based on privacy spans in the dataset
"""

import os
import json
import random
import math
import numpy as np
import torch
import argparse
import warnings
import logging
from datetime import datetime
# Filter pynndescent warnings
warnings.filterwarnings("ignore", message="pynndescent not installed")
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from typing import List, Tuple, Dict
from dataclasses import dataclass
import sys
sys.path.append(os.path.dirname(__file__))
from models_init.roberta_lora import RoBERTaLoRA
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def set_seed(seed: int):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@dataclass
class RewriteRegion:
    start: int
    end: int
    text: str
    type: str  # 'PII' or 'PRIVACY'


class CandidateSystem:
    
    def __init__(self, 
                 qwen_model_path: str = "./models/Qwen2.5-1.5b-Instruct",
                 dataset_path: str = "./datasets/Pri_DDXPlus_SLJA_dpo",
                 max_candidates: int = 10,
                 device: str = "cuda:0",
                 load_mean_std_required: bool = True):
        """
        Initialize privacy rewriting system
        
        Args:
            qwen_model_path: Path to the Qwen model
            dataset_path: Path to the dataset
            max_candidates: Maximum number of candidates
            device: Device name
        """
        self.qwen_model_path = qwen_model_path
        self.dataset_path = dataset_path
        self.max_candidates = max_candidates
        self.device = device

        
        # Load models
        self._load_models()
        
        # Load dataset
        self.dataset = load_from_disk(self.dataset_path)
        
        # Rewrite markers
        self.rewrite_start_marker = "<REWRITE>"
        self.rewrite_end_marker = "</REWRITE>"

        # Preference dataset parameters
        self.candidate_param_grid = [
            {"temperature": 0.5, "top_p": 0.90},
            {"temperature": 0.8, "top_p": 0.95},
            {"temperature": 1.0, "top_p": 0.97},
            {"temperature": 0.7, "top_p": 0.92},
            {"temperature": 0.9, "top_p": 0.90},
            {"temperature": 0.6, "top_p": 0.95},
            {"temperature": 0.2, "top_p": 1.00},
            {"temperature": 0.3, "top_p": 0.85},
            {"temperature": 1.2, "top_p": 0.95},
            {"temperature": 1.1, "top_p": 0.80}
        ]
    
    def _load_models(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.qwen_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.qwen_model_path,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            self.model.eval()
        except Exception as e:
            print(f"Warning: Unable to load Qwen model {self.qwen_model_path}: {e}")
            self.tokenizer = None
            self.model = None
    
    def mark_text_for_rewrite(self, text: str, regions: List[RewriteRegion]) -> str:
        if not regions:
            return text
        
        # Replace from back to front to avoid position offset
        marked_text = text
        for region in reversed(regions):
            before = marked_text[:region.start]
            after = marked_text[region.end:]
            marked_text = (before + 
                          self.rewrite_start_marker + 
                          region.text + 
                          self.rewrite_end_marker + 
                          after)
        
        return marked_text
    
    def rewrite_marked_text(self, marked_text: str, private_spans: List[str], temperature: float = 0.8, top_p: float = 0.95) -> str:
        private_json = json.dumps(private_spans, ensure_ascii=False)
        n_spans = marked_text.count(self.rewrite_start_marker)
        system_content = f"""You are a sophisticated privacy-focused text anonymizer. You need to rewrite ONLY the content between <REWRITE> and </REWRITE> tags.

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

        user_content = f"""
Original text: {marked_text}
Privacy spans: {private_json}
Replacement spans:"""

        messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        try:
            inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            max_length = 2048,
            return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = outputs[:, inputs.size(1):]
            generated_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            
            # Calculate expected number of replacement spans
            import re
            pattern = rf'{re.escape(self.rewrite_start_marker)}(.*?){re.escape(self.rewrite_end_marker)}'
            matches = list(re.finditer(pattern, marked_text))
            expected_count = len(matches)
            
            # Parse replacement span list
            replacement_spans = self._parse_replacement_words(generated_text, expected_count)
            
            # If replacement spans are empty (length mismatch), return None to ignore
            if not replacement_spans:
                return None
            
            # Check for empty strings, ignore this candidate if any found
            if any(not span.strip() for span in replacement_spans):
                return None
            
            # Manually reconstruct rewritten text
            rewritten_text = self._reconstruct_text_with_replacements(marked_text, replacement_spans, private_spans)
            
            return rewritten_text
            
        except Exception as e:
            print(f"Error during rewriting: {e}")
            return None
    
    def _parse_replacement_words(self, replacement_text: str, expected_count: int) -> List[str]:
        import re
        import json
        
        # Remove possible extra explanations (only remove newlines, keep JSON content)
        replacement_text = replacement_text.replace('\n', '').strip()
        
        try:
            # Try to parse JSON format
            if replacement_text.startswith('[') and replacement_text.endswith(']'):
                words = json.loads(replacement_text)
                # Filter empty strings
                words = [word.strip() for word in words if word and word.strip()]
                return words
        except json.JSONDecodeError as e:
            # Try to fix common JSON format errors
            try:
                # Use comma splitting to fix strings missing quotes
                fixed_text = replacement_text
                # Remove square brackets
                content = fixed_text[1:-1]
                # Split by comma
                parts = content.split(',')
                fixed_parts = []
                for part in parts:
                    part = part.strip()
                    # If part is not surrounded by quotes, add quotes
                    if not (part.startswith('"') and part.endswith('"')):
                        if not part.startswith('"'):
                            part = '"' + part
                        if not part.endswith('"'):
                            part = part + '"'
                    fixed_parts.append(part)
                
                # Reassemble JSON
                fixed_text = '[' + ','.join(fixed_parts) + ']'
                
                words = json.loads(fixed_text)
                words = [word.strip() for word in words if word and word.strip()]
                return words
            except json.JSONDecodeError as e2:
                pass
        
        # Handle malformed JSON (e.g., ["word1,  "word2,  "word3"])
        if replacement_text.startswith('[') and replacement_text.endswith(']'):
            # Remove square brackets
            content = replacement_text[1:-1]
            # Split by comma and clean
            words = []
            for part in content.split(','):
                part = part.strip().strip('"\'')
                if part:
                    words.append(part)
            
            if len(words) == expected_count:
                return words
        
        # If not JSON format, split by comma
        words = [word.strip().strip('"\'') for word in replacement_text.split(',')]
        # Filter empty strings
        words = [word for word in words if word]
        
        if len(words) != expected_count:
            return []
        
        return words
    
    def _reconstruct_text_with_replacements(self, marked_text: str, replacement_spans: List[str], private_spans: List[str]) -> str:
        import re
        
        # Find all marker positions
        pattern = rf'{re.escape(self.rewrite_start_marker)}(.*?){re.escape(self.rewrite_end_marker)}'
        matches = list(re.finditer(pattern, marked_text))
        
        if len(matches) != len(replacement_spans):
            return marked_text
        
        # Create mapping from original words to replacement words
        if len(replacement_spans) != len(private_spans):
            return marked_text
        
        replacement_map = {}
        for i, fragment in enumerate(private_spans):
            replacement_map[fragment] = replacement_spans[i]
        
        # Replace from back to front to avoid position offset
        result = marked_text
        for match in reversed(matches):
            # Get original content within markers
            original_content = match.group(1)
            
            # Find corresponding original fragment by word-finding approach
            matched_fragment = None
            for fragment in private_spans:
                if fragment in original_content:
                    matched_fragment = fragment
                    break
            
            if matched_fragment and matched_fragment in replacement_map:
                replacement_span = replacement_map[matched_fragment]
                start, end = match.span()
                result = result[:start] + f"{self.rewrite_start_marker}{replacement_span}{self.rewrite_end_marker}" + result[end:]
        
        return result

    # =============== Preference Dataset: Candidate Generation ===============
    def _generate_candidate_with_params(self, text: str, private_spans: List[str], temperature: float, top_p: float) -> Tuple[str, List[Tuple[str, str]]]:
        # 2. Find privacy spans and mark them
        privacy_regions = []
        for fragment in private_spans:
            if fragment in text:
                start = text.find(fragment)
                if start != -1:
                    privacy_regions.append(RewriteRegion(
                        start=start,
                        end=start + len(fragment),
                        text=fragment,
                        type='PRIVACY'
                    ))
        
        if not privacy_regions:
            return text, []
        
        # 3. Mark text
        marked_text = self.mark_text_for_rewrite(text, privacy_regions)
        
        # 4. First attempt to generate
        rewritten_text = self.rewrite_marked_text(marked_text, private_spans, temperature, top_p)
        
        # If first generation returns None, retry directly
        if rewritten_text is None:
            rewritten_text_retry = self.rewrite_marked_text(marked_text, private_spans, temperature, top_p)
            
            if rewritten_text_retry is None:
                return None, [], []
            else:
                rewritten_text = rewritten_text_retry
        
        # 4. Extract replacement pairs (from marked rewritten text)
        original_spans, rewritten_spans = self._extract_replacement_pairs_from_marked(text, rewritten_text, private_spans)
        
        # 5. If replacement pairs are empty, retry
        if not original_spans or not rewritten_spans:
            rewritten_text_retry = self.rewrite_marked_text(marked_text, private_spans, temperature, top_p)
            
            if rewritten_text_retry is None:
                return None, [], []
            
            original_spans_retry, rewritten_spans_retry = self._extract_replacement_pairs_from_marked(text, rewritten_text_retry, private_spans)
            
            if not original_spans_retry or not rewritten_spans_retry:
                return None, [], []
            else:
                original_spans = original_spans_retry
                rewritten_spans = rewritten_spans_retry
                rewritten_text = rewritten_text_retry
        
        # 6. Remove markers
        final_text = self.remove_markers(rewritten_text)
        
        return final_text, original_spans, rewritten_spans
    
    
    def _extract_replacement_pairs_from_marked(self, original_text: str, marked_rewritten_text: str, private_spans: List[str]) -> List[Tuple[str, str]]:

        original_spans = []
        rewritten_spans = []
        
        # Fix marker errors that the model might produce
        fixed_text = marked_rewritten_text
        # Fix based on marker format: RE prefix + any letters + >
        import re
        # Fix end marker: </RE + any letters + >
        fixed_text = re.sub(r'</RE[A-Z]+>', '</REWRITE>', fixed_text)
        # Fix start marker: <RE + any letters + >
        fixed_text = re.sub(r'<RE[A-Z]+>', '<REWRITE>', fixed_text)
        
        # Use regex to extract all content within markers
        import re
        pattern = rf'{re.escape(self.rewrite_start_marker)}(.*?){re.escape(self.rewrite_end_marker)}'
        matches = re.findall(pattern, fixed_text)
        
        # Match original privacy spans and rewritten content in order
        for i, fragment in enumerate(private_spans):
            if fragment in original_text and i < len(matches):
                # Get corresponding rewritten fragment
                rewritten_fragment = matches[i].strip()
                original_spans.append(fragment)
                rewritten_spans.append(rewritten_fragment)
            elif fragment in original_text:
                # If no corresponding rewritten content found, ignore this candidate
                return [], []  # Return empty list to indicate ignoring this candidate
        
        return original_spans, rewritten_spans

    def generate_candidates(self, text: str, private_spans: List[str], max_candidates: int = 10) -> List[Tuple[str, List[Tuple[str, str]], Dict[str, float]]]:

        candidates = []
        for i, params in enumerate(self.candidate_param_grid):
            if len(candidates) >= max_candidates:
                break
            result = self._generate_candidate_with_params(text, private_spans, params["temperature"], params["top_p"])
            if result[0] is not None:  # Only add valid candidates
                cand_text, original_spans, rewritten_spans = result
                candidates.append((cand_text, original_spans, rewritten_spans))    
        
        return candidates

    # =============== Create Candidate Set ===============
    def build_preference_for_item(self, item: Dict) -> Dict:

        text = item["question_init"]
        private_list = item["private_spans"]
        domain = item["domain"]
        
        # Generate candidates
        cands = self.generate_candidates(text, private_list, max_candidates=len(self.candidate_param_grid))
        
        # Convert tuples to lists to ensure serializability
        serializable_cands = []
        for cand in cands:
            cand_text, original_spans, rewritten_spans = cand
            serializable_cands.append({"cand_text": cand_text, "original_spans": original_spans, "rewritten_spans": rewritten_spans})
        
        return {
            **item,  # Keep all fields from original data
            "candidates": serializable_cands
        }

    def build_and_save_preference_dataset(self, output_dir: str, limit: int = None) -> str:

        from datasets import Dataset
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")
        records = []
        n = len(self.dataset) if limit is None else min(limit, len(self.dataset))
        for i in range(n):
            item = self.dataset[i]
            try:
                new_item = self.build_preference_for_item(item)
                records.append(new_item)
            except Exception as e:
                print(f"Sample {i} processing failed: {e}")
        ds = Dataset.from_list(records)
        os.makedirs(output_dir, exist_ok=True)
        ds.save_to_disk(output_dir)
        print(f"Preference dataset saved to: {output_dir}, total {len(ds)} records")
        return output_dir
    
    def remove_markers(self, text: str) -> str:

        return text.replace(self.rewrite_start_marker, "").replace(self.rewrite_end_marker, "")
    
    def rewrite_text(self, text: str, private_spans: List[str]) -> str:

        # 1. Find privacy fragments
        privacy_regions = []
        for fragment in private_spans:
            if fragment in text:
                start = text.find(fragment)
                if start != -1:
                    privacy_regions.append(RewriteRegion(
                        start=start,
                        end=start + len(fragment),
                        text=fragment,
                        type='PRIVACY'
                    ))
        
        if not privacy_regions:
            return text
        
        # 2. Mark parts that need to be rewritten
        marked_text = self.mark_text_for_rewrite(text, privacy_regions)
        
        # 3. Rewrite using model
        rewritten_text = self.rewrite_marked_text(marked_text)
        
        # 4. Remove markers and return final result
        final_text = self.remove_markers(rewritten_text)
        
        return final_text


def main():

    parser = argparse.ArgumentParser(description='Privacy rewriting system')
    parser.add_argument('--qwen_model_path', type=str, default='./models/Qwen2.5-1.5b-Instruct',
                        help='Path to the Qwen model')
    parser.add_argument('--dataset_path', type=str, default='./datasets/Pri_DDXPlus_SLJA_dpo',
                        help='Path to the input dataset')
    parser.add_argument('--max_candidates', type=int, default=10,
                        help='Maximum number of candidates to generate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device name (e.g., cuda:0, cpu)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for processing')
    parser.add_argument('--start_batch', type=int, default=1,
                        help='Starting batch number (1-indexed)')
    parser.add_argument('--end_batch', type=int, default=11,
                        help='Ending batch number (exclusive), None means process to the end')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Create output directory
    output_path = f'./datasets/DPO/candidate/Pri_DDXPlus_SLJA_dpo_candidate_batch'
    
    # Initialize system
    system = CandidateSystem(
        qwen_model_path=args.qwen_model_path,
        dataset_path=args.dataset_path,
        max_candidates=args.max_candidates,
        device=args.device
    )
    
    # Setup logging
    log_dir = f"./logs/candidate_dataset_construction"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'candidate_dataset_construction_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Generate preference dataset
    if system.dataset is not None and len(system.dataset) > 0:
        total_samples = len(system.dataset)
        batch_size = args.batch_size
        total_batches = (total_samples + batch_size - 1) // batch_size
        
        # Determine batch range to process
        start_batch = args.start_batch
        end_batch = args.end_batch if args.end_batch is not None else total_batches + 1
        
        # Validate batch range
        if start_batch < 1 or start_batch > total_batches:
            logger.error(f"Start batch {start_batch} out of range [1, {total_batches}]")
            return
        if end_batch < start_batch or end_batch > total_batches + 1:
            logger.error(f"End batch {end_batch} invalid, should be in range [{start_batch}, {total_batches + 1}]")
            return
        
        logger.info(f"Starting to generate preference dataset, total {total_samples} training samples")
        logger.info(f"Processing batch range: {start_batch} to {end_batch - 1} (total {end_batch - start_batch} batches)")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        successful_batches = []
        failed_batches = []
        
        # Process batches in specified range
        for batch_idx in range(start_batch - 1, end_batch - 1):
            batch_num = batch_idx + 1
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            
            logger.info(f"Starting to process batch {batch_num} (samples {start_idx + 1}-{end_idx})")
            
            batch_preference_data = []
            batch_success_count = 0
            
            try:
                # Process current batch - access samples one by one
                for i in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_num}"):
                    item = system.dataset[i]
                    try:
                        preference_data = system.build_preference_for_item(item)
                        if preference_data is not None:
                            batch_preference_data.append(preference_data)
                            batch_success_count += 1
                        else:
                            logger.warning(f"Batch {batch_num} sample {i + 1} build failed")
                            
                    except Exception as e:
                        logger.error(f"Batch {batch_num} sample {i + 1} processing error: {e}")
                        continue
                
                # Save current batch
                if batch_preference_data:
                    from datasets import Dataset
                    batch_dataset = Dataset.from_list(batch_preference_data)
                    batch_dir = os.path.join(output_path, f'batch_{batch_num}')
                    batch_dataset.save_to_disk(batch_dir)
                    
                    successful_batches.append(batch_num)
                    logger.info(f"Batch {batch_num} processing completed, success {batch_success_count}/{end_idx - start_idx} records")
                    logger.info(f"Batch {batch_num} saved to: {batch_dir}")
                else:
                    failed_batches.append(batch_num)
                    logger.error(f"Batch {batch_num} did not successfully generate any data")
                    
            except Exception as e:
                failed_batches.append(batch_num)
                logger.error(f"Batch {batch_num} processing failed: {e}")
                continue
        
        # Record processing results
        logger.info(f"Batch processing completed!")
        logger.info(f"Successful batches: {successful_batches}")
        if failed_batches:
            logger.error(f"Failed batches: {failed_batches}")
        
        logger.info("Preference dataset generation completed")
    else:
        logger.error("Dataset is empty or failed to load")


if __name__ == "__main__":
    main()
