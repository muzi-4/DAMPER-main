#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation pipeline: privacy detection and rewriting
"""

import torch
import random
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_from_disk
import numpy as np
from typing import List, Dict, Tuple, Optional
from peft import PeftModel
import re
import json
import os
from text_chunker import TextChunker
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
import re
from collections import deque
from transformers import AutoConfig
from models_init.roberta_lora import RoBERTaLoRA
from transformers import LogitsProcessor

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(log_dir: str = "./logs/rewrite_ours", log_level: str = "INFO") -> logging.Logger:

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename (with timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"rewrite_ours_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("rewrite_ours")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

from typing import List, Tuple

def add_rewrite_tags_all(text: str, spans: List[str]) -> str:
    if not spans:
        return text

    # 1. Find all span start and end positions in original text at once
    matches: List[Tuple[int, int]] = []
    search_start = 0

    for span in spans:

        # Find first occurrence of current span after search_start
        idx = text.find(span, search_start)

        start = idx
        end = idx + len(span)
        matches.append((start, end))

        # Use assumption that spans are ordered and non-overlapping, next search from current span end
        search_start = end

    # 2. Insert tags from back to front to avoid index disruption from previous insertions
    tagged = text
    for s, e in reversed(matches):
        tagged = tagged[:e] + "</REWRITE>" + tagged[e:]
        tagged = tagged[:s] + "<REWRITE>" + tagged[s:]

    return tagged


class AllowedVocabMaskProcessor(LogitsProcessor):


    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.mask = mask  # [vocab_size]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [batch, seq_len]
        scores:    [batch, vocab_size]  logits for current step
        Returns:   [batch, vocab_size]
        """
        return scores + self.mask

class ClipLogitsProcessor(LogitsProcessor):
    def __init__(self, clip_min: float, clip_max: float):
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [batch, vocab_size]
        return scores.clamp(self.clip_min, self.clip_max)

class EvaluationPipeline:
    """Evaluation pipeline class"""
    
    def __init__(self, 
                 roberta_base_path : str = "./models/roberta-base",
                 roberta_lora_path: str = "./saved_models/roberta_lora",
                 qwen_lora_path: str = "./saved_models/qwen2.5_1.5b_lora_dpo_merge",
                #  qwen_lora_path: str = "./models/Qwen2.5-1.5b-Instruct",
                 domain_prototypes_path: str = "./saved_models/domain_prototypes/domain_prototypes.pt",
                 similarity_threshold: float = 0.9,
                 epsilon: float = 50,
                 device: str = "cuda:0",
                 logger: str = "./logs/rewrite_ours",
                 log_level: str = "INFO"):
        """
        Initialize evaluation pipeline
        
        Args:
            roberta_lora_path: Path to RoBERTa adapter model
            qwen_lora_path: Path to Qwen DPO adapter model
            domain_prototypes_path: Path to domain prototype file
            similarity_threshold: Similarity threshold
            device: Device
            log_level: Log level
        """
        self.device = device
        self.similarity_threshold = similarity_threshold

        # Setup DP
        self.epsilon = epsilon
        self.clip_min = 7.3307
        self.clip_max = 22.7108
        
        # Setup logger
        self.logger = logger
        
        # Initialize text chunker
        self.chunker = TextChunker()
        
        # Load models
        self._load_models(roberta_base_path, roberta_lora_path, qwen_lora_path)
        
        # Load domain prototypes
        self._load_domain_prototypes(domain_prototypes_path)


        # 1) Construct vocab mask
        self.build_allowed_vocab_mask()

        # 2) Construct logits_processor (only one processor needed)
        self.logits_processor = [
        ClipLogitsProcessor(self.clip_min, self.clip_max),          # 1) First clip logits
        AllowedVocabMaskProcessor(self.allowed_vocab_mask),         # 2) Then add vocab mask
        ]
        
    def _load_models(self, roberta_base_path: str, roberta_lora_path: str, qwen_lora_path: str):

        self.logger.info("Loading models...")
        
        try:  
            # Load RoBERTa adapter model (semantic encoding)
            self.logger.info(f"Loading RoBERTa adapter model: {roberta_lora_path}")
            self.roberta_model = RoBERTaLoRA(roberta_base_path)
            self.roberta_model.to(self.device)
            roberta_base = AutoModel.from_pretrained(roberta_base_path)
            roberta_lora = PeftModel.from_pretrained(roberta_base, roberta_lora_path)
            self.roberta_model.backbone = roberta_lora.to(self.device)
            self.roberta_model.eval()
            
            # Directly load merged complete policy model (no need for Adapter instantiation)
            self.logger.info(f"Directly loading merged DPO model: {qwen_lora_path}")
            self.dpo_tokenizer = AutoTokenizer.from_pretrained(qwen_lora_path)
            if self.dpo_tokenizer.pad_token is None:
                self.dpo_tokenizer.pad_token = self.dpo_tokenizer.eos_token
            self.dpo_model = AutoModelForCausalLM.from_pretrained(qwen_lora_path)
            self.dpo_model.to(self.device)
            self.dpo_model.eval()
            self.logger.info("✓ DPO merged model loading completed")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
    
    def _load_domain_prototypes(self, domain_prototypes_path: str):

        self.logger.info("Loading domain prototypes...")
        try:
            prototypes_data = torch.load(domain_prototypes_path, map_location=self.device)
            
            # Extract medical and legal domain prototypes
            self.medical_prototypes = prototypes_data['medical_prototypes']  # [num_medical_prototypes, hidden_dim]
            self.legal_prototypes = prototypes_data['legal_prototypes']     # [num_legal_prototypes, hidden_dim]
            
            self.logger.info(f"✓ Domain prototypes loading completed")
            self.logger.info(f"  Medical domain prototype shape: {self.medical_prototypes.shape}")
            self.logger.info(f"  Legal domain prototype shape: {self.legal_prototypes.shape}")
            
        except Exception as e:
            self.logger.error(f"Domain prototype loading failed: {e}")
            raise

    def build_allowed_vocab_mask(self):

        # Note: Model output layer vocab size may not exactly match tokenizer.get_vocab() length
        # Here use model's actual output dimension to ensure alignment with logits' last dimension.
        vocab = self.dpo_tokenizer.get_vocab()
        model_vocab_size = self.dpo_model.get_output_embeddings().weight.size(0)

        mask = torch.zeros(model_vocab_size, dtype=torch.float32)

        # Some helpers
        def looks_ok(decoded: str) -> bool:
            # Only accept ASCII
            if any(ord(ch) > 127 for ch in decoded):
                return False

            # Allow pure whitespace (space token)
            if decoded.strip() == "" and " " in decoded:
                return True

            # Must have at least one letter
            has_alpha = any(ch.isalpha() for ch in decoded)
            if not has_alpha:
                return False

            # Cannot contain too many strange punctuation; simple rule: only letters / spaces / hyphens
            for ch in decoded:
                if not (ch.isalpha() or ch.isspace() or ch == "-"):
                    return False

            return True

        for token, idx in vocab.items():
            # If id in tokenizer exceeds model output layer range, skip directly (this id has no corresponding position in model)
            if idx >= model_vocab_size:
                continue
            # Default allow
            allow = True

            # eos token must be kept
            if hasattr(self.dpo_tokenizer, "eos_token_id") and idx == self.dpo_tokenizer.eos_token_id:
                allow = True
            else:
                # Decode single token, check real text
                try:
                    decoded = self.dpo_tokenizer.decode([idx], skip_special_tokens=False)
                except Exception:
                    decoded = token  # fallback

                if not looks_ok(decoded):
                    allow = False

            if not allow:
                mask[idx] = -1e10  # or float("-inf")

        self.allowed_vocab_mask = mask.to(self.device)


    def classify_domain(self, raw_spans: List[str]) -> int:
        medical_scores = []
        legal_scores = []
        for span in raw_spans:
            emb = self.encode_text(span)
            medical_sim = self.compute_similarity(emb, self.medical_prototypes)
            legal_sim = self.compute_similarity(emb, self.legal_prototypes)
            medical_scores.append(medical_sim)
            legal_scores.append(legal_sim)
        
        if sum(medical_scores) >= sum(legal_scores):
            return 0
        else:
            return 1
    
    def encode_text(self, text: str) -> torch.Tensor:

        return self.roberta_model.encode_span(text)
    
    def compute_similarity(self, span_embedding: torch.Tensor, domain_prototypes: torch.Tensor) -> float:
        
        # Compute cosine similarity with each prototype, then take mean
        similarities = F.cosine_similarity(span_embedding, domain_prototypes, dim=1)
        mean_similarity = torch.mean(similarities).item()
        return mean_similarity

    def privacy_span_detection(self, span: str, domain_prototypes: torch.Tensor) -> str:
        # Encode span
        span_embedding = self.encode_text(span)    

        # Compute similarity with domain prototypes
        mean_similarity = self.compute_similarity(span_embedding, domain_prototypes)

        if mean_similarity >= self.similarity_threshold:
            return True, mean_similarity
        else:
            return False, mean_similarity
    
    def detect_privacy_spans(self, question_init: str) -> List[Tuple[str, float]]:

        def get_first_match(span: str) -> Tuple[int, int]:

            pattern = re.compile(re.escape(span))
            m = pattern.search(question_init)
            return (m.start(), m.end())

        def has_overlap(a: Dict[str, float], b: Dict[str, float]) -> bool:
            return not (a["end"] <= b["start"] or a["start"] >= b["end"])

        def trim_segment(segment: Dict[str, float], cut_start: int, cut_end: int) -> List[Dict[str, float]]:

            seg_start, seg_end = segment["start"], segment["end"]

            # No overlap
            if cut_end <= seg_start or cut_start >= seg_end:
                return [segment]

            new_segments: List[Dict[str, float]] = []

            # Left residual segment
            if seg_start < cut_start:
                s1, e1 = seg_start, cut_start
                left_text = question_init[s1:e1]
                if left_text.strip():
                    is_priv, sim = self.privacy_span_detection(left_text, domain_prototypes)
                    if is_priv:
                        new_segments.append({
                            "start": s1,
                            "end": e1,
                            "score": sim,
                            "span": left_text
                        })

            # Right residual segment
            if seg_end > cut_end:
                s2, e2 = cut_end, seg_end
                right_text = question_init[s2:e2]
                if right_text.strip():
                    is_priv, sim = self.privacy_span_detection(right_text, domain_prototypes)
                    if is_priv:
                        new_segments.append({
                            "start": s2,
                            "end": e2,
                            "score": sim,
                            "span": right_text
                        })

            return new_segments

        # 1. Text chunking
        raw_spans = self.chunker.chunk_text(question_init)    

        # 2. Domain classification
        domain = self.classify_domain(raw_spans)
        if domain == 0:  # medical
            domain_prototypes = self.medical_prototypes
        else:            # legal
            domain_prototypes = self.legal_prototypes

        # 3. Pre-cache all occurrence positions of each span in text (using regex + cache)
        match_cache: Dict[str, List[Tuple[int, int]]] = {}
        

        # 4. First build initial candidate entries (already filtered once by threshold)
        entries: List[Dict[str, float]] = []

        for span in raw_spans:
            is_privacy, max_similarity = self.privacy_span_detection(span, domain_prototypes)

            if not is_privacy:
                continue

            if span not in match_cache:
                pos = get_first_match(span)
                match_cache[span] = pos
            
            pos = match_cache[span]
            start, end = pos

            entries.append({
                "start": start,
                "end": end,
                "score": max_similarity,
                "span": span
            })

        # 5. Sort by occurrence order, if start positions are same then higher score first
        entries.sort(key=lambda e: (e["start"], -e["score"]))

        resolved: List[Dict[str, float]] = []

        for entry in entries:
            pending = deque([entry])

            while pending:
                current = pending.popleft()
                if current["end"] <= current["start"]:
                    continue

                overlapped = False

                for idx, kept in enumerate(resolved):
                    if not has_overlap(current, kept):
                        continue

                    overlapped = True

                    # Higher score priority
                    if current["score"] > kept["score"]:
                        trimmed = trim_segment(kept, current["start"], current["end"])
                        resolved.pop(idx)
                        for seg in reversed(trimmed):
                            resolved.insert(idx, seg)
                        pending.appendleft(current)
                    else:
                        trimmed = trim_segment(current, kept["start"], kept["end"])
                        for seg in reversed(trimmed):
                            pending.appendleft(seg)

                    break

                if not overlapped:
                    resolved.append(current)

        # 6. Sort
        resolved.sort(key=lambda e: e["start"])

        # 7. Final: deduplicate by text content (only keep first occurrence)
        seen = set()
        privacy_spans = []
        for seg in resolved:
            t = seg["span"]
            if t not in seen:
                seen.add(t)
                privacy_spans.append(t)

        return privacy_spans

    def build_messages_dp(self, text: str, privacy_span: str) -> List[Dict]:
        """Construct prompts following test_dpo_model.py: system+user, requiring JSON array of replacements."""

        system_content = f"""You are a sophisticated privacy-focused text anonymizer.

You will rewrite exactly ONE privacy span per request.

CRITICAL INSTRUCTIONS:
1. You will receive:
   - The full original text
   - one privacy span
2. Rewrite the provided span into a generic, non-identifying alternative.
3. Maintain semantic fluency and coherence with the original text.
4. Output ONLY the rewritten span as a plain string.
5. DO NOT output JSON, lists, keys, quotes, brackets, or multiple fields.
6. The output must be a single line of text with no surrounding punctuation.
7. The replacement MUST be a short English word or phrase (1–8 words), using ONLY letters, spaces, or hyphens.
8. The output should be concise and not a long sentence.
9. The replacement should contain approximately the same number of words as the privacy span (±2 words). Do not expand it into a long description.

If anything is invalid, output an empty string.

Example:
Original text: 
A patient has fever and headache.

Privacy span: 
fever

Replacement spans:
sickness
"""

        user_content = f"""
Original text:
{text}

Privacy span:
{privacy_span}

Replacement spans:"""

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def rewrite_text_dp(self, text: str, privacy_spans: List[str]) -> Tuple[str, List[str]]:

        if len(privacy_spans) == 0:
            return text, []

        dp_temperature = self.dp_temperature

        # 1) Construct batch messages
        all_messages = [
            self.build_messages_dp(text, span)
            for span in privacy_spans
        ]

        # 2) Encode as batch input
        inputs = self.dpo_tokenizer.apply_chat_template(
            all_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        batch_size = inputs.size(0)

        word_counts = []
        for span in privacy_spans:
            # Simple word splitting by space; defend against empty string
            wc = len(span.split())
            if wc <= 0:
                wc = 1
            word_counts.append(wc)

        # Target upper limit for each span: original word count + 2
        target_word_counts = [wc + 2 for wc in word_counts]

        # generate's max_new_tokens can only be one number → take the maximum among all spans
        max_new_tokens = max(target_word_counts)

        # 3) Generate once, using temperature + logits_processor
        with torch.no_grad():
            outputs = self.dpo_model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.dpo_tokenizer.eos_token_id,
                temperature=dp_temperature,
                do_sample=True,
                top_p=1.0,
                logits_processor=self.logits_processor,
                return_dict_in_generate=True,
            )

        gen_ids = outputs.sequences[:, inputs.size(1):]  # [B, T_gen]
        if gen_ids.size(0) != batch_size:
            raise ValueError(f"[DP] Generated batch size mismatch: {gen_ids.size(0)} vs {batch_size}")

        # 4) Decode + clean + truncate
        rewritten_spans: List[str] = []

        for i in range(batch_size):
            ids_i = gen_ids[i]

            if ids_i.numel() == 0:
                raise ValueError(f"[DP] Span {i} generation length is 0, considered failure.")

            raw_out = self.dpo_tokenizer.decode(
                ids_i,
                skip_special_tokens=True
            ).strip()

            # Remove newlines
            raw_out = raw_out.replace("\n", " ").strip()

            cleaned = self._clean_span_whitespace(raw_out)

            if cleaned == "":
                raise ValueError(f"[DP] Span {i} is empty after cleaning, considered failure.")

            rewritten_spans.append(cleaned)

        # 5) Search and replace in original text in order
        original = text
        positions: List[Tuple[int, int]] = []
        search_pos = 0

        for idx, span in enumerate(privacy_spans):
            start = original.find(span, search_pos)
            if start == -1:
                raise ValueError(
                    f"[DP] Cannot find privacy_span {idx} in original text: '{span}', "
                    f"search failed starting from position {search_pos}."
                )
            end = start + len(span)
            positions.append((start, end))
            search_pos = end

        result_parts: List[str] = []
        last_end = 0
        for (start, end), replacement in zip(positions, rewritten_spans):
            result_parts.append(original[last_end:start])
            result_parts.append(replacement)
            last_end = end
        result_parts.append(original[last_end:])

        result = "".join(result_parts)

        return result, rewritten_spans

    def _clean_span_whitespace(self, generated: str) -> str:

        if generated is None:
            return ""
        # Replace all consecutive whitespace with single space
        cleaned = re.sub(r"\s+", " ", generated).strip()
        return cleaned

    def rewrite_text(self, marked_text: str, privacy_spans: List[str]) -> Tuple[str, List[str]]:

        # 1) Build chat messages
        messages = self.build_messages(marked_text, privacy_spans)

        # 2) Encode as model input
        inputs = self.dpo_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        # 3) Generate replacement spans in JSON array format
        with torch.no_grad():
            outputs = self.dpo_model.generate(
                inputs,
                max_new_tokens=256,
                pad_token_id=self.dpo_tokenizer.eos_token_id,
                # Use default configuration from generation_config.json
                # Parameters like do_sample=True, temperature=0.6, top_p=0.9 will be automatically loaded from config file
            )
        gen_ids = outputs[:, inputs.size(1):]
        model_out = self.dpo_tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        # 4) Parse JSON array, apply replacements, get final text
        try:
            # Try to fix model output: if starts with '[' but doesn't end with ']'
            # - If ends with comma: remove trailing comma and whitespace then add ']'
            # - If ends with double or single quote: directly add ']'
            fixed_out = model_out
            if fixed_out.startswith('[') and not fixed_out.endswith(']'):
                trimmed = fixed_out.rstrip()
                if trimmed.endswith(','):
                    # Only remove when there is indeed a trailing comma, avoid accidentally deleting valid separator comma
                    trimmed = trimmed[:-1].rstrip()
                    fixed_out = trimmed + "]"
                    self.logger.warning("Model output ends with comma, removed trailing comma and completed ']'.")
                elif trimmed.endswith('"') or trimmed.endswith("'"):
                    fixed_out = trimmed + "]"
                    self.logger.warning("Model output ends with quote, completed ']'.")
            replacements = json.loads(fixed_out)
            if not isinstance(replacements, list):
                raise ValueError("Model output is not a JSON array")
            # Validate replacement count must match privacy span count, otherwise consider failure, let upper layer ignore this sample
            if len(replacements) != len(privacy_spans):
                raise ValueError(
                    f"Replacement JSON length mismatch: expected {len(privacy_spans)}, actual {len(replacements)}. Output: {fixed_out}"
                )
        except Exception as e:
            # print('Privacy spans:', privacy_spans)
            # print('Rewritten spans:', model_out)
            raise ValueError(f"Replacement JSON parsing failed. Output: {model_out}, error: {e}")

        # Replace each <REWRITE>...</REWRITE> segment in marked text in order with corresponding replacements[i]
        result = marked_text
        try:
            # Find each <REWRITE>...</REWRITE> segment in order and replace
            replacement_index = 0
            while replacement_index < len(replacements):
                # Find next <REWRITE> marker
                start = result.find("<REWRITE>")
                if start == -1:
                    break
                
                # Find corresponding </REWRITE> marker
                end = result.find("</REWRITE>", start)
                if end == -1:
                    break
                
                # Replace entire <REWRITE>...</REWRITE> segment
                result = result[:start] + replacements[replacement_index] + result[end + len("</REWRITE>"):]
                replacement_index += 1
        except Exception as e:
            raise ValueError(f"Error applying replacements. Error: {e}")

        return result, replacements

    
    def evaluate_single(self, question_init: str) -> Dict:

        # 1) Text chunking
        split_spans = self.chunker.chunk_text(question_init)
        
        # 2) Detect privacy spans
        privacy_spans = self.detect_privacy_spans(question_init)

        # 4) Add <REWRITE> tags to original text based on non-overlapping matches
        marked_text = add_rewrite_tags_all(question_init, privacy_spans)

        # 5) Rewrite text (retry up to 3 times on failure)
        last_error = None
        rewritten_text = None
        rewrite_spans = None

        for attempt in range(3):
            try:
                rewritten_text, rewrite_spans = self.rewrite_text_dp(question_init, privacy_spans)
                break
            except Exception as e:
                last_error = e
                self.logger.warning(f"Rewrite attempt failed (attempt {attempt + 1}/3): {e}")

        if rewritten_text is None or rewrite_spans is None:
            # Failed 3 times consecutively, raise exception to let upper layer ignore this sample
            raise last_error if last_error else ValueError("Rewrite failed, unknown error")

        # 6) Clean rewritten text, remove markers
        clean_rewritten_text = self._clean_rewrite_tags(rewritten_text)

        # 7) Build result
        result = {
            "split_spans": split_spans,
            "detect_private_spans": privacy_spans,
            "rewrite_spans": rewrite_spans,
            "rewrite_text": rewritten_text
        }  
        
        return result
    
    def calculate_max_detect_spans(self, dataset_path: str) -> int:
        dataset = load_from_disk(dataset_path)
        max_private_spans_length = 0
        for item in tqdm(dataset):
            question_init = item["question_init"]
            private_spans = self.detect_privacy_spans(question_init)
            concatenated_spans = "".join(private_spans)
            encoding = self.dpo_tokenizer(concatenated_spans, add_special_tokens=False)
            spans_token_count = len(encoding["input_ids"])
            max_private_spans_length = max(max_private_spans_length, spans_token_count)

        return max_private_spans_length

    def _clean_rewrite_tags(self, text: str) -> str:

        import re
        # Remove all <REWRITE> and </REWRITE> markers, including incomplete marker fragments
        cleaned_text = re.sub(r'<REWRITE[^>]*>', '', text)  # Remove <REWRITE> and its variants
        cleaned_text = re.sub(r'</REWRITE[^>]*>', '', cleaned_text)  # Remove </REWRITE> and its variants
        # Additional cleanup of possible residual fragments
        return cleaned_text
    
    def evaluate_dataset(self, dataset_path: str, max_private_spans_length: int) -> List[Dict]:

        self.logger.info(f"Loading dataset: {dataset_path}")
        try:
            dataset = load_from_disk(dataset_path)
            self.logger.info(f"Dataset size: {len(dataset)}")
        except Exception as e:
            self.logger.error(f"Dataset loading failed: {e}")
            raise

        self.dp_temperature = 2*(self.clip_max - self.clip_min)*max_private_spans_length/self.epsilon
        
        results = []
        error_count = 0
        total_samples = len(dataset)
        
        for i, sample in enumerate(tqdm(dataset, desc="Evaluation progress")):
            question_init = sample["question_init"]
            
            try:
                # Get evaluation result
                evaluation_result = self.evaluate_single(question_init)
                
                # Keep original fields and add new fields
                result = dict(sample)  # Keep all original fields
                result.update(evaluation_result)  # Add new evaluation fields
                
                results.append(result)
                
                # Record progress information
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1} samples, success: {len(results)}, failed: {error_count}")
                    
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error processing sample {i + 1}, skipping this sample. Error: {e}")
                # Skip this sample, do not add to results
                continue
        
        # Statistics of final results
        successful_samples = len(results)
        self.logger.info(f"Evaluation completed!")
        self.logger.info(f"Initial sample count: {total_samples}")
        self.logger.info(f"Successfully processed: {successful_samples}")
        self.logger.info(f"Processing failed: {error_count}")
        self.logger.info(f"Success rate: {successful_samples/total_samples*100:.2f}%")
        
        return results
    
    def save_evaluation_results(self, results: List[Dict], output_path: str):

        self.logger.info(f"Saving evaluation results to: {output_path}")
        
        try:
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert results to Dataset format
            from datasets import Dataset
            dataset = Dataset.from_list(results)
            
            # Save as Arrow format
            dataset.save_to_disk(output_path)
            
            self.logger.info(f"✓ Evaluation results saved to: {output_path}")
            self.logger.info(f"  Saved sample count: {len(results)}")
            self.logger.info(f"  Field count: {len(results[0]) if results else 0}")
            
            # Display statistics for new fields
            if results:
                new_fields = ["split_spans", "detect_private_spans", "rewrite_spans", "rewrite_text"]
                for field in new_fields:
                    if field in results[0]:
                        if field == "split_spans":
                            avg_spans = sum(len(r.get(field, [])) for r in results) / len(results)
                            self.logger.info(f"  Average chunk count: {avg_spans:.2f}")
                        elif field == "detect_private_spans":
                            avg_private = sum(len(r.get(field, [])) for r in results) / len(results)
                            self.logger.info(f"  Average privacy span count: {avg_private:.2f}")
                        elif field == "rewrite_spans":
                            avg_rewrite = sum(len(r.get(field, [])) for r in results) / len(results)
                            self.logger.info(f"  Average rewrite span count: {avg_rewrite:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to save evaluation results: {e}")
            raise
    
    def print_evaluation_summary(self, results: List[Dict]):

        if not results:
            self.logger.warning("No evaluation results")
            return
        
        total_samples = len(results)
        total_spans = sum(r["detection_info"]["total_spans"] for r in results)
        total_privacy_spans = sum(r["detection_info"]["privacy_spans"] for r in results)
        
        medical_samples = sum(1 for r in results if r["detection_info"]["domain"] == "medical")
        legal_samples = sum(1 for r in results if r["detection_info"]["domain"] == "legal")
        
        summary = f"""
{'='*50} Evaluation Summary {'='*50}
Total samples: {total_samples}
Medical domain samples: {medical_samples}
Legal domain samples: {legal_samples}
Total spans: {total_spans}
Privacy spans: {total_privacy_spans}
Privacy span ratio: {total_privacy_spans/total_spans*100:.2f}%
Similarity threshold: {self.similarity_threshold}
{'='*110}
        """
        
        print(summary)
        self.logger.info(f"Evaluation summary - Total samples: {total_samples}, Medical domain: {medical_samples}, Legal domain: {legal_samples}, Privacy spans: {total_privacy_spans}/{total_spans} ({total_privacy_spans/total_spans*100:.2f}%)")


def main():

    parser = argparse.ArgumentParser(description="Evaluation pipeline: privacy detection and rewriting")
    parser.add_argument("--dataset_name", type=str, default="Pri_DDXPlus",
                       help="Path to the test dataset")
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature parameter for InfoNCE loss')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta parameter for DPO training')
    parser.add_argument('--preference_alpha', type=float, default=0.3,
                        help='Weight for domain similarity in preference scoring')
    parser.add_argument("--similarity_threshold", type=float, default=0.8,
                       help="Similarity threshold for privacy span detection")
    parser.add_argument("--epsilon", type=float, default=150,
                       help="Differential privacy epsilon parameter")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device name (e.g., cuda:0, cpu)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()

    set_seed(args.seed)

    # Create logger
    log_dir = f"./logs/rewrite_ours/temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold}/epsilon_{int(args.epsilon)}"
    logger = setup_logger(log_dir)
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(
        qwen_lora_path=f"./saved_models/qwen2.5_1.5b_dpo/temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/qwen2.5_1.5b_lora_dpo_merge",
        # qwen_lora_path=f"./models/Qwen2.5-1.5b-Instruct",
        domain_prototypes_path=f"./saved_models/domain_prototypes/temperature_{args.temperature}/domain_prototypes.pt",
        roberta_lora_path=f"./saved_models/roberta_lora/temperature_{args.temperature}",
        similarity_threshold=args.similarity_threshold,
        epsilon=args.epsilon,
        device=args.device,
        logger=logger,
        log_level=args.log_level
    )
    
    # Process dataset
    dataset_path = f"./datasets/Evaluation/{args.dataset_name}/{args.dataset_name}_eval"
    max_private_spans_length = pipeline.calculate_max_detect_spans(dataset_path)
    print(f"max_private_spans_length: {max_private_spans_length}")
    results = pipeline.evaluate_dataset(dataset_path, max_private_spans_length)
    
    # Save processed dataset
    pipeline.save_evaluation_results(results, f"./datasets/Evaluation/{args.dataset_name}/temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold}/epsilon_{int(args.epsilon)}/{args.dataset_name}_eval_rewrite_ours")

if __name__ == "__main__":
    main()
