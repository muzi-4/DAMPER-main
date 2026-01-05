#!/usr/bin/env python3
"""
Privacy rewriting system - subtask implementation
Selective rewriting based on privacy spans in the dataset
"""

import os
import json
import re
import random
import math
import bisect
import numpy as np
import torch
import argparse
import warnings
import logging
from datetime import datetime
# Filter pynndescent warnings
warnings.filterwarnings("ignore", message="pynndescent not installed")
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset, load_from_disk
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import sys
sys.path.append(os.path.dirname(__file__))
from models_init.roberta_lora import RoBERTaLoRA
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def set_seed(seed: int):
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


class PrivacyRewriteSystem:
    
    def __init__(self, 
                 qwen_model_path: str = "./models/Qwen2.5-1.5b-Instruct",
                 semantic_model_path: str = "./models/all-MiniLM-L6-v2",
                 roberta_base_path: str = "./models/roberta-base",
                 roberta_lora_path: str = "./saved_models/roberta_lora",
                 dataset_path: str = "./datasets/DPO/candidate/Pri_DDXPlus_SLJA_dpo_candidate",
                 temperature: float = 0.1,
                 preference_alpha: float = 0,
                 max_candidates: int = 10,
                 device: str = "cuda:0",
                 logger: logging.Logger = None):
        """
        Initialize privacy rewriting system
        
        Args:
            qwen_model_path: Path to the Qwen model
            dataset_path: Path to the dataset
            preference_alpha: Domain similarity weight
            max_candidates: Maximum number of candidates
            device: Device name
        """
        self.qwen_model_path = qwen_model_path
        self.semantic_model_path = semantic_model_path
        self.roberta_base_path = roberta_base_path
        self.dataset_path = dataset_path
        self.preference_alpha = preference_alpha
        self.max_candidates = max_candidates
        self.device = device
        self.logger = logger

        self.temperature = temperature
        self.roberta_lora_path = os.path.join(roberta_lora_path, f'temperature_{temperature}')

        
        # Load models
        self._load_models()
        self._load_roberta_adapter()
        self._load_semantic_model()
        
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
        
        # Preload domain prototypes
        self.domain_prototypes = {}
        self._load_domain_prototypes()

        # Score sorted arrays (set after warmup)
        self.domain_scores_sorted = []
        self.leak_scores_sorted = []
        self.domain_score_mean = 0.0
        self.domain_score_std = 1.0
        self.leak_score_mean = 0.0
        self.leak_score_std = 1.0

        # Load saved mean and std
        self._load_mean_std()
    
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
            self.logger.warning(f"Warning: Unable to load Qwen model {self.qwen_model_path}: {e}")
            self.tokenizer = None
            self.model = None
    
    def _load_roberta_adapter(self):

        try:
            self.roberta_model = RoBERTaLoRA(self.roberta_base_path)
            self.roberta_model.to(self.device)
            roberta_base = AutoModel.from_pretrained(self.roberta_base_path)
            roberta_lora = PeftModel.from_pretrained(roberta_base, self.roberta_lora_path)
            self.roberta_model.backbone = roberta_lora.to(self.device)
            self.roberta_model.eval()
            
        except Exception as e:
            self.logger.warning(f"Warning: Unable to load RoBERTa adapter model: {e}")
            self.roberta_model = None

    def _load_semantic_model(self):

        try:
            self.semantic_model = SentenceTransformer(self.semantic_model_path, device=self.device)
            self.semantic_model.eval()
        except Exception as e:
            self.logger.warning(f"Warning: Unable to load semantic model: {e}")
            self.semantic_model = None
    
    def _load_domain_prototypes(self):

        if self.dataset is not None:
            # Get all unique domains
            domains = set()
            for item in self.dataset:
                if 'domain' in item:
                    domains.add(item['domain'])
            
            # Preload prototypes for each domain
            for domain in domains:
                try:
                    prototype = self._load_single_domain_prototype(domain)
                    self.domain_prototypes[domain] = prototype
                except Exception as e:
                    self.logger.warning(f"Domain prototype loading failed {domain}: {e}")
                    self.domain_prototypes[domain] = None
    
    def _load_single_domain_prototype(self, domain: str) -> torch.Tensor:

        proto_file = os.path.join(os.path.dirname(__file__), "saved_models", "domain_prototypes", f"temperature_{self.temperature}","domain_prototypes.pt")
        
        try:
            if os.path.exists(proto_file):
                data = torch.load(proto_file, map_location=self.device)
                
                # Select corresponding prototype based on domain name
                if domain == "medical" and "medical_prototypes" in data:
                    prototypes = data["medical_prototypes"]  # [18, 768]
                elif domain == "legal" and "legal_prototypes" in data:
                    prototypes = data["legal_prototypes"]  # [128, 768]
                else:
                    self.logger.warning(f"Warning: Prototype for domain '{domain}' does not exist")
                return prototypes
            else:
                self.logger.warning(f"Warning: Domain prototype file does not exist {proto_file}")
                
        except Exception as e:
            self.logger.warning(f"Failed to load domain prototype: {e}")

    def _load_mean_std(self):

        mean_std_file = os.path.join(
            os.path.dirname(__file__), 
            "saved_models", 
            "mean_std", 
            f"temperature_{self.temperature}",
            "mean_std.json"
        )
        
        if os.path.exists(mean_std_file):
            with open(mean_std_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.domain_scores_sorted = data["domain_scores"]
            self.leak_scores_sorted = data["leak_scores"]

            # Ensure scores are in [-1, 1] range, truncate if out of range
            domain_scores_array = np.array(self.domain_scores_sorted)  # Avoid infinity from arctanh(1) or arctanh(-1)
            domain_scores_fisherz = np.arctanh(domain_scores_array)
            
            # Calculate mean and variance for transformed domain scores
            self.domain_score_mean = float(np.mean(domain_scores_fisherz))
            self.domain_score_std = float(np.std(domain_scores_fisherz))

            self.leak_score_mean = float(np.mean(self.leak_scores_sorted))
            self.leak_score_std = float(np.std(self.leak_scores_sorted))

            self.warmup_completed = True
            
            self.logger.info(f"Successfully loaded sorted arrays: {mean_std_file}")
            self.logger.info(f"  Domain Score: total {len(self.domain_scores_sorted)} scores (sorted)")
            self.logger.info(f"  Leak Score: total {len(self.leak_scores_sorted)} scores (sorted)")
        else:
            print(f"Warmup file does not exist: {mean_std_file}")
            total_samples = len(self.dataset)
            warmup_size = max(1, int(total_samples * 0.05))

            # Randomly sample indices
            indices = random.sample(range(len(self.dataset)), warmup_size)
            domain_scores = []
            leak_scores = []

            for idx in tqdm(indices, desc="Warming up"):
                try:
                    item = self.dataset[idx]
                    domain = item["domain"]
                    cands = item["candidates"]
                    
                    # Calculate raw scores (without normalization)
                    for cand in cands:
                        s_leak = self._compute_leak_score(cand["original_spans"], cand["rewritten_spans"])
                        s_domain = self._compute_domain_score(domain, cand["rewritten_spans"])
                        domain_scores.append(s_domain)
                        leak_scores.append(s_leak)
                except Exception as e:
                    self.logger.warning(f"Warmup sample {idx} processing failed: {e}")
                    continue

            domain_scores_sorted = sorted(domain_scores)
            leak_scores_sorted = sorted(leak_scores)

            # Create parent directory of the file (not the file itself)
            os.makedirs(os.path.dirname(mean_std_file), exist_ok=True)

            result = {
                "domain_scores": domain_scores_sorted,
                "leak_scores": leak_scores_sorted,
                "num_domain_scores": len(domain_scores_sorted),
                "num_leak_scores": len(leak_scores_sorted),
                "warmup_size": warmup_size,
                "total_samples": total_samples,
            }
            with open(mean_std_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to: {mean_std_file}")
            self.logger.info("Warmup process completed!")

            self.domain_scores_sorted = domain_scores_sorted
            self.leak_scores_sorted = leak_scores_sorted

            # Ensure scores are in [-1, 1] range, truncate if out of range
            domain_scores_array = np.array(self.domain_scores_sorted)  # Avoid infinity from arctanh(1) or arctanh(-1)
            domain_scores_fisherz = np.arctanh(domain_scores_array)
            
            # Calculate mean and variance for transformed domain scores
            self.domain_score_mean = float(np.mean(domain_scores_fisherz))
            self.domain_score_std = float(np.std(domain_scores_fisherz))

            self.leak_score_mean = float(np.mean(self.leak_scores_sorted))
            self.leak_score_std = float(np.std(self.leak_scores_sorted))

            self.warmup_completed = True
            
            self.logger.info(f"Successfully loaded sorted arrays: {mean_std_file}")
            self.logger.info(f"  Domain Score: total {len(self.domain_scores_sorted)} scores (sorted)")
            self.logger.info(f"  Leak Score: total {len(self.leak_scores_sorted)} scores (sorted)")

    # =============== Preference Dataset: Scoring ===============
    def _semantic_embedding(self, texts: List[str]) -> torch.Tensor:

        with torch.no_grad():
            emb = self.semantic_model.encode(texts, convert_to_tensor=True, device=self.device)
            return emb
    def _compute_leak_score(self, original_spans: List[str], rewritten_spans: List[str]) -> float:

        if not original_spans or not rewritten_spans:
            return 1.0

        orig_emb = self._semantic_embedding(original_spans)
        repl_emb = self._semantic_embedding(rewritten_spans)
        
        # Ensure tensors are 2D (if only 1 element, sentence-transformers may return 1D tensor)
        if orig_emb.dim() == 1:
            orig_emb = orig_emb.unsqueeze(0)  # [dim] -> [1, dim]
        if repl_emb.dim() == 1:
            repl_emb = repl_emb.unsqueeze(0)  # [dim] -> [1, dim]
        
        cos_sim = torch.cosine_similarity(orig_emb, repl_emb, dim=1)
        avg_sim = cos_sim.mean().item()
        return 1.0 - avg_sim

    def _text_embedding(self, text: str) -> torch.Tensor:

        with torch.no_grad():
            emb = self.roberta_model.encode_span([text])
            return emb[0]  # [768]

    def _compute_domain_score(self, domain: str, rewritten_spans: List[str]) -> float:

        if not rewritten_spans:
            return 0.0
        
        
        proto = self.domain_prototypes[domain]
        
        if proto.dim() == 1:
            proto = proto.unsqueeze(0)  # [1, 768]
        
        sims = []
        for repl in rewritten_spans:
            emb = self._text_embedding(repl)  # [768]
            if emb.numel() == 1 and emb.sum() == 0:
                continue
            
            # Expand dimensions to match prototype shape
            emb = emb.unsqueeze(0)  # [1, 768]
            
            # Compute cosine similarity with all prototypes
            # proto: [num_prototypes, 768], emb: [1, 768]
            # Need to broadcast to compute similarity between each prototype and rewritten span
            cos_sims = torch.cosine_similarity(proto, emb.expand_as(proto), dim=1)  # [num_prototypes]
            
            # Take maximum
            max_sim = torch.max(cos_sims).item()

            sims.append(max_sim)
        
        if not sims:
            return 0.0
        
        return float(sum(sims) / len(sims))

    def score_candidates(self, original_text: str, private_fragments: List[str], domain: str,
                         candidates: List[Tuple[str, List[Tuple[str, str]], Dict[str, float]]]) -> List[Dict]:
        
        if not self.warmup_completed or not self.domain_scores_sorted or not self.leak_scores_sorted:
            raise RuntimeError("Sorted arrays not loaded, please run mean_std_pre.py for warmup first")
        
        results = []
        for cand in candidates:
            s_leak_raw = self._compute_leak_score(cand["original_spans"], ["rewritten_spans"])
            s_domain_raw = self._compute_domain_score(domain, cand["rewritten_spans"])

            # Apply Fisher Z transformation to domain score (consistent with loading)
            s_domain_raw_fisherz = np.arctanh(s_domain_raw)

            # Normalize using z-score
            s_leak = sigmoid((s_leak_raw - self.leak_score_mean) / self.leak_score_std)
            s_domain = sigmoid((s_domain_raw_fisherz - self.domain_score_mean) / self.domain_score_std)

            score = self.preference_alpha * s_domain + (1.0 - self.preference_alpha) * s_leak
            results.append({
                "text": cand["cand_text"],
                "rewritten_spans": cand["rewritten_spans"],
                "s_domain": s_domain,  
                "s_leak": s_leak,     
                "s_domain_raw": s_domain_raw,  # Raw score
                "s_leak_raw": s_leak_raw,      # Raw score
                "score": score,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def build_preference_for_item(self, item: Dict) -> Dict:

        text = item["question_init"]
        private_list = item["private_spans"]
        domain = item["domain"]
        cands = item["candidates"]
        
        # Calculate scores
        ranked = self.score_candidates(text, private_list, domain, cands)
        
        if not ranked:
            # No valid candidates, return None to indicate ignore
            return None
        
        best = ranked[0]
        reject = ranked[-1]
        
        return {
            **item,  # Keep all fields from original data
            "prefer_text": best["text"],
            "prefer_private": best["rewritten_spans"],
            "reject_text": reject["text"],
            "reject_private": reject["rewritten_spans"],
            "meta": {
                "s_domain": best["s_domain"],
                "s_leak": best["s_leak"],
                "score": best["score"],
            }
        }

    def build_and_save_preference_dataset(self, output_dir: str) -> str:
 
        from datasets import Dataset
        
        records = []
        for item in tqdm(self.dataset, desc="Building preference dataset"):
            new_item = self.build_preference_for_item(item)
            if new_item is not None:
                records.append(new_item)
        
        ds = Dataset.from_list(records)
        os.makedirs(output_dir, exist_ok=True)
        ds.save_to_disk(output_dir)
        print(f"Preference dataset saved to: {output_dir}, total {len(ds)} records")
        return output_dir


def main():

    parser = argparse.ArgumentParser(description='Privacy rewriting system')
    parser.add_argument('--qwen_model_path', type=str, default='./models/Qwen2.5-1.5b-Instruct',
                        help='Path to the Qwen model')
    parser.add_argument('--semantic_model_path', type=str, default='./models/all-MiniLM-L6-v2',
                        help='Path to the similarity model')
    parser.add_argument('--dataset_path', type=str, default='./datasets/DPO/candidate/Pri_DDXPlus_SLJA_dpo_candidate',
                        help='Path to the input dataset')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature parameter for InfoNCE loss')
    parser.add_argument('--preference_alpha', type=float, default=0.5,
                        help='Weight for domain similarity in preference scoring')
    parser.add_argument('--max_candidates', type=int, default=10,
                        help='Maximum number of candidates to generate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device name (e.g., cuda:0, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Create output directory
    output_path = f'./datasets/DPO/temperature_{args.temperature}/alpha_{args.preference_alpha}/Pri_DDXPlus_SLJA_dpo_preference'
    
    # Setup logging
    log_dir = f"./logs/preference_dataset_construction/temperature_{args.temperature}/alpha_{args.preference_alpha}"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'preference_dataset_construction_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Initialize system
    system = PrivacyRewriteSystem(
        qwen_model_path=args.qwen_model_path,
        semantic_model_path=args.semantic_model_path,
        dataset_path=args.dataset_path,
        temperature=args.temperature,
        preference_alpha=args.preference_alpha,
        max_candidates=args.max_candidates,
        device=args.device,
        logger=logger
    )

    system.build_and_save_preference_dataset(output_path)


if __name__ == "__main__":
    main()
