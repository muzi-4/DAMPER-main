"""Perform FINCH clustering based on trained model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pyarrow as pa
import numpy as np
from finch import FINCH
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import random
from tqdm import tqdm
from models_init.roberta_lora import RoBERTaLoRA

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

@torch.no_grad()
def evaluate_model(model, data: List[Dict[str, Any]], device: str, logger, medical_prototypes, legal_prototypes) -> Dict[str, float]:
    # Separate different types of data
    private_spans = [item for item in data if item['private'] == 1]
    non_private_spans = [item for item in data if item['private'] == 0]
    
    # Generate meaningless spans
    meaningless_spans = generate_meaningless_spans(100)
    
    results = {}
    
    # Metric 1: Mean maximum cosine similarity of private spans
    if private_spans:
        private_similarities = []
        for item in private_spans:
            span_embedding = model.encode_span([item['span']])
            if item['domain'] == 'medical':
                similarities = F.cosine_similarity(span_embedding, medical_prototypes, dim=1)
            else:  # legal
                similarities = F.cosine_similarity(span_embedding, legal_prototypes, dim=1)
            max_sim = similarities.max().item()
            private_similarities.append(max_sim)
        
        results['private_max_similarity_mean'] = np.mean(private_similarities)
    
    # Metric 2: Mean maximum cosine similarity of non-private spans
    if non_private_spans:
        non_private_similarities = []
        for item in non_private_spans:
            span_embedding = model.encode_span([item['span']])
            if item['domain'] == 'medical':
                similarities = F.cosine_similarity(span_embedding, medical_prototypes, dim=1)
            else:  # legal
                similarities = F.cosine_similarity(span_embedding, legal_prototypes, dim=1)
            max_sim = similarities.max().item()
            non_private_similarities.append(max_sim)
        
        results['non_private_max_similarity_mean'] = np.mean(non_private_similarities)
    
    # Metric 3: Mean maximum cosine similarity of meaningless spans
    if meaningless_spans:
        meaningless_similarities = []
        for span in meaningless_spans:
            span_embedding = model.encode_span([span])
            # Compute maximum similarity with all domain prototypes
            medical_sim = F.cosine_similarity(span_embedding, medical_prototypes, dim=1).max()
            legal_sim = F.cosine_similarity(span_embedding, legal_prototypes, dim=1).max()
            max_sim = max(medical_sim.item(), legal_sim.item())
            meaningless_similarities.append(max_sim)
        
        results['meaningless_max_similarity_mean'] = np.mean(meaningless_similarities)
    
    # Metric 4: Distance from medical private spans to legal domain prototypes
    medical_private_spans = [item for item in data if item['domain'] == 'medical' and item['private'] == 1]
    if medical_private_spans and legal_prototypes is not None:
        medical_to_legal_similarities = []
        for item in medical_private_spans:
            span_embedding = model.encode_span([item['span']])
            similarities = F.cosine_similarity(span_embedding, legal_prototypes, dim=1)
            max_sim = similarities.max().item()
            medical_to_legal_similarities.append(max_sim)
        
        results['medical_private_to_legal_prototype_mean'] = np.mean(medical_to_legal_similarities)
    
    # Metric 5: Distance from legal private spans to medical domain prototypes
    legal_private_spans = [item for item in data if item['domain'] == 'legal' and item['private'] == 1]
    if legal_private_spans and medical_prototypes is not None:
        legal_to_medical_similarities = []
        for item in legal_private_spans:
            span_embedding = model.encode_span([item['span']])
            similarities = F.cosine_similarity(span_embedding, medical_prototypes, dim=1)
            max_sim = similarities.max().item()
            legal_to_medical_similarities.append(max_sim)
        
        results['legal_private_to_medical_prototype_mean'] = np.mean(legal_to_medical_similarities)
    
    return results

def generate_meaningless_spans(num_spans: int) -> List[str]:
    # Meaningless words
    meaningless_words = [
        "xyz", "abc", "qwerty", "asdf", "hjkl", "mnbv", "poiu", "lkjh", "fdsa", "rewq",
        "random", "nonsense", "gibberish", "meaningless", "invalid", "fake", "dummy", "test",
        "unknown", "undefined", "null", "empty", "void", "blank", "none", "zero", "one", "two"
    ]
    
    # Meaningless phrases
    meaningless_phrases = [
        "xyz abc", "qwerty asdf", "hjkl mnbv", "poiu lkjh", "fdsa rewq",
        "random nonsense", "gibberish meaningless", "invalid fake", "dummy test",
        "unknown undefined", "null empty", "void blank", "none zero", "one two",
        "completely random", "totally meaningless", "absolutely invalid", "purely fake",
        "entirely dummy", "wholly test", "utterly unknown", "completely undefined",
        "totally null", "absolutely empty", "purely void", "entirely blank",
        "wholly none", "utterly zero", "completely one", "totally two",
        "random gibberish text", "meaningless nonsense phrase", "invalid fake content",
        "dummy test data", "unknown undefined value", "null empty result",
        "void blank space", "none zero count", "one two three", "abc def ghi",
        "xyz qwerty asdf", "hjkl mnbv poiu", "lkjh fdsa rewq", "random text here",
        "meaningless content", "invalid data", "fake information", "dummy result",
        "unknown value", "undefined content", "null data", "empty result",
        "void content", "blank data", "none value", "zero result"
    ]
    
    # Merge all meaningless spans
    all_meaningless = meaningless_words + meaningless_phrases
    
    meaningless_spans = []
    for _ in range(num_spans):
        # Randomly select a span (word or phrase)
        span = random.choice(all_meaningless)
        meaningless_spans.append(span)
    
    return meaningless_spans

def _finch_prototypes_auto(embeddings: np.ndarray, distance: str = "cosine") -> np.ndarray:
    # FINCH returns:
    # c: (N, num_layers)  each column is a layer of cluster labels (layers from coarse to fine)
    # num_clust: number of clusters for each layer
    # req_c: None (because we no longer set req_clust)
    c, num_clust, req_c = FINCH(
        embeddings,
        distance=distance,
        verbose=False
    )

    # Use the last layer (finest clustering)
    labels = c[:, -1]          # shape (N,)
    unique_labels = np.unique(labels)

    # Compute mean vector for each cluster as prototype
    prototypes = []
    for lab in unique_labels:
        cluster_emb = embeddings[labels == lab]
        center = cluster_emb.mean(axis=0)
        prototypes.append(center)

    prototypes = np.stack(prototypes, axis=0)   # (K, D), K is the number of clusters automatically discovered by FINCH
    return prototypes

def compute_domain_prototypes(model, data: List[Dict[str, Any]], device: str) -> Tuple[torch.Tensor, torch.Tensor]:

    # Separate private spans from different domains
    medical_private_spans = [
        item['span'] for item in data
        if item['domain'] == 'medical' and item['private'] == 1
    ]
    legal_private_spans = [
        item['span'] for item in data
        if item['domain'] == 'legal' and item['private'] == 1
    ]

    medical_prototypes = None
    legal_prototypes = None

    batch_size = 32

    # Medical domain
    if len(medical_private_spans) > 0:
        medical_embeddings = []
        for i in range(0, len(medical_private_spans), batch_size):
            batch_spans = medical_private_spans[i:i + batch_size]
            batch_embeddings = model.encode_span(batch_spans)
            medical_embeddings.append(batch_embeddings.detach().cpu().numpy())
        medical_embeddings = np.vstack(medical_embeddings)

        # FINCH parameter-free clustering
        medical_proto_np = _finch_prototypes_auto(
            medical_embeddings,
            distance="cosine"
        )
        medical_prototypes = torch.tensor(
            medical_proto_np, device=device, dtype=torch.float32
        )

    # Legal domain
    if len(legal_private_spans) > 0:
        legal_embeddings = []
        for i in range(0, len(legal_private_spans), batch_size):
            batch_spans = legal_private_spans[i:i + batch_size]
            batch_embeddings = model.encode_span(batch_spans)
            legal_embeddings.append(batch_embeddings.detach().cpu().numpy())
        legal_embeddings = np.vstack(legal_embeddings)

        legal_proto_np = _finch_prototypes_auto(
            legal_embeddings,
            distance="cosine"
        )
        legal_prototypes = torch.tensor(
            legal_proto_np, device=device, dtype=torch.float32
        )

    return medical_prototypes, legal_prototypes

def setup_logging(log_dir: str, log_level: str = "INFO"):
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename (with timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"domain_prototype_{timestamp}.log")
    
    # Configure log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured, output to: {log_file}")
    return logger

class PrototypeSpanDataset(Dataset):
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'span': item['span'],
            'domain': item['domain'],
            'private': item['private']
        }

class BalancedBatchSampler:
    
    def __init__(self, dataset: PrototypeSpanDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Group data by type
        self.medical_private = []
        self.medical_non_private = []
        self.legal_private = []
        self.legal_non_private = []
        
        for i, item in enumerate(dataset.data):
            if item['domain'] == 'medical' and item['private'] == 1:
                self.medical_private.append(i)
            elif item['domain'] == 'medical' and item['private'] == 0:
                self.medical_non_private.append(i)
            elif item['domain'] == 'legal' and item['private'] == 1:
                self.legal_private.append(i)
            elif item['domain'] == 'legal' and item['private'] == 0:
                self.legal_non_private.append(i)
        
        print(f"Data distribution:")
        print(f"  Medical Private: {len(self.medical_private)}")
        print(f"  Medical Non-private: {len(self.medical_non_private)}")
        print(f"  Legal Private: {len(self.legal_private)}")
        print(f"  Legal Non-private: {len(self.legal_non_private)}")
        
        # Calculate number of batches per epoch
        self.num_batches = len(dataset) // batch_size
        
    def __iter__(self):
        # Randomly shuffle indices for each category
        import random
        random.shuffle(self.medical_private)
        random.shuffle(self.medical_non_private)
        random.shuffle(self.legal_private)
        random.shuffle(self.legal_non_private)
        
        # Create circular iterators
        medical_private_iter = iter(self.medical_private * 10)  # Repeat multiple times to ensure sufficient data
        medical_non_private_iter = iter(self.medical_non_private * 10)
        legal_private_iter = iter(self.legal_private * 10)
        legal_non_private_iter = iter(self.legal_non_private * 10)
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Ensure each batch contains various types of data
            # Calculate the number of each type (try to balance)
            medical_private_count = min(8, len(self.medical_private), self.batch_size // 4)
            medical_non_private_count = min(8, len(self.medical_non_private), self.batch_size // 4)
            legal_private_count = min(8, len(self.legal_private), self.batch_size // 4)
            legal_non_private_count = min(8, len(self.legal_non_private), self.batch_size // 4)
            
            # If a certain type of data is insufficient, supplement with other types
            remaining = self.batch_size - (medical_private_count + medical_non_private_count + 
                                        legal_private_count + legal_non_private_count)
            
            # Randomly assign remaining positions
            counts = [medical_private_count, medical_non_private_count, 
                     legal_private_count, legal_non_private_count]
            for _ in range(remaining):
                counts[random.randint(0, 3)] += 1
            
            # Sample data
            try:
                for _ in range(counts[0]):
                    batch_indices.append(next(medical_private_iter))
            except StopIteration:
                pass
                
            try:
                for _ in range(counts[1]):
                    batch_indices.append(next(medical_non_private_iter))
            except StopIteration:
                pass
                
            try:
                for _ in range(counts[2]):
                    batch_indices.append(next(legal_private_iter))
            except StopIteration:
                pass
                
            try:
                for _ in range(counts[3]):
                    batch_indices.append(next(legal_non_private_iter))
            except StopIteration:
                pass
            
            # If batch is not full enough, randomly supplement
            while len(batch_indices) < self.batch_size:
                all_indices = (self.medical_private + self.medical_non_private + 
                             self.legal_private + self.legal_non_private)
                batch_indices.append(random.choice(all_indices))
            
            # Randomly shuffle order within batch
            random.shuffle(batch_indices)
            yield batch_indices[:self.batch_size]
    
    def __len__(self):
        return self.num_batches

def load_prototype_data(file_path: str) -> List[Dict[str, Any]]:
    try:
        # Try to load Arrow file
        with pa.OSFile(file_path, 'rb') as source:
            with pa.RecordBatchFileReader(source) as reader:
                table = reader.read_all()
        data = table.to_pylist()
    except:
        # If Arrow loading fails, try JSON
        with open(file_path.replace('.arrow', '.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    return data

def parse_args():

    parser = argparse.ArgumentParser(description='Domain prototype contrastive learning training')
    
    # Data-related parameters
    parser.add_argument('--data_path', type=str, default='./datasets/prototype_span/prototype_spans.arrow',
                        help='Path to the dataset')
    
    # Model-related parameters
    parser.add_argument('--roberta_base_path', type=str, default='./models/roberta-base',
                        help='Path to the pretrained RoBERTa base model')
    parser.add_argument('--roberta_lora_dir', type=str, default='./saved_models/roberta_lora',
                        help='Directory path to the RoBERTa LoRA adapter model')

    parser.add_argument('--temperature', type=float, default=0.1,
                        help='InfoNCE temperature parameter')
    
    # Device-related parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Training device')
    
    # Output-related parameters
    parser.add_argument('--output_dir', type=str, default='./saved_models',
                        help='Output directory for saved models')
    
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def main():

    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_dir = f"./logs/domain_prototype/temperature_{args.temperature}"
    logger = setup_logging(log_dir)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # Load data
    data = load_prototype_data(args.data_path)
    
    
    # Load model
    roberta_lora_path = os.path.join(args.roberta_lora_dir, f"temperature_{args.temperature}")
    logger.info(f"Loading RoBERTa adapter model: {roberta_lora_path}")
    roberta_model = RoBERTaLoRA(args.roberta_base_path)
    roberta_model.to(device)
    roberta_base = AutoModel.from_pretrained(args.roberta_base_path)
    roberta_lora = PeftModel.from_pretrained(roberta_base, roberta_lora_path)
    roberta_model.backbone = roberta_lora.to(device)
    roberta_model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Domain prototype directory and file
    prototypes_dir = os.path.join(args.output_dir, f'domain_prototypes', f'temperature_{args.temperature}')
    prototype_file = os.path.join(prototypes_dir, 'domain_prototypes.pt')
    
    # Skip if already exists
    if os.path.exists(prototype_file):
        logger.info(f"Domain prototype file already exists: {prototype_file}")
        return
    
    # Compute domain prototypes
    medical_prototypes, legal_prototypes = compute_domain_prototypes(roberta_model, data, device)

    # Save domain prototypes
    os.makedirs(prototypes_dir, exist_ok=True)

    torch.save({
        'medical_prototypes': medical_prototypes.detach().cpu(),
        'legal_prototypes': legal_prototypes.detach().cpu()
    }, os.path.join(prototypes_dir, f'domain_prototypes.pt'))

    logger.info(f"Domain prototypes saved to: {prototypes_dir}")
    
    # Evaluation
    post_train_results = evaluate_model(roberta_model, data, device, logger, medical_prototypes, legal_prototypes)
    logger.info(f"Post-training evaluation: {post_train_results}")

if __name__ == "__main__":
    main()
