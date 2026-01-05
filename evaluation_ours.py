import os
import re
import sys
import torch
import random
from torch.nn import functional as F
import logging
import argparse
import numpy as np
from torch.nn.functional import log_softmax
from datetime import datetime
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel
from models_init.roberta_lora import RoBERTaLoRA
from bert_score import score as bertscore_score
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

def build_messages_mix(text, question, options):

    question_text = text + question

    system_content = """You are a helpful and precise multiple-choice question answering assistant.
Read the question carefully and choose the correct answer from the given options.
Only output one capital letter (A, B, C, D, E, F, G or H) without any explanation."""

    user_content = f"""
Question: 
{question_text}

Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
E. {options[4]}
F. {options[5]}
G. {options[6]}
H. {options[7]}
Answer:"""

    return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

@torch.no_grad()
def _evaluate_downstream_mix(args, model, tokenizer, dataset, question_key):
    
    correct_count = 0
    valid_count = 0
    total_count = len(dataset)

    domain_error = 0
    error_total = 0
    
    for item in tqdm(dataset):
        messages = build_messages_mix(item[question_key], item["question"], item["options"])
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(args.device)
        
        outputs = model.generate(
                inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_ids = outputs[:, inputs.size(1):]
        model_out = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        
        # Check if inference result is a valid ABCD option
        if model_out in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            valid_count += 1
            if model_out == item['answer_letter']:
                correct_count += 1
            else:
                error_total += 1
                if model_out in ['E', 'F', 'G', 'H']:
                    domain_error += 1
    
    # Calculate accuracy
    if valid_count > 0:
        accuracy = correct_count / valid_count

    print("valid_count:", valid_count)
    
    return valid_count, accuracy, domain_error/error_total


def build_messages(text, question, options):

    question_text = text + question

    system_content = """You are a helpful and precise multiple-choice question answering assistant.
Read the question carefully and choose the correct answer from the given options.
Only output one capital letter (A, B, C, or D) without any explanation."""

    user_content = f"""
Question: 
{question_text}

Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Answer:"""

    return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

@torch.no_grad()
def _evaluate_downstream(args, model, tokenizer, dataset, question_key):
    
    correct_count = 0
    valid_count = 0
    total_count = len(dataset)
    
    for item in tqdm(dataset):
        if len(item["options"]) == 4:
            messages = build_messages(item[question_key], item["question"], item["options"])
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(args.device)
            
            outputs = model.generate(
                    inputs,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_ids = outputs[:, inputs.size(1):]
            model_out = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            
            # Check if inference result is a valid ABCD option
            if model_out in ['A', 'B', 'C', 'D']:
                valid_count += 1
                if model_out == item['answer_letter']:
                    correct_count += 1
    
    # Calculate accuracy
    if valid_count > 0:
        accuracy = correct_count / valid_count
    
    return valid_count, accuracy

@torch.no_grad()
def _semantic_leakage_score(args, semantic_model, dataset, question_key):

    sls = []

    for item in tqdm(dataset):

        private_spans = item["detect_private_spans"]
        rewrite_spans = item["rewrite_spans"]
        
        private_spans_emb = semantic_model.encode(private_spans, convert_to_tensor=True, device=args.device)
        rewrite_spans_emb = semantic_model.encode(rewrite_spans, convert_to_tensor=True, device=args.device)

        cos_sim = torch.cosine_similarity(private_spans_emb, rewrite_spans_emb, dim=1)
        cos_sim = cos_sim.clamp(min=0.0, max=1.0)
        avg_sim = cos_sim.mean().item()

        sls.append(avg_sim)
    
    return np.mean(sls)

@torch.no_grad()
def _bert_score(args, dataset, question_key):

    refs = [item['question_init'] for item in dataset]
    cands = [item[question_key] for item in dataset]

    P, R, F1 = bertscore_score(
        cands, refs,
        lang="en",
        rescale_with_baseline=True,
        batch_size=16,
        device = args.device
    )
    return (float(F1.mean().item())+1)/2

@torch.no_grad()
def _domain_style_span(args, logger, dataset, model, prototypes_data, question_key):

    def compute_similarity(span_embedding: torch.Tensor, domain_prototypes: torch.Tensor) -> float:
        # Calculate cosine similarity with each prototype, then take maximum
        similarities = F.cosine_similarity(span_embedding, domain_prototypes, dim=1)
        max_similarity = torch.max(similarities).item()
        return max_similarity

    spans_key = 'rewrite_spans'

    total_similarity = 0.0
    total = 0

    for item in tqdm(dataset):

        prototypes = prototypes_data['medical_prototypes'] if item['domain'] == 'medical' else prototypes_data['legal_prototypes']

        for span in item[spans_key]:
            span_emb = model.encode_span(span)
            similarity = compute_similarity(span_emb, prototypes)

            total_similarity += similarity
            total += 1
    
    return total_similarity/total

def _normalize(text: str) -> str:

    s = text.lower().strip()
    # Remove sentence punctuation (replace with space uniformly to avoid words sticking together)
    s = re.sub(r"[，,;；。\.!?？]", " ", s)
    # Remove conjunctions (\b is word boundary, avoid deleting "or" in "ordinary")
    s = re.sub(r"\b(and|or)\b", " ", s)
    # Merge extra whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _privacy_coverage_ratio(args, logger, dataset):

    covered = 0
    total = 0

    for item in dataset:
        private_spans = item["private_spans"]
        detect_spans = item["detect_private_spans"]

        # 1) Concatenate all detect spans into a long text, then normalize
        detect_concat = " ".join(detect_spans)
        norm_detect_long = _normalize(detect_concat)

        # 2) Check each GT span to see if it appears in this long text
        for span in private_spans:
            norm_span = _normalize(span)
            if norm_span in norm_detect_long:
                covered += 1

        total += len(private_spans)

    return covered/total

@torch.no_grad()
def _over_prediction_ratio(args, logger, dataset):
    
    over_predicted = 0
    total = 0

    for item in dataset:
        private_spans = item["private_spans"]
        detect_spans = item["detect_private_spans"]

        # First preprocess GT spans into token sets to avoid repeated processing
        norm_private_tokens_list = []
        for p in private_spans:
            norm_p = _normalize(p)
            norm_private_tokens_list.append(set(norm_p.split()))

        total += len(detect_spans)

        for span in detect_spans:
            norm_span = _normalize(span)

            det_tokens = set(norm_span.split())

            matched = False
            for priv_tokens in norm_private_tokens_list:
                
                if det_tokens & priv_tokens:
                    matched = True
                    break

            if not matched:
                over_predicted += 1

    return over_predicted/total

def setup_logging(log_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")
        
        # Configure log format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        return logger

def main():
    parser = argparse.ArgumentParser(description="Evaluation for all methods")
    parser.add_argument("--server_model_path", type=str, default="./models/Qwen2.5-7b-Instruct",
                       help="Path to the downstream task inference model")
    parser.add_argument("--roberta_base_path", type=str, default="./models/roberta-base",
                       help="Path to the base encoding model")
    parser.add_argument("--roberta_lora_dir", type=str, default="./saved_models/roberta_lora",
                       help="Directory path to domain prototype encoding model")
    parser.add_argument("--semantic_model_path", type=str, default="./models/all-MiniLM-L6-v2",
                       help="Path to the similarity model")
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
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device name (e.g., cuda:0, cpu)")
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup logging
    log_dir = f"./logs/evaluation/temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold_medical}_{args.similarity_threshold_legal}/epsilon_{int(args.epsilon)}"

    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir)

    logger.info(f"temperature:{args.temperature}, alpha:{args.preference_alpha}, beta:{args.beta}, gamma:{args.similarity_threshold_medical}_{args.similarity_threshold_legal}, epsilon:{args.epsilon}")

    # Load server model
    logger.info(f"Loading server model: {args.server_model_path}")
    server_tokenizer = AutoTokenizer.from_pretrained(args.server_model_path)
    if server_tokenizer.pad_token is None:
            server_tokenizer.pad_token = server_tokenizer.eos_token
    server_model = AutoModelForCausalLM.from_pretrained(args.server_model_path)
    server_model.to(args.device)
    server_model.eval()

    # Load RoBERTa LoRA
    roberta_lora_path = os.path.join(args.roberta_lora_dir, f"temperature_{args.temperature}")
    logger.info(f"Loading RoBERTa LoRA: {roberta_lora_path}")
    roberta_model = RoBERTaLoRA(args.roberta_base_path)
    roberta_model.to(args.device)
    roberta_base = AutoModel.from_pretrained(args.roberta_base_path)
    roberta_lora = PeftModel.from_pretrained(roberta_base, roberta_lora_path)
    roberta_model.backbone = roberta_lora.to(args.device)
    roberta_model.eval()

    # Load similarity model
    logger.info(f"Loading semantic model: {args.semantic_model_path}")
    semantic_model = SentenceTransformer(args.semantic_model_path, device=args.device)
    semantic_model.eval()

    # Load domain prototypes
    domain_prototypes_path = f"./saved_models/domain_prototypes/temperature_{args.temperature}/domain_prototypes.pt"
    logger.info(f"Loading domain prototypes: {domain_prototypes_path}")
    prototypes_data = torch.load(domain_prototypes_path, map_location=args.device)


    for dataset_name in ['Pri_DDXPlus', 'Pri_SLJA']:
        logger.info(f"                                                                                                                                                        ")
        logger.info(f"========================================================================================================================================================")
        logger.info(f"                                                             Evaluation for {dataset_name}                                                              ")
        logger.info(f"========================================================================================================================================================")
        # Load batch data
        if dataset_name == 'Pri_DDXPlus':
            dataset_path = f"./datasets/Evaluation/{dataset_name}/temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold_medical}/epsilon_{int(args.epsilon)}/{dataset_name}_eval_rewrite_ours"
        elif dataset_name == 'Pri_SLJA':
            dataset_path = f"./datasets/Evaluation/{dataset_name}/temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold_legal}/epsilon_{int(args.epsilon)}/{dataset_name}_eval_rewrite_ours"
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        logger.info(f"Dataset path: {dataset_path}")

        dataset = load_from_disk(dataset_path)


        # evaluation
        MIN_KEY_WIDTH = 10
        MAX_KEY_WIDTH = 10**9
        raw_max = max((len(str(k)) for k in ['rewrite_text']), default=MIN_KEY_WIDTH)
        KEY_W = max(MIN_KEY_WIDTH, min(raw_max, MAX_KEY_WIDTH))

        header = (
            f"| {'Key':<{KEY_W}} | {'Valid':>7} | {'Acc':>8} |"
            f"{'SOI':>10} | {'BScore':>10} | {'DFS':>10} | {'Recall':>8} |  {'Precision':>8} |"
        )
        sep = "-" * len(header)
        
        logger.info(sep)
        logger.info(header)
        logger.info(sep)

        for key in ['rewrite_text']:
            valid_count, accuracy = _evaluate_downstream(args, server_model, server_tokenizer, dataset, question_key=key)
            semantic_leakage_score = _semantic_leakage_score(args, semantic_model, dataset, question_key=key)
            bert_score = _bert_score(args, dataset, question_key=key)
            domain_style_span = _domain_style_span(args, logger, dataset, roberta_model, prototypes_data, question_key=key)
            pcr = _privacy_coverage_ratio(args, logger, dataset)
            opr = _over_prediction_ratio(args, logger, dataset)

            logger.info(
                f"| {str(key):<{KEY_W}} | "
                f"{(f'{valid_count:>7d}' if isinstance(valid_count, (int, float)) else f'{valid_count:>7}') } | "
                f"{(f'{accuracy:>7.2%}' if isinstance(accuracy, (int, float)) else f'{accuracy:>7}') } | "
                f"{(f'{semantic_leakage_score:>10.4f}' if isinstance(semantic_leakage_score, (int, float)) else f'{semantic_leakage_score:>10}') } | "
                f"{(f'{bert_score:>10.4f}' if isinstance(bert_score, (int, float)) else f'{bert_score:>10}') } | "
                f"{(f'{domain_style_span:>10.4f}' if isinstance(domain_style_span, (int, float)) else f'{domain_style_span:>10}') } |"
                f"{(f'{pcr:>7.2%}' if isinstance(pcr, (int, float)) else f'{pcr:>7}') } | "
                f"{(f'{opr:>7.2%}' if isinstance(opr, (int, float)) else f'{opr:>7}') } | "
            )

        logger.info(sep)

    logger.info(f"                                                                                                                                                        ")
    logger.info(f"========================================================================================================================================================")
    logger.info(f"                                                               Evaluation for Pri_Mixture                                                               ")
    logger.info(f"========================================================================================================================================================")
    dataset_path = f"./datasets/Evaluation/Pri_Mixture/temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold_medical}_{args.similarity_threshold_legal}/epsilon_{int(args.epsilon)}/Pri_Mixture_eval_rewrite_ours"

    logger.info(f"Dataset path: {dataset_path}")

    dataset = load_from_disk(dataset_path)

    # evaluation
    MIN_KEY_WIDTH = 10
    MAX_KEY_WIDTH = 10**9
    raw_max = max((len(str(k)) for k in ['rewrite_text']), default=MIN_KEY_WIDTH)
    KEY_W = max(MIN_KEY_WIDTH, min(raw_max, MAX_KEY_WIDTH))

    header = (
        f"| {'Key':<{KEY_W}} | {'Valid':>7} | {'Acc':>8} |"
        f"{'SOI':>10} | {'BScore':>10} | {'DFS':>10} | {'Recall':>8} |  {'Precision':>8} |"
    )
    sep = "-" * len(header)
    
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for key in ['rewrite_text']:
        valid_count, accuracy, domain_error_rate = _evaluate_downstream_mix(args, server_model, server_tokenizer, dataset, question_key=key)
        semantic_leakage_score = _semantic_leakage_score(args, semantic_model, dataset, question_key=key)
        bert_score = _bert_score(args, dataset, question_key=key)
        domain_style_span = _domain_style_span(args, logger, dataset, roberta_model, prototypes_data, question_key=key)
        pcr = _privacy_coverage_ratio(args, logger, dataset)
        opr = _over_prediction_ratio(args, logger, dataset)

        logger.info(
            f"| {str(key):<{KEY_W}} | "
            f"{(f'{valid_count:>7d}' if isinstance(valid_count, (int, float)) else f'{valid_count:>7}') } | "
            f"{(f'{accuracy:>7.2%}' if isinstance(accuracy, (int, float)) else f'{accuracy:>7}') } | "
            f"{(f'{semantic_leakage_score:>10.4f}' if isinstance(semantic_leakage_score, (int, float)) else f'{semantic_leakage_score:>10}') } | "
            f"{(f'{bert_score:>10.4f}' if isinstance(bert_score, (int, float)) else f'{bert_score:>10}') } | "
            f"{(f'{domain_style_span:>10.4f}' if isinstance(domain_style_span, (int, float)) else f'{domain_style_span:>10}') } |"
            f"{(f'{pcr:>7.2%}' if isinstance(pcr, (int, float)) else f'{pcr:>7}') } | "
            f"{(f'{opr:>7.2%}' if isinstance(opr, (int, float)) else f'{opr:>7}') } | "
        )

    logger.info(sep)

if __name__ == "__main__":
    main()