import os
import re
import sys
import torch
import random
import logging
import argparse
import numpy as np
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# Disable OpenAI-related HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import re

def parse_rating(judge_out):
    # 1. Most strict: match [[x]]
    m = re.search(r'\[\s*(\d{1,2})\s*\]', judge_out)
    if m:
        rating = int(m.group(1))
        if 1 <= rating <= 10:
            return rating

    # 2. Match numeric rating appearing alone on a line
    for line in judge_out.split("\n")[:3]:  # Only check first 3 lines
        m = re.match(r'^\s*(\d{1,2})\s*$', line.strip())
        if m:
            rating = int(m.group(1))
            if 1 <= rating <= 10:
                return rating

    return None

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

def build_messages_inference(item, question_key):

    question_text = item[question_key] + item["question"]

    if item['domain'] == 'medical':

        system_content = """You are a licensed medical doctor.
When answering medical-exam diagnosis questions, base all reasoning on standard clinical guidelines, evidence-based medicine, and known pathophysiology.
Give:

1.Brief clinical reasoning (key findings, differentials, why the final answer fits).
2.Final diagnosis or best answer.

Do not hallucinate unsupported findings; answer only from given information + established medical knowledge."""

        user_content = f"""
Question: 
{question_text}

Answer:"""

        return [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

    if item['domain'] == 'legal':

        system_content = """You are a legal judgment expert.
When evaluating a case, use the four components—Subject, Subjective Element, Object, Objective Elements—to determine criminal liability according to standard criminal law principles.

Provide:

1.Brief legal reasoning (analysis of Subject, Subjective Element, Object, Objective Elements, and why the conduct fits a specific offense).
2.Final judgment (crime constituted, sentencing rationale, and final sentence).

Do not invent facts; rely only on the given information and established legal doctrine."""

        user_content = f"""
Case:
{question_text}

Answer:"""

        return [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

def build_messages_llmj(item, answer_text):
    question_text = item['question_init'] + item["question"]

    if item['domain'] == 'medical':
        system_content = """
You are a neutral medical evaluator.

You will be provided with:
1) A medical question from a patient.
2) An AI assistant's response to that question.

Your task:
- Assess whether the AI assistant's response is useful and accurately addresses the diagnosis.
- Consider: relevance, medical correctness, clarity, completeness, and safety.
- Then give an overall quality score.

Scoring rules:
- Score is an integer from 1 to 10 (no decimals).
- 1 = very poor / misleading / harmful
- 10 = excellent, medically sound, clear, and helpful.

STRICT OUTPUT FORMAT (VERY IMPORTANT):
- You MUST follow this exact format:
    Line 1: [[rating]]
    Line 2+: A brief explanation in natural language.

- `[[rating]]` MUST:
    * Appear ONLY ONCE.
    * Be the FIRST thing in the output.
    * Contain ONLY the integer score (1–10) inside the brackets.
- Do NOT output JSON.
- Do NOT output Markdown headings or bullet lists.
- Do NOT add any text before or after this pattern.
- Do NOT include any other `[[` or `]]` patterns.

Examples of CORRECT output:
[[3]]
The response is vague, misses key differential diagnoses, and may be misleading.

[[9]]
The response is medically accurate, clearly explains the diagnosis and red-flag symptoms, and gives appropriate follow-up advice.

Examples of INCORRECT output (DO NOT DO THIS):
Rating: [[7]]
[[7]] The score is...
The rating is 7 because...
"""

        user_content = f"""
User Question:
{question_text}

Assistant Response:
{answer_text}

Now produce your evaluation following the STRICT OUTPUT FORMAT.
"""

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    if item['domain'] == 'legal':
        system_content = """
You are a neutral legal evaluator.

You will be provided with:
1) A legal question describing a case.
2) An AI assistant's judgment/analysis of the case.

Your task:
- Assess whether the AI assistant's judgment is useful and accurately identifies the relevant crime(s) or legal issues.
- Consider: relevance, legal correctness (at a general level, not jurisdiction-specific), clarity, and reasoning quality.
- Then give an overall quality score.

Scoring rules:
- Score is an integer from 1 to 10 (no decimals).
- 1 = very poor / irrelevant / clearly wrong
- 10 = excellent, well-reasoned, and legally sound at a high level.

STRICT OUTPUT FORMAT (VERY IMPORTANT):
- You MUST follow this exact format:
    Line 1: [[rating]]
    Line 2+: A brief explanation in natural language.

- `[[rating]]` MUST:
    * Appear ONLY ONCE.
    * Be the FIRST thing in the output.
    * Contain ONLY the integer score (1–10) inside the brackets.
- Do NOT output JSON.
- Do NOT output Markdown headings or bullet lists.
- Do NOT add any text before or after this pattern.
- Do NOT include any other `[[` or `]]` patterns.

Examples of CORRECT output:
[[2]]
The judgment misidentifies the crime and does not apply the facts to the relevant legal elements.

[[8]]
The judgment correctly identifies the likely offense and provides clear reasoning based on the facts.

Examples of INCORRECT output (DO NOT DO THIS):
Rating: [[7]]
[[7]] The score is...
The rating is 7 because...
"""

        user_content = f"""
User Question:
{question_text}

Assistant Response:
{answer_text}

Now produce your evaluation following the STRICT OUTPUT FORMAT.
"""

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]


@torch.no_grad()
def _evaluate_llmj(args, logger, model, tokenizer, dataset, question_key, api_key=None):

    import re

    ratings = []
    total_count = len(dataset)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for i, item in enumerate(tqdm(dataset)):

        try:
            # ===== Stage 1: Generate answer using local model =====
            messages_inference = build_messages_inference(item, question_key)
            inputs = tokenizer.apply_chat_template(
                messages_inference,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(args.device)

            outputs = model.generate(
                    inputs,
                    max_new_tokens=1024,
                    pad_token_id=tokenizer.eos_token_id
                )

            gen_ids = outputs[:, inputs.size(1):]
            model_out = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

            # ===== Stage 2: Use DeepSeek API as judge =====
            messages_llmj = build_messages_llmj(item, model_out)

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages_llmj,
                stream=False
            )

            judge_out = response.choices[0].message.content

            # ===== Parse rating =====
            rating = parse_rating(judge_out)
            if rating is None:
                print(f"Warning: Failed to parse rating from judge output. Skipping sample")
                print(f"Judge output: {judge_out[:200]}...")
                continue

            # Only add valid ratings to list
            ratings.append(rating)

        except Exception as e:
            print(f"Error processing item: {e}. Skipping sample")
            continue

    # ===== Calculate statistics =====
    if ratings:
        mean_rating = sum(ratings) / len(ratings)
        valid_count = len(ratings)
        print(f"LLMJ Evaluation: {valid_count}/{total_count} valid ratings, mean score: {mean_rating:.2f}")
    else:
        mean_rating = 0.0
        valid_count = 0
        print(f"LLMJ Evaluation: No valid ratings found")

    return mean_rating, valid_count, total_count


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
    parser.add_argument('--deepseek_api_key', type=str, default='',
                       help='DeepSeek API key for LLMJ evaluation')
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup logging
    log_dir = f"./logs/evaluation_llmj/temperature_{args.temperature}/alpha_{args.preference_alpha}/beta_{args.beta}/gamma_{args.similarity_threshold_medical}_{args.similarity_threshold_legal}/epsilon_{int(args.epsilon)}"

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
        for key in ['rewrite_text']:
            mean_rating, valid_count, total_count = _evaluate_llmj(args, logger, server_model, server_tokenizer, dataset, question_key=key, api_key=args.deepseek_api_key)

            logger.info(f"Mean rating: {mean_rating:.2f}, Valid count: {valid_count}, Total count: {total_count}")

if __name__ == "__main__":
    main()