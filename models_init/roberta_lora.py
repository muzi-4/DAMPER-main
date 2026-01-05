import torch
import torch.nn as nn
from typing import List
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

class RoBERTaLoRA(nn.Module):
    
    def __init__(self, model_name: str, lora_r: int = 16, lora_alpha: int = 32, 
                 lora_dropout: float = 0.1, max_length: int = 128):
        super().__init__()
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        
        # Add LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value", "key", "dense"]
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        
        self.max_length = max_length
    
    def encode_span(self, spans: List[str]) -> torch.Tensor:

        # Tokenize
        inputs = self.tokenizer(
            spans,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get encodings
        outputs = self.backbone(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Calculate mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled
    
    def forward(self, spans: List[str]):

        # Encode spans
        span_embeddings = self.encode_span(spans)
        
        return span_embeddings

    def save_lora_adapter(self, save_dir: str):

        self.backbone.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)