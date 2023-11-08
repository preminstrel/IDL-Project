import token
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

def build_model(config):
    
    model = AutoModelForCausalLM.from_pretrained(config['arch'])
    tokenizer = AutoTokenizer.from_pretrained(config['arch'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    return model, tokenizer