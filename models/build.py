import token
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import os
from pathlib import Path

def build_model(config):
    
    if config['dataset'] == 'imdb':
        model = AutoModelForSequenceClassification.from_pretrained(config['arch'], num_labels=2)
    
    elif config['dataset'] == 'twitter_complaints' or config['dataset'] == 'wikitext2':
        model = AutoModelForCausalLM.from_pretrained(config['arch'])
    else:
        raise ValueError("Invalid dataset name")
    
    tokenizer = AutoTokenizer.from_pretrained(config['arch'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id



    return model, tokenizer