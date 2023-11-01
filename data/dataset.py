from transformers import GPT2Tokenizer
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader


def get_dataloader(config):

    if config['dataset'] == 'imdb':
        dataset = load_dataset("imdb")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        train_dataset = train_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=512, return_tensors="pt"))
        test_dataset = test_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=512, return_tensors="pt"))

        def collate_fn(batch):
            input_ids = torch.stack([torch.tensor(item['input_ids']).squeeze(0) for item in batch])
            attention_mask = torch.stack([torch.tensor(item['attention_mask']).squeeze(0) for item in batch])
            labels = torch.stack([torch.tensor(item['label']) for item in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)
    
    return train_dataloader, test_dataloader