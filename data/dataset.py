from transformers import GPT2Tokenizer
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader


def get_dataloader(config):

    if config['dataset'] == 'squad':
        dataset = load_dataset("squad")
        train_dataset = dataset["train"]
        test_dataset = dataset["validation"]

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    def encode_example(example):
        # Tokenize the context/question, and return their ids and attention masks
        inputs = tokenizer(example['context'], example['question'], truncation=True, padding='max_length', max_length=512)
        answers = example['answers']
        print(len(example['context']))
        answer_tokens = tokenizer.tokenize(answers['text'][0])
       
        return {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'answers': answer_tokens
        }

    # Apply function to training and test datasets
    # train_dataset = train_dataset.map(encode_example)
    # test_dataset = test_dataset.map(encode_example)

    def collate_fn(batch):
        context = [item['context'] for item in batch]
        question = [item['question'] for item in batch]
        answers = [item['answers']['text'][0] for item in batch]
        # Tokenize the context/question, and return their ids and attention masks
        inputs = tokenizer(context, question, truncation=True, padding='max_length', max_length=512)

        # tokenize the answers
        answer_tokens = tokenizer(answers, truncation=True, padding='max_length', max_length=32)

        input_ids = torch.stack([torch.tensor(inputs['input_ids'][i]) for i in range(len(inputs['input_ids']))])
        attention_mask = torch.stack([torch.tensor(inputs['attention_mask'][i]) for i in range(len(inputs['attention_mask']))])
        answers = torch.stack([torch.tensor(answer_tokens['input_ids'][i]) for i in range(len(answer_tokens['input_ids']))])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'answers': answers
        }

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)
    
    return train_dataloader, test_dataloader