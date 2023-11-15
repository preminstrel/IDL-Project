from datasets import load_dataset
from transformers import default_data_collator
import torch
from torch.utils.data import DataLoader
import random


def get_dataloader(config, tokenizer):

    if config['dataset'] == 'twitter_complaints':
        dataset = load_dataset("ought/raft", config['dataset'])
        classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
        dataset = dataset.map(
            lambda x: {"text_label": [classes[label] for label in x["Label"]]},
            batched=True,
            num_proc=1,
        )

        text_column = "Tweet text"
        label_column = "text_label"
        max_length = 64

        target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])

        def preprocess_function(examples):
            batch_size = len(examples[text_column])
            inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
            targets = [str(x) for x in examples[label_column]]
            model_inputs = tokenizer(inputs)
            labels = tokenizer(targets)
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
                # print(i, sample_input_ids, label_input_ids)
                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
            # print(model_inputs)
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i]
                model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
                ) + sample_input_ids
                model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                    "attention_mask"
                ][i]
                labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
                model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
                model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
                labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["test"]


        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=config['batch_size'], pin_memory=True
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=config['batch_size'], pin_memory=True)

        return train_dataloader, eval_dataloader
    
    elif config['dataset'] == 'imdb':
        dataset = load_dataset("imdb")
        train_dataset = dataset["train"]
        test_dataset = dataset["validation"]

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
    