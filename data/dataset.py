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
        test_dataset = dataset["test"]

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
    
    elif config['dataset'] == 'wikitext2':
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

        trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

        def split_sequence(sequence, max_length):
            sequence = sequence.tolist() if isinstance(sequence, torch.Tensor) else sequence
            sequences = [sequence[i:i + max_length] for i in range(0, len(sequence), max_length)]
            # 对最后一个序列进行填充
            last_seq = sequences[-1]
            if len(last_seq) < max_length:
                sequences[-1] = last_seq + [0] * (max_length - len(last_seq))  # 填充0
            return sequences

        max_length = 512
        split_input_ids = split_sequence(trainenc['input_ids'][0], max_length)
        split_attention_mask = split_sequence(trainenc['attention_mask'][0], max_length)
        labels = trainenc['input_ids'][0][1:].tolist() + [0]
        split_labels = split_sequence(labels, max_length)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(split_input_ids), 
            torch.tensor(split_attention_mask),
            torch.tensor(split_labels)
        )

        split_input_ids = split_sequence(testenc['input_ids'][0], max_length)
        split_attention_mask = split_sequence(testenc['attention_mask'][0], max_length)
        labels = testenc['input_ids'][0][1:].tolist() + [0]
        split_labels = split_sequence(labels, max_length)

        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(split_input_ids), 
            torch.tensor(split_attention_mask),
            torch.tensor(split_labels)
        )

        
        def collate_fn(batch):
            input_ids = torch.stack([item[0] for item in batch])
            attention_mask = torch.stack([item[1] for item in batch])
            labels = torch.stack([item[2] for item in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

        return train_dataloader, test_dataloader

