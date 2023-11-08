import torch
from tqdm import tqdm
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(config, model, optimizer, train_dataloader, eval_dataloader, device, tokenizer):
    if config['dataset'] == 'twitter_complaints':
        train_loss, val_loss, train_ppl, eval_ppl= train_twitter(model, optimizer, train_dataloader, eval_dataloader, device, tokenizer)
    elif config['dataset'] == 'imdb':
        train_loss, val_loss, train_ppl, eval_ppl= train_imdb(model, optimizer, train_dataloader, eval_dataloader, device, tokenizer)

    return train_loss, val_loss, train_ppl, eval_ppl

def train_twitter(model, optimizer, train_dataloader, eval_dataloader, device, tokenizer):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    return train_epoch_loss, eval_epoch_loss, train_ppl, eval_ppl


def train_imdb(model, optimizer, train_dataloader, test_dataloader, device, tokenizer):
    
    positive_index = tokenizer.convert_tokens_to_ids("positive")
    negative_index = tokenizer.convert_tokens_to_ids("negative")
    
    train_loss = 0
    train_acc = 0
    model.train()  # set the model to training mode
    batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i, batch in enumerate(train_dataloader):
        # Move data to the device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # [batch_size, seq_len, vocab_size]
        selected_logits = logits[:, -1, [positive_index, negative_index]] # [batch_size, 2]
        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(selected_logits, labels)
        train_loss += loss.item()

        train_acc += torch.sum(torch.argmax(selected_logits, dim= 1) == labels).item()/selected_logits.shape[0]

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(train_loss / (i + 1))),
            acc="{:.04f}%".format(float(train_acc*100 / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        # acc = (selected_logits.argmax(dim=1) == labels).float().mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    batch_bar.close()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
            selected_logits = outputs[:, -1, [positive_index, negative_index]]
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(selected_logits, labels)
            total_loss += loss.item()
            total_acc += torch.sum(torch.argmax(selected_logits, dim= 1) == labels).item()/selected_logits.shape[0]


    total_loss /= len(test_dataloader)
    total_acc /= len(test_dataloader)
    
    return train_loss, total_loss, train_acc, total_acc