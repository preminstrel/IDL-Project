import torch
from tqdm import tqdm
import numpy as np
import random

import wandb


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(config, model, optimizer, train_dataloader, eval_dataloader, device, tokenizer):
    if config['dataset'] == 'twitter_complaints' or config['dataset'] == 'wikitext2':
        train_loss, val_loss, train_ppl, eval_ppl= train_wikitext2(config, model, optimizer, train_dataloader, eval_dataloader, device, tokenizer)
    elif config['dataset'] == 'imdb':
        train_loss, val_loss, train_ppl, eval_ppl= train_imdb(model, optimizer, train_dataloader, eval_dataloader, device, tokenizer)
    else:
        raise ValueError("Invalid dataset name")

    return train_loss, val_loss, train_ppl, eval_ppl


def train_wikitext2(config, model, optimizer, train_dataloader, eval_dataloader, device, tokenizer):
    model.train()
    total_loss = 0
    bar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        #loss = outputs.loss

        logits = outputs.logits
        # shift logits [:, config['soft_prompt_tokens']:]
        logits = logits[:, config['soft_prompt_tokens']:]
        loss_fct = torch.nn.CrossEntropyLoss()
        # print(logits.size(), batch["labels"].size()) # torch.Size([10, 512, 50272]) torch.Size([10, 512])
        #loss = loss_fct(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), batch["labels"].view(-1))
        
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (step + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])),
            train_ppl="{:.04f}".format(float(torch.exp(total_loss / (step + 1)))),
            )
        bar.update()
    bar.close()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        logits = logits[:, config['soft_prompt_tokens']:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))

        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)

    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)

    print(f"{eval_epoch_loss=} {eval_ppl=}")

    if config['use_wandb']:
        wandb.log({
            "Eval Loss": eval_epoch_loss,
            "Eval PPL": eval_ppl,
        })

    return train_epoch_loss, eval_epoch_loss, train_ppl, eval_ppl


def test_wikitext(config, model, optimizer, train_dataloader, eval_dataloader, device, tokenizer):
    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        logits = logits[:, config['soft_prompt_tokens']:]
        loss_fct = torch.nn.CrossEntropyLoss()
        print(logits.size(), batch["labels"].size())
        loss = loss_fct(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))

        #loss = outputs.loss
        # print(loss)
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)

    eval_ppl = torch.exp(eval_epoch_loss)

    print(f"{eval_epoch_loss=} {eval_ppl=}")

def train_imdb(model, optimizer, train_dataloader, test_dataloader, device, tokenizer):
    train_loss = 0
    # train_acc = 0
    model.train()  # set the model to training mode
    batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i, batch in enumerate(train_dataloader):
        # Move data to the device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answers = batch["answers"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predictions = torch.argmax(outputs.logits, dim=-1)
        train_acc += (predictions == labels).sum().item()/labels.size(0)
        train_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(train_loss / (i + 1))),
            # acc="{:.04f}%".format(float(train_acc*100 / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

    batch_bar.close()
    train_loss /= len(train_dataloader)
    # train_acc /= len(train_dataloader)

    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["answers"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            predictions = torch.argmax(outputs.logits, dim=-1)

            test_acc += (predictions == labels).sum().item()/labels.size(0)
            test_loss += outputs.loss.item()


    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    
    return train_loss, test_loss, train_acc, test_acc
