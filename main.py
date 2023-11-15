import torch
import gc
import wandb
import os
import pandas as pd
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType

from data.dataset import get_dataloader

from models.build import build_model

from engine import setup_seed, train, test_wikitext

from utils.info import epic_start, get_device, terminal_msg, config_to_string
from utils.model import get_config, count_parameters
from utils.parser import ParserArgs

import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="OPTForSequenceClassification")
    parser = ParserArgs()
    args = parser.get_args()

    device = get_device()
    config = get_config(args.config)
    epic_start("CMU 11-785 Project", config=config)
    setup_seed(config['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    config['name'] = f"{config['arch']}-{config['dataset']}-{config['soft_prompt_tokens']}"
    if config['use_wandb']:
        wandb.init(
            name=config['name'],
            project= config['project'],
            config=config
        )
    
    model, tokenizer = build_model(config)

    if config['dataset'] == 'twitter_complaints' or config['dataset'] == 'wikitext2':
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=config['soft_prompt_tokens'],
            prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
            tokenizer_name_or_path=config['arch'],
        )
    elif config['dataset'] == 'imdb':
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=config['soft_prompt_tokens'],
            prompt_tuning_init_text="Classify if the movie review is positive or negative:",
            tokenizer_name_or_path=config['arch'],
        )



    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    model = model.to(device)

    if config['dataset'] == 'twitter_complaints':
        val_loader, train_loader = get_dataloader(config, tokenizer)
    else:
        train_loader, val_loader = get_dataloader(config, tokenizer)
    terminal_msg(f"Data Loaded! Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}", 'C')

    if config['regularization'] == 'l2':
        optimizer = torch.optim.AdamW(model.parameters(), lr= config['lr'], weight_decay=config['weight_decay']) # l2 loss
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr= config['lr'])

    terminal_msg(f'Init scheduler {config["scheduler"]}...', 'E')
    if config['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience= 5, threshold= 1e-4)
    elif config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0.0001)
    else:
        terminal_msg(f'Invalid scheduler {config["scheduler"]}...', "F")


    torch.cuda.empty_cache()
    gc.collect()

    if config['mode'] == 'eval':
        test_wikitext(config, model, optimizer, train_loader, val_loader, device, tokenizer)


    elif config['mode'] == 'train':
        # save best
        best_lev_dist = float("inf")
        save_best = False
        best_epoch = 0

        # plot settings
        training_loss = []
        validation_loss = []
        training_performance = []
        validation_performance = []

        start_epoch = 0

        for epoch in range(start_epoch, config['epochs']):
            print("\nEpoch {}/{}".format(epoch+1, config['epochs']))
            curr_lr = float(optimizer.param_groups[0]['lr'])
            train_loss, val_loss, train_ppl, eval_ppl = train(config, model, optimizer, train_loader, val_loader, device, tokenizer)
            print(f"{epoch=}: {train_ppl=} {train_loss=} {eval_ppl=} {val_loss=}")
            if config['scheduler'] == 'StepLR':
                scheduler.step()
            elif config['scheduler'] == 'cosine':
                scheduler.step()

            training_loss.append(train_loss)
            validation_loss.append(val_loss)

            if config['use_wandb']:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": curr_lr,
                    "train_performance": train_ppl,
                    "eval_performance": eval_ppl,
                })
        
        terminal_msg("Training phase finished!", "C")
    
    else:
        terminal_msg("Invalid mode input", mode="F")