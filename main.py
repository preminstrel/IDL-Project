import torch
from torchsummaryX import summary
import gc
import wandb
import os
import pandas as pd

from data.dataset import get_dataloader

from models.build import build_model

from engine import setup_seed, train #, eval, test

from utils.info import epic_start, get_device, terminal_msg, config_to_string
from utils.model import get_config, count_parameters
from utils.parser import ParserArgs

if __name__ == "__main__":
    parser = ParserArgs()
    args = parser.get_args()

    device = get_device()
    config = get_config(args.config)
    epic_start("CMU 11-785 Project", config=config)
    setup_seed(config['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if config['use_wandb']:
        wandb.init(
            name=config['name'],
            project= config['project'],
            config=config
        )
    
    model = build_model(config).to(device)
    # params check
    num_params, num_trainable_params = count_parameters(model)
    terminal_msg(f"Model Built! Params in {model.name}: {num_params / 1e6:.4f}M ({num_trainable_params / 1e6:.4f}M trainable). ", 'C')
    train_loader, val_loader = get_dataloader(config)
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
        val_loss, val_acc = eval(model, val_loader, criterion, device)
        print(val_acc)
        exit()


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
            train_loss, val_loss, train_acc, val_acc = train(model, optimizer, train_loader, val_loader, device)
            
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
                })

            print("Train Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
            print("\tVal Loss {:.04f}".format(val_loss))
        
        terminal_msg("Training phase finished!", "C")
    
    else:
        terminal_msg("Invalid mode input", mode="F")