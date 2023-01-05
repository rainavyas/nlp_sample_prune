import torch
import torch.nn as nn
import sys
import os
import argparse
import logging

from src.tools.tools import get_default_device, set_seeds
from src.models.model_selector import select_model
from src.data.data_selector import select_data
from src.training.batch_trainer import BatchTrainer

def base_name_creator(args):
    if args.prune_method:
        base_name = f'{args.model_name}_{args.data_name}_prune_{args.prune_method}_pretrained{not args.not_pretrained}_pruneval{args.prune_val}_seed{args.seed}'
        if args.prune_method == 'pos':
            pos = ''.join(args.kept_pos)
            base_name = f'{args.model_name}_{args.data_name}_prune_{args.prune_method}-{pos}_pretrained{not args.not_pretrained}_pruneval{args.prune_val}_seed{args.seed}'
    else:
        base_name = f'{args.model_name}_{args.data_name}_pretrained{not args.not_pretrained}_seed{args.seed}'
    return base_name

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. rt')
    commandLineParser.add_argument('--data_dir_path', type=str, required=False, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--bs', type=int, default=8, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=5, help="Specify max epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.00001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=[3], nargs='+', help="Specify scheduler cycle, e.g. 10 100 1000")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--num_classes', type=int, default=2, help="Specify number of classes")
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--prune_method', required=False, type=str, help="How to prune each sample")
    commandLineParser.add_argument('--kept_pos', type=str, default=['N', 'V', 'A', 'D'], nargs='+', help="If prune method is pos, specifiy pos to keep")
    commandLineParser.add_argument('--prune_val', action='store_true', help='prune validation data too')
    commandLineParser.add_argument('--not_pretrained', action='store_true', help='do not use pretrained_model')
    args = commandLineParser.parse_args()

    set_seeds(args.seed)
    # file naming
    base_name = base_name_creator(args)
    out_file = f'{args.out_dir}/{base_name}.th'

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Initialise logging
    if not os.path.isdir('LOGs'):
        os.mkdir('LOGs')
    fname = f'LOGs/{base_name}.log'
    logging.basicConfig(filename=fname, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info('LOG created')

    # Get the device
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the training data
    val_data, train_data = select_data(args, train=True, prune_method=args.prune_method, prune_val=args.prune_val, kept_pos=args.kept_pos)

    # Initialise model
    model = select_model(args.model_name, pretrained=not args.not_pretrained, num_labels=args.num_classes)
    model.to(device)

    # Define learning objects
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch)
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    trainer = BatchTrainer(device, model, optimizer, criterion, scheduler)
    trainer.train_process(train_data, val_data, out_file, max_epochs=args.epochs, bs=args.bs)