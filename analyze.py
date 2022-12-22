import sys
import os
import argparse
from src.models.model_selector import select_model
from src.data.data_selector import select_data
from statistics import mean, stdev

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. rt')
    commandLineParser.add_argument('--data_dir_path', type=str, required=False, help='path to data directory, e.g. data')
    commandLineParser.add_argument('--prune_method', required=False, type=str, help="How to prune each sample")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analyze.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the training data
    _, train_data = select_data(args, train=True, prune_method=args.prune_method)

    # Load tokenizer
    model = select_model(args.model_name)
    tokenizer = model.tokenizer

    # Get avg token length
    tkn_lens = []
    for sample in train_data:
        # tkn_lens.append(len(model.tokenizer.tokenize(sample['text'], max_length=512, truncation=True)))
        tkn_lens.append(len(model.tokenizer.tokenize(sample['text'])))
    
    len_mean = mean(tkn_lens)
    len_std = stdev(tkn_lens)
    print(f'Length {len_mean:.3f}+-{len_std:.3f}')
    
    


    
