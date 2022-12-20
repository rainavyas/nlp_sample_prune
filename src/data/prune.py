import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os
import json

class Pruner():
    @classmethod
    def prune(cls, ds, prune_method, cache_dir, data_name, part='train'):
        # check to see if pruned data is already cached
        fname = f'{cache_dir}/pruned/{data_name}_{part}_{prune_method}.json'
        if not os.path.isdir(f'{cache_dir}/pruned'):
            os.mkdir(f'{cache_dir}/pruned')
        try:
            with open(fname, 'r') as f:
                pruned_data = json.loads(f.read())
        except:
            print("First time pruning here")
            prune_map = {
                'stopword'  :   cls.remove_stopword
            }
            pruned_data = prune_map[prune_method](ds)
            with open(fname, 'w') as f:
                json.dump(pruned_data, f)
        return pruned_data
    
    @staticmethod
    def remove_stopword(ds):
        nltk.download('stopwords')
        new_ds = []
        for sample in tqdm(ds):
            text = ' '.join([w for w in word_tokenize(sample['text']) if not w in stopwords.words()])
            new_ds.append({'text':text, 'label':sample['label']})
        return new_ds