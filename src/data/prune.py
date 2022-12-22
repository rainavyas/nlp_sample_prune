import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from collections import Counter

class Pruner():
    @classmethod
    def prune(cls, ds, prune_method):
        prune_map = {
            'stopword'  :   cls.remove_stopword
        }
        pruned_data = prune_map[prune_method](ds)
        return pruned_data
    
    @staticmethod
    def remove_stopword(ds):
        nltk.download('stopwords')
        stop_words = stopwords.words()
        stopwords_dict = Counter(stop_words)
        new_ds = []
        for sample in tqdm(ds):
            text = ' '.join([w for w in word_tokenize(sample['text']) if not w in stopwords_dict])
            new_ds.append({'text':text, 'label':sample['label']})
        return new_ds