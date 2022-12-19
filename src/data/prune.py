from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from tqdm import tqdm

class Pruner():
    @classmethod
    def prune(cls, ds, prune_method):
        prune_map = {
            'stopword'  :   cls.remove_stopword
        }
        return prune_map[prune_method](ds)
    
    @staticmethod
    def remove_stopword(ds):
        new_ds = []
        for sample in tqdm(ds):
            text = ' '.join([w for w in word_tokenize(sample['text']) if not w in stopwords.words()])
            new_ds.append({'text':text, 'label':sample['label']})
        return new_ds