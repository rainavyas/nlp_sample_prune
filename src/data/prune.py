import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from collections import Counter

class Pruner():
    @classmethod
    def prune(cls, ds, prune_method, kept_pos=['N', 'V', 'A', 'D']):
        if prune_method == 'stopword':
            return cls.remove_stopword(ds)
        elif prune_method == 'pos':
            return cls.keep_pos(ds, kept_pos)
    
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
    
    @staticmethod
    def keep_pos(ds, kept_pos):
        '''
            A: Adjective
            D: Adverb
            N: Noun
            V: Verb
        '''
        name_map = {'A':'ADJ', 'D':'ADV', 'N':'NOUN', 'V':'VERB'}
        kept_pos = [name_map[p] for p in kept_pos]

        nltk.download('averaged_perceptron_tagger')
        nltk.download('universal_tagset')

        print("Pruning to only keep specified pos")
        new_ds = []
        for sample in tqdm(ds):
            tkns = word_tokenize(sample['text'])
            tags = nltk.pos_tag(tkns, tagset = 'universal')
            text = ' '.join([i for (i, POS) in tags if POS in kept_pos])
            new_ds.append({'text':text, 'label':sample['label']})
        return new_ds