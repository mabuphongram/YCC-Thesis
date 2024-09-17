import pandas as pd
from sklearn.model_selection import train_test_split
import nltk

def load_data(filepath):
    custom_reader = CustomTaggedCorpusReader('.', filepath, encoding='utf-8')
    tagged_sentences = custom_reader.tagged_sents()
    return tagged_sentences

def split_data(corpus, train_size=0.85, test_size=0.15, val_size=0.20, random_state=101):
    train_set, test_set = train_test_split(corpus, train_size=train_size, test_size=test_size, random_state=random_state)
    train_set, val_set = train_test_split(train_set, train_size=1-val_size, test_size=val_size, random_state=random_state)
    return train_set, val_set, test_set

class  CustomTaggedCorpusReader(nltk.corpus.reader.TaggedCorpusReader): 
    def __init__(self, root, fileids, sep='/', encoding='utf-8'):
        nltk.corpus.reader.TaggedCorpusReader.__init__(self, root, fileids, sep, encoding=encoding)



