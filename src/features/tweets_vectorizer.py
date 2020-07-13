import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook

import sys
sys.path.append('..')

from src.features.vocabulary import *



class TweetsVectorizer(object):
    def __init__(self, text_vocab, target_vocab):
        self.text_vocab = text_vocab
        self.target_vocab = target_vocab
        
    def vectorize(self, text, vector_length=-1):
        indices = [self.text_vocab.begin_seq_index]
        indices.extend(self.text_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.text_vocab.end_seq_index)
        
        if vector_length < 0:
            vector_length = len(indices)
            
        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.text_vocab.mask_index
        
        return out_vector
    
    @classmethod
    def from_dataframe(cls, tweets_df, cutoff=25):
        target_vocab = Vocabulary()
        for target in sorted(set(tweets_df.target)):
            target_vocab.add_token(target)
            
        word_counts = Counter()
        for text in tweets_df.text:
            for token in text.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1
                    
        text_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                text_vocab.add_token(word)
                
        return cls(text_vocab, target_vocab)
    
    @classmethod
    def from_serializable(cls, contents):
        text_vocab = \
            SequenceVocabulary.from_serializable(contents['text_vocab'])
        target_vocab =  \
            Vocabulary.from_serializable(contents['target_vocab'])

        return cls(text_vocab=text_vocab, target_vocab=target_vocab)

    def to_serializable(self):
        return {'text_vocab': self.text_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable()}