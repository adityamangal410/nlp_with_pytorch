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

from src.features.tweets_vectorizer import *

class TweetsDataset(Dataset):
    def __init__(self, tweets_df, vectorizer):
        """
        Args:
            tweets_df (pandas.DataFrame): the dataset
            vectorizer (TweetsVectorizer): vectorizer instatiated from dataset
        """
        self.tweets_df = tweets_df
        self._vectorizer = vectorizer

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, tweets_df.text)) + 2
        

        self.train_df = self.tweets_df[self.tweets_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.tweets_df[self.tweets_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.tweets_df[self.tweets_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        # Class weights
        class_counts = tweets_df.target.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.target_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls, tweets_csv):
        tweets_df = pd.read_csv(tweets_csv)
        train_tweets_df = tweets_df[tweets_df.split == 'train']
        return cls(train_tweets_df, TweetsVectorizer.from_dataframe(train_tweets_df))
    
    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        text_vector = self._vectorizer.vectorize(row.text, self._max_seq_length)
        
        target_index = self._vectorizer.target_vocab.lookup_token(row.target)
        
        return {'x_data': text_vector, 'y_target': target_index}
    
    @classmethod
    def load_dataset_and_load_vectorizer(cls, tweets_csv, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer. 
        Used in the case in the vectorizer has been cached for re-use
        
        Args:
            tweets_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of SurnameDataset
        """
        tweets_df = pd.read_csv(tweets_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(tweets_csv, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file
        
        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of SurnameVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return NameVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json
        
        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size
    
    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict