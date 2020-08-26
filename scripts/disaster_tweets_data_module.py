import string
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch.utils.data import Dataset


class Vocabulary:
    def __init__(self):
        self._token_to_idx = {}
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError('the index {} is not in the vocab'.format(index))
        return self._idx_to_token[index]

    def __len__(self):
        return len(self._token_to_idx)

    def __str__(self):
        return 'Vocabulary(size={})'.format(len(self))

    def get_token_to_idx(self):
        return self._token_to_idx


class SequenceVocabulary(Vocabulary):
    def __init__(self, unk_token="<UNK>", mask_token="<MASK>",
                 begin_seq_token="<BEGIN>", end_seq_token="<END"):
        super(SequenceVocabulary, self).__init__()
        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


class TweetsVectorizer:
    def __init__(self, text_vocab):
        self.text_vocab = text_vocab

    def vectorize(self, text, vector_length=-1):
        indices = [self.text_vocab.begin_seq_index]
        indices.extend(self.text_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.text_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.text_vocab.mask_index

        return out_vector, len(indices)

    @classmethod
    def from_dataframe(cls, tweets_df, cutoff=1):
        word_counts = Counter()
        for text in tweets_df.text:
            for token in text.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1

        text_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count > cutoff:
                text_vocab.add_token(word)

        return cls(text_vocab)


class TweetsDataset(Dataset):
    def __init__(self, tweets_df, vectorizer, max_seq_length=-1):
        self.tweets_df = tweets_df
        self._vectorizer = vectorizer
        self._max_seq_length = max_seq_length

    @classmethod
    def load_dataset_and_make_vectorizer(cls, tweets_df):
        train_tweets_df = tweets_df[tweets_df.split == 'train']
        measure_len = lambda context: len(context.split(" "))
        max_seq_length = max(map(measure_len, tweets_df.text)) + 2  # for begin and end tokens
        return cls(train_tweets_df, TweetsVectorizer.from_dataframe(train_tweets_df), max_seq_length)

    def get_vectorizer(self):
        return self._vectorizer

    def get_max_seq_length(self):
        return self._max_seq_length

    def __len__(self):
        return len(self.tweets_df)

    def __getitem__(self, idx):
        row = self.tweets_df.iloc[idx]

        text_vector, vec_length = self._vectorizer.vectorize(row.text, self._max_seq_length)
        return {'x_data': torch.tensor(text_vector),
                'y_target': torch.tensor(row.target),
                'text': row.text,
                'x_length': vec_length}


class DisasterTweetsDataModule(pl.LightningDataModule):
    def __init__(self, tweets_data_path,
                 embeddings_path, batch_size, num_workers):
        super().__init__()
        self.num_workers = num_workers
        self.tweets_data_path = tweets_data_path
        self.embeddings_path = embeddings_path
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        tweets_df = pd.read_csv(self.tweets_data_path)
        self.train_ds = TweetsDataset.load_dataset_and_make_vectorizer(tweets_df)
        self.val_ds = TweetsDataset(tweets_df[tweets_df.split == 'val'],
                                    self.train_ds.get_vectorizer(),
                                    self.train_ds.get_max_seq_length())
        self.test_ds = TweetsDataset(tweets_df[tweets_df.split == 'test'],
                                     self.train_ds.get_vectorizer(),
                                     self.train_ds.get_max_seq_length())

        def load_glove_from_file(glove_filepath):
            word_to_index = {}
            embeddings = []
            with open(glove_filepath, "r") as fp:
                for index, line in enumerate(fp):
                    line = line.split(" ")
                    word_to_index[line[0]] = index
                    embedding_i = np.array([float(val) for val in line[1:]])
                    embeddings.append(embedding_i)
            return word_to_index, np.stack(embeddings)

        def make_embedding_matrix(glove_filepath, words):
            word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
            embedding_size = glove_embeddings.shape[1]

            final_embeddings = np.zeros((len(words), embedding_size))

            for i, word in enumerate(words):
                if word in word_to_idx:
                    final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
                else:
                    embedding_i = torch.ones(1, embedding_size)
                    torch.nn.init.xavier_uniform_(embedding_i)
                    final_embeddings[i, :] = embedding_i
            return final_embeddings

        words = self.train_ds.get_vectorizer().text_vocab.get_token_to_idx().keys()
        embeddings = make_embedding_matrix(self.embeddings_path, words)
        self.pretrained_embeddings = torch.from_numpy(embeddings).float()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )
