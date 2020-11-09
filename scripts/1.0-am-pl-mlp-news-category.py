import re
import string
from collections import Counter
import numpy as np

import nlp
import transformers

import torch
import torch.utils.data
import pytorch_lightning as pl

from argparse import ArgumentParser


class Vocabulary:
    def __init__(self, token_to_idx=None):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """Retrieve the index associated with the token

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        """
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


class NewsCategoryClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        nlp.load_dataset('ag_news')

    def setup(self, stage):
        def _clean_text(text):
            text = text.lower()
            text = re.sub(r"([.,!?])", r" \1 ", text)
            text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
            return text

        def _preprocess(x):
            x['text'] = _clean_text(x['text'])
            return x

        def _add_text_to_counter(x, counter):
            for token in x['text'].split(" "):
                if token not in string.punctuation:
                    counter[token] += 1

        def _create_vocab(counter, cutoff=1):
            vocab = SequenceVocabulary()
            for word, word_count in counter.items():
                if word_count >= cutoff:
                    vocab.add_token(word)
            return vocab

        def _vectorize(x):
            indices = [self.vocab.begin_seq_index]
            indices.extend(self.vocab.lookup_token(token) for token in x['text'].split(" "))
            indices.append(self.vocab.end_seq_index)

            out_vector = np.zeros(self._max_seq_len, dtype=np.int64)
            if self._max_seq_len > len(indices):
                out_vector[:len(indices)] = indices
                out_vector[len(indices):] = self.vocab.mask_index
            else:
                out_vector = indices[:self._max_seq_len]

            x['input_ids'] = out_vector
            return x

        def _measure_len(x):
            return len(x['text'].split(" "))

        def _prepare_ds_train(ds):
            ds = ds.map(_vectorize)

            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        def _get_train_ds_and_vocab():
            ds = nlp.load_dataset('ag_news',
                                  split='train[:{}]'.format(args.batch_size if args.fast_dev_run else '5%'))
            ds = ds.map(_preprocess)
            _max_seq_len = max(_measure_len(sample) for sample in ds) + 2

            word_counts = Counter()
            for sample in ds:
                _add_text_to_counter(sample, word_counts)

            vocab = _create_vocab(word_counts)

            return ds, vocab, _max_seq_len

        def _prepare_ds_val():
            ds = nlp.load_dataset('ag_news',
                                  split='test[:{}]'.format(args.batch_size if args.fast_dev_run else '5%'))
            ds = ds.map(_preprocess)

            ds = ds.map(_vectorize)

            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.vocab, self._max_seq_len = _get_train_ds_and_vocab()
        self.train_ds = _prepare_ds_train(self.train_ds)
        self.val_ds = _prepare_ds_val()
        self.fc1 = torch.nn.Linear(self._max_seq_len, self._hidden_dim)

    def forward(self, batch, apply_softmax=False):
        intermediate = torch.nn.functional.relu(self.fc1(batch.float()))
        output = self.fc2(intermediate)
        if apply_softmax:
            output = torch.nn.functional.softmax(output, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        y_pred = self(batch['input_ids'])
        loss = self.loss(y_pred, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch['input_ids'])
        loss = self.loss(y_pred, batch['label'])
        acc = (y_pred.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=True
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=args.learning_rate)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--seq_length', default=32, type=str)
    parser.add_argument('--hidden_dim', default=16, type=str)

    args = parser.parse_args()

    num_classes = 4
    model = NewsCategoryClassifier(input_dim=args.seq_length, hidden_dim=args.hidden_dim, output_dim=num_classes)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.configure_logger(pl.loggers.TensorBoardLogger('lightning_logs/', name='ag_news', version=0))
    trainer.fit(model)
