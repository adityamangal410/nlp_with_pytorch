from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch.utils.data import Dataset
import sh
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

sh.rm('-r', '-f', 'lightning_logs/disaster_tweets')


class DisasterTweetsClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(26, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 8)
        self.fc4 = torch.nn.Linear(8, 2)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def setup(self, stage):
        class TweetsDataset(Dataset):
            def __init__(self, split='train'):
                df = pd.read_csv('../data/processed/nlp_with_disaster_tweets/train_with_splits.csv')
                df = df[df.split == split]

                def _get_char_counts(text):
                    vec = np.zeros(26)
                    for word in text.split(' '):
                        for letter in word:
                            if letter.isalpha():
                                vec[ord(letter) - ord('a')] += 1
                    return vec

                X = df.text.apply(_get_char_counts).apply(pd.Series).values
                y = df.target.values

                # Convert to tensors
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y)

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return {'x_data': self.X[idx], 'y_target': self.y[idx]}

        self.train_ds = TweetsDataset()
        self.val_ds = TweetsDataset(split='val')
        self.test_ds = TweetsDataset(split='test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )

    def forward(self, batch):
        int1 = torch.nn.functional.relu(self.fc1(batch.float()))
        int2 = torch.nn.functional.relu(self.fc2(int1))
        int3 = torch.nn.functional.relu(self.fc3(int2))
        output = self.fc4(int3)
        return output

    def training_step(self, batch, batch_idx):
        y_pred = self(batch['x_data'])
        loss = self.loss(y_pred, batch['y_target']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch['x_data'])
        loss = self.loss(y_pred, batch['y_target'])
        acc = (y_pred.argmax(-1) == batch['y_target']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=args.learning_rate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )

    def test_step(self, batch, batch_idx):
        y_pred = self(batch['x_data'])
        loss = self.loss(y_pred, batch['y_target'])
        acc = (y_pred.argmax(-1) == batch['y_target']).float()
        return {'loss': loss, 'acc': acc}

    def test_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'test_loss': loss, 'test_acc': acc}
        return {**out, 'log': out}


def predict_target(text, classifier, text_vector=None):
    if text_vector is None:
        vec = np.zeros(26)
        for word in text.split(' '):
            for letter in word:
                if letter.isalpha():
                    vec[ord(letter) - ord('a')] += 1

        text_vector = torch.tensor(vec, dtype=torch.float32)

    pred = torch.nn.functional.softmax(classifier(text_vector.unsqueeze(dim=0)), dim=1)
    probability, target = pred.max(dim=1)

    return {'pred': target.item(), 'probability': probability.item()}


def predict_on_dataset(classifier, ds):
    df = pd.DataFrame(columns=["target", "pred", "probability"])
    for sample in iter(ds):
        result = predict_target(text=None, classifier=classifier, text_vector=sample['x_data'])
        result['target'] = sample['y_target'].item()
        df = df.append(result, ignore_index=True)
    f1 = f1_score(df.target, df.pred)
    acc = accuracy_score(df.target, df.pred)
    roc_auc = roc_auc_score(df.target, df.probability)
    print("Result metrics - \n Accuracy={} \n F1-Score={} \n ROC-AUC={}".format(acc, f1, roc_auc))
    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)

    args = parser.parse_args()

    model = DisasterTweetsClassifier()
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.configure_logger(pl.loggers.TensorBoardLogger('lightning_logs/', name='disaster_tweets'))
    trainer.fit(model)
