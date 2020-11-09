import nlp
import transformers
from argparse import ArgumentParser
import torch
import torch.utils.data
import pytorch_lightning as pl


class NewsCategoryClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(args.model)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(args.model)

        def _tokenize(x):
            x['input_ids'] = tokenizer.encode(x['text'], max_length=args.seq_length, truncation=True)
            return x

        def _prepare_ds(split):
            ds = nlp.load_dataset('ag_news',
                                  split='{}[:{}]'.format(split, args.batch_size if args.fast_dev_run else '5%'))
            ds = ds.map(_tokenize)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            return ds

        self.train_ds, self.val_ds = map(_prepare_ds, ('train', 'test'))
        print("here")

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        outputs = self.model(input_ids, mask)
        return outputs

    def training_step(self, batch, batch_idx):
        print("here")
        pass

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'])
        loss = self.loss(outputs, batch['label'])
        print("here")

    def validation_epoch_end(self, outputs):
        print("here")
        pass

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
            shuffle=False
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=args.learning_rate
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--model', default='bert-base-uncased', type=str)
    parser.add_argument('--seq_length', default=32, type=str)

    args = parser.parse_args()

    model = NewsCategoryClassifier()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
