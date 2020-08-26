from argparse import ArgumentParser

import sh
import torch.utils.data
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from disaster_tweets_data_module import *

# sh.rm('-r', '-f', 'lightning_logs/disaster_tweets_cnn_glove')


class DisasterTweetsClassifierCNN(pl.LightningModule):

    def __init__(self, num_channels, hidden_dim, num_classes,
                 dropout_p, pretrained_embeddings, learning_rate):
        super().__init__()

        self.save_hyperparameters('num_classes', 'dropout_p', 'learning_rate', 'num_channels', 'hidden_dim')

        embedding_dim = pretrained_embeddings.size(1)
        num_embeddings = pretrained_embeddings.size(0)

        self.emb = torch.nn.Embedding(embedding_dim=embedding_dim,
                                      num_embeddings=num_embeddings,
                                      padding_idx=0,
                                      _weight=pretrained_embeddings)

        self.convnet = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=embedding_dim,
                            out_channels=num_channels, kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                            kernel_size=3),
            torch.nn.ELU()
        )
        self._dropout_p = dropout_p
        self.fc1 = torch.nn.Linear(num_channels, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch):
        x_embedded = self.emb(batch).permute(0, 2, 1)
        features = self.convnet(x_embedded)

        remaining_size = features.size(dim=2)
        features = torch.nn.functional.max_pool1d(features, remaining_size).squeeze(dim=2)

        int1 = torch.nn.functional.relu(torch.nn.functional.dropout(self.fc1(features),
                                                                    p=self._dropout_p))
        output = self.fc2(int1)
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_channels', default=64, type=int)
        parser.add_argument('--hidden_dim', default=32, type=int)
        parser.add_argument('--dropout_p', default=0.7, type=float)
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        return parser


class DisasterTweetsClassifierCNNResidual(DisasterTweetsClassifierCNN):
    def __init__(self, num_channels, hidden_dim, num_classes,
                 dropout_p, pretrained_embeddings, learning_rate, max_seq_length):
        super().__init__(num_channels, hidden_dim, num_classes,
                         dropout_p, pretrained_embeddings, learning_rate)

        self.save_hyperparameters('max_seq_length')

        embedding_dim = pretrained_embeddings.size(1)

        self.convnet = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=embedding_dim,
                            out_channels=max_seq_length,
                            kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=max_seq_length,
                            out_channels=max_seq_length,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=max_seq_length,
                            out_channels=max_seq_length,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=max_seq_length,
                            out_channels=max_seq_length,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=max_seq_length,
                            out_channels=max_seq_length,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=max_seq_length,
                            out_channels=max_seq_length,
                            kernel_size=3),
            torch.nn.ELU()
        )
        self.fc1 = torch.nn.Linear(max_seq_length,
                                   self.hparams.hidden_dim)

    def forward(self, batch):
        x_embedded = self.emb(batch).permute(0, 2, 1)
        features = self.convnet(x_embedded)

        remaining_size = features.size(dim=2)
        features = torch.nn.functional.max_pool1d(features, remaining_size).squeeze(dim=2)

        features = features + batch
        int1 = torch.nn.functional.relu(torch.nn.functional.dropout(self.fc1(features),
                                                                    p=self._dropout_p))
        output = self.fc2(int1)
        return output


def predict_target(text, classifier, vectorizer, max_seq_length):
    text_vector, _ = torch.tensor(vectorizer.vectorize(text, max_seq_length))
    pred = torch.nn.functional.softmax(classifier(text_vector.unsqueeze(dim=0)), dim=1)
    probability, target = pred.max(dim=1)

    return {'pred': target.item(), 'probability': probability.item()}


def predict_on_dataset(classifier, ds):
    classifier.eval()
    df = pd.DataFrame(columns=["text", "target", "pred", "probability"])
    for sample in iter(ds):
        result = predict_target(sample['text'], classifier, ds.get_vectorizer(), ds.get_max_seq_length())
        result['target'] = sample['y_target'].item()
        result['text'] = sample['text']
        df = df.append(result, ignore_index=True)
    df.target = df.target.astype(np.int32)
    df.pred = df.pred.astype(np.int32)

    f1 = f1_score(df.target, df.pred)
    acc = accuracy_score(df.target, df.pred)
    roc_auc = roc_auc_score(df.target, df.probability)
    print("Result metrics - \n Accuracy={} \n F1-Score={} \n ROC-AUC={}".format(acc, f1, roc_auc))
    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--tweets_data_path',
                        default='../data/processed/nlp_with_disaster_tweets/train_with_splits.csv',
                        type=str)
    parser.add_argument('--embeddings_path',
                        default='/Users/amangal/Desktop/machine-learning/nlp_with_disaster_tweets/data/glove.twitter'
                                '.27B/glove.twitter.27B.25d.txt',
                        type=str)

    parser = DisasterTweetsClassifierCNN.add_model_specific_args(parser)

    args = parser.parse_args()
    print("Args: \n {}".format(args))

    pl.seed_everything(42)

    dm = DisasterTweetsDataModule(tweets_data_path=args.tweets_data_path,
                                  embeddings_path=args.embeddings_path,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    dm.setup('fit')

    # model = DisasterTweetsClassifierCNN(num_channels=args.num_channels,
    #                                     hidden_dim=args.hidden_dim,
    #                                     num_classes=2,
    #                                     dropout_p=args.dropout_p,
    #                                     pretrained_embeddings=dm.pretrained_embeddings,
    #                                     learning_rate=args.learning_rate)

    model = DisasterTweetsClassifierCNNResidual(num_channels=args.num_channels,
                                                hidden_dim=args.hidden_dim,
                                                num_classes=2,
                                                dropout_p=args.dropout_p,
                                                pretrained_embeddings=dm.pretrained_embeddings,
                                                learning_rate=args.learning_rate,
                                                max_seq_length=dm.train_ds.get_max_seq_length())

    trainer = pl.Trainer.from_argparse_args(args, deterministic=True,
                                            weights_summary='full')

    trainer.configure_logger(pl.loggers.TensorBoardLogger('lightning_logs/',
                                                          name='disaster_tweets_cnn_glove'))
    trainer.fit(model, dm)
