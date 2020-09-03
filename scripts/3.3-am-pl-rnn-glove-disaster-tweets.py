from argparse import ArgumentParser

import sh
import torch.nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from disaster_tweets_data_module import *


# sh.rm('-r', '-f', 'lightning_logs/disaster_tweets_rnn_glove')


class DisasterTweetsClassifierRNN(pl.LightningModule):

    def __init__(self, rnn_hidden_size, num_classes,
                 dropout_p, pretrained_embeddings, learning_rate):
        super().__init__()

        self.save_hyperparameters('num_classes', 'dropout_p', 'learning_rate', 'rnn_hidden_size')

        embedding_dim = pretrained_embeddings.size(1)
        num_embeddings = pretrained_embeddings.size(0)

        self.emb = torch.nn.Embedding(embedding_dim=embedding_dim,
                                      num_embeddings=num_embeddings,
                                      padding_idx=0,
                                      _weight=pretrained_embeddings)

        self.rnn = torch.nn.RNNCell(embedding_dim, rnn_hidden_size)
        self._dropout_p = dropout_p
        self.fc1 = torch.nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.fc2 = torch.nn.Linear(rnn_hidden_size, num_classes)

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def _initialize_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hparams.rnn_hidden_size))

    def forward(self, batch, batch_lengths):
        x_embedded = self.emb(batch)
        batch_size, seq_size, feat_size = x_embedded.size()
        x_embedded = x_embedded.permute(1, 0, 2)

        hiddens = []
        initial_hidden = self._initialize_hidden(batch_size)

        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn(x_embedded[t], hidden_t)
            hiddens.append(hidden_t)

        hiddens = torch.stack(hiddens).permute(1, 0, 2)

        features = self.column_gather(hiddens, batch_lengths)

        int1 = torch.nn.functional.relu(torch.nn.functional.dropout(self.fc1(features),
                                                                    p=self._dropout_p))
        output = self.fc2(torch.nn.functional.dropout(int1, p=self._dropout_p))
        return output

    def column_gather(self, hiddens, batch_lengths):
        batch_lengths = batch_lengths.long().detach().cpu().numpy() - 1
        out = []

        for batch_index, column_index in enumerate(batch_lengths):
            out.append(hiddens[batch_index, column_index])
        return torch.stack(out)

    def training_step(self, batch, batch_idx):
        y_pred = self(batch['x_data'], batch['x_length'])
        loss = self.loss(y_pred, batch['y_target']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch['x_data'], batch['x_length'])
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
        y_pred = self(batch['x_data'], batch['x_length'])
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
        parser.add_argument('--rnn_hidden_size', default=32, type=int)
        parser.add_argument('--dropout_p', default=0.7, type=float)
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        return parser


class DisasterTweetsClassifierGRU(DisasterTweetsClassifierRNN):
    def __init__(self, rnn_hidden_size, num_classes,
                 dropout_p, pretrained_embeddings, learning_rate):
        super().__init__(rnn_hidden_size, num_classes,
                         dropout_p, pretrained_embeddings, learning_rate)

        embedding_dim = pretrained_embeddings.size(1)
        self.rnn = torch.nn.GRUCell(embedding_dim, rnn_hidden_size)


class DisasterTweetsClassifierLSTM(DisasterTweetsClassifierRNN):
    def __init__(self, rnn_hidden_size, num_classes,
                 dropout_p, pretrained_embeddings, learning_rate):
        super().__init__(rnn_hidden_size, num_classes,
                         dropout_p, pretrained_embeddings, learning_rate)

        embedding_dim = pretrained_embeddings.size(1)
        self.rnn = torch.nn.LSTMCell(embedding_dim, rnn_hidden_size)


def predict_target(text, classifier, vectorizer, max_seq_length):
    text_vector, vec_length = vectorizer.vectorize(text, max_seq_length)
    text_vector = torch.tensor(text_vector)
    vec_length = torch.tensor(vec_length)
    pred = torch.nn.functional.softmax(classifier(text_vector.unsqueeze(dim=0), vec_length.unsqueeze(dim=0)), dim=1)
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

    parser = DisasterTweetsClassifierRNN.add_model_specific_args(parser)

    args = parser.parse_args()
    print("Args: \n {}".format(args))

    pl.seed_everything(42)

    dm = DisasterTweetsDataModule(tweets_data_path=args.tweets_data_path,
                                  embeddings_path=args.embeddings_path,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    dm.setup('fit')

    # model = DisasterTweetsClassifierRNN(rnn_hidden_size=args.rnn_hidden_size,
    #                                     num_classes=2,
    #                                     dropout_p=args.dropout_p,
    #                                     pretrained_embeddings=dm.pretrained_embeddings,
    #                                     learning_rate=args.learning_rate)

    model = DisasterTweetsClassifierGRU(rnn_hidden_size=args.rnn_hidden_size,
                                        num_classes=2,
                                        dropout_p=args.dropout_p,
                                        pretrained_embeddings=dm.pretrained_embeddings,
                                        learning_rate=args.learning_rate)

    # model = DisasterTweetsClassifierLSTM(rnn_hidden_size=args.rnn_hidden_size,
    #                                     num_classes=2,
    #                                     dropout_p=args.dropout_p,
    #                                     pretrained_embeddings=dm.pretrained_embeddings,
    #                                     learning_rate=args.learning_rate)

    trainer = pl.Trainer.from_argparse_args(args,
                                            deterministic=True,
                                            weights_summary='full',
                                            early_stop_callback=True)

    trainer.configure_logger(pl.loggers.TensorBoardLogger('lightning_logs/',
                                                          name='disaster_tweets_rnn_glove'))
    trainer.fit(model, dm)
