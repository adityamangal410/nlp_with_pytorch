from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, file_name):
        df = pd.read_csv(file_name)

        df = df.select_dtypes(include=['int64'])
        X_df = df.drop(['Id', 'SalePrice'], axis=1)
        self._columns = X_df.columns
        X = X_df.values
        y = df.loc[:, 'SalePrice'].values
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)

        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'x_data': self.X[idx], 'y_target': self.y[idx]}

    def get_scaler(self):
        return self._scaler

    def get_columns(self):
        return self._columns


class HousePricePredictor(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(33, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 8)
        self.fc4 = torch.nn.Linear(8, 1)
        self.loss = torch.nn.MSELoss(reduction='none')

    def setup(self, stage):
        self.train_ds = FeatureDataset('../data/raw/house_prices/train.csv')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True
        )

    def forward(self, batch):
        int1 = torch.nn.functional.relu(self.fc1(batch))
        int2 = torch.nn.functional.relu(self.fc2(int1))
        int3 = torch.nn.functional.relu(self.fc3(int2))
        output = self.fc4(int3)
        return output

    def training_step(self, batch, batch_idx):
        y_pred = self(batch['x_data'])
        loss = self.loss(y_pred, batch['y_target']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=args.learning_rate)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)

    args = parser.parse_args()

    model = HousePricePredictor()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.configure_logger(pl.loggers.TensorBoardLogger('lightning_logs/', name='house_price', version=0))
    trainer.fit(model)
