import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class Model(pl.LightningModule):
    def __init__(self, hidden_size=1, dropout=0.2, num_layers=1,
                 bidirectional=False,lr=1e-3):
        super(Model, self).__init__()
        self.gru = nn.GRU(
            input_size=35,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.lr = lr
        output_dim_gru = hidden_size if not bidirectional else 2 * hidden_size
        self.fc0 = nn.Linear(in_features=output_dim_gru, out_features=10)
        self.fc1 = nn.Linear(in_features=10, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        o, h = self.gru(x)
        y_hat = F.relu(self.fc0(o[:, -1, :]))
        y_hat = F.relu(self.fc1(y_hat))
        y_hat = self.fc2(y_hat)

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/train", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log("loss/val", loss)
        return loss

    def test_step(self, batch):
        pass
