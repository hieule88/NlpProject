import torch
from torchcrf import CRF
import torch.nn as nn
import pytorch_lightning as pl

class LSTM_CRF(pl.LightningModule):
    def __init__(self, in_dim, out_dim, 
                num_labels: int,
                hidden_size: int = 128,
                dropout: float = 0.1,
                learning_rate: float = 2e-5,
                epsilon: float = 1e-8,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                train_batch_size: int = 32,
                eval_batch_size: int = 32,
                freeze_embed=False, 
                num_val_dataloader: int = 1,
                device=None
    ):

        super().__init__()
        self.out_dim = out_dim
        self.device = device
        self.num_labels = num_labels
        self.num_val_dataloader = num_val_dataloader

        self.lstm0 = nn.LSTM(in_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p= dropout)

        self.crf = CRF(self.num_labels, batch_first=self.hparams.batch_first)

        self.linear = nn.Sequential(nn.Linear(in_features=128, out_features=out_dim), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(1, batch_size, 32).to(self.device)
        c_0 = torch.zeros(1, batch_size, 32).to(self.device)

        recurrent_features, (h_1, c_1) = self.lstm0(input, (h_0, c_0))
        recurrent_features, (h_2, c_2) = self.lstm1(recurrent_features)
        recurrent_features = self.drop_out(recurrent_features)
        recurrent_features, _ = self.lstm2(recurrent_features)
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 128))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs, recurrent_features

class LSTMLinear():
    def __init__(self):
        pass