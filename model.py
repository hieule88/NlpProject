import torch
from torchcrf import CRF
import torch.nn as nn
import pytorch_lightning as pl
from preprocess import Preprocessor

class LSTM_CRF(pl.LightningModule):
    def __init__(self, embed_size, 
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
                device=None,
                use_crf = True,
                batch_first = True,
                biLSTM = True
    ):

        super().__init__()
        self.device = device
        self.num_labels = num_labels
        self.num_val_dataloader = num_val_dataloader
        self.use_crf = use_crf
        self.hidden_size = hidden_size

        # Preprocess + embed
        self.preprocessor = Preprocessor()
        self.embed = self.preprocessor.w2vModel_word_to_vector

        self.lstm = nn.LSTM(embed_size, hidden_size=hidden_size, bidirectional=biLSTM, batch_first=batch_first)

        self.dropout = nn.Dropout(p= dropout)

        self.crf = CRF(self.num_labels, batch_first=batch_first)

        self.cls_head = SimpleClsHead(hidden_size, self.num_labels)

    def forward(self, input):
        # get label 
        x = input['sentences']
        labels = input['labels']
        
        # embed
        batch_size = x.size(0)
        x = self.embed(x)

        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)

        recurrent_features, (h_1, c_1) = self.lstm(input, (h_0, c_0))
        recurrent_features = self.dropout(recurrent_features)

        after_lstm = self.cls_head(recurrent_features)
        # if after_lstm.isnan().any():
        #     raise NanException(f"NaN after CLS head")

        if not self.use_crf:
            logits = after_lstm
            loss_fct = nn.CrossEntropyLoss(ignore_index= -2)

            loss = loss_fct(logits.reshape((logits.shape[0]*logits.shape[1], logits.shape[2])),\
                                            labels.reshape((labels.shape[0]*labels.shape[1])))

        else:   

            logits = torch.tensor(self.crf.decode(after_lstm))
            mask = torch.tensor([[1 if labels[j][i] != -2 else 0 \
                                    for i in range(len(labels[j]))] \
                                    for j in range(len(labels))], dtype=torch.uint8)

            loss = self.crf(after_lstm, labels, mask=mask)
 
        return loss, logits

class SimpleClsHead(nn.Module):

    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, x, **kwargs):
        # x = torch.tanh(x)
        x = self.dense(x)
        return x

    def reset_parameters(self):
        self.dense.reset_parameters()

