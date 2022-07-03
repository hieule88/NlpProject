import torch
from torchcrf import CRF
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from argparse import ArgumentParser

class LSTM_CRF(pl.LightningModule):
    def __init__(self, 
                num_labels: int = 15,
                embed_size: int = 100,
                hidden_size: int = 128,
                dropout: float = 0.1,
                learning_rate: float = 2e-5,
                epsilon: float = 1e-8,
                warmup_steps: int = 0,
                weight_decay: float = 0.0,
                train_batch_size: int = 32,
                eval_batch_size: int = 32,
                num_val_dataloader: int = 1,
                device=None,
                use_crf: bool = True,
                batch_first = True,
                bidirection = True,
                max_seq_length = 50
    ):

        super().__init__()
        self.device = device
        self.num_labels = num_labels
        self.num_val_dataloader = num_val_dataloader
        self.use_crf = use_crf
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # Preprocess + embed
        self.lstm = nn.LSTM(embed_size, hidden_size=hidden_size, bidirectional=bidirection, batch_first=batch_first)

        self.dropout = nn.Dropout(p= dropout)

        self.crf = CRF(self.num_labels, batch_first=batch_first)

        self.cls_head = SimpleClsHead(hidden_size, self.num_labels)

    def init_metric(self, metric):
        self.metric = metric

    def forward(self, input):
        # get label 
        x = input['sentences']
        labels = input['labels']

        # embed
        batch_size = x.size(0)

        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)

        recurrent_features, (h_1, c_1) = self.lstm(x, (h_0, c_0))
        recurrent_features = self.dropout(recurrent_features)

        after_lstm = self.cls_head(recurrent_features)
        if after_lstm.isnan().any():
            raise Exception(f"NaN after CLS head")

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

    def training_step(self, batch):
        loss, _ = self(**batch)
        return {"loss": loss}

    def validation_step(self, batch):
        val_loss, logits = self(None, **batch)
        if self.num_labels >= 1:
            # preds = torch.argmax(logits, dim=-1)
            preds = logits
        elif self.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
                
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)

        preds = [i for j in range(len(preds)) for i in preds[j][:labels[j][-1]] ]
        labels = [i for j in range(len(labels)) for i in labels[j][:labels[j][-1]] ]

        metrics = {}
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['f1'] = f1_score(labels, preds, average='macro')
        metrics['recall'] = recall_score(labels, preds, average='macro')
        metrics['precision'] = precision_score(labels, preds, average='macro')

        self.log_dict(metrics, prog_bar=True)

        callbacks = metrics
        callbacks['val_loss'] = loss.item()
        self.callbacks.append(callbacks)
        print(f'epoch: {self.current_epoch}, val_loss: {loss}, accuracy: {metrics} ')

        return

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def setup(self, stage):
        if stage == "fit":
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (
                    len(train_loader.dataset)
                    // (self.train_batch_size)
                )
                * float(self.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.lstm
        fc = self.cls_head
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in fc.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ]
                + [
                    p
                    for n, p in fc.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.epsilon,
        )

        return [optimizer]

    def total_params(self):
        return sum(p.numel() for p in self.recurrent_model.parameters())

    def reset_weights(self):
        self.cls_head.reset_parameters()
        self.lstm.reset_parameters()

    @staticmethod
    def add_learning_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_first", default=True, type=bool)
        parser.add_argument("--bidirection", default=True, type=bool)
        parser.add_argument("--hidden_size", default=128, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)
        parser.add_argument("--use_crf", default=True, type=bool)
        return parser

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

