import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn as nn
from argparse import ArgumentParser

from data_module import DataModule
from model import LSTM_CRF

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, args):
        self.hparams = args

        self.dm = DataModule.from_argparse_args(self.hparams)
        self.dm.setup()

        self.metric_name = 'f1'

        self.progress_bar = 0
        self.weights_summary = None
        self.early_stop = None
        self.save_path = args.save_path
        self.baseline = False

    @staticmethod
    def add_arguments(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        return parser

    @staticmethod
    def total_params(model):
        return sum(p.numel() for p in model.parameters())


    def setup_model_trainer(self):
        glue_pl = self.setup_model()
        trainer = self.setup_trainer()
        return glue_pl, trainer

    def setup_trainer(self):
        if type(self.early_stop) == int:
            early_stop = EarlyStopping(
                monitor=self.metric_name,
                min_delta=0.00,
                patience=self.early_stop,
                verbose=False,
                mode="max",
            )
            early_stop = [early_stop]
        else:
            early_stop = None

        trainer = pl.Trainer.from_argparse_args(
            self.hparams,
            progress_bar_refresh_rate=self.progress_bar,
            weights_summary=self.weights_summary,
            checkpoint_callback=False,
            callbacks= early_stop,
            max_epochs = self.hparams.max_epochs,
        )
        return trainer

    def setup_model(self):

        glue_pl = LSTM_CRF(
            max_sequence_length= self.dm.max_seq_length,
            num_labels=self.dm.num_labels,
            **vars(self.hparams),
        )

        glue_pl.init_metric(self.dm.metric)
        return glue_pl

    def train(self, model):
        
        trainer = self.setup_trainer()
        train_dataloader = DataLoader(self.dm.dataset['train'], batch_size= self.hparams.train_batch_size, shuffle= True, num_workers= self.hparams.num_workers)
        val_dataloader = DataLoader(self.dm.dataset['validation'], batch_size= self.hparams.eval_batch_size, num_workers= self.hparams.num_workers)
        # self.lr_finder(model, trainer, train_dataloader, val_dataloader)
        trainer.fit(
            model, 
            train_dataloaders= train_dataloader,
            val_dataloaders= val_dataloader,
        )
        num_epoch = len(model.callbacks)
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize= (21, 7), dpi=120)
        ax1.plot([i for i in range(1, num_epoch+1)], [i['accuracy'] for i in model.callbacks], color= 'g')
        ax2.plot([i for i in range(1, num_epoch+1)], [i['f1'] for i in model.callbacks], color= 'b')
        ax3.plot([i for i in range(1, num_epoch+1)], [i['val_loss'] for i in model.callbacks], color= 'r')

        ax1.set(title='Accuracy', xlabel='Epochs', ylabel='Accuracy')
        ax2.set(title='F1', xlabel='Epochs', ylabel='F1')
        ax3.set(title='Val Loss', xlabel='Epochs', ylabel='Loss')

        plt.show()
        trainer.save_checkpoint(self.save_path)
        # print(model.callbacks)

    def evaluate(self):
        glue_pl = self.setup_model()
        print('TOTAL PRAMS:')
        print(self.total_params(glue_pl))
        self.train(glue_pl)
        