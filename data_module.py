from argparse import ArgumentParser
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from preprocess import Preprocessor
from datasets import Dataset
import datasets
import pickle

import warnings
warnings.filterwarnings("ignore")

class DataModule(pl.LightningDataModule):

    loader_columns = [
        "input_ids",
        "labels",
    ]

    num_labels = 15

    max_seq_length = 50

    def __init__(
        self,
        root_path_other: str,
        max_seq_length: int = 50,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.root_path_other = '/content/NlpProject/dataset'
        preprocessor = Preprocessor(train_path= self.root_path_other + '/train_update_10t01.pkl',\
                            mode= 'init',\
                            val_path= self.root_path_other + '/dev_update_10t01.pkl',\
                            test_path= self.root_path_other + '/test_update_10t01.pkl',)
        self.embed = preprocessor.batch_to_matrix

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.text_fields = 'sentences'
        self.num_labels = 15
        
        # self.dataset = datasets.load_dataset(*self.dataset_names[self.task_name])
        # self.max_seq_length = self.tokenizer.model_max_length

    def setup(self, stage):

        dataset = {}

        with open(self.root_path_other + '/processed_train_data.pkl', 'rb') as f:
            dataset['train'] = pickle.load(f)
        with open(self.root_path_other + '/processed_dev_data.pkl', 'rb') as f:
            dataset['validation'] = pickle.load(f)
        with open(self.root_path_other + '/processed_test_data.pkl', 'rb') as f:
            dataset['test'] = pickle.load(f)

        dataset['train'] = Dataset.from_dict(dataset['train'])
        dataset['validation'] = Dataset.from_dict(dataset['validation'])
        dataset['test'] = Dataset.from_dict(dataset['test'])

        dataset = datasets.DatasetDict({"train":dataset['train'], "validation": dataset['validation'], "test":dataset['test']})
        self.dataset = dataset

        for split in self.dataset.keys():

            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                batch_size= 32,
                remove_columns=['embeddings', 'sentences'],
                drop_last_batch=True
            )

            self.columns = [
                c
                for c in self.dataset[split].column_names
                if c in self.loader_columns
            ]
            
        self.dataset.set_format(type="torch", columns=self.columns)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def train_dataset(self):
        return self.dataset["train"]

    @property
    def val_dataset(self):
        return self.dataset["validation"]

    @property
    def metric(self):
        return f1_score

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        texts_or_text_pairs = example_batch[self.text_fields]

        # Tokenize the text/text pairs
        features = {}
        features['input_ids'] = np.array(self.embed(data= texts_or_text_pairs, max_seq_length= self.max_seq_length))

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = np.array(self.embed(example_batch['labels'], max_seq_length= self.max_seq_length, mode='labels'))

        return features

    @staticmethod
    def add_cache_arguments(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--root_path_other", default='', type=str)
        return parser