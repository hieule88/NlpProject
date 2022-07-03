import argparse

import pytorch_lightning as pl

from data_module import DataModule
from model import LSTM_CRF
from trainer import Trainer
from preprocess import Preprocessor

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def parse_args():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = DataModule.add_cache_arguments(parser)
    parser = LSTM_CRF.add_model_specific_args(parser)
    parser = LSTM_CRF.add_learning_specific_args(parser)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args


def main():
    # get args
    args = parse_args()

    # solve problems
    problem = Trainer(args)
    problem.weights_summary = "top"

    print('Run')

    problem.evaluate()  

if __name__ == "__main__":
    preprocessor = Preprocessor(train_path= './dataset/train_update_10t01.pkl',\
                            mode= 'test',\
                            val_path= './dataset/dev_update_10t01.pkl',\
                            test_path= './dataset/test_update_10t01.pkl',)
    main()