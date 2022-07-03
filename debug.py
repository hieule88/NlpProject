import datasets 
from datasets import Dataset
import pickle
from preprocess import Preprocessor

# preprocessor = Preprocessor(train_path= 'C:/Users/ThinkPro/OneDrive/Máy tính/NlpProject/dataset/train_update_10t01.pkl',\
#                             mode= 'test',\
#                             val_path= 'C:/Users/ThinkPro/OneDrive/Máy tính/NlpProject/dataset/dev_update_10t01.pkl',\
#                             test_path= 'C:/Users/ThinkPro/OneDrive/Máy tính/NlpProject/dataset/test_update_10t01.pkl',)
dataset = {}

with open('C:/Users/ThinkPro/OneDrive/Máy tính/NlpProject/dataset/processed_train_data.pkl', 'rb') as f:
    dataset['train'] = pickle.load(f)
# with open('C:/Users/ThinkPro/OneDrive/Máy tính/NlpProject/dataset/processed_dev_data.pkl', 'rb') as f:
#     dataset['validation'] = pickle.load(f)
# with open('C:/Users/ThinkPro/OneDrive/Máy tính/NlpProject/dataset/processed_test_data.pkl', 'rb') as f:
#     dataset['test'] = pickle.load(f)

data = []
print(dataset['train'].keys())
# for i in range(len(dataset['train']['sentences'])):

#     data.append(len(dataset['train']['sentences'][i]))
print(len(dataset['train']['labels'][0][0]))

import statistics as sta


# using mean() and mode() to calculate average and mode of list elements
mean = sta.mean(data) # 2
mode = sta.mode(data) # 2

print(mean)