from cProfile import label
from email.policy import default
import json
import pickle
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import time


DATASET_FOLDER = './dataset'
RESULT_FOLDER = './eda_result'
DATASETS = [ "train", "test", "dev", "all" ]


class EDAtor():

    def __init__(self, dataset, result_folder):
        self.dataset = dataset
        self.result_folder = result_folder


    # feel free to adjustment
    # just remember to save EDA result (Ex: image) to eda_result
    def eda_method(self):    
        pass


    def filter_selected_indexes(self, data, row=()):
        if len(row) > 0: data = [data[i] for i in row]
        return data

    def load_database(self, dataset, index):
        with open('./dataset/specification.json', 'r') as f:
            data = json.load(f)
        dataset = DATASETS[dataset]
        name = data.get('dataset').get(dataset)[index].get('name')
        path = self.dataset + '/' + name
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return [name, data]


    def export_to_csv(self, dataset=0, index=0):
        [name, data] = self.load_database(dataset, index)
        
        i = 0
        indexes = []
        sentences = []
        tags = []
        
        for row in data:
            indexes.append(i)
            i+=1
            sentence = []
            tag = []
            for tuple in row:
                sentence.append(tuple[0])
                tag.append(tuple[1])
            sentences.append(sentence)
            tags.append(tag)

        df = pd.DataFrame({
            'Index': indexes,
            'Sentence': sentences,
            'NER_Tags': tags
        })
        df.to_csv(RESULT_FOLDER+'/output-'+name+'.csv',index=False)


    def export_to_console(self, dataset=0, index=0, row=()):
        [name, data] = self.load_database(dataset, index)
        print("DATABASE", name+':')
        data = self.filter_selected_indexes(data, row)
        print(data)


    def export_original_sentences(self, dataset=0, index=0, row=(), to_csv=False):
        [name, data] = self.load_database(dataset, index)
        data = self.filter_selected_indexes(data, row)

        sentences = list(map(lambda dataRow: re.sub(r'\s+([?.,;:!"])', r'\1', ' '.join(list(map(lambda tp: tp[0], dataRow)))).replace('_', ' '), data))
        sentences.sort()

        if to_csv:
            df = pd.DataFrame({
                'Sentence': sentences
            })
            df.to_csv(RESULT_FOLDER+'/origin-sentences-'+name+'.csv',index=False)
        else: 
            print("DATABASE", name+':')
            print(sentences)


    def get_all_sentences(self, loaded_dataset):
        data = loaded_dataset[1]
        sentences = list(map(lambda dataRow: re.sub(r'\s+([?.,;:!"])', r'\1', ' '.join(list(map(lambda tp: tp[0], dataRow)))).replace('_', ' '), data))
        return sorted(sentences)


    def get_all_of_type(self, loaded_dataset, type):
        data = loaded_dataset[1]

        if type == 'token':
            tokens = set()
            list(map(lambda row: list(map(lambda tpl: tokens.add(tpl[0]), row)) ,data))
            return sorted(list(tokens))
        if type == 'tag':
            tags = set()
            list(map(lambda row: list(map(lambda tpl: tags.add(tpl[1]), row)) ,data))
            return sorted(list(tags))
        if type == 'pair':
            pairs = set()
            list(map(lambda row: list(map(lambda tpl: pairs.add(tpl), row)) ,data))
            return sorted(list(pairs))

        print('Invalid type')
        return []


    def get_tokens_per_tag(self, loaded_dataset, tag):
        data = loaded_dataset[1]
        count = 0
        tokens = set()
        for row in data:
            for tpl in row:
                if tpl[1] == tag:
                    count+=1
                    tokens.add(tpl[0])

        return [count, tokens]


    def get_tags_per_token(self, loaded_dataset, token):
        data = loaded_dataset[1]
        count = 0
        tags = set()
        for row in data:
            for tpl in row:
                if tpl[0] == token:
                    count+=1
                    tags.add(tpl[1])

        return [count, tags]

    
    def get_pairs_frequency(self, loaded_dataset, token="", tag=""):
        pass


    def get_tags_frequency(self, loaded_dataset, selected_tags=()):
        # tic = time.perf_counter()

        all_tags = self.get_all_of_type(loaded_dataset, 'tag')
        tags = self.filter_selected_indexes(all_tags, selected_tags)
        tags_keys = dict(zip(tags, [0]*len(tags)))

        freqs = {}
        [freqs.update({tag: 0}) for tag in tags]

        data = loaded_dataset[1]
        for row in data:
            [freqs.update({tpl[1]: freqs[tpl[1]]+1}) for tpl in row if tpl[1] in tags_keys]
        
        freqs = [freqs[tag] for tag in sorted(freqs.keys())]
        tags_freq = list(zip(tags, freqs))

        tags_freq.sort(key=lambda i:i[1], reverse=True)

        # toc = time.perf_counter()
        # print(f"Performance: {toc - tic:0.4f} seconds")
        return tags_freq

    
    def get_tokens_frequency(self, loaded_dataset, selected_tokens=()):
        # tic = time.perf_counter()

        all_tokens = self.get_all_of_type(loaded_dataset, 'token')
        tokens = self.filter_selected_indexes(all_tokens, selected_tokens)
        tokens_keys = dict(zip(tokens, [0]*len(tokens)))
        
        freqs = {}
        [freqs.update({token: 0}) for token in tokens]
        
        data = loaded_dataset[1]
        for row in data:
            [freqs.update({tpl[0]: freqs[tpl[0]]+1}) for tpl in row if tpl[0] in tokens_keys]

        freqs = [freqs[token] for token in sorted(freqs.keys())]
        tokens_freq = list(zip(tokens, freqs))        

        tokens_freq.sort(key=lambda i:i[1], reverse=True)

        # toc = time.perf_counter()
        # print(f"Performance: {toc - tic:0.4f} seconds")

        return tokens_freq


    def get_dataset_info(self, loaded_dataset):
        name = loaded_dataset[0]

        sentences = self.get_all_sentences(loaded_dataset)        
        tokens = self.get_all_of_type(loaded_dataset, 'token')
        tags = self.get_all_of_type(loaded_dataset,'tag')
        pairs = self.get_all_of_type(loaded_dataset, 'pair')

        return {
            "name": name,
            "sentences": {
                "quantity": len(sentences),
                "data": sentences
            },
            "tokens": {
                "quantity": len(tokens),
                "data": tokens
            },
            "tags": {
                "quantity": len(tags),
                "data": tags
            },
            "pairs": {
                "quantity": len(pairs),
                "data": pairs
            }
        }


    def get_all_dataset_info(self):
        with open('./dataset/specification.json', 'r') as f:
            data = json.load(f)
        datasets = data.get('dataset')
        for index, purpose in enumerate(datasets):
            print("Datasets for", purpose, "purpose:")
            for i in range(len(datasets[purpose])):
                load_dataset = self.load_database(index, i)
                dataset_info = self.get_dataset_info(load_dataset)
                print("-----")
                print("Dataset", dataset_info.get("name"))
                print("Number of sentences:", dataset_info.get("sentences").get("quantity"))
                print("Number of tokens:", dataset_info.get("tokens").get("quantity"))
                print("Number of tags:", dataset_info.get("tags").get("quantity"))
                print("Number of (token, tag) pairs:", dataset_info.get("pairs").get("quantity"))
        return
    
    
    def plot_tags_freq(self, loaded_dataset, start=0, end=0, only_entity=True):
        tags_freq = self.get_tags_frequency(loaded_dataset)

        if only_entity: tags_freq = tags_freq[1:] 
        tags_freq = tags_freq[start:]
        if end > 0: tags_freq = tags_freq[:end]
        
        fig, ax = plt.subplots()
        tags, freqs = zip(*tags_freq)
        y_pos = [i for i in range(len(tags))]

        ax.barh(y_pos, freqs, align='center')
        ax.set_yticks(y_pos, labels=tags)
        ax.invert_yaxis()
        ax.set_xlabel('Tags Frequency')
        
        plt.show()


    # get tokens freq by tags not work yet!
    def plot_tokens_freq(self, loaded_dataset, selected_tags=(), start=0, end=0, only_entity=True, none_punc=True):
        selected_tokens = set()
        if len(selected_tags) > 0:
            list(map(lambda tag: selected_tokens.update(self.get_tokens_per_tag(loaded_dataset, tag)[1]), selected_tags))
        print(len(selected_tokens))
        tokens_freq = self.get_tokens_frequency(loaded_dataset, selected_tokens)

        PUNCS = [',' , '(', ')', '!', '?', '-', '.', ';', ':', '"', "'", '".']
        NONE_ENTITY_TAGS = ["PAD", "O"]
        NONE_ENTITY_TOKENS = [self.get_tokens_per_tag(loaded_dataset, tag)[1] for tag in NONE_ENTITY_TAGS]
        NONE_ENTITY_TOKENS = NONE_ENTITY_TOKENS[0] | NONE_ENTITY_TOKENS[1]
        
        if only_entity: tokens_freq = [token_freq for token_freq in tokens_freq if token_freq[0] not in NONE_ENTITY_TOKENS]
        if none_punc: tokens_freq = [token_freq for token_freq in tokens_freq if token_freq[0] not in PUNCS]
        tokens_freq = tokens_freq[start:]
        if end > 0: tokens_freq = tokens_freq[:end]

        fig, ax = plt.subplots()
        tokens, freqs = zip(*tokens_freq)
        y_pos = [i for i in range(len(tokens))]

        ax.barh(y_pos, freqs, align='center')
        ax.set_yticks(y_pos, labels=tokens)
        ax.invert_yaxis()
        ax.set_xlabel('Tokens Frequency')
        
        plt.show()


def main():
    eda = EDAtor(DATASET_FOLDER, RESULT_FOLDER)
    
    dataset = 0
    index = 0
    row = ()

    test_exp_console = False
    test_exp_csv = False
    test_exp_sentences = False

    if test_exp_console: eda.export_to_console(dataset, index, row)
    if test_exp_csv: eda.export_to_csv(dataset, index)
    if test_exp_sentences: eda.export_original_sentences(dataset, index, row, False)

    load_dataset = eda.load_database(dataset, index)

    # tokens = eda.get_all_of_type(load_dataset, 'token')
    # tags = eda.get_all_of_type(load_dataset, 'tag')
    # pairs = eda.get_all_of_type(load_dataset, 'pair')

    # print("List", len(tokens), "tokens:", tokens)
    # print("List", len(tags), "tags:", tags)
    # print("List", len(pairs), "pairs:", pairs)

    # eda.get_all_dataset_info()
    # print(eda.get_tags_frequency(load_dataset))
    # print(eda.get_tokens_frequency(load_dataset))
    # eda.plot_tags_freq(load_dataset)
    eda.plot_tokens_freq(load_dataset, (0,1,2), 0, 20)

    pass

if __name__ == '__main__':
    main()