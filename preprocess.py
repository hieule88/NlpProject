from audioop import avg
from concurrent.futures import process
import numpy as np
import pickle
import string
from pprint import pprint
from gensim.models import Word2Vec
from collections import Counter
from vncorenlp import VnCoreNLP
import vnlpc
import torch


class Preprocessor():
    embedding_len = 100

    def __init__(self, train_path, mode='train', val_path= None, test_path= None):
        # train, test, dev
        self.make_tag_lookup_table()

        self.listpunctuation = string.punctuation.replace('_', '')
        tmp = []
        for i in range(0, len(self.listpunctuation)):
            tmp.append(self.listpunctuation[i])
        self.listpunctuation = tmp

        self.dataset = {}
        self.processed_data = {}

        # VncoreNLP
        # self.annotator = VnCoreNLP(
        # "./VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx5g')
        # self.vc = vnlpc.VNLPClient("http://localhost:39000")

        self.w2vModel = None
        self.avg_vector = None

        self.load_raw_data(train_path, "train")
        self.w2vModel_from_data(self.dataset["train"]["sentences"])
        self.make_one_hot_vector_for_tag("train")

        if mode != 'train':
            self.preprocess_dev(val_path)
            self.preprocess_test(test_path)


    def preprocess_test(self, path):
        self.load_raw_data(path, "test")
        self.make_one_hot_vector_for_tag("test")
        return self.processed_data["test"]

    def preprocess_dev(self, path):
        self.load_raw_data(path, "dev")
        self.make_one_hot_vector_for_tag("dev")
        return self.processed_data["dev"]

    def batch_to_matrix(self, data, max_seq_length, mode='sentences'):
        rs = []
        if mode == 'sentences':
            for sentence in data:
                processed_sentence = []
                if len(sentence) > max_seq_length:
                    for word_index in range(max_seq_length):  
                        vector = self.w2vModel_word_to_vector(sentence[word_index])
                        processed_sentence.append(vector)
                    rs.append(processed_sentence)
                else:
                    for word in sentence:  
                        vector = self.w2vModel_word_to_vector(word)
                        processed_sentence.append(vector)
                    for i in range(max_seq_length - len(sentence)):
                        vector = [0 for j in range(self.embedding_len)]
                        processed_sentence.append(vector)
                    rs.append(processed_sentence)
        
        else:
            for label in data:
                if len(label) > max_seq_length:
                    rs.append(label[:max_seq_length])
                else:
                    for i in range(max_seq_length - len(label)):
                        padding_label = [-2 for i in range(15)]
                        label.append(padding_label)
                    rs.append(label)
        return rs

    def load_raw_data(self, input_path, name):
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        sentences = []
        tags = []
        self.dataset[name] = {}
        for item in data:
            pre_sentence = []
            tag = []
            for token in item:
                if token[0] not in self.listpunctuation:
                    pre_sentence.append(token[0].lower())
                    tag.append(token[1])
            sentences.append(pre_sentence)
            tags.append(tag)

        self.dataset[name]["sentences"] = sentences
        self.dataset[name]["tags"] = tags
        return self.dataset[name]

    def load_processed_data(self, input_path, name):
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        self.processed_data[name] = data
        return data

    def make_tag_lookup_table(self):
        # ner_labels = ["PAD", "ADDRESS", "SKILL", "EMAIL", "PERSON", "PHONENUMBER", "QUANTITY", "PERSONTYPE",
        #               "ORGANIZATION", "PRODUCT", "IP", "LOCATION", "O", "DATETIME", "EVENT", "URL", "MISCELLANEOUS"]
        ner_labels = ["PAD", "ADDRESS", "SKILL", "PERSON", "PHONENUMBER", "QUANTITY", "PERSONTYPE",
                      "ORGANIZATION", "PRODUCT", "LOCATION", "O", "DATETIME", "EVENT", "RULE", "MISCELLANEOUS"]
        table = {}
        i = 0
        for label in ner_labels:
            table[label] = i
            i += 1
        self.tag_table = table
        return table

    def make_one_hot_vector_for_tag(self, name):
        self.make_tag_lookup_table()
        rs = {}
        rs["sentences"] = []
        rs["embeddings"] = []
        rs["labels"] = []
        sentences = self.dataset[name]["sentences"]
        tags = self.dataset[name]["tags"]
        EUI_tag = ["EMAIL", "URL", "IP"]
        for i in range(len(sentences)):
            sentence = sentences[i]
            tag = tags[i]
            processed_sentence = []
            processed_embedding = []
            processed_label = []

            for k in range(len(sentence)):
                if tag[k] in EUI_tag:
                    tag[k] = "RULE"
                vector = [0 for i in range(15)]
                vector[self.tag_table[tag[k]]] = 1
                vector = np.array(vector)
                sentence[k] = sentence[k].lower()
                w2vVector = self.w2vModel_word_to_vector(sentence[k])
                processed_sentence.append(sentence[k])
                processed_embedding.append(w2vVector)
                processed_label.append(vector)
            rs["sentences"].append(processed_sentence)
            rs["embeddings"].append(processed_embedding)
            rs["labels"].append(processed_label)
        self.processed_data[name] = rs
        with open("/content/NlpProject/dataset/processed_" + name + "_data.pkl", 'wb') as f:
            pickle.dump(rs, f, protocol=pickle.HIGHEST_PROTOCOL)
        return rs

    # def tokenize(self, sentence):
    #     tmps = self.vc.tokenize(sentence.lower())
    #     rs = []
    #     for tmp in tmps:
    #         if tmp not in self.listpunctuation:
    #             rs.append(tmp)
    #     return rs


    def w2vModel_from_data(self, data):
        self.w2vModel = Word2Vec(
            sentences=data, min_count=1, vector_size=100, window=5, sg=1)
        vocabs = self.w2vModel_get_vocab()
        sum_vector = np.array(self.w2vModel_word_to_vector(vocabs[0]))
        for index in range(1, len(vocabs)):
            sum_vector += np.array(self.w2vModel_word_to_vector(vocabs[index]))
        self.avg_vector = sum_vector / self.w2vModel_get_vocab_length()
        self.w2vModel.save("word2vec.model")

    def w2vModel_from_file(self, model_path):
        self.w2vModel = Word2Vec.load(model_path)
        vocabs = self.w2vModel_get_vocab()
        sum_vector = np.array(self.w2vModel_word_to_vector(vocabs[0]))
        for index in range(1, len(vocabs)):
            sum_vector += np.array(self.w2vModel_word_to_vector(vocabs[index]))
        self.avg_vector = sum_vector / self.w2vModel_get_vocab_length()

    def w2vModel_word_to_vector(self, word):
        word = word.lower()
        if(self.w2vModel != None):
            try:
                return self.w2vModel.wv[word]
            except:
                return self.avg_vector

    def w2vModel_id_to_vector(self, id):
        if(self.w2vModel != None):
            word = self.w2vModel_id_to_word(id)
            return self.w2vModel_word_to_vector(word)

    def w2vModel_word_to_id(self, word):
        if(self.w2vModel != None):
            return self.w2vModel.wv.key_to_index[word]

    def w2vModel_id_to_word(self, id):
        if(self.w2vModel != None):
            return self.w2vModel.wv.index_to_key[id]

    def w2vModel_get_vocab(self):
        if(self.w2vModel != None):
            return self.w2vModel.wv.index_to_key

    def w2vModel_get_vocab_length(self):
        if(self.w2vModel != None):
            return len(self.w2vModel.wv.index_to_key)


# preprocessor = Preprocessor("./dataset/train_update_10t01.pkl")
# preprocessor.w2vModel_from_file("./word2vec.model");
# w2vs = preprocessor.w2vModel.wv
# w2vs = np.array(w2vs)
# avgVector = np.average(w2vs, axis = 0)
# print(avgVector)