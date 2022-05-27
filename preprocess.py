import numpy as np
import pickle
import string
from pprint import pprint
from gensim.models import Word2Vec
from collections import Counter


class Preprocessor():
    def __init__(self, dataset):
        self.dataset = dataset
        self.listpunctuation = string.punctuation.replace('_', '')
        tmp = []
        for i in range(0, len(self.listpunctuation)):
            tmp.append(self.listpunctuation[i])
        self.listpunctuation = tmp

        self.dataset = {}
        self.processed_data = {}
        self.w2vModel = None

    def preprocess_method(self):
        pass

    def load_input_data(self, input_path, name):
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

    def make_tag_lookup_table(self):
        ner_labels = ["PAD", "ADDRESS", "SKILL", "EMAIL", "PERSON", "PHONENUMBER", "QUANTITY", "PERSONTYPE",
                      "ORGANIZATION", "PRODUCT", "IP", "LOCATION", "O", "DATETIME", "EVENT", "URL", "MISCELLANEOUS"]
        table = {}
        i = 0
        for label in ner_labels:
            table[label] = i
            i += 1
        self.tag_table = table
        return table

    def make_one_hot_vector_for_tag(self, name):
        self.make_tag_lookup_table()
        rs = []
        sentences = self.dataset[name]["sentences"]
        tags = self.dataset[name]["tags"]
        for i in range(0, len(sentences)):
            sentence = sentences[i]
            tag = tags[i]
            processed_sentence = []
            for k in range(0, len(sentence)):
                vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                vector[self.tag_table[tag[k]]] = 1
                # print(vector)
                vector = np.array(vector)
                # print(vector)
                pair = (sentence[k], str(vector))
                processed_sentence.append(pair)
            rs.append(processed_sentence)
        self.processed_data[name] = rs
        with open("./dataset/processed_" + name + "_data.pkl", 'wb') as f:
            pickle.dump(rs, f, protocol=pickle.HIGHEST_PROTOCOL)
        return rs

    def w2vModel_from_data(self, data):
        self.w2vModel = Word2Vec(
            sentences=data, min_count=1, vector_size=100, window=5, sg=1)
        self.w2vModel.save("word2vec.model")

    def w2vModel_from_file(self, model_path):
        self.w2vModel = Word2Vec.load(model_path)

    def w2vModel_get_vector(self, word):
        if(self.w2vModel != None):
            return self.w2vModel.wv[word]


preprocessor = Preprocessor("")
mapping = preprocessor.make_tag_lookup_table()
input_path = "./dataset/train_vnc_15t02.pkl"
model_path = "./word2vec.model"

data = preprocessor.load_input_data(input_path=input_path, name="train")
preprocessor.make_one_hot_vector_for_tag(name="train")
preprocessor.w2vModel_from_data(data["sentences"])
# preprocessor.w2vModel_from_file(model_path=model_path)
print(preprocessor.w2vModel_get_vector("vua"))
