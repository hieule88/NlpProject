import numpy as np
import pickle
import string
from pprint import pprint
from gensim.models import Word2Vec

class Preprocessor():
    def __init__(self, dataset):
        self.dataset = dataset
        self.listpunctuation = string.punctuation.replace('_','')
        tmp = []
        for i in range(0, len(self.listpunctuation)):
            tmp.append(self.listpunctuation[i])
        self.listpunctuation = tmp
        self.w2vModel = None

    def preprocess_method(self):
        pass

    def make_tag_lookup_table(self):
        ner_labels = ["PAD", "ADDRESS", "SKILL", "EMAIL", "PERSON", "PHONENUMBER", "QUANTITY", "PERSONTYPE",
                  "ORGANIZATION", "PRODUCT", "IP", "LOCATION", "O", "DATETIME", "EVENT", "URL"]
        table = {}
        i = 0
        for label in ner_labels:
            table[i] = label
            i+=1
        return table
    
    def load_input_data(self, input_path):
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        sentences = []
        for item in data:
            pre_sentence = []
            for token in item:
                if token[0] not in self.listpunctuation:
                    pre_sentence.append(token[0].lower())
            sentences.append(" ".join(pre_sentence))
        return sentences

    def w2vModelFromData(self, data):
        self.w2vModel = Word2Vec(data, window=5, min_count = 2, workers = 4, sg = 0)
        self.w2vModel.wv.save("word2vec.model")

    def similarity(self, word1, word2):
        return self.w2vModel.wv.similarity(word1, word2)

preprocessor = Preprocessor("")
mapping = preprocessor.make_tag_lookup_table()
input_path = "./dataset/test_vnc_15t02.pkl"
data = preprocessor.load_input_data(input_path=input_path)
preprocessor.w2vModelFromData(data)
print(preprocessor.similarity('nếu', 'đây'))