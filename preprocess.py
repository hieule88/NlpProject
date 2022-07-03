import numpy as np
import pickle
import string
from pprint import pprint
from gensim.models import Word2Vec
from collections import Counter
from vncorenlp import VnCoreNLP
import vnlpc

class Preprocessor():
    def __init__(self, dataset):
        self.dataset = dataset
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
        self.vc = vnlpc.VNLPClient("http://localhost:39000")

        # w2v model
        self.w2vModel = None

        # GloVe model
        # words = []
        # idx = 0
        # word2idx = {}
        # vectors = []
        # with open('./GloVe/glove.6B.50d.txt', 'rb') as f:
        # # with open('./GloVe/glove.42B.300d.txt', 'rb') as f:
        #     for l in f:
        #         line = l.decode().split()
        #         word = line[0]
        #         words.append(word)
        #         word2idx[word] = idx
        #         idx += 1
        #         vect = np.array(line[1:]).astype(np.float64)
        #         vectors.append(vect)
        # self.gloveModel = {w: vectors[word2idx[w]] for w in words}

    def preprocess_method(self):
        pass

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
                pair = (sentence[k], self.w2vModel_get_vector(sentence[k]), str(vector))
                processed_sentence.append(pair)
            rs.append(processed_sentence)
        self.processed_data[name] = rs
        with open("./dataset/processed_" + name + "_data.pkl", 'wb') as f:
            pickle.dump(rs, f, protocol=pickle.HIGHEST_PROTOCOL)
        return rs

    def tokenize(self, sentence):
        return self.vc.tokenize(sentence)

    def w2vModel_from_data(self, data):
        self.w2vModel = Word2Vec(
            sentences=data, min_count=1, vector_size=100, window=5, sg=1)
        self.w2vModel.save("word2vec.model")

    def w2vModel_from_file(self, model_path):
        self.w2vModel = Word2Vec.load(model_path)

    def w2vModel_get_vector(self, word):
        if(self.w2vModel != None):
            return self.w2vModel.wv[word]

    # def gloveModel_get_vector(self, word):
    #     return self.gloveModel[word]


# preprocessor = Preprocessor("")

# # load raw data
# preprocessor.load_raw_data("./dataset/train_update_10t01.pkl","train")
# preprocessor.load_raw_data("./dataset/test_update_10t01.pkl","test")
# preprocessor.load_raw_data("./dataset/dev_update_10t01.pkl","dev")

# # build word2vec model
# preprocessor.w2vModel_from_data(preprocessor.dataset["train"]["sentences"] + preprocessor.dataset["test"]["sentences"] + preprocessor.dataset["dev"]["sentences"])

# # make on hot vector
# preprocessor.make_one_hot_vector_for_tag("train")
# preprocessor.make_one_hot_vector_for_tag("test")
# preprocessor.make_one_hot_vector_for_tag("dev")

# # load model to use
# preprocessor.w2vModel_from_file("./word2vec.model")

