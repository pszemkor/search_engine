import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import *
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os
import pickle
import PySimpleGUI as sg
import tkinter


class Engine:
    def __init__(self, data_directory, k):
        self.documents_count = 0
        self.directory = data_directory
        self.bag_of_words = {}
        self.reduced_list_of_words = []
        self.data_matrix = None
        self.index_document_dict = {}
        self.word_index_dict = {}
        self.used_common_words = k
        # _______________
        self.data_matrix_SVD = None
        self.svd_components = None

    # read/write dictionaries
    # ***********************************************************
    def save_data(self, directory):
        keys = list(Engine.__dict__.keys())
        for key in keys:
            with open(directory + '/' + key + '.pkl', 'wb') as f:
                pickle.dump(getattr(self, key), f, pickle.HIGHEST_PROTOCOL)

    def load_data(self, directory):
        keys = Engine.__dict__.keys()
        dir_content = [w for w in keys if w + '.pkl' in os.listdir(directory)]
        for key in dir_content:
            print(key)
            if key == "__weakref__":
                continue
            with open(directory + '/' + key + '.pkl', 'rb') as f:
                setattr(self, key, pickle.load(f))

    # ***********************************************************

    def get_files_dicts(self):
        # dictionary filename -> (dictionary: word -> freq)
        bag = dict()
        # dictionary: word -> amount of docs in which word was
        word_document = dict()
        for filename in os.listdir(self.directory):
            f = open(self.directory + "/" + filename)
            words = f.read()
            bag[filename] = self.get_word_frequency_for_file(words)
            for k in bag.keys():
                word_document[k] = word_document.get(k, 0) + 1
        return bag, word_document

    def get_word_frequency_for_file(self, words):
        word_freq = dict()
        words = ''.join(ch for ch in words if ch not in set(string.punctuation))
        # print(words)
        words = word_tokenize(words)
        ps = PorterStemmer()
        for w in words:
            w = w.lower()
            if w in stopwords.words('english'):
                continue
            stemmed = ps.stem(w)
            word_freq[stemmed] = word_freq.get(stemmed, 0) + 1
        return word_freq

    # union of all dictionaries with freq of words
    def get_bag_of_words(self):
        files_dicts, _ = self.get_files_dicts()
        word_freq_dicts = files_dicts.values()
        bag_of_words = dict()
        for dictionary in word_freq_dicts:
            for k in dictionary.keys():
                bag_of_words[k] = bag_of_words.get(k, 0) + dictionary[k]
        return bag_of_words

    def get_k_most_common(self, k):
        self.bag_of_words = self.get_bag_of_words()
        sorted_words = list(map(lambda tuple: tuple[0], sorted(self.bag_of_words.items(), key=lambda t: t[1])))
        # print(sorted_words)
        first_k_words = sorted_words[::-1][:k]
        return first_k_words

    def compute_IDF(self, word, word_freq_in_doc, documents_count):
        import math
        return math.log10(documents_count / word_freq_in_doc.get(word, 1))

    def create_sparse_matrix(self):
        self.reduced_list_of_words = self.get_k_most_common(self.used_common_words)
        # print(self.reduced_list_of_words)
        document_dicts, word_freq_in_doc = self.get_files_dicts()
        index_document_dict = dict()
        word_index_dict = dict()
        documents_count = len(document_dicts.keys())
        reduced_words_count = len(self.reduced_list_of_words)
        data_matrix = lil_matrix((reduced_words_count, documents_count))
        i = 0
        for w in self.reduced_list_of_words:
            word_index_dict[w] = i
            i += 1

        i = 0
        print(self.reduced_list_of_words)
        for d in document_dicts.keys():
            self.documents_count += 1
            d_dict = document_dicts[d]
            for w in self.reduced_list_of_words:
                freq = d_dict.get(w, 0)
                # print(freq)
                # print(word_index_dict[w])
                # print(i)
                if freq > 0:
                    data_matrix[word_index_dict[w], i] = freq * self.compute_IDF(w, word_freq_in_doc, documents_count)
            index_document_dict[i] = d
            i += 1
        # print("elo")
        # print(data_matrix)
        data_matrix = csr_matrix(data_matrix)
        data_matrix = normalize(data_matrix, norm='l2', axis=0)
        self.data_matrix = data_matrix
        self.index_document_dict = index_document_dict
        self.word_index_dict = word_index_dict

    def create_query_vector(self, query, word_index_dict):
        ps = PorterStemmer()
        query = word_tokenize(query)
        # print(query)
        query_vector = lil_matrix((len(word_index_dict.keys()), 1), dtype=np.double)

        for word in query:
            word = ps.stem(word).lower()
            # print(word)
            if word in word_index_dict.keys():
                query_vector[word_index_dict[word]] = 1
        # print(query_vector)
        query_vector = csr_matrix(query_vector)
        query_vector = normalize(query_vector)
        return query_vector

    def get_cosine_similarity_vector(self, query):
        return query.T.dot(self.data_matrix)

    def get_n_best_articles(self, query, n):
        query_vec = self.create_query_vector(query, self.word_index_dict)
        correlation = self.get_cosine_similarity_vector(query_vec).toarray()
        corr_by_id = sorted([(i, correlation[0, i]) for i in range(self.documents_count)], key=(lambda x: x[1]),
                            reverse=True)[:n]
        result = ''
        for i, el in enumerate(corr_by_id):
            # print(i + 1, ':', self.index_document_dict[el[0]], ' (correlation: ', round(el[1] * 100, 2), ')')
            result += str(i + 1) + ':' + str(self.index_document_dict[el[0]]) + ' (correlation: ' + str(
                round(el[1] * 100, 2)) + ')'
            result += '\n'
        return result

    # #######################################################################

    def create_svd_matrix(self):
        svd = TruncatedSVD(n_components=700)
        self.data_matrix_SVD = svd.fit_transform(self.data_matrix)
        self.svd_components = svd.components_
        print(self.data_matrix_SVD.shape)
        print(self.data_matrix.shape)

    def get_cosine_similarity_svd(self, query_vec):
        return query_vec.T.dot(self.data_matrix_SVD).dot(self.svd_components)

    def get_n_best_articles_svd(self, query, n):
        query_vec = self.create_query_vector(query, self.word_index_dict)
        correlation = self.get_cosine_similarity_svd(query_vec)
        corr_by_id = sorted([(i, correlation[0, i]) for i in range(self.documents_count)], key=(lambda x: x[1]),
                            reverse=True)[:n]
        result = ""
        for i, el in enumerate(corr_by_id):
            # print(i + 1, ':', self.index_document_dict[el[0]], ' (correlation: ', round(el[1] * 100, 2), ')')
            result += str(i + 1) + ':' + str(self.index_document_dict[el[0]]) + ' (correlation: ' + str(
                round(el[1] * 100, 2)) + ')'
            result += '\n'
        return result


if __name__ == "__main__":
    engine = Engine("data", 10000)
    # engine.create_sparse_matrix()
    engine.load_data('obj')
    # engine.create_svd_matrix()
    # engine.save_data('obj')
    # engine.load_data('obj')
    while True:
        # query = input("your query:\n > ")
        # query = sg.PopupGetText('Type in your query:', 'Gogle')

        layout = [[sg.Text('Please enter tour query')],
                  [sg.Input(do_not_clear=True)],
                  [sg.Button('Search'), sg.Exit()]]
        window = sg.Window('Search Engine', layout)

        event, query = window.Read()
        if event is None or event == 'Exit':
            break

        query = str(query)
        print(query[2:-2])
        query = query[2:-2]
        res = engine.get_n_best_articles(query, 20)
        sg.Popup("Your results: \n ", res)

