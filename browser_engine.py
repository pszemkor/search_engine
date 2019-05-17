from scipy.spatial import distance
import numpy as np
import nltk
from scipy.sparse import *
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os


class Engine:
    def __init__(self, data_directory):
        self.directory = data_directory

    # todo -> slownik, ktory przechowa w ilu dokumentach wystepuje slowo
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
        bag_of_words = self.get_bag_of_words()
        sorted_words = list(map(lambda tuple: tuple[0], sorted(bag_of_words.items(), key=lambda t: t[1])))
        # print(sorted_words)
        first_k_words = sorted_words[::-1][:k]
        return first_k_words

    def compute_IDF(self, word, word_freq_in_doc, documents_count):
        import math
        return math.log10(documents_count / word_freq_in_doc.get(word, 1))

    def create_sparse_matrix(self, k):
        reduced_list_of_letters = self.get_k_most_common(k)
        # print(reduced_list_of_letters)
        document_dicts, word_freq_in_doc = self.get_files_dicts()
        index_document_dict= dict()
        word_index_dict = dict()
        documents_count = len(document_dicts.keys())
        reduced_words_count = len(reduced_list_of_letters)
        data_matrix = csc_matrix((reduced_words_count, documents_count)).toarray()
        # print(documents_count)
        # print(reduced_words_count)
        # print(data_matrix.shape)
        i = 0
        for w in reduced_list_of_letters:
            word_index_dict[w] = i
            i += 1

        i = 0
        print(reduced_list_of_letters)
        for d in document_dicts.keys():
            d_dict = document_dicts[d]
            for w in reduced_list_of_letters:
                freq = d_dict.get(w, 0)
                # print(freq)
                # print(word_index_dict[w])
                # print(i)
                if freq > 0:
                    data_matrix[word_index_dict[w]][i] = freq * self.compute_IDF(w, word_freq_in_doc, documents_count)
            index_document_dict[i] = d
            i += 1
        # print("elo")
        # print(data_matrix)
        return data_matrix, index_document_dict, word_index_dict

    def get_L2_norm(self, vector):
        import math
        norm = 0
        for i in range(len(vector)):
            norm += vector[i] ** 2
        return math.sqrt(norm)

    def normalize_vector(self, vector):
        norm = self.get_L2_norm(vector)
        if norm == 0:
            raise Exception("result not found")
        vector = vector / norm
        return vector

    def create_query_vector(self, query, word_index_dict):
        ps = PorterStemmer()
        query = word_tokenize(query)
        # print(query)
        query_vector = csc_matrix((len(word_index_dict.keys()), 1)).toarray()

        for word in query:
            word = ps.stem(word).lower()
            # print(word)
            if word in word_index_dict.keys():
                query_vector[word_index_dict[word]] = 1
        # print(query_vector)
        return query_vector

    def get_cosine_similarity_vector(self, query, matrix):
        return query.transpose() @ matrix

    def get_n_best_indices(self, n, similarity_vec):
        corr_dict = dict()
        for i in range(len(similarity_vec)):
            val = similarity_vec[0][i]
            corr_dict[val] = i
        best = sorted(corr_dict.items(), key=lambda tuple: tuple[0])
        best = list(map(lambda tuple: tuple[1] ,best))
        return best[::-1][:n]

    def get_n_best_articles(self, query, n, k):
        data_matrix, index_document_dict, word_index_dict = engine.create_sparse_matrix(k)
        query_vec = self.create_query_vector(query, word_index_dict)

        # normalization
        query_vec = self.normalize_vector(query_vec)
        for i in range(data_matrix.shape[1]):
            data_matrix[:, i] = self.normalize_vector(data_matrix[:, i])

        similarity_vec = self.get_cosine_similarity_vector(query_vec, data_matrix)
        best_doc_indices = self.get_n_best_indices(n, similarity_vec)
        for i in best_doc_indices:
            print(index_document_dict[i])


if __name__ == "__main__":
    engine = Engine("test_data")

    # query:
    q = "movies lol bla bla xd"
    engine.get_n_best_articles(q, 2, 10)
