# -*- coding: utf-8 -*-

from gensim.models import KeyedVectors
import csv

__author__ = "ujihirokazuya"
__date__ = "2018/09/24"


class WordEmbeddings(object):

    def __init__(self, vec_path):
        self._model = KeyedVectors.load_word2vec_format(vec_path)

    def get_similar_words(self, target_word) -> list:
        # Finding out similar words [default = top 10]
        return self._model.similar_by_word(target_word, topn=30)


if __name__ == '__main__':

    target_words = ["deep", "leap", "mind", "intelligence",
                    "AI", "ai", "neuron", "neural", "network",
                    "finance", "financial", "technology", "information", "IT",
                    "professional", "engineering", "engineer", "system", "prediction",
                    "inference", "model", "learn", "learning", "deeplearning", "machine",
                    "statistics", "gaussian", "imagination",
                    "general", "strong", "exceed", "over", "above", "beyond"]

    root_directory = "/home/res/"
    # root_directory = ""
    embeddings = WordEmbeddings(vec_path=root_directory + "wiki.en.vec")
    headers = ["word", "similar_word", "similarity"]
    with open("similar_words.csv", mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        for w in target_words:
            try:
                similar_words = embeddings.get_similar_words(w)
                for values in similar_words:
                    writer.writerow([w, values[0], str(values[1])])
            except KeyError:
                writer.writerow([w, "", "0"])
