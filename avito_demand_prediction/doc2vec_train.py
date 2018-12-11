# -*- coding: utf-8 -*-
from typing import List
import collections
from gensim import models
from gensim.models.doc2vec import TaggedDocument
import csv
import random
from kaggle_logger import logger
from datetime import datetime

__author__ = "ujihirokazuya"


class RowData(object):

    def __init__(self, row: list):
        self.id = row[0]
        self.words = row[1].split(" ")


class Description2Vec(object):

    # _data_file = "top1000_nlp_all.csv"
    # _data_file = "preprocessed_nlp_all.csv"
    _data_file = "preprocessed_nlp_lemma_stop_all.csv"
    _passing_precision = 95

    def __init__(self):
        pass

    def _read_line(self) -> List[RowData]:
        with open(self._data_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                row_data = RowData(row)
                yield row_data

    def _corpus_to_sentences(self):
        sentences = list()
        for row_data in self._read_line():
            name = row_data.id
            words = row_data.words
            # yield TaggedDocument(words=words, tags=[name])
            sentence = TaggedDocument(words=words, tags=[name])
            sentences.append(sentence)
        return sentences

    def train(self):
        sentences = self._corpus_to_sentences()
        sentence_ids = range(len(sentences))
        # model = models.Doc2Vec(dm=1, size=400, window=15, alpha=0.0015, sample=1e-4, min_count=1, workers=4)

        # memory error
        # model = models.Doc2Vec(dm=1, size=3000, window=15, alpha=0.0015, sample=1e-4, min_count=2, workers=5)
        # model = models.Doc2Vec(dm=1, size=1800, window=15, alpha=0.0015, sample=1e-4, min_count=3, workers=5)
        # model = models.Doc2Vec(dm=1, size=80, window=15, alpha=0.0015, sample=1e-5, min_count=3, workers=5)
        model = models.Doc2Vec(dm=1, size=30, window=15, alpha=0.0015, sample=1e-4, min_count=3, workers=5)

        # model = models.Doc2Vec(dm=1, size=400, window=15, alpha=0.0015, sample=1e-4, min_count=2, workers=4)
        # model = models.Doc2Vec(dm=1, size=400, window=15, alpha=0.0015, sample=1e-2, min_count=2, workers=4)
        # model = models.Doc2Vec(dm=1, size=400, window=15, alpha=0.0015, sample=1e-5, min_count=2, workers=4)
        # model = models.Doc2Vec(dm=1, size=400, window=15, alpha=0.0015, sample=1e-5, min_count=1, workers=4)
        # model = models.Doc2Vec(dm=1, size=400, window=15, alpha=0.0015, sample=1e-5, min_count=3, workers=4)
        # def infer_vector(self, doc_words, alpha=0.1, min_alpha=0.0001, steps=5):
        # model = models.Doc2Vec(dm=0, size=300, window=15, alpha=.025, min_alpha=.025, min_count=1, sample=1e-6)
        model.build_vocab(sentences)
        self._inner_train(model, sentences, sentence_ids)

    def _inner_train(self, model: models.Doc2Vec, sentences: List[TaggedDocument], sentence_ids: List[str]):
        for x in range(250):
        # for x in range(60):
            logger.info("epoch:{}".format(x))
            model.train(sentences, total_examples=model.corpus_count, epochs=1)
            if x % 50 != 0:
            # if x % 1 != 0:
                continue
            ranks = []
            for doc_id in random.sample(sentence_ids, 100):
                inferred_vector = model.infer_vector(sentences[doc_id].words)
                sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
                rank = [doc_id_candidate for doc_id_candidate, sim in sims].index(sentences[doc_id].tags[0])
                ranks.append(rank)
            counter = collections.Counter(ranks)
            result = "epoch:{}, rank value:{}, counter:{}".format(x, counter[0], counter)
            logger.info(result)
            if counter[0] >= self._passing_precision:
                break
        datetime_pattern = "%Y%m%d%H%M"
        datetime_str = datetime.now().strftime(datetime_pattern)
        file_name_format = "doc2vec_{}.model"
        file_name = file_name_format.format(datetime_str)
        model.save(file_name)
        return model

    def train_with_model(self):
        sentences = self._corpus_to_sentences()
        sentence_ids = range(len(sentences))
        # model = models.Doc2Vec.load('doc2vec_20180617.model')
        # model = models.Doc2Vec.load('doc2vec_201806180808.model')
        # model = models.Doc2Vec.load('doc2vec_201806190049.model')
        # model = models.Doc2Vec.load('doc2vec_201806191613.model')
        # model = models.Doc2Vec.load('doc2vec_201806261000.model')
        # model = models.Doc2Vec.load('doc2vec_201806261548.model')
        model = models.Doc2Vec.load('doc2vec_201806261850.model')
        # model.build_vocab(sentences)
        self._inner_train(model, sentences, sentence_ids)

    def test_train(self):
        model = models.Doc2Vec.load('doc2vec.model')
        sentences = self._corpus_to_sentences()
        sentence_ids = range(len(sentences))
        ranks = []
        for doc_id in random.sample(sentence_ids, 100):
            inferred_vector = model.infer_vector(sentences[doc_id].words)
            sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
            rank = [doc_id_candidate for doc_id_candidate, sim in sims].index(sentences[doc_id].tags[0])
            ranks.append(rank)
        counter = collections.Counter(ranks)
        logger.info(counter)
        logger.info("rank value:{}".format(counter[0]))


if __name__ == '__main__':
    logger.info("start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    executor = Description2Vec()
    try:
        executor.train()
        # executor.train_with_model()
    except Exception as e:
        logger.error("Unexpected error has occurred.", exc_info=e)
    logger.info("end>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
