# -*- coding: utf-8 -*-
import csv
from itertools import chain
from typing import List

import pandas as pd
from gensim import models

from kaggle_logger import logger

__author__ = "ujihirokazuya"


class RowData(object):

    def __init__(self, row: list):
        self.id = row[0]
        self.words = row[1].split(" ")


class VectorGenerator(object):

    _doc2vec_model_path = "dim_90_doc2vec/doc2vec_201806271212.model"
    mode = "train"
    summary_with_image_file = "preprocessed_{}_with_image.csv".format(mode)
    summary_with_image_and_text_file = "preprocessed_{}_with_image_and_text.csv".format(mode)
    text_vectors_file = "text_vectors_train.csv"

    def __init__(self):
        self.model = models.Doc2Vec.load(self._doc2vec_model_path)

    def _read_line(self) -> List[RowData]:
        with open(self._data_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                row_data = RowData(row)
                yield row_data

    def _get_vectors(self, doc_id):
        offset = self.model.docvecs.doctags.get(doc_id).offset
        result = self.model.docvecs.vectors_docs[offset]
        return result

    def execute(self):
        df = pd.read_csv(self.summary_with_image_file)
        item_ids = df["item_id"].values
        with open(self.text_vectors_file, mode='w', encoding='utf-8') as f:
            for item_id in item_ids:
                values = [item_id]
                text_vector = self._get_vectors(item_id)
                values.extend(text_vector.tolist())
                values = [str(v) for v in values]
                f.write(",".join(values))
                f.write("\n")
            headers = ["item_id"]
            for i in range(len(values) - 1):
                headers.append("v_{}".format(i))
            f.write(",".join(headers))


if __name__ == '__main__':
    logger.info("start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    executor = VectorGenerator()
    try:
        executor.execute()
    except Exception as e:
        logger.error("Unexpected error has occurred.", exc_info=e)
    logger.info("end>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
