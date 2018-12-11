# -*- coding: utf-8 -*-

import pandas as pd

from kaggle_logger import logger
import numpy as np

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

cols_to_drop = ["user_id", "item_id", "image"]


class DataLoader(object):

    def __init__(self):
        # self._train_file_name = "preprocessed_top1000_train.csv"
        # self._test_file_name = "preprocessed_top1000_test.csv"
        # self._train_file_name = "preprocessed_train.csv"
        # self._test_file_name = "preprocessed_test.csv"
        self._train_file_name = "preprocessed_train_with_image.csv"
        self._test_file_name = "preprocessed_test_with_image.csv"
        # self._train_file_name = "preprocessed_train_with_image_and_text.csv"
        # self._test_file_name = "preprocessed_test_with_image_and_text.csv"

    def load_train(self):
        train_df = pd.read_csv(self._train_file_name)
        logger.info("Train file rows and columns are : {}".format(train_df.shape))
        train_y = train_df["deal_probability"].values
        train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)
        train_X = self._convert(train_X)
        train_X = train_X.fillna(-1)
        return train_X, train_y

    def load_test(self):
        test_df = pd.read_csv(self._test_file_name)
        logger.info("Test file rows and columns are : {}".format(test_df.shape))
        test_id = test_df["item_id"].values
        test_X = test_df.drop(cols_to_drop, axis=1)
        test_X = self._convert(test_X)
        test_X = test_X.fillna(-1)
        return test_X, test_id

    @staticmethod
    def _convert(df):
        columns = "dullness, whiteness, average_pixel_width, dominant_red, dominant_green, dominant_blue, average_red, average_green, average_blue, image_size, width, height, blurriness"
        columns = columns.split(",")
        columns = [c.strip() for c in columns]
        for c in columns:
            df[c] = np.log(df[c] + 0.001)
        return df


if __name__ == '__main__':
    DataLoader().load_test()
