# -*- coding: utf-8 -*-

import os

import pandas as pd

from kaggle_logger import logger

__author__ = "ujihirokazuya"

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


class DataSummarizer(object):

    data_directory = "/home/src"
    # data_directory = "/Users/ujihirokazuya/unirobot/kaggle/kaggle-master/avito_demand_prediction/image_features"
    # file_name = "preprocessed_image_train_{}.csv"
    mode = "train"
    # mode = "train"
    feature_file_name = "preprocessed_{}.csv".format(mode)
    image_file_name = "preprocessed_image_{}_all.csv".format(mode)
    summary_with_image_file = "preprocessed_{}_with_image.csv".format(mode)
    summary_with_text_file = "preprocessed_{}_with_text.csv".format(mode)
    summary_with_image_and_text_file = "preprocessed_{}_with_text.csv".format(mode)
    text_file_name = "text_vectors_{}.csv".format(mode)

    def __init__(self):
        pass

    def summarize_image(self):
        feature_df = pd.read_csv(os.path.join(self.data_directory, self.feature_file_name))
        image_df = pd.read_csv(os.path.join(self.data_directory, self.image_file_name))
        image_df["image_im"] = image_df["image"]
        image_df = image_df.drop(["image"], axis=1)
        image_df["abc"] = "abc"
        summary_df = pd.merge(feature_df, image_df, on="item_id", how="left")
        if len(summary_df[summary_df["abc"] != "abc"]["item_id"].values) != 0:
            import pdb; pdb.set_trace()
            raise ValueError()
        output_df = summary_df.drop(["abc", "image_im"], axis=1)
        output_df.fillna(-1, inplace=True)
        output_df.to_csv(self.summary_with_image_file, index=False)
        summary_df = summary_df.fillna(0)
        if len(summary_df[summary_df["image"] != summary_df["image_im"]]["item_id"].values) != 0:
            import pdb; pdb.set_trace()
            raise ValueError()
        logger.info("end")

    def summarize_text(self):
        feature_df = pd.read_csv(os.path.join(self.data_directory, self.summary_with_image_file))
        text_df = pd.read_csv(os.path.join(self.data_directory, self.text_file_name))
        text_df["abc"] = "abc"
        summary_df = pd.merge(feature_df, text_df, on="item_id", how="left")
        if len(summary_df[summary_df["abc"] != "abc"]["item_id"].values) != 0:
            import pdb; pdb.set_trace()
            raise ValueError()
        summary_df = summary_df.drop(["abc"], axis=1)
        summary_df.fillna(-1, inplace=True)
        summary_df.to_csv(self.summary_with_image_and_text_file, index=False)
        logger.info("end")


if __name__ == '__main__':
    logger.info("start>>>>>>>>>>>>>>>>>>>>>")
    try:
        summarizer = DataSummarizer()
        # summarizer.summarize_image()
        summarizer.summarize_text()
    except Exception as e:
        logger.error("Unexpected error has occurred.", exc_info=e)
    logger.info("end>>>>>>>>>>>>>>>>>>>>>")
