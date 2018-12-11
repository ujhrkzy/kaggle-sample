# -*- coding: utf-8 -*-
import os

import pandas as pd

from kaggle_logger import logger

__author__ = "ujihirokazuya"

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


class ImageSummary(object):

    data_directory = "/home/src/image_features"
    # data_directory = "/Users/ujihirokazuya/unirobot/kaggle/kaggle-master/avito_demand_prediction/image_features"
    # file_name = "preprocessed_image_train_{}.csv"
    file_name = "preprocessed_image_train_{}.csv"
    file_name_2 = "preprocessed_image_train_x_{}.csv"
    file_count = 508 + 1

    def __init__(self):
        pass

    def execute(self):
        summary_df = None
        for i in range(self.file_count):
            file_name = self.file_name.format(i)
            df = pd.read_csv(os.path.join(self.data_directory, file_name))
            if summary_df is None:
                summary_df = df
                continue
            summary_df = pd.concat([summary_df, df])
        file_name = self.file_name.format("all")
        summary_df.to_csv(file_name, index=False)

    def execute_2(self):
        file_name = self.file_name.format("all")
        df = pd.read_csv(file_name)
        file_name = self.file_name_2.format("0")
        df2 = pd.read_csv(file_name)
        summary_df = pd.concat([df, df2])
        summary_df.to_csv("preprocessed_image_train_all_1.csv", index=False)


if __name__ == '__main__':
    logger.info("start>>>>>>>>>>>>> >>>>>>>>")
    try:
        summary = ImageSummary()
        # summary.execute()
        summary.execute_2()
    except Exception as e:
        logger.error("Unexpected error has occurred.", exc_info=e)
    logger.info("end>>>>>>>>>>>>>>>>>>>>>")
