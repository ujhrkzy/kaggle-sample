# -*- coding: utf-8 -*-

import os
from datetime import datetime
import gc

import numpy as np
import pandas as pd

from kaggle_logger import logger

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


class DataSummarizer(object):

    data_directory = "/home/src/result"
    # data_directory = "/Users/ujihirokazuya/unirobot/kaggle/kaggle-master/avito_demand_prediction/results"
    result_file_names = ["lgb_20180626_1843.csv",
                         "lgb_20180625_2238.csv",
                         "lgb_20180627_2022.csv",
                         "lgb_20180627_2024.csv"]

    def __init__(self):
        pass

    def execute(self):
        summary_df = None
        for result_file_name in self.result_file_names:
            result_df = pd.read_csv(os.path.join(self.data_directory, result_file_name))
            if summary_df is None:
                summary_df = result_df
                continue
            summary_df = pd.merge(summary_df, result_df, on="item_id", how="left")

        del result_df
        gc.collect()

        probability_columns = list()
        for c in summary_df.columns:
            if c == "item_id" or c == "mean":
                continue
            probability_columns.append(c)

        datetime_pattern = "%Y%m%d_%H%M"
        datetime_str = datetime.now().strftime(datetime_pattern)
        with open("mean_{}.csv".format(datetime_str), mode="w") as f:
            headers = ["item_id", "deal_probability"]
            f.write(",".join(headers))
            f.write("\n")
            for _, row in summary_df.iterrows():
                values = list()
                for c in probability_columns:
                    values.append(row[c])
                mean = np.array(values).mean()
                items = [row["item_id"], str(mean)]
                f.write(",".join(items))
                f.write("\n")


if __name__ == '__main__':
    logger.info("start>>>>>>>>>>>>>>>>>>>>>")
    try:
        summarizer = DataSummarizer()
        summarizer.execute()
    except Exception as e:
        logger.error("Unexpected error has occurred.", exc_info=e)
    logger.info("end>>>>>>>>>>>>>>>>>>>>>")

