# -*- coding: utf-8 -*-

import pandas as pd
import gc
from datetime import datetime

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

data_directory = "/home/res/"

# _date_columns = ["activation_date", "date_from", "date_to"]
_date_columns = ["date_from", "date_to"]


def _datetime_converter(value: str):
    if not isinstance(value, str):
        return datetime.now()
    return datetime.strptime(value, "%Y-%m-%d")


def _calc_days(value: str):
    if not isinstance(value, str):
        return -1
    values = value.split(" ")
    t1 = _datetime_converter(values[0])
    t2 = _datetime_converter(values[1])
    return (t1 - t2).days


def _convert(data_frame):
    data_frame["period_from_act"] = data_frame["date_from"] + " " + data_frame["activation_date"]
    data_frame["period_from_act"] = data_frame["period_from_act"].apply(_calc_days)
    data_frame["period_to_act"] = data_frame["date_to"] + " " + data_frame["activation_date"]
    data_frame["period_to_act"] = data_frame["period_to_act"].apply(_calc_days)
    data_frame["period_to_from"] = data_frame["date_to"] + " " + data_frame["date_from"]
    data_frame["period_to_from"] = data_frame["period_to_from"].apply(_calc_days)
    return data_frame.drop(_date_columns, axis=1)


# train_df = pd.read_csv(data_directory + "periods_train.csv", parse_dates=_date_columns)
train_df = pd.read_csv(data_directory + "periods_train.csv")
train_df = _convert(train_df)
train_df.to_csv("preprocessed_period_train.csv", index=False)
del train_df
gc.collect()

# test_df = pd.read_csv(data_directory + "periods_test.csv", parse_dates=_date_columns)
test_df = pd.read_csv(data_directory + "periods_test.csv")
test_df = _convert(test_df)
test_df.to_csv("preprocessed_period_test.csv", index=False)
