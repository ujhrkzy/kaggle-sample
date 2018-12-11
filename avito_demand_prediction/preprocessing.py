# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import preprocessing
import gc
import numpy as np

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

data_directory = "/home/res/"

train_df = pd.read_csv(data_directory + "train.csv", parse_dates=["activation_date"])
test_df = pd.read_csv(data_directory + "test.csv", parse_dates=["activation_date"])
# train_df = pd.read_csv("top1000_train.csv", parse_dates=["activation_date"])
# test_df = pd.read_csv("top1000_test.csv", parse_dates=["activation_date"])


class ActivationDateCount(object):

    def __init__(self):
        self.base_df = None
        self.period_df = None

    def _apply_period_from_act(self, item_id: str):
        act_date = self.base_df[self.base_df.item_id == item_id].activation_date
        row = self.period_df[self.period_df.item_id == item_id]
        if len(row.values) == 0:
            return -1
        row = row[row.activation_date == act_date]
        values = row["period_from_act"].values
        if len(values) == 0:
            return -1
        return values[0]

    def _apply_period_to_act(self, item_id: str):
        act_date = self.base_df[self.base_df.item_id == item_id].activation_date
        row = self.period_df[self.period_df.item_id == item_id]
        if len(row.values) == 0:
            return -1
        row = row[row.activation_date == act_date]
        values = row["period_to_act"].values
        if len(values) == 0:
            return -1
        return values[0]

    def _apply_period_to_from(self, item_id: str):
        act_date = self.base_df[self.base_df.item_id == item_id].activation_date
        row = self.period_df[self.period_df.item_id == item_id]
        if len(row.values) == 0:
            return -1
        row = row[row.activation_date == act_date]
        values = row["period_to_from"].values
        if len(values) == 0:
            return -1
        return values[0]

    def exec(self):
        # self.base_df["period_from_act"] = self.base_df["item_id"].map(self._apply_period_from_act)
        # self.base_df["period_to_act"] = self.base_df["item_id"].apply(self._apply_period_to_act)
        # self.base_df["period_to_from"] = self.base_df["item_id"].apply(self._apply_period_to_from)
        self.base_df = self.base_df.drop("activation_date", axis=1)


# New variable
def _sentence_length(value):
    if isinstance(value, str):
        return len(value)
    return -1


def _word_counts(value):
    if isinstance(value, str):
        return len(value.split())
    return -1


def _apply_data(data_frame):
    data_frame["title_word_length"] = data_frame["title"].apply(_sentence_length)
    data_frame["description_word_length"] = data_frame["description"].apply(_sentence_length)
    data_frame["title_word_counts"] = data_frame["title"].apply(_word_counts)
    data_frame["description_word_counts"] = data_frame["description"].apply(_word_counts)
    # df_both[txt_col + '_len'] = df_both[txt_col].str.len()
    # df_both[txt_col + '_wc'] = df_both[txt_col].str.count(' ')
    data_frame["has_image"] = data_frame["image"].values.astype("str") != "nan"
    data_frame["activation_weekday"] = data_frame["activation_date"].dt.weekday
    data_frame["activation_days_in_month"] = data_frame["activation_date"].dt.daysinmonth
    data_frame["price"] = np.log(data_frame["price"] + 0.001)
    data_frame["price"].fillna(-999, inplace=True)
    return data_frame


train_df = _apply_data(train_df)
test_df = _apply_data(test_df)

# Label encode the categorical variables
# cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
cat_vars = ["user_id", "image_top_1",
            "region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
for col in cat_vars:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

# cols_to_drop = ["user_id", "title", "description"]
cols_to_drop = ["title", "description"]
train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)
gc.collect()


# train_period_df = pd.read_csv("preprocessed_period_train.csv")
train_period_df = None
counter = ActivationDateCount()
counter.base_df = train_df
counter.period_df = train_period_df
counter.exec()
counter.base_df.to_csv("preprocessed_train.csv", index=False)

del train_df
del train_period_df
del counter
gc.collect()

# test_period_df = pd.read_csv("preprocessed_period_test.csv")
test_period_df = None
counter = ActivationDateCount()
counter.base_df = test_df
counter.period_df = test_period_df
counter.exec()
counter.base_df.to_csv("preprocessed_test.csv", index=False)
