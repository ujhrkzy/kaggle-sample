# -*- coding: utf-8 -*-

import pandas as pd
import re

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

data_directory = "/home/res/"

train_df = pd.read_csv(data_directory + "train.csv")
test_df = pd.read_csv(data_directory + "test.csv")
# train_df = pd.read_csv("top1000_train.csv")
# test_df = pd.read_csv("top1000_test.csv")


train_df = train_df.drop("deal_probability", axis=1)

valid_columns = ["item_id", "title", "description"]
invalid_columns = [name for name in test_df.columns.values if name not in valid_columns]
train_df = train_df.drop(invalid_columns, axis=1)
test_df = test_df.drop(invalid_columns, axis=1)


def _concat(data_frame):
    data_frame["title_description"] = data_frame["title"] + " title_end " + data_frame["description"].astype(str)
    return data_frame.drop(["title", "description"], axis=1)


train_df = _concat(train_df)
test_df = _concat(test_df)


def _replace_word(word: str):
    word = word.replace(":", " ")
    word = word.replace(";", " ")
    word = word.replace(",", " ")
    word = word.replace("/", " ")
    word = word.replace("!", " ")
    word = word.replace("?", " ")
    word = word.replace('"', " ")
    word = word.replace("'", " ")
    word = word.replace("-", " ")
    word = word.replace("+", " ")
    word = word.replace("\n", " ")
    word = word.replace("\r\n", " ")
    word = word.replace("(", " ")
    word = word.replace(")", " ")
    word = word.replace("[", " ")
    word = word.replace("]", " ")
    word = word.replace("=", " ")
    word = word.replace(".", " ")
    word = re.sub("[0-9]+", " ", word)
    word = re.sub(" +", " ", word)
    word = word.lower()
    return word


train_df["title_description"] = train_df["title_description"].map(_replace_word)
test_df["title_description"] = test_df["title_description"].map(_replace_word)

train_df.to_csv("preprocessed_nlp_train.csv", index=False)
test_df.to_csv("preprocessed_nlp_test.csv", index=False)
