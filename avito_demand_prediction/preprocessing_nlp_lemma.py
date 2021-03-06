# -*- coding: utf-8 -*-

import pandas as pd
import re
from pymystem3 import Mystem

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
mystem = Mystem()


def _replace_word(word: str):
    if not isinstance(word, str):
        return ""
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
    word = word.replace("_", " ")
    word = word.replace("ー", " ")
    word = word.replace("=", " ")
    word = word.replace(".", " ")
    word = re.sub("[0-9]+", " ", word)
    word = re.sub(" +", " ", word)
    words = mystem.lemmatize(word)
    return " ".join(words)


train_df["title"] = train_df["title"].map(_replace_word)
test_df["title"] = test_df["title"].map(_replace_word)
train_df["description"] = train_df["description"].map(_replace_word)
test_df["description"] = test_df["description"].map(_replace_word)


def _replace_word_2(word: str):
    word = word.replace("\n", " ")
    word = word.replace("\r\n", " ")
    word = re.sub(" +", " ", word)
    word = word.strip()
    word = word.lower()
    return word


def _concat(data_frame):
    data_frame["title_description"] = data_frame["title"] + " title_end " + data_frame["description"].astype(str)
    data_frame = data_frame.drop(["title", "description"], axis=1)
    data_frame["title_description"] = data_frame["title_description"].map(_replace_word_2)
    return data_frame


train_df = _concat(train_df)
test_df = _concat(test_df)


train_df.to_csv("preprocessed_nlp_lemma_train.csv", index=False)
test_df.to_csv("preprocessed_nlp_lemma_test.csv", index=False)
