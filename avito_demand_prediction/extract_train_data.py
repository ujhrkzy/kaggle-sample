# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

train_df = pd.read_csv("preprocessed_train.csv")
train_df = train_df[:1000]
train_df.to_csv(path_or_buf="preprocessed_top1000_train.csv", index=False)

test_df = pd.read_csv("preprocessed_test.csv")
test_df = test_df[:1000]
test_df.to_csv(path_or_buf="preprocessed_top1000_test.csv", index=False)
