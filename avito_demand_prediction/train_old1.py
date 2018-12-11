# -*- coding: utf-8 -*-

import pandas as pd
import lightgbm as lgb

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# validation_index = -200
# train_df = pd.read_csv("preprocessed_top1000_train.csv")
# test_df = pd.read_csv("preprocessed_top1000_test.csv")
validation_index = -200000
train_df = pd.read_csv("preprocessed_train.csv")
test_df = pd.read_csv("preprocessed_test.csv")
print("Train file rows and columns are : ", train_df.shape)
print("Test file rows and columns are : ", test_df.shape)

train_y = train_df["deal_probability"].values
test_id = test_df["item_id"].values

cols_to_drop = ["item_id", "image"]
train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 30,
        "learning_rate": 0.1,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=20,
                      evals_result=evals_result)

    predicted_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return predicted_test_y, model, evals_result


# Splitting the data for model training#
dev_X = train_X.iloc[:validation_index,:]
val_X = train_X.iloc[validation_index:,:]
dev_y = train_y[:validation_index]
val_y = train_y[validation_index:]
print(dev_X.shape, val_X.shape, test_X.shape)

# Training the model #
# predicted_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
predicted_test, model, evals_result = run_lgb(train_X, train_y, val_X, val_y, test_X)

# Making a submission file #
predicted_test[predicted_test > 1] = 1
predicted_test[predicted_test < 0] = 0
sub_df = pd.DataFrame({"item_id": test_id})
sub_df["deal_probability"] = predicted_test
sub_df.to_csv("baseline_lgb.csv", index=False)
