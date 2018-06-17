# -*- coding: utf-8 -*-

import pandas as pd
import lightgbm
from sklearn.model_selection import GridSearchCV
from kaggle_logger import logger

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


def run_lightgbm(train_X, train_y, val_X, val_y, test_X):
    param_grid = {
        'max_depth': [-1, 8, 16],
        'num_leaves': [15, 31, 62, 93],
        'learning_rate': [0.1, 0.01, 0.03],
        'n_estimators': [50, 100, 150, 1000]
    }
    model = lightgbm.LGBMRegressor(objective="regression")
    gbm = GridSearchCV(model,
                       param_grid,
                       cv=5,
                       scoring="neg_mean_squared_error",
                       n_jobs=-1)
    fit_params = {'early_stopping_rounds': 50,
                  'eval_metric': 'rmse',
                  'eval_set': [(val_X, val_y)]
                  }
    """
    gbm.fit(train_X,
            train_y,
            **fit_params)
    """
    gbm.fit(train_X, train_y)
    print('Best parameters found by grid search are:', gbm.best_params_)
    best_model = gbm.best_estimator_
    predicted_test_y = best_model.predict(test_X)
    # save
    means = gbm.cv_results_['mean_test_score']
    stds = gbm.cv_results_['std_test_score']
    with open("best_params.txt", encoding="utf-8", mode="w") as f:
        f.write(str(gbm.best_estimator_))
        f.write("\n")
        for mean, std, params in zip(means, stds, gbm.cv_results_['params']):
            f.write("mean:{}, std:{}, params:{}".format(mean, std, params))
            f.write("\n")
    best_model.booster_.save_model("best_lgb_model.txt")
    # load
    # bst = lgb.Booster(model_file='mode.txt')bst = lgb.Booste
    return predicted_test_y


# Splitting the data for model training#
dev_X = train_X.iloc[:validation_index,:]
val_X = train_X.iloc[validation_index:,:]
dev_y = train_y[:validation_index]
val_y = train_y[validation_index:]
print(dev_X.shape, val_X.shape, test_X.shape)

# Training the model #
# predicted_test = run_lightgbm(dev_X, dev_y, val_X, val_y, test_X)
# TODO
predicted_test = run_lightgbm(train_X, train_y, val_X, val_y, test_X)

# Making a submission file #
predicted_test[predicted_test > 1] = 1
predicted_test[predicted_test < 0] = 0
sub_df = pd.DataFrame({"item_id": test_id})
sub_df["deal_probability"] = predicted_test
sub_df.to_csv("baseline_lightgbm.csv", index=False)

# TODO
# see discussion
# see kernel
# see パラメーターチューニング
# アンサンブル
# 画像処理
# description, title解析
