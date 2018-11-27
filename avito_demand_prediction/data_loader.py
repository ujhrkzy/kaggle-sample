# -*- coding: utf-8 -*-

import pandas as pd
import lightgbm
from sklearn.model_selection import GridSearchCV
from kaggle_logger import logger
from datetime import datetime

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# validation_index = -200
# train_df = pd.read_csv("preprocessed_top1000_train.csv")
# test_df = pd.read_csv("preprocessed_top1000_test.csv")
validation_index = -200000
train_df = pd.read_csv("preprocessed_train.csv")
test_df = pd.read_csv("preprocessed_test.csv")
logger.info("Train file rows and columns are : {}".format(train_df.shape))
logger.info("Test file rows and columns are : {}".format(test_df.shape))

datetime_pattern = "%Y%m%d_%H%M"
datetime_str = datetime.now().strftime(datetime_pattern)

train_y = train_df["deal_probability"].values
test_id = test_df["item_id"].values

cols_to_drop = ["user_id", "item_id", "image"]
train_X = train_df.drop(cols_to_drop + ["deal_probability"], axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

# cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
cat_vars = ["image_top_1",
            "region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]


def run_lightgbm(train_X, train_y, val_X, val_y, test_X):
    param_grid = {
        'max_depth': [-1, 8, 16],
        'num_leaves': [31, 62, 93, 150],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [150, 1000, 1500]
    }
    model = lightgbm.LGBMRegressor(objective="regression")
    gbm = GridSearchCV(model,
                       param_grid,
                       cv=5,
                       scoring="neg_mean_squared_error",
                       n_jobs=1)
    fit_params = {'early_stopping_rounds': 50,
                  'eval_metric': 'rmse',
                  'eval_set': [(val_X, val_y)]
                  }
    """
    gbm.fit(train_X,
            train_y,
            **fit_params)
    """
    # TODO specify category columns
    # gbm.fit(train_X, train_y, categorical_feature=cat_vars)
    gbm.fit(train_X, train_y)
    print('Best parameters found by grid search are:', gbm.best_params_)
    best_model = gbm.best_estimator_
    predicted_test_y = best_model.predict(test_X)
    # save
    means = gbm.cv_results_['mean_test_score']
    stds = gbm.cv_results_['std_test_score']

    best_params_file_name_format = "best_params_{}.txt"
    best_params_file_name = best_params_file_name_format.format(datetime_str)
    with open(best_params_file_name, encoding="utf-8", mode="w") as f:
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

file_name_format = "baseline_lightgbm_{}.csv"
file_name = file_name_format.format(datetime_str)
sub_df.to_csv(file_name, index=False)
logger.info("end")

# TODO
# see discussion
# see kernel
# see パラメーターチューニング
# アンサンブル
# 画像処理
# description, title, param1, param2, param3解析
