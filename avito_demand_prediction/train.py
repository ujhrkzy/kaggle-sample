# -*- coding: utf-8 -*-

import pandas as pd
import lightgbm
from sklearn.model_selection import GridSearchCV
from kaggle_logger import logger
from datetime import datetime
from data_loader import DataLoader

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


datetime_pattern = "%Y%m%d_%H%M"
datetime_str = datetime.now().strftime(datetime_pattern)

# cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
cat_vars = ["image_top_1",
            "region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]


def run_lightgbm(train_X, train_y, test_X):
    param_grid = {
        'max_depth': [-1, 8],
        'num_leaves': [150, 200],
        'learning_rate': [0.1],
        'n_estimators': [1500]
    }
    """
    param_grid = {
        'max_depth': [-1],
        'num_leaves': [31, 62],
        'learning_rate': [0.1],
        'n_estimators': [150]
    }
    """
    model = lightgbm.LGBMRegressor(objective="regression")
    gbm = GridSearchCV(model,
                       param_grid,
                       cv=3,
                       scoring="neg_mean_squared_error",
                       n_jobs=1)
    # TODO specify category columns
    gbm.fit(train_X, train_y)

    means = gbm.cv_results_['mean_test_score']
    stds = gbm.cv_results_['std_test_score']

    best_params_file_name_format = "best_params_{}.txt"
    best_params_file_name = best_params_file_name_format.format(datetime_str)
    with open(best_params_file_name, encoding="utf-8", mode="w") as f:
        f.write(str(gbm.best_estimator_))
        f.write("\n")
        f.write("best params: {}".format(gbm.best_params_))
    for mean, std, params in zip(means, stds, gbm.cv_results_['params']):
        logger.info("    mean:{}, std:{}, params:{}".format(mean, std, params))
    logger.info("best params: {}".format(gbm.best_params_))

    model = lightgbm.LGBMRegressor(objective="regression", **gbm.best_params_)
    model.fit(X=train_X, y=train_y)
    predicted_test_y = model.predict(test_X)
    # load
    # bst = lgb.Booster(model_file='mode.txt')bst = lgb.Booste
    return predicted_test_y


# Splitting the data for model training#
data_loader = DataLoader()
train_X, train_y = data_loader.load_train()
# validation_index = -200
# validation_index = -200000
# dev_X = train_X.iloc[:validation_index,:]
# val_X = train_X.iloc[validation_index:,:]
# dev_y = train_y[:validation_index]
# val_y = train_y[validation_index:]

test_X, test_id = data_loader.load_test()

# Training the model #
predicted_test = run_lightgbm(train_X, train_y, test_X)

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
