# -*- coding: utf-8 -*-

import pickle
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, ParameterGrid

from data_loader import DataLoader
from kaggle_logger import logger

# import xgboost as xgb

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


datetime_pattern = "%Y%m%d_%H%M"
datetime_str = datetime.now().strftime(datetime_pattern)

# cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
cat_vars = ["image_top_1",
            "region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]


class TrainModel(object):

    def __init__(self):
        pass

    def exec(self, train_X, train_y, evaluation_X):
        """
        LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
       learning_rate=0.1, max_depth=8, min_child_samples=20,
       min_child_weight=0.001, min_split_gain=0.0, n_estimators=1000,
       n_jobs=-1, num_leaves=150, objective='regression',
       random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
       subsample=1.0, subsample_for_bin=200000, subsample_freq=1)
        """
        all_params = {"objective": ["regression"],
                      "metric": ["rmse"],
                      "num_iterations": [100, 1000, 1500],
                      "learning_rate": [0.1, 0.01],
                      "num_leaves": [31, 93, 150],
                      "max_depth": [-1],
                      "min_data_in_leaf": [0, 20, 100],
                      "max_bin": [63, 255, 511]
                      }
        """
        all_params = {"objective": ["regression"],
                      "metric": ["rmse"],
                      "num_iterations": [100, 1000]
                      }
        """
        min_score = 100
        min_params = None
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for params in ParameterGrid(all_params):
            logger.info('params: {}'.format(params))

            loss_function_scores = list()
            best_iterations = list()
            for train_idx, valid_idx in cv.split(train_X, train_y):
                trn_x = train_X.iloc[train_idx, :]
                val_x = train_X.iloc[valid_idx, :]
                trn_y = train_y[train_idx]
                val_y = train_y[valid_idx]

                train_set = lgb.Dataset(trn_x, label=trn_y, categorical_feature=cat_vars)
                test_set = lgb.Dataset(val_x, label=val_y, categorical_feature=cat_vars)
                model = lgb.train(params=params, train_set=train_set,
                                  valid_sets=[test_set], early_stopping_rounds=100)

                predicted_val = model.predict(val_x, num_iteration=model.best_iteration)
                rmse = np.sqrt(mean_squared_error(y_true=val_y, y_pred=predicted_val))
                loss_function_scores.append(rmse)
                best_iterations.append(model.best_iteration)
                logger.debug("   loss: {}".format(rmse))

            # params['num_iteration'] = int(np.mean(best_iterations))
            model_score = np.mean(loss_function_scores)
            if min_score > model_score:
                min_score = model_score
                min_params = params
            logger.info('current min score: {}, params: {}'.format(min_score, min_params))

        logger.info('minimum score: {}'.format(min_score))
        logger.info('minimum params: {}'.format(min_params))

        train_set = lgb.Dataset(train_X, label=train_y, categorical_feature=cat_vars)
        model = lgb.train(params=params, train_set=train_set)
        model_file_name = "model_{}.pkl".format(datetime_str)
        with open(model_file_name, 'wb') as f:
            pickle.dump(model, f, -1)
        # with open(model_file_name, 'rb') as f:
        #     model = pickle.load(f)

        predicted_test = model.predict(data=evaluation_X)
        return predicted_test


def _main():
    data_loader = DataLoader()
    train_X, train_y = data_loader.load_train()
    test_X, test_id = data_loader.load_test()

    train_model = TrainModel()
    predicted_test = train_model.exec(train_X, train_y, test_X)

    # Making a submission file #
    predicted_test[predicted_test > 1] = 1
    predicted_test[predicted_test < 0] = 0
    sub_df = pd.DataFrame({"item_id": test_id})
    sub_df["deal_probability"] = predicted_test

    file_name_format = "lgb_{}.csv"
    file_name = file_name_format.format(datetime_str)
    sub_df.to_csv(file_name, index=False)
    logger.info("end")


if __name__ == '__main__':
    logger.info("start>>>>>>>>>>>>>>>>>>>>>")
    try:
        _main()
    except Exception as e:
        logger.error("Unexpected error has occurred.", exc_info=e)
    logger.info("end>>>>>>>>>>>>>>>>>>>>>")
