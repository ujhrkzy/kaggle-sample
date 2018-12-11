# -*- coding: utf-8 -*-

import pandas as pd

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

result_file = "baseline_lightgbm_20180618_01.csv"
result_df = pd.read_csv(result_file)

result_df["deal_probability"] = result_df["deal_probability"].apply(lambda x: 0.0 if x < 0.02 else x)
result_df.to_csv("baseline_lightgbm_20180618_01_under_002.csv", index=False)


result_df["deal_probability"] = result_df["deal_probability"].apply(lambda x: 0.0 if x < 0.04 else x)
result_df.to_csv("baseline_lightgbm_20180618_01_under_004.csv", index=False)

result_df["deal_probability"] = result_df["deal_probability"].apply(lambda x: 0.0 if x < 0.06 else x)
result_df.to_csv("baseline_lightgbm_20180618_01_under_006.csv", index=False)

result_df["deal_probability"] = result_df["deal_probability"].apply(lambda x: 0.0 if x < 0.1 else x)
result_df.to_csv("baseline_lightgbm_20180618_01_under_01.csv", index=False)

result_df["deal_probability"] = result_df["deal_probability"].apply(lambda x: 0.0 if x < 0.2 else x)
result_df.to_csv("baseline_lightgbm_20180618_01_under_02.csv", index=False)
