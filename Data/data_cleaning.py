
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import pytz

#%%
idx = pd.DataFrame(
    pd.read_csv(
        "./raw/SPY_hourly_bars.csv", 
        parse_dates=["datetime"]
    )['datetime']
)

symbols = [
    "AAL", "DAL", "LUV", "ALGT", "UAL", #airlines
    "CVX", "MRO", "MUR", "XOM", "DVN", #oil
    "SPY", "TLT" #market and rates
]

#%% MERGE INTO SPY DATA TO GET CONSISTENT DATA OF SAME LENGTH
for symbol in symbols:
    to_clean = pd.read_csv(
        "./raw/{s}_hourly_bars.csv".format(s=symbol), 
        parse_dates=["datetime"]
    )

    clean_df = pd.merge_asof(idx, to_clean, on="datetime", direction="backward")
    clean_df.fillna(method="backfill", inplace=True)
    clean_df.to_csv("./cleaned/{s}_hourly_bars.csv".format(s=symbol), index=False)
# %%
