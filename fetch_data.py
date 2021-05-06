#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import requests
import datetime
import pytz

#%%
BASE_URL = "https://data.alpaca.markets/v2"
headers = {
    "APCA-API-KEY-ID": os.environ.get("APCA_KEY_ID"), 
    "APCA-API-SECRET-KEY": os.environ.get("APCA_SECRET_KEY")
}

#%%
def get_bars(dt_start, dt_end, ticker, timeframe):
    df = None
    page_token = None
    count = 0

    while True:
        resp = requests.get(
            BASE_URL + "/stocks/{ticker}/bars".format(ticker=ticker), 
            headers=headers, 
            params={
                "start": dt_start.isoformat(),
                "end": dt_end.isoformat(),
                "limit": 10000,
                "page_token": page_token,
                "timeframe": timeframe
            }
        )

        page_token = resp.json()["next_page_token"]
        new_df = pd.DataFrame(resp.json()["bars"])

        if df is None:
            df = new_df
        else:
            df = df.append(new_df, ignore_index=True)

        print(count, page_token)
        count+=1

        if page_token is None:
            break

    df.columns = [
        "datetime", "open", "high", "low", "close", "volume"
    ]
    
    return df


#%%
symbol = "UAL"
if __name__ == "__main__":
    data = get_bars(
        datetime.datetime(2016, 1, 1, 0, 0, tzinfo=pytz.UTC),
        datetime.datetime(2021, 5, 1, 0, 0, tzinfo=pytz.UTC),
        symbol,
        "1Hour" #1Hour, 1Day
    )
    data.to_csv("./Data/raw/"+symbol+"_hourly_bars.csv", index=False)

# %%
data
# %%
