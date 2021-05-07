#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
cmap = matplotlib.cm.get_cmap("cividis")
import seaborn as sns
import datetime
import pytz

#%%
symbols = [
    "AAL", "DAL", "LUV", "ALGT", "UAL", #airlines
    "CVX", "MRO", "MUR", "XOM", "DVN", #oil
    "SPY", "TLT" #market and rates
]

EVAL_TYPE = "TEST"
test_start = 16078

bars={}
for symbol in symbols:
    bars[symbol] = pd.read_csv(
        "../Data/cleaned/{s}_hourly_bars.csv".format(s=symbol), 
        parse_dates=["datetime"]
    )[
        (test_start if EVAL_TYPE=="TEST" else 0):(None if EVAL_TYPE=="TEST" else test_start)
    ].reset_index(drop=True)

predictions = None
if EVAL_TYPE == "TEST":
    predictions = pd.read_csv('../Models/model_out/Full_PCA_Predict_Test.csv')
else:
    predictions = pd.read_csv('../Models/model_out/Full_PCA_Predict_Train.csv')

fcast = "SPY"

corr_trade = pd.DataFrame()
corr_trade["datetime"] = bars[fcast]["datetime"]
corr_trade[fcast] = bars[fcast]["close"]
corr_trade[fcast+"_return"] = corr_trade[fcast].diff()/corr_trade[fcast].shift()


corr_trade[fcast+"_markout_1h"] = corr_trade[fcast].shift(-1) - corr_trade[fcast]
corr_trade[fcast+"_markout_2h"] = corr_trade[fcast].shift(-2) - corr_trade[fcast]
corr_trade[fcast+"_markout_4h"] = corr_trade[fcast].shift(-4) - corr_trade[fcast]
corr_trade["signal"] = corr_trade[fcast+"_return"] - predictions[fcast]

# %%
#if signal is high, sell, else buy
pnls = []
rng = np.linspace(-.1, .1, 1000)
for threshold in rng:
    buy_pnl_1h = corr_trade[corr_trade.signal < threshold][fcast+"_markout_1h"].sum()
    buy_pnl_2h = corr_trade[corr_trade.signal < threshold][fcast+"_markout_2h"].sum()
    buy_pnl_4h = corr_trade[corr_trade.signal < threshold][fcast+"_markout_4h"].sum()
    pnls.append([buy_pnl_1h, buy_pnl_2h, buy_pnl_4h])

pnls = np.asarray(pnls)

fig, axs = plt.subplots(1,1, figsize=(8,6))
axs.plot(rng, pnls[:,0], label="1h markout", c=cmap(1/3))
axs.plot(rng, pnls[:,1], label="2h markout", c=cmap(2/3))
axs.plot(rng, pnls[:,2], label="4h markout", c=cmap(3/3))
axs.legend()
axs.set_xlabel("Signal Threshold")
axs.set_ylabel("Sum of Markouts")
axs.set_title("PnL Curve for buying " + fcast + " on PCs of Full Data Returns")

# %%
#if signal is high, sell, else buy
pnls = []
rng = np.linspace(-.1, .1, 1000)
for threshold in rng:
    sell_pnl_1h = -corr_trade[corr_trade.signal > threshold][fcast+"_markout_1h"].sum()
    sell_pnl_2h = -corr_trade[corr_trade.signal > threshold][fcast+"_markout_2h"].sum()
    sell_pnl_4h = -corr_trade[corr_trade.signal > threshold][fcast+"_markout_4h"].sum()
    pnls.append([sell_pnl_1h, sell_pnl_2h, sell_pnl_4h])

pnls = np.asarray(pnls)

fig, axs = plt.subplots(1,1, figsize=(8,6))
axs.plot(rng, pnls[:,0], label="1h markout", c=cmap(1/3))
axs.plot(rng, pnls[:,1], label="2h markout", c=cmap(2/3))
axs.plot(rng, pnls[:,2], label="4h markout", c=cmap(3/3))
axs.legend()
axs.set_xlabel("Signal Threshold")
axs.set_ylabel("Sum of Markouts")
axs.set_title("PnL Curve for selling " + fcast + " on PCs of Full Data Returns")
# %%
