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
    predictions = pd.read_csv('./model_out/Full_PCA_Predict_Test.csv')
else:
    predictions = pd.read_csv('./model_out/Full_PCA_Predict_Train.csv')

#%%
def run_strat_analysis(fcast, predict, threshold):
    corr_trade = pd.DataFrame()
    corr_trade["datetime"] = bars[fcast]["datetime"]
    corr_trade[fcast] = bars[fcast]["close"]
    corr_trade[fcast+"_return"] = corr_trade[fcast].diff()/corr_trade[fcast].shift()

    corr_trade[fcast+"_markout_1h"] = corr_trade[fcast].shift(-1) - corr_trade[fcast]
    corr_trade[fcast+"_markout_2h"] = corr_trade[fcast].shift(-2) - corr_trade[fcast]
    corr_trade[fcast+"_markout_4h"] = corr_trade[fcast].shift(-4) - corr_trade[fcast]

    corr_trade["signal"] = corr_trade[fcast+"_return"] - predict[fcast]
    
    buy_pnl_1h = corr_trade[corr_trade.signal < threshold][fcast+"_markout_1h"].sum()
    buy_pnl_2h = corr_trade[corr_trade.signal < threshold][fcast+"_markout_2h"].sum()
    buy_pnl_4h = corr_trade[corr_trade.signal < threshold][fcast+"_markout_4h"].sum()
    sell_pnl_1h = -corr_trade[corr_trade.signal > threshold][fcast+"_markout_1h"].sum()
    sell_pnl_2h = -corr_trade[corr_trade.signal > threshold][fcast+"_markout_2h"].sum()
    sell_pnl_4h = -corr_trade[corr_trade.signal > threshold][fcast+"_markout_4h"].sum()

    return (
        [buy_pnl_1h, buy_pnl_2h, buy_pnl_4h],
        [sell_pnl_1h,sell_pnl_2h,sell_pnl_4h]
    )


#%%
buy_mat = []
sell_mat = []
for symbol in symbols:
        buy_results, sell_results = run_strat_analysis(symbol, predictions, 0)
        buy_mat.append(buy_results)
        sell_mat.append(sell_results)

#separate out results
buy_mat = np.asarray(buy_mat)
sell_mat = np.asarray(sell_mat)

buy_mat = pd.DataFrame(
    buy_mat, index=symbols, columns=["Markout 1H", "Markout 2H", "Markout 4H"]
)

sell_mat = pd.DataFrame(
    sell_mat, index=symbols, columns=["Markout 1H", "Markout 2H", "Markout 4H"]
)

#%% Plot results
figsize = (12,4)
fig, axs = plt.subplots(1,1, figsize=figsize)
sns.heatmap(round(buy_mat.T,2), annot=True, ax=axs, cmap = cmap, fmt='g')
axs.set_title("0 Threshold PnL - Buy Side")
axs.set_xlabel("Traded Symbol")

fig, axs = plt.subplots(1,1, figsize=figsize)
sns.heatmap(round(sell_mat.T,2), annot=True, ax=axs, cmap = cmap, fmt='g')
axs.set_title("0 Threshold PnL - Sell Side")
axs.set_xlabel("Traded Symbol")
# %%
