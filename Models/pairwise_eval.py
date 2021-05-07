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
    )[(test_start if EVAL_TYPE=="TEST" else 0):(None if EVAL_TYPE=="TEST" else test_start)]

slopes = pd.read_csv("../Models/model_out/pairwise_slopes.csv", index_col=0)
intercepts = pd.read_csv("../Models/model_out/pairwise_intercepts.csv", index_col=0)

#%%
def run_strat_analysis(fcast, using, threshold):
    params = [intercepts[using][fcast], slopes[using][fcast]] #intercept, slope
    corr_trade = pd.DataFrame()
    corr_trade["datetime"] = bars[using]["datetime"]
    corr_trade[using] = bars[using]["close"]
    corr_trade[fcast] = bars[fcast]["close"]
    corr_trade[using+"_return"] = corr_trade[using].diff()/corr_trade[using].shift()
    corr_trade[fcast+"_return"] = corr_trade[fcast].diff()/corr_trade[fcast].shift()
    corr_trade[fcast+"_markout_1h"] = corr_trade[fcast].shift(-1) - corr_trade[fcast]
    corr_trade[fcast+"_markout_2h"] = corr_trade[fcast].shift(-2) - corr_trade[fcast]
    corr_trade[fcast+"_markout_4h"] = corr_trade[fcast].shift(-4) - corr_trade[fcast]
    corr_trade["signal"] = corr_trade[fcast+"_return"] - (params[0] + params[1]*corr_trade[using+"_return"])
    
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
for symbol_y in symbols:
    sym_res_buy = []
    sym_res_sell = []
    for symbol_x in symbols:
        buy_results, sell_results = run_strat_analysis(symbol_y, symbol_x, 0)
        sym_res_buy.append(buy_results)
        sym_res_sell.append(sell_results)

    buy_mat.append(sym_res_buy)
    sell_mat.append(sym_res_sell)

#separate out results
buy_mat = np.asarray(buy_mat)
sell_mat = np.asarray(sell_mat)

buy_mat_1h = pd.DataFrame(buy_mat[:,:,0], columns=symbols, index=symbols)
buy_mat_2h = pd.DataFrame(buy_mat[:,:,1], columns=symbols, index=symbols)
buy_mat_4h = pd.DataFrame(buy_mat[:,:,2], columns=symbols, index=symbols)

sell_mat_1h = pd.DataFrame(sell_mat[:,:,0], columns=symbols, index=symbols)
sell_mat_2h = pd.DataFrame(sell_mat[:,:,1], columns=symbols, index=symbols)
sell_mat_4h = pd.DataFrame(sell_mat[:,:,2], columns=symbols, index=symbols)


#%% Plot results
figsize = (16,16)
fig, axs = plt.subplots(1,1, figsize=figsize)
sns.heatmap(round(buy_mat_1h,2), annot=True, ax=axs, cmap = cmap, fmt='g', square=True)
axs.set_title("0 Threshold PnL - Buy 1h")
axs.set_ylabel("Traded Symbol")
axs.set_xlabel("Observed Symbol")
fig.savefig('../Figures/pair_buy_pnl_1h.png', pad_inches=0, bbox_inches='tight')

fig, axs = plt.subplots(1,1, figsize=figsize)
sns.heatmap(round(buy_mat_2h,2), annot=True, ax=axs, cmap = cmap, fmt='g', square=True)
axs.set_title("0 Threshold PnL - Buy 2h")
axs.set_ylabel("Traded Symbol")
axs.set_xlabel("Observed Symbol")
fig.savefig('../Figures/pair_buy_pnl_2h.png', pad_inches=0, bbox_inches='tight')

fig, axs = plt.subplots(1,1, figsize=figsize)
sns.heatmap(round(buy_mat_4h,2), annot=True, ax=axs, cmap = cmap, fmt='g', square=True)
axs.set_title("0 Threshold PnL - Buy 4h")
axs.set_ylabel("Traded Symbol")
axs.set_xlabel("Observed Symbol")
fig.savefig('../Figures/pair_buy_pnl_4h.png', pad_inches=0, bbox_inches='tight')

fig, axs = plt.subplots(1,1, figsize=figsize)
sns.heatmap(round(sell_mat_1h,2), annot=True, ax=axs, cmap = cmap, fmt='g', square=True)
axs.set_title("0 Threshold PnL - Sell 1h")
axs.set_ylabel("Traded Symbol")
axs.set_xlabel("Observed Symbol")
fig.savefig('../Figures/pair_sell_pnl_1h.png', pad_inches=0, bbox_inches='tight')

fig, axs = plt.subplots(1,1, figsize=figsize)
sns.heatmap(round(sell_mat_2h,2), annot=True, ax=axs, cmap = cmap, fmt='g', square=True)
axs.set_title("0 Threshold PnL - Sell 2h")
axs.set_ylabel("Traded Symbol")
axs.set_xlabel("Observed Symbol")
fig.savefig('../Figures/pair_sell_pnl_2h.png', pad_inches=0, bbox_inches='tight')

fig, axs = plt.subplots(1,1, figsize=figsize)
sns.heatmap(round(sell_mat_4h,2), annot=True, ax=axs, cmap = cmap, fmt='g', square=True)
axs.set_title("0 Threshold PnL - Sell 4h")
axs.set_ylabel("Traded Symbol")
axs.set_xlabel("Observed Symbol")
fig.savefig('../Figures/pair_sell_pnl_4h.png', pad_inches=0, bbox_inches='tight')
# %%
