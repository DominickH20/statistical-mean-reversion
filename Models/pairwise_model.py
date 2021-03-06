import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
cmap = matplotlib.cm.get_cmap("cividis")
import statsmodels.api as sm
import datetime
import pytz

#SETUP
symbols = [
    "AAL", "DAL", "LUV", "ALGT", "UAL", #airlines
    "CVX", "MRO", "MUR", "XOM", "DVN", #oil
    "SPY", "TLT" #market and rates
]
test_start = 16078

#GET TRAIN DATA
bars={}
for symbol in symbols:
    bars[symbol] = pd.read_csv(
        "../Data/cleaned/{s}_hourly_bars.csv".format(s=symbol), 
        parse_dates=["datetime"]
    )[:test_start].reset_index(drop=True)

#%%
for symbol in symbols:
    bars[symbol]['return'] = bars[symbol]['close'].diff()/bars[symbol]['close'].shift()

returns = pd.DataFrame()
for symbol in symbols:
    returns[symbol] = bars[symbol]['return'].dropna()

#TRIM THE OUTLIERS
# Want to learn a model that is representative of most periods
# and trade on convergence of outlier days
upper_bound = 0.995
lower_bound = 0.005

#Flag outliers
outlier_flag = pd.DataFrame()
for symbol in symbols:
    qt = np.quantile(returns[symbol], q=[lower_bound, upper_bound])
    outlier_flag[symbol] = (returns[symbol] > qt[1]) | (returns[symbol] < qt[0])
    print(symbol, qt)

#flag a bar if any of the data points are outliers
or_count = outlier_flag[symbols[0]]
for symbol in symbols:
    or_count = or_count | outlier_flag[symbol]

outlier_flag["OUT"] = or_count

#see how many values were lost
print("OUTLIERS TRIMMED:")
print(outlier_flag.values.shape)
print(outlier_flag[outlier_flag.OUT==False].values.shape)



#FIT PAIRWISE OLS
models = {}
for symbol_y in symbols:
    submodels = {}
    for symbol_x in symbols:
        Y = returns[outlier_flag.OUT==False][symbol_y]
        X = returns[outlier_flag.OUT==False][symbol_x]
        X = sm.add_constant(X)
        print("Model for X =", symbol_x, X.shape,"y =", symbol_y, Y.shape)
        model = sm.OLS(Y,X)
        results = model.fit()
        submodels[symbol_x] = results
    models[symbol_y] = submodels

#COLLECT INTERCEPT AND SLOPE COEFFICIENTS AND SAVE
intercepts = []
slopes = []
for symbol_y in symbols:

    sl_part = []
    int_part = []

    for symbol_x in symbols:
        int_part.append(models[symbol_y][symbol_x].params[0])
        sl_part.append(models[symbol_y][symbol_x].params[1])

    intercepts.append(int_part)
    slopes.append(sl_part)

intercepts = pd.DataFrame(intercepts, columns=symbols, index=symbols)
slopes = pd.DataFrame(slopes, columns=symbols, index=symbols)

slopes.to_csv("./model_out/pairwise_slopes.csv")
intercepts.to_csv("./model_out/pairwise_intercepts.csv")