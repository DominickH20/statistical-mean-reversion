#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import pytz

#%%
symbols = [
    "AAL", "DAL", "LUV", "ALGT", "UAL", #airlines
    "CVX", "MRO", "MUR", "XOM", "DVN", #oil
    "SPY", "TLT" #market and rates
]

bars={}
for symbol in symbols:
    bars[symbol] = pd.read_csv(
        "../Data/cleaned/{s}_hourly_bars.csv".format(s=symbol), 
        parse_dates=["datetime"]
    )

#%%
for symbol in symbols:
    bars[symbol]['return'] = bars[symbol]['close'].diff()/bars[symbol]['close'].shift()

returns = pd.DataFrame()
for symbol in symbols:
    returns[symbol] = bars[symbol]['return'].dropna()
returns.head()

# %%
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
print(outlier_flag.values.shape)
print(outlier_flag[outlier_flag.OUT==False].values.shape)

#%%
#get targets
returns_future = returns.shift(-1)
new_col = [symbol+"_fut" for symbol in symbols]
returns_future.columns = new_col
full_data = pd.concat([returns, returns_future], axis=1)[:-1]
outlier_flag_full = outlier_flag[:-1]

# %%
#PLOTTING
fig, axs = plt.subplots(1,1, figsize=(12,12))
x_sym = "AAL"
y_sym = "SPY"
axs.scatter(x=bars[x_sym]["return"], y=bars[y_sym]["return"], color="black")
axs.set_xlabel(x_sym)
axs.set_ylabel(y_sym)
axs.set_title("Return Plot")

# %%
# #plot to check - LONG
# for i in range(len(symbols)):
#     for j in range(i, len(symbols)):
#         fig, axs = plt.subplots(1,1, figsize=(8,8))
#         x_sym = symbols[i]
#         y_sym = symbols[j]
#         axs.scatter(
#             x=returns[outlier_flag.OUT==False][x_sym], 
#             y=returns[outlier_flag.OUT==False][y_sym], 
#             color="black"
#         )
#         axs.set_xlabel(x_sym)
#         axs.set_ylabel(y_sym)
#         axs.set_title("Return Plot " + y_sym + " on " + x_sym)




#%%
import seaborn as sns
fig, axs = plt.subplots(1,1, figsize=(12,12))
sns.heatmap(round(returns.corr(),2), square=True, annot=True, ax=axs)
axs.set_title("Correlation Coefficients")

fig, axs = plt.subplots(1,1, figsize=(12,12))
sns.heatmap(round(returns[outlier_flag.OUT==False].corr(),2), square=True, annot=True, ax=axs)
axs.set_title("Correlation Coefficients (No Outlier)")


# %%
import statsmodels.api as sm

models = {}
for symbol_y in symbols:
    submodels = {}
    for symbol_x in symbols:
        print("Model for X =", symbol_x, "y =", symbol_y)
        Y = returns[outlier_flag.OUT==False][symbol_y]
        X = returns[outlier_flag.OUT==False][symbol_x]
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        submodels[symbol_x] = results
    models[symbol_y] = submodels

# %%
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
#export model slopes and coefficients
slopes.to_csv("./Models/pairwise_slopes.csv")
intercepts.to_csv("./Models/pairwise_intercepts.csv")

# %%
fig, axs = plt.subplots(1,1, figsize=(12,12))
sns.heatmap(round(intercepts,2), square=True, annot=True, ax=axs)
axs.set_title("Intercepts")

fig, axs = plt.subplots(1,1, figsize=(12,12))
sns.heatmap(round(slopes,2), square=True, annot=True, ax=axs)
axs.set_title("Slopes")



# %%
# level = 0.05

# intercepts_sig = []
# slopes_sig = []
# for symbol_y in symbols:
#     sl_part = []
#     int_part = []
#     for symbol_x in symbols:
#         int_part.append(models[symbol_y][symbol_x].pvalues[0] < level)
#         sl_part.append(models[symbol_y][symbol_x].pvalues[1] < level)
#     intercepts_sig.append(int_part)
#     slopes_sig.append(sl_part)

# intercepts_sig = pd.DataFrame(intercepts_sig, columns=symbols, index=symbols)
# slopes_sig = pd.DataFrame(slopes_sig, columns=symbols, index=symbols)

# # %%
# fig, axs = plt.subplots(1,1, figsize=(12,12))
# sns.heatmap(round(intercepts_sig,2), square=True, annot=True, ax=axs)
# axs.set_title("Intercept Significance")

# fig, axs = plt.subplots(1,1, figsize=(12,12))
# sns.heatmap(round(slopes_sig,2), square=True, annot=True, ax=axs)
# axs.set_title("Slope Significance")

# %%
y_sym = "UAL"
x_sym = "DAL"
slr = models[y_sym][x_sym]
line_x = np.linspace(-.05, .05, 1000)
line_y = slr.params[0] + slr.params[1]*line_x

#THE LINE IS BIASED, WANT TO FIGURE OUT WHY
# THIS IS BECAUSE OLS FITS on Y, USE ODR FOR PERPENDICULAR RESIDUALS
plt.scatter(
    returns[outlier_flag.OUT==False][x_sym],
    returns[outlier_flag.OUT==False][y_sym]
)
plt.plot(line_x, line_y, color="black")

#%%
# FIT FUTURE RETURN ON RESIDUALS
resid_models = {}
for symbol_y in symbols:
    submodels = {}
    for symbol_x in symbols:
        print("Resid Model for X =", symbol_x, "y =", symbol_y)
        slr = models[symbol_y][symbol_x]
        Y = full_data[outlier_flag_full.OUT==False][symbol_y+"_fut"]
        X = slr.resid[:-1].rename(symbol_x)
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        submodels[symbol_x] = results
    resid_models[symbol_y] = submodels

# %%
r_intercepts = []
r_slopes = []
for symbol_y in symbols:

    sl_part = []
    int_part = []

    for symbol_x in symbols:
        #print(symbol_y, symbol_x, resid_models[symbol_y][symbol_x].params)
        int_part.append(resid_models[symbol_y][symbol_x].params[0])
        sl_part.append(resid_models[symbol_y][symbol_x].params[1])

    r_intercepts.append(int_part)
    r_slopes.append(sl_part)

r_intercepts = pd.DataFrame(r_intercepts, columns=symbols, index=symbols)
r_slopes = pd.DataFrame(r_slopes, columns=symbols, index=symbols)

# %%
fig, axs = plt.subplots(1,1, figsize=(12,12))
sns.heatmap(round(r_intercepts,2), square=True, annot=True, ax=axs)
axs.set_title("Resid Intercepts")

fig, axs = plt.subplots(1,1, figsize=(12,12))
sns.heatmap(round(r_slopes,2), square=True, annot=True, ax=axs)
axs.set_title("Resid Slopes")

# %%
y_sym = "XOM"
x_sym = "CVX"
slr = resid_models[y_sym][x_sym]
line_x = np.linspace(-.05, .05, 1000)
line_y = slr.params[0] + slr.params[1]*line_x
plt.scatter(
    models[x_sym][y_sym].resid[:-1], 
    full_data[outlier_flag_full.OUT==False][y_sym+"_fut"]
)
plt.plot(line_x, line_y, color="black")