#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
cmap = matplotlib.cm.get_cmap("cividis")
import seaborn as sns

import datetime
import pytz

from sklearn.decomposition import PCA

#%%
symbols = [ #UNITED
    "AAL", "DAL", "LUV", "ALGT", "UAL", #airlines
    "CVX", "MRO", "MUR", "XOM", "DVN", #oil
    "SPY", "TLT" #market and rates
]

test_start = 16078

airlines = symbols[0:5]
oil = symbols[5:10]
airlines_oil = symbols[0:10]

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

#%% COMPUTE PCA 
data = symbols #change this to run on different sets of data
pca = PCA()

PCs = {}
components = {}
var_exp = {}
var_exp_r = {}
for i in range(len(data)):
    y_label = data[i]
    x_labels = data[0:i] + data[i+1:len(data)]
    print("Y =",y_label, ", X =",x_labels)
    #fit PCA
    new_x = pca.fit_transform(returns[outlier_flag.OUT==False][x_labels])
    #record transformed X
    PCs[y_label] = pd.DataFrame(new_x)
    PCs[y_label].columns = ["PC" + str(i) for i in range(1,len(data))]
    #record other metrics
    components[y_label] = pca.components_
    var_exp[y_label] = pca.explained_variance_
    var_exp_r[y_label] = pca.explained_variance_ratio_

# %% PLOT THE DATA
symbol = "CVX"
corrs = round(
    PCs[symbol].corrwith(
        returns[outlier_flag.OUT==False][symbol].reset_index(drop=True)
    ),3
)
fig, axs = plt.subplots(2,2, figsize = (12,12))
for i in range(2):
    for j in range(2):
        axs[i][j].scatter(
            PCs[symbol]["PC"+str(2*i+j+1)], 
            returns[outlier_flag.OUT==False][symbol], 
            color=cmap((2*i+j+1)/4), s=1
        )
        axs[i][j].set_title("PC"+str(2*i+j+1)+" vs " + symbol + " Return (r=" + str(corrs[2*i+j]) + ")")
        axs[i][j].set_xlabel("PC"+str(2*i+j+1))
        axs[i][j].set_ylabel(symbol + " Return")

# %%
pc_pves = pd.DataFrame(index=["PC" + str(i) for i in range(1,len(data))])
for symbol in data:
    pc_pves[symbol] = var_exp_r[symbol]

pc_corrs = pd.DataFrame()
for symbol in data:
    corrs = PCs[symbol].corrwith(
        returns[outlier_flag.OUT==False][symbol].reset_index(drop=True)
    )
    pc_corrs[symbol] = corrs

fig, axs = plt.subplots(1,1, figsize=(12,8))
sns.heatmap(round(pc_pves,3), annot=True, ax=axs, cmap = cmap)
axs.set_title("PC Percentage of Variance Explained")
axs.set_xlabel("Symbol Held Out")

fig, axs = plt.subplots(1,1, figsize=(12,8))
sns.heatmap(round(abs(pc_corrs),3), annot=True, ax=axs, cmap = cmap)
axs.set_title("PC Correlations")
axs.set_xlabel("Response Variable")

#%% FIT OLS REGRESSION MODEL ON PCs
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

RMSES = {}
RMSES["PC1"] = {}
RMSES["Full"] = {}
RMSES["Lasso"] = {}

reg_pc1_models = {}
for symbol in data:
    Y = returns[outlier_flag.OUT==False][symbol].reset_index(drop=True)
    X = PCs[symbol]["PC1"].reset_index(drop=True)
    X = sm.add_constant(X)
    print("PC1 model for y =", symbol, Y.shape, "on PCs and Intercept:",X.shape)
    model = sm.OLS(Y,X)
    results = model.fit()
    reg_pc1_models[symbol] = results
    
    #print rmse
    ypred = results.predict(X)
    RMSES["PC1"][symbol] = rmse(Y, ypred)

# %%
reg_pc1_params = []
for symbol in data:
    reg_pc1_params.append(reg_pc1_models[symbol].params.values)

reg_pc1_params = pd.DataFrame(
    np.asarray(reg_pc1_params), 
    index=data, columns=["Int", "PC1"]
)

fig, axs = plt.subplots(1,1, figsize=(12,8))
sns.heatmap(round(abs(reg_pc1_params),3), annot=True, ax=axs, cmap = cmap)
axs.set_title("OLS PC1 Regression Coefficients")
axs.set_xlabel("Coefficient on PC")

#%% FIT OLS REGRESSION MODEL ON PCs
reg_pc_models = {}
for symbol in data:
    Y = returns[outlier_flag.OUT==False][symbol].reset_index(drop=True)
    X = PCs[symbol].reset_index(drop=True)
    X = sm.add_constant(X)
    print("Full model for y =", symbol, Y.shape, "on PCs and Intercept:",X.shape)
    model = sm.OLS(Y,X)
    results = model.fit()
    reg_pc_models[symbol] = results

    #print rmse
    ypred = results.predict(X)
    RMSES["Full"][symbol] = rmse(Y, ypred)

# %%
reg_pc_params = []
for symbol in data:
    reg_pc_params.append(reg_pc_models[symbol].params.values)

reg_pc_params = pd.DataFrame(
    np.asarray(reg_pc_params), 
    index=data, columns=["Int"] + ["PC"+str(i) for i in range(1,len(data))]
)

fig, axs = plt.subplots(1,1, figsize=(12,8))
sns.heatmap(round(abs(reg_pc_params),3), annot=True, ax=axs, cmap = cmap)
axs.set_title("OLS Full Regression Coefficients")
axs.set_xlabel("Coefficient on PC")

#%%
from sklearn import linear_model

lasso_pc_models = {}
for symbol in data:
    Y = returns[outlier_flag.OUT==False][symbol].reset_index(drop=True)
    X = PCs[symbol].reset_index(drop=True)
    X = sm.add_constant(X)
    print("Lasso model for y =", symbol, Y.shape, "on PCs and Intercept:",X.shape)
    reg = linear_model.LassoCV(cv=5, random_state=0).fit(X,Y)
    lasso_pc_models[symbol] = reg

    #print rmse
    ypred = reg.predict(X)
    RMSES["Lasso"][symbol] = rmse(Y, ypred)

# %%
lasso_pc_params = []
for symbol in data:
    lasso_pc_params.append(lasso_pc_models[symbol].coef_)

lasso_pc_params = pd.DataFrame(
    np.asarray(lasso_pc_params), 
    index=data, columns=["Int"] + ["PC"+str(i) for i in range(1,len(data))]
)

fig, axs = plt.subplots(1,1, figsize=(12,8))
sns.heatmap(round(abs(reg_pc_params),3), annot=True, ax=axs, cmap = cmap)
axs.set_title("Lasso Regression Coefficients")
axs.set_xlabel("Coefficient on PC")


#%%
RMSES = pd.DataFrame(RMSES)
RMSES

#%%
#get targets
returns_future = returns.shift(-1)
new_col = [symbol+"_fut" for symbol in symbols]
returns_future.columns = new_col
full_data = pd.concat([returns, returns_future], axis=1)[:-1]
outlier_flag_full = outlier_flag[:-1]
# %%
