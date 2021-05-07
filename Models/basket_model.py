#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
cmap = matplotlib.cm.get_cmap("cividis")
import statsmodels.api as sm
from sklearn.decomposition import PCA
import datetime
import pytz

#SETUP
symbols = [
    "AAL", "DAL", "LUV", "ALGT", "UAL", #airlines
    "CVX", "MRO", "MUR", "XOM", "DVN", #oil
    "SPY", "TLT" #market and rates
]
test_start = 16078

def get_bars(dataset_type):
    bars = {}
    for symbol in symbols:
        bars[symbol] = pd.read_csv(
            "../Data/cleaned/{s}_hourly_bars.csv".format(s=symbol), 
            parse_dates=["datetime"]
        )[
            (test_start if dataset_type=="TEST" else 0):(None if dataset_type=="TEST" else test_start)
        ].reset_index(drop=True)

    return bars

def get_returns(bars):
    for symbol in symbols:
        bars[symbol]['return'] = bars[symbol]['close'].diff()/bars[symbol]['close'].shift()

    returns = pd.DataFrame()
    for symbol in symbols:
        returns[symbol] = bars[symbol]['return'].dropna()
    
    return returns

returns_train = get_returns(get_bars("TRAIN"))
returns_test = get_returns(get_bars("TRAIN"))

#TRIM THE OUTLIERS
# Want to learn a model that is representative of most periods
# and trade on convergence of outlier days
upper_bound = 0.995
lower_bound = 0.005

#Flag outliers
outlier_flag = pd.DataFrame()
for symbol in symbols:
    qt = np.quantile(returns_train[symbol], q=[lower_bound, upper_bound])
    outlier_flag[symbol] = (returns_train[symbol] > qt[1]) | (returns_train[symbol] < qt[0])
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

# COMPUTE PCA AND REGRESSION

train_bars = get_bars("TRAIN")
test_bars = get_bars("TEST")

train_fcast = pd.DataFrame()
test_fcast = pd.DataFrame()

for i in range(len(symbols)):
    pca = PCA()
    y_label = symbols[i]
    x_labels = symbols[0:i] + symbols[i+1:len(symbols)]
    print("Y =",y_label, ", X =",x_labels)

    #FIT PCA
    new_x = pca.fit_transform(returns_train[outlier_flag.OUT==False][x_labels])
    new_x = pd.DataFrame(new_x)
    new_x.columns = ["PC" + str(i) for i in range(1,len(symbols))]

    #FIT MODEL, NO OUTLIERS
    Y = returns_train[outlier_flag.OUT==False][symbol].reset_index(drop=True)
    X = new_x
    X = sm.add_constant(X)
    print("PC reg for y =", symbol, Y.shape, "on PCs and Intercept:", X.shape)
    model = sm.OLS(Y,X)
    results = model.fit()

    #PREDICT ON TRAIN
    train_df = pd.DataFrame()
    train_df["datetime"] = train_bars[symbol]["datetime"]
    train_df[y_label] = train_bars[y_label]["close"]
    for x_label in x_labels:
        train_df[x_label] = train_bars[x_label]["close"]

    #get returns
    train_rets = train_df[x_labels].diff()/train_df[x_labels].shift()
    train_rets.dropna(inplace = True) #now indices 1:end rather than 0:end

    #compute transform
    train_pcs = pca.transform(train_rets)
    train_pcs = pd.DataFrame(train_pcs)
    train_pcs.columns = ["PC" + str(i) for i in range(1,len(symbols))]

    #now make indices 0:end after transform
    train_pcs.loc[-1] = [np.NaN for x in x_labels]  # adding a row
    train_pcs.index = train_pcs.index + 1  # shifting index
    train_pcs.sort_index(inplace=True) 

    #predict and store
    train_pcs = sm.add_constant(train_pcs)
    f_cast_train = results.predict(train_pcs)
    train_fcast[y_label] = f_cast_train


    #PREDICT ON TEST
    test_df = pd.DataFrame()
    test_df["datetime"] = test_bars[symbol]["datetime"]
    test_df[y_label] = test_bars[y_label]["close"]
    for x_label in x_labels:
        test_df[x_label] = test_bars[x_label]["close"]

    #get returns
    test_rets = test_df[x_labels].diff()/test_df[x_labels].shift()
    test_rets.dropna(inplace = True) #now indices 1:end rather than 0:end

    #compute transform
    test_pcs = pca.transform(test_rets)
    test_pcs = pd.DataFrame(test_pcs)
    test_pcs.columns = ["PC" + str(i) for i in range(1,len(symbols))]

    #now make indices 0:end after transform
    test_pcs.loc[-1] = [np.NaN for x in x_labels]  # adding a row
    test_pcs.index = test_pcs.index + 1  # shifting index
    test_pcs.sort_index(inplace=True) 

    #predict and store
    test_pcs = sm.add_constant(test_pcs)
    f_cast_test = results.predict(test_pcs)
    test_fcast[y_label] = f_cast_test    

train_fcast.to_csv('./model_out/Full_PCA_Predict_Train.csv', index=False)
test_fcast.to_csv('./model_out/Full_PCA_Predict_Test.csv', index=False)
