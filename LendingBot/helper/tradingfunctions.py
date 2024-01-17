import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential

def compute_sma(df, window, colname):
    '''Computes Simple Moving Average column on a dataframe'''
    df[colname] = df['close'].rolling(window=window, center=False).mean()
    return(df)

def compute_rsi(df, window, colname):
    '''Computes RSI column for a dataframe. http://stackoverflow.com/a/32346692/3389859'''
    series = df['close']
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    # first value is sum of avg gains
    u[u.index[window - 1]] = np.mean(u[:window])
    u = u.drop(u.index[:(window - 1)])
    # first value is sum of avg losses
    d[d.index[window - 1]] = np.mean(d[:window])
    d = d.drop(d.index[:(window - 1)])
    rs = u.ewm(com=window - 1,ignore_na=False,
               min_periods=0,adjust=False).mean() / d.ewm(com=window - 1, ignore_na=False,
                                            min_periods=0,adjust=False).mean()
    df[colname] = 100 - 100 / (1 + rs)
    df[colname].fillna(df[colname].mean(), inplace=True)
    return(df)

def compute_variables1(df):
    print("Let's compute predictive variables : ")
    df["date"] = pd.to_datetime(df["date"])
    df['bodysize'] = df['close'] - df['open']
    df['shadowsize'] = df['high'] - df['low']
    #TODO evtl Reihe erhöhen und auf auswirkungen schauen
    for window in [3, 8, 21, 55, 144, 377]: # several Fibonacci numbers
        print(window)
        df = compute_sma(df, window, colname = 'sma_{}'.format(window))
        df = compute_rsi(df, window, colname = 'rsi_{}'.format(window))
        df["Min_{}".format(window)] = df["low"].rolling(window).min()
        df["Max_{}".format(window)] = df["high"].rolling(window).max()
        df["volume_{}".format(window)] = df["volume"].rolling(window).mean()
        df['percentChange_{}'.format(window)] = df['close'].pct_change(periods = window)
        df['RelativeSize_sma_{}'.format(window)] = df['close'] / df['sma_{}'.format(window)]
        df['Diff_{}'.format(window)] = df['close'].diff(window)

    # (a) Add modulo 10, 100, 1000, 500, 50
    df["Modulo_10"] = df["close"].copy() % 10
    df["Modulo_100"] = df["close"].copy() % 100
    df["Modulo_1000"] = df["close"].copy() % 1000
    df["Modulo_500"] = df["close"].copy() % 500
    df["Modulo_50"] = df["close"].copy() % 50
    # (b) Add weekday and day of the month
    df["WeekDay"] = df["date"].dt.weekday
    df["Day"] = df["date"].dt.day
    df.dropna(inplace=True)
    return(df)

def check_outcome(df, line, stoploss, takeprofit):
    '''0 means we reached stoploss
    1 means we reached takeprofit
    -1 means still in between'''
    price0 = df["close"].iloc[line]
    upper_lim = price0*(1+takeprofit)
    down_lim = price0*(1-stoploss)
    for i in range(line, df["close"].size):
        if df["low"].iloc[i] < down_lim :
            return(0)
        elif df["high"].iloc[i] > upper_lim :
            return(1)
    return(-1)

# def compute_result(df, stoploss, takeprofit):
#     df['result'] = 0
#     for i in range(df["close"].size):
#         if i%500 == 0:
#             print(i, '/', df.shape[0])
#         df['result'].iloc[i] = check_outcome(df, i, stoploss, takeprofit)
#     return(df)

from multiprocessing import Pool

def f(df, stoploss, takeprofit):
    print('start')
    tmp = []
    for i in range(df["close"].size):
        # df['result'].iloc[i] = check_outcome(df, i, stoploss, takeprofit)
        tmp.append(check_outcome(df, i, stoploss, takeprofit))
    print("end")
    df['result'] = tmp
    return df
# https://docs.python.org/3/library/multiprocessing.html
def compute_result(df, stoploss, takeprofit):
    global dfs
    p = Pool(5)
    df['result'] = 0
    n_worker = 20
    N = df["close"].size
    dfs = []

    print("####",df.shape)

    def adder(_df):
        global dfs
        dfs.append(_df)

    for i in range(n_worker):
        print(int(i*N/n_worker),int((i+1)*N/n_worker))
        _df = df.iloc[int(i*N/n_worker):int((i+1)*N/n_worker)]
        # print(_df.head())
        p.apply_async(f, args=(_df, stoploss, takeprofit,), callback=adder)
    p.close()
    p.join()

    df = pd.concat(dfs)
    print("CONCATENATION",df.shape)
    return (df)


def compute_earnings_loss(stoploss, takeprofit, fees):
    '''Compute earnings and loss with given fees, stoploss, takeprofit'''
    win = (1-fees)*(1+takeprofit)*(1-fees) -1
    loss = (1-fees)*(1-stoploss)*(1-fees) -1
    return(win, loss)

def predict_and_backtest_bullish(df, df_final, model, stoploss, takeprofit, fees, nPCs, plotting = True):
    '''This functin takes the test set as input (in both shapes) + the model, computes predictions and  probabilities, then compute the earnings according to the fees. Finally it can plot the strategy'''
    # Compute predictions on testset
    clf = Sequential()
    df['preds'] = (clf.predict(df_final.iloc[:, :nPCs]) > 0.5)*1
    df['proba1'] = clf.predict(df_final.iloc[:, :nPCs])

    # keep only the timesteps in which the model predicts a bullish trend
    testset1 = df[df['preds'] == 1].copy()

    # Compute earnings column
    a = compute_earnings_loss(stoploss, takeprofit, fees)
    testset1['EarningsBullish'] = (testset1['preds'] == testset1['result'])*a[0] + (testset1['preds'] != testset1['result'])*a[1]

    if plotting:
        # Now plot our trading strategy
        plt.plot(pd.to_datetime(testset1['date']), np.cumsum(testset1['EarningsBullish']))
        plt.title('Approach over the test set \n ROI = {} %'.format(100*np.mean(testset1['EarningsBullish'])))
        plt.xlabel('Date')
        plt.xlabel('Cumulative Earnings')
        plt.show()

        # Display the entry points
        plt.plot(pd.to_datetime(df['date']), df['close'])
        plt.scatter(pd.to_datetime(testset1['date']), testset1['close'], c = (testset1['EarningsBullish']>0))
        plt.title('Entry points \n Yellow = Win, Blue = Loss')
        plt.show()

    return(testset1)

def table_recap(df, stoploss, takeprofit, nPCs, columnA = 'proba1', columnB = 'EarningsBullish'):
    ''' Summarize the strategy by steps of 0.05, depending on which column (i.e. strategy)
    we choose'''
    recap = pd.DataFrame(np.zeros((int(10), 0)))
    recap['stoploss'] = stoploss
    recap['takeprofit'] = takeprofit
    recap['nPCs'] = nPCs
    recap['Min'] = [0.5 + k*0.05 for k in range(10)]
    recap['Max'] = 1
    recap['ROI%'] = 0
    recap['nTrades'] = 0
    for i in range(len(recap['Min'])):
        min, max = recap['Min'].iloc[i], recap['Max'].iloc[i]
        df2 = df[(df[columnA] > min) & (df[columnA] < max)]
        recap['ROI%'].iloc[i] = 100 * np.mean(df2[columnB])
        recap['nTrades'].iloc[i] = df2.shape[0]

    return(recap)