########### PARTS I to III
import os
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import requests
import json
import time
import csv
import helper.tradingfunctions
import helper.requestcsv
from keras.models import Sequential
from keras.layers import Dense
from scipy.stats import ttest_1samp

# Creating subfolders if they don't exist ...
if not os.path.exists('./Data'):
        os.makedirs('./Data')
if not os.path.exists('./Models'):
        os.makedirs('./Models')
if not os.path.exists('./Results'):
        os.makedirs('./Results')

CurrencyPair = "USDT_BTC"
Period = "7200"

helper.requestcsv.request(CurrencyPair,Period)

df = pd.read_csv('./Data/' + CurrencyPair + '_Poloniex_all_' + Period + '.csv')
print("Bug 1", df.shape)

#df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"],unit='s')
df = df.sort_values(by = 'date')

# Keep only 5 basic information + the date
df = df[['close', 'date', 'high', 'low', 'open', 'volume']]

# Define limit between train set and testset
#TODO variable limits
start_validation = '2018-12-21 12:00:00'
start_test = '2019-12-21 12:00:00'
stoploss = 0.05
takeprofit = 0.1

#TODO Hier eine Schleife bauen, die verschiedene Stoploss and Takeprofits ausprobiert
#################### II - Build the function to compute variables #################
df = helper.tradingfunctions.compute_variables1(df)
df.to_csv('./Data/DatasetWithVariables.csv', index = False)
print("Bug 2", df.shape)

#################### III - Compute the output #######################################
df = helper.tradingfunctions.compute_result(df, stoploss, takeprofit)
df = df[df['result']>=0] # Only keep observations where we also have the result
df.to_csv('./Data/DatasetWithVariablesAndY_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
print("Bug 3", df.shape)

#################### IV - Apply PCA and save results ##############################
# First we define the trainset, validation set, testset. This is important in this step to avoid causality issues.
#TODO mit in der Schleife
k = df.shape[0]
trainset = df[:int(k*.25)] #df[df['date'] < start_validation]
validation_set = df[int(k*.25):int(k*.75)] #df[(df['date'] >= start_validation) & (df['date'] < start_test)]
testset = df[int(k*.75):] #df[df['date'] > start_test]

trainset.to_csv('./Data/TrainSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
validation_set.to_csv('./Data/ValidationSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
testset.to_csv('./Data/TestSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)

# Display the splitting
# plt.plot(pd.to_datetime(trainset['date']), trainset['close'], c = 'orange')
# plt.plot(pd.to_datetime(validation_set['date']), validation_set['close'], c = 'b')
# plt.plot(pd.to_datetime(testset['date']), testset['close'], c = 'g')
# plt.title('Repartition between trainset, validation set, and test set')
# plt.show()

# (i) Scale the variables
scale_fct = StandardScaler()
scale_fct.fit(trainset.drop('date', 1).drop('result', 1))
pk.dump(scale_fct, open('./Models/scaler.pkl','wb'))

# (ii) Apply PCA
pca = PCA(n_components=trainset.shape[1] - 2) # remove the result and the date
pca.fit(scale_fct.transform(trainset.drop('date', 1).drop('result', 1)))
pk.dump(pca, open('./Models/pca.pkl',"wb"))

# (iii) Scale PCA components (this accelerates training process in Deep Learning)
pca_scaler = StandardScaler()
pca_scaler.fit(pca.transform(scale_fct.transform(trainset.drop('date', 1).drop('result', 1))))
pk.dump(pca_scaler, open('./Models/pca_scaler.pkl','wb'))

# (iv) Save ready-to-use versions (i.e. datasets after applying scalers and PCA)
trainset_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(trainset.drop('date', 1).drop('result', 1)))))
validation_set_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(validation_set.drop('date', 1).drop('result', 1)))))
testset_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(testset.drop('date', 1).drop('result', 1)))))

trainset_final.to_csv('./Data/TrainSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
validation_set_final.to_csv('./Data/ValidationSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
testset_final.to_csv('./Data/TestSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)


list_nPCs = [5, 10, 15, 20, 25, 30, 35, 40]

# (a) Load previously built datasets : we just need train sets and validation sets here
trainset_final = pd.read_csv('./Data/TrainSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
trainset = pd.read_csv('./Data/TrainSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

validation_set_final = pd.read_csv('./Data/ValidationSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
validation_set = pd.read_csv('./Data/ValidationSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

# (b) Build and train several models different amounts of PCs
for nPCs in list_nPCs:
    print(nPCs)
    X = trainset_final.iloc[:, :nPCs]
    y = trainset["result"]

    # Build model and train it
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=nPCs))
    #Second, third and fourth  hidden Layers
    classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    
    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    #Compiling the neural network
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    #Fitting the data to the training dataset
    classifier.fit(X,y, batch_size=500, epochs=75, verbose =1)

    # pk.dump(classifier, open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit),"wb"))
    classifier.save_weights('./Models/DL_model_{}PC_stoploss{}_takeprofit{}.h5'.format(nPCs, stoploss, takeprofit))

# (c) Test onto the testset : we compare all models and store results in a csv file
accuracies, nPCs_list = [], []
for nPCs in list_nPCs:
    print(nPCs)
     # Build model and train it
     #TODO wird das hier überhaupt abgearbeitet?
    clf = Sequential()
    #First Hidden Layer
    clf.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=nPCs))
    #Second, third and fourth  hidden Layers
    clf.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

    #Output Layer
    clf.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    # with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
    #     clf = pk.load(f)
    clf.load_weights('./Models/DL_model_{}PC_stoploss{}_takeprofit{}.h5'.format(nPCs, stoploss, takeprofit))
    # Compute predictions on testset
    preds = (clf.predict(validation_set_final.iloc[:, :nPCs]) > 0.5)*1

    # Assess accuracy on Bullish predictions only (because we will only perform Bullish trades IRL) : we prioritize selectivity
    validation_set1 = validation_set[preds == 1].copy()
    accuracies.append(np.mean(preds == list(validation_set1['result'])))
    nPCs_list.append(nPCs)

recap = pd.DataFrame({'nPCs' : list(nPCs_list), 'Accuracy' : (list(accuracies))})
recap.to_csv('./Results/Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
print(recap)


################### (a) Load previously built datasets
fees = 0.00155 # transaction fees : 0.125% for example

nTrades_mini = 50 # minimal number of trades we want over the test set: this is for second approach

trainset_final = pd.read_csv('./Data/TrainSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
trainset = pd.read_csv('./Data/TrainSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

validation_set_final = pd.read_csv('./Data/ValidationSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
validation_set = pd.read_csv('./Data/ValidationSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

testset_final = pd.read_csv('./Data/TestSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
testset = pd.read_csv('./Data/TestSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

#################### (b) Basic strategy : pick the best model and bet on bullish trends over the testset
recap = pd.read_csv('./Results/Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit)).sort_values('Accuracy', ascending = False)
nPCs = recap['nPCs'].iloc[0]

# with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
#     clf = pk.load(f)
clf = Sequential()
#First Hidden Layer
#TODO Läuft hier Deep Leraning überhaupt?
clf.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=nPCs))
#Second, third and fourth  hidden Layers
clf.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

#Output Layer
clf.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
# with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
#     clf = pk.load(f)
clf.load_weights('./Models/DL_model_{}PC_stoploss{}_takeprofit{}.h5'.format(nPCs, stoploss, takeprofit))
# Compute predictions on testset

def predict_and_backtest_bullish(df, df_final, model, stoploss, takeprofit, fees, plotting = True):
    '''This functin takes the test set as input (in both shapes) + the model, computes predictions and  probabilities, then compute the earnings according to the fees. Finally it can plot the strategy'''
    # Compute predictions on testset
    df['preds'] = (clf.predict(df_final.iloc[:, :nPCs]) > 0.5)*1
    df['proba1'] = clf.predict(df_final.iloc[:, :nPCs])

    # keep only the timesteps in which the model predicts a bullish trend
    testset1 = df[df['preds'] == 1].copy()

    # Compute earnings column
    a = helper.tradingfunctions.compute_earnings_loss(stoploss, takeprofit, fees)
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

testset1 = predict_and_backtest_bullish(testset, testset_final, clf, stoploss, takeprofit, fees, plotting = True)

# Assess the performance by comparing to if we always traded bullish blindly over the period
a = helper.tradingfunctions.compute_earnings_loss(stoploss, takeprofit, fees)
testset_benchmark = testset.copy()
testset_benchmark['EarningsBullish'] = (testset['result'] == 1)*a[0] + (testset['result'] == 0)*a[1]
avg_return_benchmark = np.mean(testset_benchmark['EarningsBullish'])

# Now let's look at our approach's performance and std
p_value = ttest_1samp(testset1['EarningsBullish'], popmean = avg_return_benchmark)[1]
print('Our model has an average ROI of {} %, while trading blindly bullish over the same period yielded a ROI of {} %, when we perform statistical testing of difference there is a p-value of {}.'.format(100*np.mean(testset1['EarningsBullish']), 100*avg_return_benchmark, p_value))

################### (c) More evolved strategy : look for the threshold to limit to the cases where p>a (for the best model)
# (i) Identify best threshold
# Load best model
recap = pd.read_csv('./Results/Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit)).sort_values('Accuracy', ascending = False)
nPCs = recap['nPCs'].iloc[0]
# with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
#     clf = pk.load(f)
clf = Sequential()
#First Hidden Layer
clf.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=nPCs))
#Second, third and fourth  hidden Layers
clf.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

#Output Layer
clf.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
# with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
#     clf = pk.load(f)
clf.load_weights('./Models/DL_model_{}PC_stoploss{}_takeprofit{}.h5'.format(nPCs, stoploss, takeprofit))
# Compute predictions on testset

# Compute predictions on validation_set
validation_set = predict_and_backtest_bullish(validation_set, validation_set_final, clf, stoploss, takeprofit, fees, plotting = False)

# Compute recapitulative table
recap = helper.tradingfunctions.table_recap(validation_set, stoploss, takeprofit, nPCs)
recap.to_csv('./Results/Recapitulative_result_stoploss{}_takeprofit{}_{}PCs.csv'.format(stoploss, takeprofit, nPCs), index = False)

# Pick the most profitable
recap = recap.sort_values('ROI%', ascending = False)
recap = recap[recap['nTrades'] > nTrades_mini]
print(recap)
min, max = recap['Min'].iloc[0], recap['Max'].iloc[0]


# (ii) Now that we have identified the best threshold, filter the predictions
# ...Finally : we plot our strategy
testset2 = predict_and_backtest_bullish(testset, testset_final, clf, stoploss, takeprofit, fees, plotting = False)
testset2 = testset2[(testset2['proba1'] > min) & (testset2['proba1'] < max)].copy()

# Now plot our trading strategy
plt.plot(pd.to_datetime(testset2['date']), np.cumsum(testset2['EarningsBullish']))
plt.title('Approach n°2 over the test set \n ROI = {} %'.format(100*np.mean(testset2['EarningsBullish'])))
plt.xlabel('Date')
plt.xlabel('Cumulative Earnings')
plt.show()

# Display the entry points
plt.plot(pd.to_datetime(testset['date']), testset['close'])
plt.scatter(pd.to_datetime(testset2['date']), testset2['close'], c = (testset2['EarningsBullish']>0))
plt.title('Entry points \n Yellow = Win, Blue = Loss')
plt.show()

# (iii) Assess the performance by comparing to if we always traded bullish over the period
a = helper.tradingfunctions.compute_earnings_loss(stoploss, takeprofit, fees)
testset_benchmark = testset.copy()
testset_benchmark['EarningsBullish'] = (testset['result'] == 1)*a[0] + (testset['result'] == 0)*a[1]
avg_return_benchmark = np.mean(testset_benchmark['EarningsBullish'])

# Now let's look at our approach's performance and std
p_value = ttest_1samp(testset2['EarningsBullish'], popmean = avg_return_benchmark)[1]
print('Our model has an average ROI of {} %, while trading blindly bullish over the same period yielded a ROI of {} %, when we perform statistical testing of difference there is a p-value of {}.'.format(100*np.mean(testset2['EarningsBullish']), 100*avg_return_benchmark, p_value))