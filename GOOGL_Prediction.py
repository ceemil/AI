import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #cross_validation
from sklearn.model_selection import cross_val_score
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
#from sklearn import preprocessing, svm
#cross_validation can't be found

#use ploting style
style.use('ggplot')

#get data from quandl
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. High','Adj. Low','Adj. Open','Adj. Close','Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /df['Adj. Close'] * 100.0
df['HL_change'] = (df['Adj. Close'] - df['Adj. Open']) /df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','HL_change','Adj. Volume',]]

forcast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forcast_out = int(math.ceil(0.01*len(df)))

print ("forcast into the future: " + str(forcast_out))

df['label'] = df[forcast_col].shift(-forcast_out)


x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x = x[:-forcast_out]
x_lately = x[-forcast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
#y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print("prediction accuracy for linear regression: "+ str(accuracy))

forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy, forcast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
