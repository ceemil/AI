import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #cross_validation
from sklearn.model_selection import cross_val_score
#from sklearn import preprocessing, svm
#cross_validation can't be found

#get data from quandl
df = quandl.get('WIKI/GOOGL')

#print heads to see how the data looks like
#print(df.head())

#select data that we need
# they are called feaures
df = df[['Adj. High','Adj. Low','Adj. Open','Adj. Close','Adj. Volume',]]

#create 2 new fields (HL_PCT high procent; PCT_change procent change)
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /df['Adj. Close'] * 100.0
df['HL_change'] = (df['Adj. Close'] - df['Adj. Open']) /df['Adj. Open'] * 100.0

#create new set with the new fields
df = df[['Adj. Close','HL_PCT','HL_change','Adj. Volume',]]

#print to see how the new model looks like
#print(df.head())

#XXXXto fill comemnt when he mentions what the fuck this is
#what is the field that we are looking for
forcast_col = 'Adj. Close'

#fill out missing data with -99999
df.fillna(-99999, inplace=True)

#this will be the timeframe for the forcast (nr of days)
#0.1 is the timeline len definer (not exact days)
#ceil - Floor and ceiling functions
forcast_out = int(math.ceil(0.01*len(df)))

print ("forcast into the future: " + str(forcast_out))

#labers are the estimations that we are trying to do
#shift https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html
#we are shifting estimation by forcast_out interval  forcast_out = int(math.ceil(0.1*len(df)))
#add extra elements
df['label'] = df[forcast_col].shift(-forcast_out)

#lowering the number accuracy
df.dropna(inplace=True)

#print(df.head())

#show tail of data
#print(df.tail())

#features are #
#everything is a feature except the label
x = np.array(df.drop(['label'],1))
#our labels
y = np.array(df['label'])
#scaling the data
#may cost processing time
x = preprocessing.scale(x)
y = np.array(df['label'])

#creating the testing and training sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#creating a classifier
#n_jobs run multi instances; -1 run max
clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print("prediction accuracy for linear regression: "+ str(accuracy))


clf = svm.SVR(gamma='auto')
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print("prediction accuracy for SVM SRV: "+ str(accuracy))
