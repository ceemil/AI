import pandas as pd
import quandl
import math

#get data from quandl
df = quandl.get('WIKI/GOOGL')

#print heads to see how the data looks like
print(df.head())

#select data that we need
# they are called feaures
df = df[['Adj. High','Adj. Low','Adj. Open','Adj. Close','Adj. Volume',]]

#create 2 new fields (HL_PCT high procent; PCT_change procent change)
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /df['Adj. Close'] * 100.0
df['HL_change'] = (df['Adj. Close'] - df['Adj. Open']) /df['Adj. Open'] * 100.0

#create new set with the new fields
df = df[['Adj. Close','HL_PCT','HL_change','Adj. Volume',]]

#print to see how the new model looks like
print(df.head())

#XXXXto fill comemnt when he mentions what the fuck this is
#what is the field that we are looking for
forcast_col = 'Adj. Close'

#fill out missing data with -99999
df.fillna(-99999, inplace=True)

#this will be the timeframe for the forcast (nr of days)
#0.1 is the timeline len definer (not exact days)
#ceil - Floor and ceiling functions
forcast_out = int(math.ceil(0.1*len(df)))

#labers are the estimations that we are trying to do
#shift https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shift.html
#we are shifting estimation by forcast_out interval  forcast_out = int(math.ceil(0.1*len(df)))
df['label'] = df[forcast_col].shift(-forcast_out)

#lowering the number accuracy
df.dropna(inplace=True)

print(df.head())

#show tail of data
#print(df.tail())
