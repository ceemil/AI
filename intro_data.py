import pandas as pd
import quandl

#get data from quandl
df = quandl.get('WIKI/GOOGL')

#print heads to see how the data looks like
print(df.head())

#select data that we need
df = df[['Adj. High','Adj. Low','Adj. Open','Adj. Close','Adj. Volume',]]

#create 2 new fields (HL_PCT high procent; PCT_change procent change)
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /df['Adj. Close'] * 100.0
df['HL_change'] = (df['Adj. Close'] - df['Adj. Open']) /df['Adj. Open'] * 100.0

#create new set with the new fields
df = df[['Adj. Close','HL_PCT','HL_change','Adj. Volume',]]

#print to see how the new model looks like
print(df.head())
