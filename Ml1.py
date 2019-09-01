import pandas as pd
import numpy as np
import quandl, math #math for ceil and quandl for taking dataset of googl alphabet dataset on us stocks
from sklearn import preprocessing#preprocessing is used for scaling the data
#from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression#Using Linear Regression from sklearn
df=quandl.get('WIKI/GOOGL')
print(df.head(10))
#choosing the specific data 
#adjustment data is the one which the company specifies on seeing the sale of their stocks 
df=df[['Adj. Open','Adj. Low','Adj. High','Adj. Close','Adj. Volume']]
#making a new column in which we are taking percent of high stock price and closing price of that day
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
#percentage change of opening and closing data 
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
df=df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume',]]
print(df.head())
#duplicating the adj close in forecast col
forecast_col='Adj. Close'
#fill all the empty data with -9999
df.fillna(-99999,inplace=True)
#now this will ceil all the data + takes the 10 % of data and train the machine
forecast_out=int(math.ceil(0.01*len(df)))
print(forecast_out)
#now this basically is shifting the labelled data upwards cuz we used 10 per of data for training
df['label']=df[forecast_col].shift(-forecast_out)
#again dropping all the 0 data or data which is not complete
df.dropna(inplace=True)
print(df.head())
#this is testing and training data on the basis of what we sepeared the 10 per of our data earlier. 
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

#train and test the data using train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#no of jobs specify how many jobs or threads it will take at a time -1 is maximum job it can
clf=LinearRegression(n_jobs=2)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)

#prediction of the 10 per data which we used earlier to forecast it. 
forecast_set=clf.predict(X_lately)
print(forecast_set,accuracy,forecast_out)