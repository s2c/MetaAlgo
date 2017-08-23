#Setup
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from helper import *

# Reads data from Zerodha API historical data files and returns a Pandas DataFrame
# Input: Zerodha Api CSV file
# Return: Pandas Dataframe of CSV file with correct timezone
def readData(filename):
    convertfunc = lambda x: (pd.to_datetime(x,utc=True)).tz_convert('Asia/Kolkata')
    return pd.read_csv(filename,
                    names=["datetime","open","high","low","close","volume"],
                    dtype=None,
                    delimiter = ',',
                    converters = {0:convertfunc},
                  #  index_col = 0
                   )

# Making sure that 2 timeseries are synced to the smaller time series
# Goes through 2 timeseries and eliminates any data from any date that is not in both
# Input: TimeSeries 1, TimeSeries 2
# Output: synced TimeSeries
def sycTimeSeries(ts1,ts2):
    # If TS1 is not bigger, then make TS1 the bigger one and TS2 the smaller one.
    flipped = 0
    if len(ts2) > len(ts1):
        flipped = 1
        ts1,ts2 = ts2,ts1
    for dt in ts1["DateTime"].values:
        if dt in ts2['DateTime'].values:
            continue
        else:
            #print(dt)
            ts1.drop(ts1[ts1["DateTime"]==dt].index,inplace = True)
    if flipped:
        return ts2, ts1.reset_index(drop = True)
    else:
        return ts1.reset_index(drop = True), ts2
    

#Creates Lagged series
#Goes through a series and generates an lag+1 dimensional pandas DataFrame that has each previous lag timeunit
#as a column and current as the last column
#Input: Pandas Series
#Output: lag+1 dimensional DataFrame

def timeseriesLagged(data, lag=60):
    df = data
    columns = [df.shift(i) for i in range(1, lag+2)] 
    df = pd.concat(columns,axis=1)
    df.fillna(0, inplace=True)
    df.columns = [str(lag+2-x) for x in range(1,lag+2)]
    df = df[df.columns[::-1]] #Flip because we want newer data on the right
    df= df.iloc[lag+1:] # drop the first 'lag' columns because zeroes.
    df.reset_index(drop=True,inplace=True)
    return df


# Binarizes the last column into 1 or 0.
# dif is the minimum difference between buy and sell that triggers a 1 or a 0. 
# Rate is the percent per transasction cost 
# lag is the time unit of lag. 
# Input: lagged pandas DataFrame, uint lag, double dif, double flat
# Output : Pandas Dataframe with last column binarized
def binarizeTime(series,lag,dif=0,rate=0.01):
    #-1 is autocalculate the dif 
    if dif != 0:
        raise AssertionError("dif not yet baked in! ")
    series[str(lag+1)] = np.where(series[str(lag)] + dif < series[str(lag+1)], 1, 0)
    return series

# Finds the right lag given a target correlation.
# data is the time series
# targetCorr is the targetCorr
# Inputs: Pandas Series, float targetCorr between -1 and 1
# Outputs: optimal lag 
def findLag(data, targetCorr):
    if targetCorr > 1 or targetCorr < -1:
        raise ValueError("targetCorr must be between -1 and 1!")
    lag = 0
    for i in range(1, len(data)):
        curCorr = data.autocorr(i)
        if curCorr < targetCorr:
            lag = i-1
            break
    return lag
