#Setup
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from helper import *



# Returns the instrument token given a trading symbol [IN NSE] from the instruments.csv
# Input: stockName (trading symbol) , instFile(Zerodha instruments list) 
# Return: integer instrument token

def findInstToken(stockName, instFile):
    inst =  pd.read_csv(instFile) # read the file
    tokenCrit = inst['segment'] == "NSE" # filter out only nse
    actStock = inst['tradingsymbol'] == stockName  #filter out the trading symbol

    return int(inst[tokenCrit & actStock]['instrument_token'].values[0]) #return

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
# Goes through 2 timeseries and eliminates data which are not present on the same date on both the timeseries
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
#as a column and current as the last cobilumn
#Input: Pandas Series
#Output: lag+1 dimensional DataFrame

def timeseriesLagged(data, lag=60):
    df = data
    columns = [df.shift(i) for i in range(1, lag+2)] 
    df = pd.concat(columns,axis=1)
    df.fillna(0, inplace=True)
    df.columns = [str(lag+2-x) for x in range(1,lag+2)]
    # df.reset_index(inplace=True,drop=False)
    df = df[df.columns[::-1]] #Flip because we want newer data on the right
    df= df.iloc[lag+1:] # drop the first 'lag' columns because zeroes.
    df.reset_index(drop=True,inplace=True)
    return df


# Binarizes the last column into 1, 0, -1. 1 = buy 0 = do nothing -1 = sell
# Rate is the percent increase or decrease that should trigger a buy or a sell
# lag is the time unit of lag. 
# Input: lagged pandas DataFrame, uint lag, double dif, double flat
# Output : Pandas Dataframe with last column binarized

def binarizeTime(resLagged,rate = 0,lookahead = 0, flat = 0):
    if lookahead <= 0 :
        raise Exception("lookahead Must be 1 or higher!")
    resLagged = resLagged.copy() # Make a deep copy
    last = np.shape(resLagged)[1] # find the length of the data 
    last = last-lookahead # convert it to string for loc
    colsLookAhead = list(resLagged.loc[:,str(last+1):str(last + lookahead)])
    colsLast = resLagged[str(last)]
    diffs = resLagged[colsLookAhead].subtract(colsLast,axis=0)
    changeToBuy = np.any(diffs >= flat,axis=1)
    changeToSell = np.any(diffs <= -flat,axis=1)
    changeToHold = ~changeToBuy & ~changeToSell
    resLagged = resLagged.drop(colsLookAhead,1)
    resLagged.loc[changeToSell,str(last+1)] = -1 # Set sell to -1
    resLagged.loc[changeToBuy,str(last+1)] = 1 # Set buy to 1
    resLagged.loc[changeToHold,str(last+1)] = 0 # Set to 0
    return resLagged

# Finds the right lag given a target correlation.
# data is the time series
# targetCorr is the targetCorr
# Supressed: Supresses message about lag being greater than 99, if a lag of above 99 is about to be used.
# Inputs: Pandas Series, float targetCorr between -1 and 1
# Outputs: lag that matches the targetCorr, limited to a max of 100
def findLag(data, targetCorr,suppressed=True):
    if targetCorr > 1 or targetCorr < -1:
        raise ValueError("targetCorr must be between -1 and 1!")
    lag = 0
    for i in range(1, len(data)):
        if i >= 99:
            if suppressed != True:
                print("GREATER THAN 99,returning 99") 
            return i
        curCorr = data.autocorr(i)
        if curCorr < targetCorr:
            lag = i-1
            break
    return lag
