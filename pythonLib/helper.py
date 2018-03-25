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
    dataInit = pd.read_csv(filename,
                    names=["datetime","open","high","low","close","volume"],
                    dtype=None,
                    delimiter = ',',
                    converters = {0:convertfunc},
                  #  index_col = 0
                   )
    # dataInit = dataInit.dropna(axis=0,how='any')
    return dataInit

# Making sure that 2 timeseries are synced to the smaller time series
# Goes through 2 timeseries and eliminates data which are not present on the same date on both the timeseries
# Input: TimeSeries 1, TimeSeries 2
# Output: synced TimeSeries
def sycTimeSeries(ts1,ts2):
    for dt in ts1["DateTime"].values: # clean ts1
        if dt in ts2['DateTime'].values:
            continue
        else:
            #print(dt)
            ts1.drop(ts1[ts1["DateTime"]==dt].index,inplace = True)
    for dt in ts2["DateTime"].values: # clean ts2
        if dt in ts1['DateTime'].values:
            continue
        else:
            #print(dt)
            ts2.drop(ts1[ts1["DateTime"]==dt].index,inplace = True)

    return ts1.reset_index(drop = True), ts2.reset_index(drop = True)
    

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
# atleast is how many of the lookahead need to be atleast the same or greater than flat+rat
# Input: lagged pandas DataFrame, uint lag, double dif, double flat, double atleast between 0 and 1
# Output : Pandas Dataframe with last column binarized

def binarizeTime(resLagged,rate = 0,lookahead = 0, flat = 0,atleast = 0.5):
    if lookahead <= 0 :
        raise Exception("lookahead Must be 1 or higher!")
    resLagged = resLagged.copy() # Make a deep copy
    last = np.shape(resLagged)[1] # find the length of the data 
    last = last-lookahead # convert it to string for loc
    colsLookAhead = list(resLagged.loc[:,str(last+1):str(last + lookahead)])
    colsLast = resLagged[str(last)]
    diffs = resLagged[colsLookAhead].subtract(colsLast,axis=0)
#     print(diffs)
    greater = diffs>=flat  # all the times the price changed higer than flat
    greater = np.count_nonzero(greater,axis=1).reshape((1,-1))
    lesser = diffs<=-flat # all the times the price fell lower than fat
    lesser = np.count_nonzero(lesser,axis=1).reshape((1,-1))
#     return greater,lesser
#     print(greater)
    greater = greater.reshape(1,-1)
    changeToBuy = np.any(greater > lesser & np.greater(greater,atleast*lookahead),axis=0) # make sure more rises than falls and atleast half rises
    changeToSell = np.any(lesser > greater & np.greater(lesser,atleast*lookahead),axis=0)      # make sure more falls than rises and atleast half rises
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
