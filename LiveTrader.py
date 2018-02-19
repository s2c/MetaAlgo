from kiteconnect import KiteConnect
import datetime as dt
from time import sleep
import pytz
import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
import json
from helper import *
from keras.models import load_model
from pythonLib.layer_utils import AttentionLSTM
from pythonLib.helper import *

# Log in to Kite
vals = json.load(open('config.json')) # read the config
kite = KiteConnect(api_key=vals['API_KEY']) # 
print("Here")              
try:
    user = kite.request_access_token(request_token=vals['REQ_TOKEN'],
                                            secret=vals['API_SECRET'])
    kite.set_access_token(user["access_token"])
except Exception as e:
    print("Authentication failed", str(e))
    raise

print(user["user_id"], "has logged in") # connected to API                  

instFile = "instruments.csv" # location of the instrument list for current instruments
curInstList = "tradeList.txt" # location of all instruments currently being traded
spreadsFile = "spreads.txt" 
spreadList = pd.read_csv(spreadsFile,header=None).values
stockList = [] # list

with open (curInstList) as f: #populate list of all current stocks
    i = 0
    for each_csv in f:
        each_csv = each_csv.rstrip('\n') # read csv
        curTicker = each_csv # store ticker
        stockList.append(curTicker)
        i+=1
        if i > 2: # first 3 stocks for now
            break
print("Here")
buyModels = [] # list of buy models
sellModels = [] # list of sell Models
instTokens = [] # All the instrument tokens
lags = []   #all the lags
historiesPrices = [] # shape = (stock,cost,vols)
historiesVols = []

#Load some of the bsaics
for i,curStock in enumerate(stockList):
    # print(curStock)
    buyModel = load_model('modelsFin/%sbuyModel.h5' % curStock, custom_objects={'AttentionLSTM': AttentionLSTM}) # load the buy model
    sellModel = load_model('modelsFin/%ssellModel.h5' % curStock, custom_objects={'AttentionLSTM': AttentionLSTM})
    currLag = int(buyModel.layers[0].input.shape[1]) # get how much lag is being used based on the input to the model
    curInstr = findInstToken(curStock, instFile)
    buyModels.append(buyModel)
    sellModels.append(sellModel)
    instTokens.append(curInstr)
    lags.append(currLag)
    historiesPrices.append(np.zeros(currLag))
    historiesVols.append(np.zeros(currLag))

curVol = np.zeros(len(curStock))
# build basic history
for t in range(0,max(lags) + 1):
    tot = 0 # total time spent
    for i,curStock in enumerate(stockList):
        rt = 0     
        while rt < 5: # 5 retries   
            try:
                if t == 0:  # dont put anything in the first time round cause we are calculating volume manually
                    quote = kite.quote(exchange="NSE",tradingsymbol= curStock )
        #             print(quote['volume'])
                    curVol[i] = quote['volume']
                elif t > lags[i] + 1: # skip this one if it doesnt need any more history
                    continue
                else: # build history like regular
                    quote = kite.quote(exchange="NSE",tradingsymbol= curStock )
                    newVol = quote['volume'] - curVol[i] # calculate the new volume
                    curVol[i] = quote['volume'] # curVol is now the volume retrieved
                    curClose = quote['last_price']
                    historiesPrices[i][t-1] = curClose
                    historiesVols[i][t-1] = newVol
            except :
                print("RETRYING")
                sleep(1) 
                rt += 1
                continue
            tot += rt
            break
        sleep(1)                             
    print("min %d Done" % (t))  
    sleep(57-tot)
historiesPrices = np.array(historiesPrices)
historiesVols = np.array(historiesVols)

def updateHistories(historiesPrices,historiesVols,stockList,curVol,kite):
    tot = 0
    for i,curStock in enumerate(stockList):
        rt = 0 
        while rt < 5:
            try:
                quote = kite.quote(exchange="NSE",tradingsymbol= curStock )
                newVol = quote['volume'] - curVol[i] # calculate the new volume
                curVol[i] = quote['volume'] # curVol is now the volume retrieved
                curClose = quote['last_price']
                historiesPrices[i][0] = curClose
                historiesVols[i][0] = newVol
                historiesPrices[i] = np.roll(historiesPrices[i],-1)
                historiesVols[i] = np.roll(historiesVols[i],-1)
                sleep(1)
            except:
                sleep(1)
                rt += 1
                continue
            break
        tot += rt
    return tot

def buyOrd(kiteCli,tSymbol,price,sqVal,stpVal,quant):
    order = kiteCli.order_place(tradingsymbol = tSymbol,
                                    exchange = "NSE",
                                    quantity = int(quant),
                                    transaction_type = "BUY",
                                    product = "MIS",
                                    order_type = "LIMIT",
                                    price = float(price),
                                    squareoff_value = float(sqVal),
                                    stoploss_value =  float(stpVal),
                                    variety = "bo",
                                    validity = "DAY",
                                    disclosed_quantity = int(quant/10))
    return order

def sellOrd(kiteCli,tSymbol,price,sqVal,stpVal,quant):
    order = kiteCli.order_place(tradingsymbol = tSymbol,
                                    exchange = "NSE",
                                    quantity = int(quant),
                                    transaction_type = "SELL",
                                    product = "MIS",
                                    order_type = "LIMIT",
                                    squareoff_value = float(sqVal),
                                    stoploss_value = float(stpVal),
                                    variety = "bo",
                                    price = float(price),
                                    validity = "DAY",
                                    disclosed_quantity = int(quant/10))
    return order

def placeOrder(kiteCli,historiesPrices,historiesVols,bMod,sMod,curStock,lag,spreads):
    """kiteCli = kite client
    historiesPrices = price history
    historiesVols = volume history
    curStock = trading Symbol of current stock
    lag = lag of stock
    spreads = list of [sqVal,stpVal,quant]
    """
    close = skp.minmax_scale(historiesPrices)
    vols = skp.minmax_scale(historiesVols)
    data = np.zeros((1,lag,2))
    data[0,:,0] = close
    data[0,:,1] = vols
    buyProb = bMod.predict([data,data])[0][0] 
    sellProb = sMod.predict([data,data])[0][0]
    sqVal = spreads[0]
    stpVal = spreads[1]
    quant = spreads[2]
    bHigh = spreads[3]
    bLow = spreads[4]
    sHigh = spreads[5]
    sLow = spreads[6]
    print("BuyProb = %.2f Sellprob = %.2f" % (buyProb,sellProb))
    if buyProb > bHigh and sellProb < bLow: # if buy probability is greater than 0.6
        print("Buyprob greater than %.2f at %.2f and SellProb less than %.2f at %.2f" % (bHigh,buyProb,bLow,sellProb))
        print("Buying")
        orderId =  buyOrd(kiteCli,curStock,historiesPrices[-1],sqVal,stpVal,quant) # place a buy order

    elif sellProb > sHigh and buyProb < sLow:
        print("SellProb greater than %.2f at %.2f and Buyprob less than %.2f at %.2f" % (sHigh,sellProb,sLow,buyProb))
        print("Selling  ")
        orderId =  sellOrd(kiteCli,curStock,historiesPrices[-1],sqVal,stpVal,quant) # place a sell order
    else:
        print("No probabilities greater than thresholds, skipping")
    
#     return waitT



while int(dt.datetime.now(pytz.timezone('Asia/Kolkata')).hour) < 15: # Last order goes in at 2 PM
    spreadList = pd.read_csv(spreadsFile,header=None).values # Maybe not needed every minute, we'll see
    for i,curStock in enumerate(stockList):
        print(curStock)
        print(historiesPrices[i])
        print(historiesVols[i])
        placeOrder(kite,historiesPrices[i],historiesVols[i],
                   buyModels[i],sellModels[i],curStock,lags[i],spreadList[i])
    t = updateHistories(historiesPrices,historiesVols,stockList,curVol,kite)
    sleep(57-t) # sleep for 60 - whatever time w was running for seconds

print("TRADING DAY IS OVER!")
