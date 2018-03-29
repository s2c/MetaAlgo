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

# print("Here")              
try:
    user = kite.generate_session(vals['REQ_TOKEN'], api_secret=vals['API_SECRET'])
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
        # if i > 0: # first  stock for now
        #     break
print(stockList)
buyModels = [] # list of buy models
sellModels = [] # list of sell Models
instTokens = [] # All the instrument tokens
lags = []   #all the lags
historiesPrices = [] # shape = (stock,cost,vols)
historiesVols = []
# print(kite.positions())
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
print(instTokens)
print("Initial Setup Complete") # connected to API
print(lags)


def buyOrd(kiteCli,tSymbol,price,sqVal,stpVal,quant):
    order = kiteCli.place_order(tradingsymbol = tSymbol,
                                    exchange = kite.EXCHANGE_NSE,
                                    quantity = int(quant),
                                    transaction_type = kite.TRANSACTION_TYPE_BUY,
                                    product = kite.PRODUCT_MIS,
                                    order_type =  kite.ORDER_TYPE_LIMIT,
                                    price = float(price),
                                    squareoff = sqVal,
                                    stoploss =  stpVal,
                                    variety = kite.VARIETY_BO,
                                    validity = kite.VALIDITY_DAY,
                                    # disclosed_quantity = int(quant/10)
                                    )
    return order

def sellOrd(kiteCli,tSymbol,price,sqVal,stpVal,quant):
    order = kiteCli.place_order(tradingsymbol = tSymbol,
                                    exchange = kite.EXCHANGE_NSE,
                                    quantity = int(quant),
                                    transaction_type = kite.TRANSACTION_TYPE_SELL,
                                    product = kite.PRODUCT_MIS,
                                    order_type = kite.ORDER_TYPE_LIMIT,
                                    squareoff = sqVal,
                                    stoploss = stpVal,
                                    variety = kite.VARIETY_BO,
                                    price = float(price),
                                    validity = kite.VALIDITY_DAY,
                                    # disclosed_quantity = int(quant/10)
                                    )
    return order

def placeOrder(kiteCli,instToken,bMod,sMod,curStock,lag,spreads):
    """kiteCli = kite client
    historiesPrices = price history
    historiesVols = volume history
    curStock = trading Symbol of current stock
    lag = lag of stock
    spreads = list of [sqVal,stpVal,quant]
    """
    history = kiteCli.historical_data(instToken, 
                                      str(dt.datetime.now().date()),
                                      str(dt.datetime.now().date() + dt.timedelta(days=1)), "minute", continuous=False)

    # print(history)
    curr = history[-lag:]
    historiesClose = np.array([x['close'] for x in curr])
    historiesVols = np.array([x['volume'] for x in curr] )
    historiesOpen = np.array([x['open'] for x in curr] )
    historiesHigh = np.array([x['high'] for x in curr] )
    historiesLow = np.array([x['low'] for x in curr] )
    print(dt.datetime.now(pytz.timezone('Asia/Kolkata')))
    print(historiesClose)
    print(historiesVols)
    sqVal = spreads[0]
    stpVal = spreads[1]
    quant = spreads[2]
    # print(quant)
    bHigh = spreads[3]
    bLow = spreads[4]
    sHigh = spreads[5]
    sLow = spreads[6]
    cont = spreads[7]
    maxHeld = spreads[8]

    close = skp.minmax_scale(historiesClose)
    vols = skp.minmax_scale(historiesVols)
    open = skp.minmax_scale(historiesOpen)
    high = skp.minmax_scale(historiesHigh)
    low = skp.minmax_scale(historiesLow)
    data = np.zeros((1,lag,5))
    data[0,:,0] = close
    data[0,:,1] = open
    data[0,:,2] = high
    data[0,:,3] = low
    data[0,:,4] = vols
    x1 = data[:,:,0].reshape(-1,lag,1)
    # x2 = data[:,:,1].reshape(-1,lag,1)
    # x3 = data[:,:,2].reshape(-1,lag,1)
    # x4 = data[:,:,3].reshape(-1,lag,1)
    x5 = data[:,:,4].reshape(-1,lag,1)
    buyProb = bMod.predict([x1,x1,x5,x5])[0][0] 
    sellProb = sMod.predict([x1,x1,x5,x5])[0][0]
    print("BuyProb = %.2f Sellprob = %.2f" % (buyProb,sellProb))


    if cont == 0: # manual switch if we want to turn this particular scrip off
    	print("Manually Skipping")
    	sleep(1)
    	return
    # print(spreads)
    try:
        positions = kiteCli.positions()['net'] # get already held positions
        for position in positions:
            if position['tradingsymbol'] == curStock:
                held = np.absolute(position['quantity'])
                print("We already hold %d of %s " %(held, curStock))
                if held + quant > maxHeld: # if the new position would go over the max position
                    quant = maxHeld - held # check how much more we can add
                    if quant <= 0: # if we can add only a negative or 0 amount then we can't really add so skip this iteration
                        print("MAX ALREADY HELD")
                        sleep(1)
                        return
    except Exception as e:
        print(e)
        if curStock not in kiteCli.positions()['net']:
            pass
        else:
            raise
    quote = kite.quote("NSE:%s" % curStock)
    # print(quote)
    curClose = quote["NSE:%s" % curStock]['last_price']
    if np.absolute(curClose - history[-1]['close']) > 0.3:
        print("Price differential too great between data and current, skipping analysis")
        sleep(1)

    else:

        if buyProb >= bHigh and sellProb < bLow: # if buy probability is greater than 0.6 Complete
            print("Buyprob greater than %.2f at %.2f and SellProb less than %.2f at %.2f" % (bHigh,buyProb,bLow,sellProb))
            print("BUYING")
            orderId =  buyOrd(kiteCli,curStock,curClose,sqVal,stpVal,quant) # place a buy order

        elif sellProb >= sHigh and buyProb < sLow:
            print("SellProb greater than %.2f at %.2f and Buyprob less than %.2f at %.2f" % (sHigh,sellProb,sLow,buyProb))
            print("SELLING")
            orderId =  sellOrd(kiteCli,curStock,curClose,sqVal,stpVal,quant) # place a sell order
        else:
            print("No probabilities greater than thresholds, skipping")
    sleep(1)
    return



while int(dt.datetime.now(pytz.timezone('Asia/Kolkata')).hour) < 15: # Last order goes in at 2 PM
    t = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
    sleeptime = 60 - (t.second)	
    print(stockList)
    sleep(sleeptime)
    spreadList = pd.read_csv(spreadsFile,header=None).values # Maybe not needed every minute, we'll see
    for i,curStock in enumerate(stockList):
        print(curStock)
        h = placeOrder(kite,instTokens[i],
                   buyModels[i],sellModels[i],curStock,lags[i],spreadList[i])
    print ("_________________________________________________________________")


print("TRADING DAY IS OVER!")
