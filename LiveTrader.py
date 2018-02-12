
# coding: utf-8

# In[1]:


from kiteconnect import KiteConnect
import datetime as dt
from time import sleep
import pytz
import numpy as np
import json
import paho.mqtt.client as mqtt
import sklearn.preprocessing as skp
from keras.models import load_model
from pythonLib.layer_utils import AttentionLSTM						


# In[2]:


lastPrice = "" # global variable that is updated with the last price every minute
buyModel = load_model('modelsFin/buyModel.h5', custom_objects={'AttentionLSTM': AttentionLSTM}) # load the buy model
sellModel = load_model('modelsFin/sellModel.h5', custom_objects={'AttentionLSTM': AttentionLSTM})
lag = int(buyModel.layers[0].input.shape[1]) # get how much lag is being used based on the input to the model
hist = np.zeros(lag)
curInstr = 2933761


vals = json.load(open('config.json'))
kite = KiteConnect(api_key=vals['API_KEY'])
                   
try:
    user = kite.request_access_token(request_token=vals['REQ_TOKEN'],
                                            secret=vals['API_SECRET'])
    kite.set_access_token(user["access_token"])
except Exception as e:
    print("Authentication failed", str(e))
    raise
                   
print(user["user_id"], "has logged in")

# sleep(60*15)

# In[3]:


def on_connect(client, userdata, flags, rc): 
    client.subscribe("priceData")

def on_disconnect(client, userdata, rc):
    if rc != 0 :
        try:
            client.reconnect()
        except:
            print("Couldn't Reconnect")


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, message):
    global lastPrice
    lastPrice = message # update lastPrice
    client.loop_stop()

def updateLastPrice():
    client = mqtt.Client() # connect
    client.on_connect = on_connect # print connected callback
    client.on_message = on_message # update message callback
    client.connect("localhost", 1883, 60)
    client.loop_start()
    sleep(0.1)


# In[4]:


# To start off build the first 30 minutes of history
print("Building first lag minute history")
for i in range(0,lag):
    sleep(60) # sleep for 60 seconds so we get the price after a minute 
    updateLastPrice()
    x = json.loads(lastPrice.payload.decode('utf-8'))
    if x['instrument_token'] == curInstr and x['tradeable'] == True:
        hist[i] = x['last_price']
    else:
        print("ERROR, ERROR, CONTACT SUPERVISOR")
        
    print("min %d Done" % (i+1))

print ("History built")
print(hist)

# In[16]:


def placeOrder(kiteCli,hist,bMod,sMod,tSymbol):
    print(hist)
    histScaled = skp.scale(hist)
    histScaled = histScaled.reshape(1,-1,1) # create scaled version for keras
    # print(np.array(histScaled).shape)
    waitT = 0# wait for it to complete       
    buyProb = bMod.predict([histScaled,histScaled])[0][0] 
    sellProb = sMod.predict([histScaled,histScaled])[0][0]
    print("BuyProb = %.2f Sellprob = %.2f" % (buyProb,sellProb))
    if buyProb > 0.6 and sellProb < 0.45: # if buy probability is greater than 0.6
        print("Buyprob greater than 0.6 at %.2f" % buyProb)
        print("Buying")
        orderId =  buyOrd(kiteCli,tSymbol,hist[-1],30000) # place a buy order
        # orderId = sellOrd(kiteClimtSymbol,hist[-1]+0.1,300)
        # while ((kite.orders(orderId)[-1]['status']) != "COMPLETE") and waitT < 30: # wait upto 30 seconds
        #     sleep(1)
        #     waitT += 1
        # if kite.orders(orderID)[-1]['status'] =="COMPLETE" : # when completed
        #     print("Bracket Buy Placed successfully")     
    elif sellProb > 0.6 and buyProb < 0.45:
        print ("Sellprob greater than 0.6 at %.2f" % sellProb)
        print("Selling  ")
        orderId =  sellOrd(kiteCli,tSymbol,hist[-1],30000) # place a sell order
        # orderId = buyOrd(kiteClimtSymbol,hist[-1]-0.1,300)
        # while ((kite.orders(orderId)[-1]['status']) != "COMPLETE") and waitT < 30: # wait upto 30 seconds
        #     sleep(1)
        #     waitT += 1
        # if kite.orders(orderID)[-1]['status'] =="COMPLETE" : # when completed
        #     print("Bracket sell completed succesfully")
    else:
        print("No probabilities greater than thresholds, skipping")
    
    return waitT
    
    


# In[17]:


def buyOrd(kiteCli,tSymbol,price,quant):
    order = kiteCli.order_place(tradingsymbol = tSymbol,
                                    exchange = "NSE",
                                    quantity = quant,
                                    transaction_type = "BUY",
                                    product = "MIS",
                                    order_type = "LIMIT",
                                    price = price,
                                    squareoff_value = 0.1,
                                    stoploss_value =  0.2,
                                    variety = "bo",
                                    validity = "DAY",
                                    disclosed_quantity = int(quant/10))
    return order

def sellOrd(kiteCli,tSymbol,price,quant):
    order = kiteCli.order_place(tradingsymbol = tSymbol,
                                    exchange = "NSE",
                                    quantity = quant,
                                    transaction_type = "SELL",
                                    product = "MIS",
                                    order_type = "LIMIT",
                                    squareoff_value = 0.1,
                                    stoploss_value = 0.2,
                                    variety = "bo",
                                    price = price,
                                    validity = "DAY",
                                    disclosed_quantity = int(quant/10))
    return order


# In[20]:



tSymbol = "GMRINFRA"
print("Starting Trading Engine")
while int(dt.datetime.now(pytz.timezone('Asia/Kolkata')).hour) < 15: # Last order goes in at 2 PM
    w = placeOrder(kite,hist,buyModel,sellModel,tSymbol)
    sleep(60-w) # sleep for 60 - whatever time w was running for seconds
    updateLastPrice() # update the last price
    cur = json.loads(lastPrice.payload.decode('utf-8'))['last_price'] # get the latest price
    # noise = np.random.normal(0,0.05,lag)
    hist[0] = cur      # replace oldest price with newes
    hist = np.roll(hist,-1) # left shift the array so newest price is in front
    # hist = hist + noise


print ("TRADE DAY IS OVER")


# In[ ]:




