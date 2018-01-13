
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


# In[2]:


lastPrice = "" # global variable that is updated with the last price every minute
buyModel = load_model('buyModel.h5') # load the buy model
sellModel = load_model('sellModel.h5') # load the sell model
lag = int(buyModel.layers[0].input.shape[1]) # get how much lag is being used based on the input to the model
hist = np.zeros(lag)
curInstr = 3076609


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
print("Building first 30 minute history")
for i in range(0,30):
    sleep(60) # sleep for 60 seconds so we get the price after a minute 
    updateLastPrice()
    x = json.loads(lastPrice.payload.decode('utf-8'))
    if x['instrument_token'] == curInstr and x['tradeable'] == True:
        hist[i] = x['last_price']
    else:
        print("ERROR, ERROR, CONTACT SUPERVISOR")
        
    print("min %d Done" % i)

print ("History built")
print(hist)


# In[16]:


def placeOrder(kiteCli,hist,bMod,sMod,tSymbol):
    histScaled = skp.scale(hist)
    histScaled = histScaled.reshape(1,-1,1) # create scaled version for keras
    waitT = 0# wait for it to complete       
    if bMod.predict([histScaled,histScaled])[0][0] > 0.5: # if buy probability is greater than 0.5
        print("Buying")
        orderId =  buyOrd(kiteCli,tSymbol,hist[-1],10) # place a buy order
        while ((kite.orders(orderId)[-1]['status']) != "COMPLETE") and waitT < 30: # wait upto 30 seconds
            sleep(1)
            waitT += 1
        # if kite.orders(orderID)[-1]['status'] =="COMPLETE" : # when completed
        #     print("Bracket Buy Placed successfully")
        
    elif sMod.predict([histScaled,histScaled])[0][0] > 0.5:
        print("Selling")
        orderId =  sellOrd(kiteCli,tSymbol,hist[-1],10) # place a sell order
        while ((kite.orders(orderId)[-1]['status']) != "COMPLETE") and waitT < 30: # wait upto 30 seconds
            sleep(1)
            waitT += 1
        # if kite.orders(orderID)[-1]['status'] =="COMPLETE" : # when completed
        #     print("Bracket sell completed succesfully")
    
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
                                    squareoff_value = price + 0.1,
                                    stoploss_value = price - 0.1,
                                    validity = "DAY")
    return order

def sellOrd(kiteCli,hist,tSymbol,price,quant):
    order = kiteCli.order_place(tradingsymbol = tSymbol,
                                    exchange = "NSE",
                                    quantity = 1,
                                    transaction_type = "SELL",
                                    product = "MIS",
                                    order_type = "LIMIT",
                                    squareoff_value = price - 0.1,
                                    stoploss_value = price + 0.1,
                                    price = price,
                                    validity = "DAY")
    return order


# In[20]:



tSymbol = "SUZLON"
print("Starting Trading Engine")
while int(dt.datetime.now(pytz.timezone('Asia/Kolkata')).hour) < 15:
    w = placeOrder(kite,hist,buyModel,sellModel,tSymbol)
    sleep(60-w) # sleep for 60 - whatever time w was running for
    updateLastPrice() # update the last price
    cur = json.loads(lastPrice.payload.decode('utf-8'))['last_price'] # get the latest price
    hist[0] = cur      # replace oldest price with newest
    hist = np.roll(hist,-1) # left shift the array

print ("TRADE DAY IS OVER")


# In[ ]:




