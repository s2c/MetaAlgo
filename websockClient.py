
# coding: utf-8

# In[8]:


from kiteconnect import WebSocket
import threading
from time import sleep
import json
import paho.mqtt.client as mqtt


# In[5]:


vals = json.load(open('config.json'))


# In[14]:


# def on_connect(client, userdata, flags, rc):
#     print("Connected with result code "+str(rc))

#     # Subscribing in on_connect() means that if we lose the connection and
#     # reconnect then subscriptions will be renewed.
#     client.subscribe("priceData")

# # The callback for when a PUBLISH message is received from the server.
# def on_message(client, userdata, msg):
#     print(msg.topic+" "+str(msg.payload))



# In[ ]:



kws = WebSocket(vals['API_KEY'],vals['API_SECRET'],vals['USER_ID'])
client = mqtt.Client()
client.connect("localhost", 1883, 60)
# Callback for tick reception.


def on_tick(tick, ws):
#     print (type(len(tick)))
    jsonString = json.dumps(tick[0])
    client.publish("priceData",jsonString,retain=True) 

# Callback for successful connection.
def on_connect(ws):
        # Subscribe to a list of instrument_tokens (GMRINFRA).
        ws.subscribe([3463169])

        # Set Suzlon to tick in `full` mode.
        ws.set_mode(ws.MODE_LTP, [3463169])
        
    # Assign the callbacks.
kws.on_tick = on_tick
kws.on_connect = on_connect

# To enable auto reconnect WebSocket connection in case of network failure
# - First param is interval between reconnection attempts in seconds.
# Callback `on_reconnect` is triggered on every reconnection attempt. (Default interval is 5 seconds)
# - Second param is maximum number of retries before the program exits triggering `on_noreconnect` calback. (Defaults to 50 attempts)
# Note that you can also enable auto reconnection        while initialising websocket.
# Example `kws = WebSocket("your_api_key", "your_public_token", "logged_in_user_id", reconnect=True, reconnect_interval=5, reconnect_tries=50)`
kws.enable_reconnect(reconnect_interval=5, reconnect_tries=50)

# Infinite loop on the main thread. Nothing after this will run.
# You have to use the pre-defined callbacks to manage subscriptions.
kws.connect()
# print('hi')


# In[ ]:




