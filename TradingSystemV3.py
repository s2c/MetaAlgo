
# coding: utf-8

# ## Importing basic libraries

# In[868]:


#Setup
# get_ipython().magic('matplotlib inline')
# get_ipython().magic('config IPCompleter.greedy=True')
import datetime as dt
import pytz
import time
import os
import psycopg2
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from pythonLib.helper import *
import sqlalchemy
import backtrader.plot as pLaut

from keras.models import Sequential,Model
from keras.layers import Activation,Dense,LSTM, Dropout,Conv1D,MaxPooling1D,Permute,Merge,Input
from keras.layers import Flatten,BatchNormalization,LeakyReLU,GlobalAveragePooling1D,concatenate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.losses import binary_crossentropy
from keras.optimizers import SGD,Adam
from keras.models import load_model
from pythonLib.layer_utils import AttentionLSTM

import h5py

from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn.preprocessing as skp
import tensorflow as tf
import tempfile

tf.__version__
DATA_DIR = 'data' 
# np.random.seed(seed)
dbString = 'postgresql://s2c:JANver95@localhost:5432/stockdata'
curInstList = 'tradeList.txt'
engine = sqlalchemy.create_engine(dbString) 
stockList = []
with open (curInstList) as f:
    for each_csv in f:
        each_csv = each_csv.rstrip('\n') # read csv
        curTicker = each_csv # store ticker
        stockList.append(curTicker)
cur = 0
stockList
utc = pytz.UTC
starDate = utc.localize(dt.datetime(2014,3,8))
endDate = utc.localize(dt.datetime(2018,1,20))
portVals = []
TransVals = []
startCash = 1000000
size = 15000
curIter = 0
drop = 0.1

while curIter==0 or (startDate.year == 2016) or (startDate.year == 2014) or (startDate.year == 2015) or (endDate.year == 2018 and endDate.month == 1 and endDate.day <= 6) :
    query = "SELECT * FROM histdata WHERE ticker = 'GMRINFRA' ORDER BY datetime ASC"
    dat = pd.read_sql(query,engine)


    startDate = starDate + dt.timedelta(days=7*curIter)
    endDate = startDate +dt.timedelta(days = 365*3 + 7*4*6) # 3 years
    print("backTestStart :")
    backTestStart = endDate
    print(backTestStart)
    backTestEnd = endDate + dt.timedelta(days=7*1*1)
    res = dat[(dat['datetime'] > startDate) & (dat['datetime'] < endDate)]
    curIter += 1

    vol = res['volume']

    # # Setup Parameters
    dataInit = res # Read the stock price data. This is 1 minute data
    data = dataInit['close'] # extract the 'close' column as a Pandas series
    # plt.figure()
    # pd.tools.plotting.lag_plot(data) # Lag plot to check randomness
    # plt.figure()
    # pd.tools.plotting.autocorrelation_plot(data) # Auto correlation plot to check if series is autocorrelated at all

    # # Find the right lag manually
    # targetCorr = 0.99 # autocorrelation we want
    # lag = findLag(data,targetCorr,True) # Lag that is indicative 
    # if lag == 99: #if lag is 99 then we can just use any number above it as autocorrelation is guaranteed.
    #     lag = 120 #nice round 2  hour intervals
    # print(lag)
    lag = 45
    lookahead = 15
    flat = 0.1
    series = timeseriesLagged(data,lag + lookahead-1) # Generate the lagged series
    vols = timeseriesLagged(vol,lag + lookahead-1)
    # res.tail(10)


        # generate the series for volumes. We need to drop the last column at some point as it is irrelevant.
    volsSeries = binarizeTime(vols,0,lookahead = lookahead, flat= flat)
    volsSeries = volsSeries.drop(str(lag+1),axis=1)
    #standardize
    volsSeries = skp.minmax_scale(volsSeries,axis=1)
    # volsSeries[0,:]


    # In[926]:


    # Create binary series where 0 = hold and 1 = buy
    # Create binary series where 0 = hold and 1 = buy
    buySeries = binarizeTime(series,0,lookahead = lookahead, flat= flat)
    change = buySeries.iloc[:,-1]== -1 # convert to binary
    buySeries.loc[change,str(lag+1)]=0 # convert to binary
                                       # clean up post binary

    buySeriesLabs = buySeries[str(lag+1)] # labels
    buySeriesFeats = buySeries.drop(str(lag+1),axis=1) #features
    buySeriesFeats = buySeriesFeats.values
    # stanardize
    buySeriesFeats = skp.minmax_scale(buySeriesFeats,axis=1)


    # Convert the data into a suitable format

    buySeries = np.zeros((len(volsSeries),lag,2))
    buySeries[:,:,0] = buySeriesFeats
    buySeries[:,:,1] = volsSeries
    # buySeries[0,:,1]


    # In[927]:
 

     # Create binary series where 0 = hold and 1 = sell
    sellSeries = binarizeTime(series,0,lookahead=lookahead,flat=flat)
    change = sellSeries.iloc[:,-1]== 1 # find 1s and convert to 0
    sellSeries.loc[change,str(lag+1)]=0 # 
    change = sellSeries.iloc[:,-1]== -1 # find -1 and conver to 1s
    sellSeries.loc[change,str(lag+1)]= 1 # convert to
                                         # cleanup post binary

    # Convert the data into a suitable format
    sellSeriesLabs = sellSeries[str(lag+1)]
    sellSeriesFeats = sellSeries.drop(str(lag+1),axis=1)


    # stanardize
    sellSeriesFeats = skp.minmax_scale(sellSeriesFeats,axis=1)


    sellSeries = np.zeros((len(volsSeries),lag,2))
    sellSeries[:,:,0] = sellSeriesFeats
    sellSeries[:,:,1] = volsSeries

    # # Generate Training Data
    # 
    # Now that we have an idea of what's going on in the dataset, it is a good time to generate training data. We do an 90:20 training:testing split, and then we randomize the training set because we assume that only the last LAG minutes matter

    # In[928]:



    # In[929]:

    # Get values from pandas series as we need a numpy array for our classifier
    x,y = shuffle(buySeries,buySeriesLabs)
    tot = len(x)
    y = y.values
    trainPercent = 0.8 # majority of data used for training
    testPercent = 0.95 # 
    valPercent = 1.00  #

    # Test Train Val Split

    xTrain = x[0:int(trainPercent*tot),:,:]
    yTrain = y[0:int(trainPercent*tot)]

    xTest = x[int(trainPercent*tot): int(testPercent*tot),:,:]
    yTest = y[int(trainPercent*tot): int(testPercent*tot)]

    xVal = x[int(testPercent*tot):,:,:]
    yVal = y[int(testPercent*tot):]

    # # # encode class values as integers
    # encoder = LabelEncoder()
    # encoder.fit(yTrain)
    # encodedyTrain = encoder.transform(yTrain)
    # encodedyTest = encoder.transform(yTest)
    # encodedyVal = encoder.transform(yVal)
    # # convert integers to one hot encoded
    # yTrain = np_utils.to_categorical(encodedyTrain)
    # yTest = np_utils.to_categorical(encodedyTest)
    # yVal = np_utils.to_categorical(encodedyVal)



    # In[11]:


    # Compute Class weights
    classWeight = class_weight.compute_class_weight('balanced', np.unique(yTrain), yTrain)
    classWeight = dict(enumerate(classWeight))
    print(classWeight)


    # In[12]:


    assert xTrain.shape[0] == yTrain.shape[0]
    assert xTest.shape[0] == yTest.shape[0]
    assert xVal.shape[0] == yVal.shape[0]
    xTrain.shape


    # # ConvNet for Buy
    # 
    # A CNN to predict buy signals from the above generated data

    # In[13]:


    learnRate = 0.05
    batchSize = 300
    totalBatches = (xTrain.shape[0]//batchSize)
    epochs = 5

    nClasses = 2
    nLength = xTrain.shape[1]
    inputShape = (nLength,2)
    # xTrainDataSet = tf.data.Dataset.from_tensors(xTrain)
    # xTrainIter = xTrainDataSet.make_one_shot_iterator()


    # In[14]:

    print("BUY TRAINING")
    # Keras
    #https://arxiv.org/pdf/1709.05206.pdf LSTM-FCN
    buyModelConv = Sequential()
    buyModelConv.add(Conv1D(15,kernel_size= 2, strides=1,
                     input_shape=inputShape,
                     batch_size = None
                       ))
    buyModelConv.add(BatchNormalization())
    buyModelConv.add(Activation('relu'))


    buyModelConv.add(Conv1D(30, kernel_size= 2, strides=1))
    buyModelConv.add(BatchNormalization())
    buyModelConv.add(Activation('relu'))


    buyModelConv.add(Conv1D(15,kernel_size= 2, strides=1))
    buyModelConv.add(BatchNormalization())
    buyModelConv.add(Activation('relu'))

    buyModelConv.add(GlobalAveragePooling1D())
    im = buyModelConv.layers[0].input
    buyConvInput = buyModelConv(im)
     ########################################
    buyModelLSTM = Sequential()

    buyModelLSTM.add(Permute((1, 2), input_shape=inputShape))
    buyModelLSTM.add(AttentionLSTM(2))
    buyModelLSTM.add(Dropout(0.5))


    im2 = buyModelLSTM.layers[0].input
    buyLstmInput = buyModelLSTM(im2)
    #############################

    merged = concatenate([buyConvInput, buyLstmInput])
    output = Dense(1, activation='sigmoid')(merged)
    buyModel = Model(inputs=[im,im2],outputs=output)
    # In[ ]:


    # buyModel.summary()
    buyModel.compile(loss=binary_crossentropy,
                  optimizer=Adam(lr=learnRate),
                  metrics=['accuracy'])


    # In[ ]:


    buyModel.fit(x=[xTrain,xTrain],
                 y=yTrain, 
                 class_weight=classWeight,
                 validation_data = ([xVal,xVal],yVal),
                 epochs = epochs,
                 verbose = 0,
                 batch_size = batchSize   
                  )


    # In[ ]:


    score = buyModel.evaluate([xTest,xTest], yTest, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    # ## ConvNet for Sell

    # In[ ]:


    # Get values from pandas series as we need a numpy array for our classifier
    x,y = shuffle(sellSeries,sellSeriesLabs)
    tot = len(x)
    y = y.values
    trainPercent = 0.8 # majority of data used for training
    testPercent = 0.95 # 
    valPercent = 1.00  #

    # Test Train Val Split

    xTrain = x[0:int(trainPercent*tot),:,:]
    yTrain = y[0:int(trainPercent*tot)]

    xTest = x[int(trainPercent*tot): int(testPercent*tot),:,:]
    yTest = y[int(trainPercent*tot): int(testPercent*tot)]

    xVal = x[int(testPercent*tot):,:,:]
    yVal = y[int(testPercent*tot):]

# #Reshape for keras
# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1],1)
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[1],1)
# xVal = xVal.reshape(xVal.shape[0],xVal.shape[1],1)


# # # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(yTrain)
# encodedyTrain = encoder.transform(yTrain)
# encodedyTest = encoder.transform(yTest)
# encodedyVal = encoder.transform(yVal)
# # convert integers to one hot encoded
# yTrain = np_utils.to_categorical(encodedyTrain)
# yTest = np_utils.to_categorical(encodedyTest)
# yVal = np_utils.to_categorical(encodedyVal)



    # # # encode class values as integers
    # encoder = LabelEncoder()
    # encoder.fit(yTrain)
    # encodedyTrain = encoder.transform(yTrain)
    # encodedyTest = encoder.transform(yTest)
    # encodedyVal = encoder.transform(yVal)
    # # convert integers to one hot encoded
    # yTrain = np_utils.to_categorical(encodedyTrain)
    # yTest = np_utils.to_categorical(encodedyTest)
    # yVal = np_utils.to_categorical(encodedyVal)




    # In[ ]:


    # Compute Class weights
# Compute Class weights
    classWeight = class_weight.compute_class_weight('balanced', np.unique(yTrain), yTrain)
    classWeight = dict(enumerate(classWeight))
    xTest.shape
    assert xTrain.shape[0] == yTrain.shape[0]
    assert xTest.shape[0] == yTest.shape[0]
    assert xVal.shape[0] == yVal.shape[0]
    yTrain
    learnRate = 0.05
    batchSize = 300
    totalBatches = (xTrain.shape[0]//batchSize)
    epochs = 5

    nClasses = 2
    nLength = xTrain.shape[1]
    inputShape = (nLength,2)
# xTrainDataSet = tf.data.Dataset.from_tensors(xTrain)
# xTrainIter = xTrainDataSet.make_one_shot_iterator()
    # xTrainDataSet = tf.data.Dataset.from_tensors(xTrain)
    # xTrainIter = xTrainDataSet.make_one_shot_iterator()


    # In[ ]:

    print("SELL TRAINING")
    # Keras
    #https://arxiv.org/pdf/1709.05206.pdf LSTM-FCN
    # Keras
    #https://arxiv.org/pdf/1709.05206.pdf LSTM-FCN
    sellModelConv = Sequential()
    sellModelConv.add(Conv1D(15,kernel_size= 2, strides=1,
                     input_shape=inputShape,
                     batch_size = None
                       ))
    sellModelConv.add(BatchNormalization())
    sellModelConv.add(Activation('relu'))


    sellModelConv.add(Conv1D(30, kernel_size= 2, strides=1))
    sellModelConv.add(BatchNormalization())
    sellModelConv.add(Activation('relu'))

    sellModelConv.add(Conv1D(15,kernel_size= 2, strides=1))
    sellModelConv.add(BatchNormalization())
    sellModelConv.add(Activation('relu'))

    sellModelConv.add(GlobalAveragePooling1D())
    # convInput = Input(shape=(None,8))
    im = sellModelConv.layers[0].input
    sellConvInput = sellModelConv(im)
     ########################################
    sellModelLSTM = Sequential()
    sellModelLSTM.add(Permute((1, 2), input_shape=inputShape))
    sellModelLSTM.add(AttentionLSTM(2))
    sellModelLSTM.add(Dropout(0.5))
    im2 = sellModelLSTM.layers[0].input
    sellLstmInput = sellModelLSTM(im2)
    #############################

    merged = concatenate([sellConvInput, sellLstmInput])
    output = Dense(1, activation='sigmoid')(merged)
    sellModel = Model(inputs=[im,im2],outputs=output)


    # In[ ]:


    # sellModel.summary()
    sellModel.compile(loss=binary_crossentropy,
                  optimizer=Adam(lr=learnRate),
                  metrics=['accuracy'])


    # In[ ]:


    sellModel.fit(x=[xTrain,xTrain],
                 y=yTrain, 
                 class_weight=classWeight,
                 validation_data = ([xVal,xVal],yVal),
                 epochs = epochs,
                 batch_size = batchSize,  
                 verbose = 0)


    # In[ ]:


    score = sellModel.evaluate([xTest,xTest], yTest, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    # In[ ]:


    # buyModel.save('buyModel.h5')
    # sellModel.save('sellModel.h5')


    import backtrader as bt
    from kiteconnect import KiteConnect
    import datetime as dt
    import pytz
    import math

    # buyModel = load_model('modelsFin/buyModel.h5')
    # sellModel = load_model('modelsFin/buyModel.h5')
    # In[947]:
    def printTradeAnalysis(analyzer):
        #Get the results we are interested in
        total_open = analyzer.total.open
        total_closed = analyzer.total.closed
        total_won = analyzer.won.total
        total_lost = analyzer.lost.total
        win_streak = analyzer.streak.won.longest
        lose_streak = analyzer.streak.lost.longest
        pnl_net = round(analyzer.pnl.net.total,2)
        strike_rate = round((total_won / total_closed) * 100,2)
        #Designate the rows
        h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost']
        h2 = ['Strike Rate','Win Streak', 'Losing Streak', 'PnL Net']
        r1 = [total_open, total_closed,total_won,total_lost]
        r2 = [strike_rate, win_streak, lose_streak, pnl_net]
        #Check which set of headers is the longest.
        if len(h1) > len(h2):
            header_length = len(h1)
        else:
            header_length = len(h2)
        #Print the rows
        print_list = [h1,r1,h2,r2]
        row_format ="{:<15}" * (header_length + 1)
        print("Trade Analysis Results:")
        for row in print_list:
            print(row_format.format('',*row))

    class neuralModel(bt.Indicator):
        lines = ('Ind',)
        params = (('period', 30),('neuralModel',None))

        def __init__(self):
            self.addminperiod(self.params.period)
    #         self.i = 0

        def next(self):
            vols = np.array(self.data.volume.get(size=self.p.period)) # get the volumes
            close = np.array(self.data.close.get(size=self.p.period)) # get the closing prices
            # scale them
            vols = skp.minmax_scale(vols)
            close = skp.minmax_scale(close)
            # make an array of the 2
    #         print(self.p.period)
            data = np.zeros((1,self.p.period,2))
    #         print(data.shape)
            data[0,:,0] = close
            data[0,:,1] = vols
            prob = self.p.neuralModel.predict([data,data])[0][0]
    #         print(prob)
            self.lines.Ind[0] = prob # predict and round to 0 for no action and 1 for buy

    class TestStrategy(bt.Strategy):
        params = (
            ('lagPeriod', lag),
            ('buyNeural',buyModel),
            ('SellNeural',sellModel)
        )

        def __init__(self):

            self.dataclose = self.datas[0].close

            self.neuralBuy = neuralModel(
                self.datas[0], 
                period=self.params.lagPeriod, 
                neuralModel = self.params.buyNeural,
                plot = False
            )

            self.neuralSell = neuralModel(
                self.datas[0], 
                period=self.params.lagPeriod, 
                neuralModel = self.params.SellNeural,
                plot = False
            )


        def next(self):


            if self.neuralBuy[0] > 0.55 and self.neuralSell[0] < 0.5:
    #             print(self.neuralBuy[0])
    #             print(self.neuralSell[0])

                buyOrd = self.buy_bracket(limitprice=self.dataclose+0.1,
                                          price=self.dataclose,
                                          stopprice=self.dataclose-0.1,
                                          size = 15000,
                                          valid = 0
                                         )




            elif self.neuralSell[0] > 0.55 and self.neuralBuy[0] < 0.5:
    #             print(self.neuralBuy[0])
    #             print(self.neuralSell[0])
                sellOrd = self.sell_bracket(limitprice=self.dataclose-0.1,
                              price=self.dataclose,
                              stopprice=self.dataclose + 0.1,
                              size = 15000,
                              valid = 0)


    class Plotter(bt.plot.Plot):

        def __init__(self):
            super().__init__()  # custom color for volume up bars 

        def show(self):
            mng = plt
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            # fig.savefig('test2png.png', dpi=100)
            title = str(backTestStart.date()) + " to " + str(backTestEnd.date()) 
            plt.title(title)

            plt.tight_layout()

            plt.savefig("plots/"+title)
            # plt.show()


    # In[950]:


    fed = bt.feeds.GenericCSVData(dataname='data/GMRINFRA.csv',
                                  dtformat="%Y-%m-%dT%H:%M:%S%z",
                                  openinterest=-1,
                                  headers=False,
                                  fromdate= backTestStart,
                                  todate= backTestEnd,
    #                               timeframe=bt.TimeFrame.Minutes,
    #                               tzinput = pytz.timezone('Asia/Kolkata'),
                                  plot=True)

    # brokerageCom = ((0.0001 +0.0000325)*0.18) + (0.0001 +0.0000325) + 0.00025
    # print(brokerageCom)
    cerebro = bt.Cerebro()
    cerebro.broker.set_shortcash(False)
    cerebro.broker.setcash(startCash)
    cerebro.broker.setcommission(commission=0.000425  ,margin = False)
    cerebro.adddata(fed) 
    cerebro.addstrategy(TestStrategy,plot=False)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.BuySell)
    # cerebro.addanalyzer(bt.analyzers.SharpeRatio , _name='Sharpe',timeframe = bt.TimeFrame.Minutes)
    cerebro.addanalyzer(bt.analyzers.Returns , _name='Transactions', timeframe = bt.TimeFrame.Minutes)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    thestrats = cerebro.run(stdstats=False)

    thestrat = thestrats[0]

    print('returns:', thestrat.analyzers.Transactions.get_analysis())
  

    try:
        printTradeAnalysis(thestrat.analyzers.ta.get_analysis())

    except:
        print ("No trades!")
    
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    startCash = cerebro.broker.getvalue()
    cerebro.plot(start=backTestStart , end=backTestEnd,plotter = Plotter())
    if (cerebro.broker.getvalue() < 9000):
        portVals.append(9000)
    else:
        portVals.append(cerebro.broker.getvalue())

    TransVals.append(thestrat.analyzers.Transactions.get_analysis())

import pickle

with open('portVals.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(portVals,f)

with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(TransVals,f)

    