
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

import h5py

from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as skp
import tensorflow as tf
import tempfile

tf.__version__

# fix random seed for reproducibility
# seed = 7
DATA_DIR = 'data' 
# np.random.seed(seed)
dbString = 'postgresql://s2c:JANver95@localhost:5432/stockdata'
engine = sqlalchemy.create_engine(dbString) 
utc = pytz.UTC
starDate = utc.localize(dt.datetime(2016,1,2))
endDate = utc.localize(dt.datetime(2016,1,2))
portVals = []
TransVals = []

curIter = 0

while curIter==0 or (startDate.year == 2016) or (endDate.year == 2018 and endDate.month == 1 and endDate.day <= 6) :
    query = "SELECT * FROM histdata WHERE ticker = 'GMRINFRA' ORDER BY datetime ASC"
    dat = pd.read_sql(query,engine)


    startDate = starDate + dt.timedelta(days=7*curIter)
    endDate = startDate +dt.timedelta(days = 7*4*18) # 18 months of Training of training
    backTestStart = endDate
    backTestEnd = endDate + dt.timedelta(days=7)
    res = dat[(dat['datetime'] > startDate) & (dat['datetime'] < endDate)]
    curIter += 1

    # res


    # ## Some Helper Functions
    # 
    # These functions are more or less general functions that should prove to be fairly useful
    # 
    # 
    # - **ReadData(filename)** : Reads data from Zerodha API historical data files and returns a Pandas DataFrame
    # - **sycTimeSeries(ts1,ts2)** : Making sure that 2 timeseries are synced to the smaller time series
    # - **timeseriesLagged(data, lag=60)**: Creates Lagged series.Goes through a series and generates an lag+1  dimensional   pandas DataFrame that has each previous lag timeunit.
    # - **binarizeTime(resLagged, rate=0.01)** : Binarizes the last column into 1,-1 or 0 depending whether the price increased, decreased or stayed the same from the beginning to the end of the lag period (triggers on changes by magnitutde = rate*current price).
    # - **findLag(data, targetCorr,suppressed)** :  Finds the right lag given a target correlation.

    # ## Reading some Data and Getting a feel 
    # 
    # We use an autocorrelation plot to help us figure out what is an optimal amount of lag. We are really looking for a lag that correlates highly. We go through the lags till we reach the last lag that guarantees 0.97 autocorrelation
    # 
    # ## THIS DID NOT WORK AS EXPECTED. REPLACE WITH FALSE NEAREST NEIGHBOUR

    # In[925]:


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
    lag = 30 
    lookahead = 60
    flat = 0.1
    series = timeseriesLagged(data,lag + lookahead-1) # Generate the lagged series


    # In[926]:


    # Create binary series where 0 = hold and 1 = buy
    buySeries = binarizeTime(series,0,lookahead = lookahead, flat= flat)
    change = buySeries.iloc[:,-1]== -1 # convert to binary
    buySeries.loc[change,str(lag+1)]=0 # convert to binary


    # In[927]:


    # Create binary series where 0 = hold and 1 = sell
    sellSeries = binarizeTime(series,0,lookahead=lookahead,flat=flat)
    change = sellSeries.iloc[:,-1]== 1 # find 1s and convert to 0
    sellSeries.loc[change,str(lag+1)]=0 # 
    change = sellSeries.iloc[:,-1]== -1 # find -1 and conver to 1s
    sellSeries.loc[change,str(lag+1)]= 1 # convert to

    # # Generate Training Data
    # 
    # Now that we have an idea of what's going on in the dataset, it is a good time to generate training data. We do an 90:20 training:testing split, and then we randomize the training set because we assume that only the last LAG minutes matter

    # In[928]:



    # In[929]:

    # Get values from pandas series as we need a numpy array for our classifier
    BuySeriesVals = buySeries.values
    np.random.shuffle(BuySeriesVals) #shuffle the entire dataset
    trainPercent = 0.999 # first 80% of the data is used for training
    # np.random.shuffle(BuySeriesVals)
    #Split into train and test
    trainBegin = int(trainPercent*len(BuySeriesVals)) 
    trains = BuySeriesVals[0:trainBegin]
    train,val = train_test_split(trains)
    test = BuySeriesVals[trainBegin:]
    # np.random.shuffle(train) # shuffle the training dataset

    # Split into x and y
    xTrain,yTrain = train[:,0:-1],train[:,-1] # X is the first lag elements. Y is the lag+1 element
    xVal,yVal = val[:,0:-1],val[:,-1] # Same for Validation
    xTest,yTest = test[:,0:-1],test[:,-1] # Same for testing data

    #scale function to local normalize each row between 0 and 1 so as to amplify any changes
    # standardize = lambda row: skp.normalize(row)
    xTrain =skp.scale(xTrain,axis=1) #np.apply_along_axis(standardize,1,xTrain) #scale to 01
    xTest = skp.scale(xTest,axis=1) #scale to 0 1
    xVal = skp.scale(xVal,axis=1) #scale to 0 1

    #Reshape for keras
    xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1],1)
    xTest = xTest.reshape(xTest.shape[0], xTest.shape[1],1)
    xVal = xVal.reshape(xVal.shape[0],xVal.shape[1],1)


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
    classWeight


    # In[12]:


    assert xTrain.shape[0] == yTrain.shape[0]
    assert xTest.shape[0] == yTest.shape[0]
    assert xVal.shape[0] == yVal.shape[0]
    xTrain.shape


    # # ConvNet for Buy
    # 
    # A CNN to predict buy signals from the above generated data

    # In[13]:


    learnRate = 0.5
    batchSize = 10
    totalBatches = (xTrain.shape[0]//batchSize)
    epochs = 5

    nClasses = 2
    nLength = xTrain.shape[1]
    inputShape = (nLength,1)
    # xTrainDataSet = tf.data.Dataset.from_tensors(xTrain)
    # xTrainIter = xTrainDataSet.make_one_shot_iterator()


    # In[14]:


    # Keras
    #https://arxiv.org/pdf/1709.05206.pdf LSTM-FCN
    buyModelConv = Sequential()
    buyModelConv.add(Conv1D(12,kernel_size= 1, strides=1,
                     input_shape=inputShape,
                     batch_size = None
                       ))
    buyModelConv.add(BatchNormalization())
    buyModelConv.add(Activation('relu'))


    buyModelConv.add(Conv1D(6, kernel_size= 1, strides=1))
    buyModelConv.add(BatchNormalization())
    buyModelConv.add(Activation('relu'))

    buyModelConv.add(Conv1D(6,kernel_size= 1, strides=1))
    buyModelConv.add(BatchNormalization())
    buyModelConv.add(Activation('relu'))

    buyModelConv.add(GlobalAveragePooling1D())
    im = buyModelConv.layers[0].input
    buyConvInput = buyModelConv(im)
     ########################################
    buyModelLSTM = Sequential()
    buyModelLSTM.add(Permute((2, 1), input_shape=inputShape))
    buyModelLSTM.add(LSTM(5))
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
                  optimizer=SGD(lr=learnRate),
                  metrics=['accuracy'])


    # In[ ]:


    buyModel.fit(x=[xTrain,xTrain],
                 y=yTrain, 
                 class_weight=classWeight,
                 validation_data = ([xVal,xVal],yVal),
                 epochs = 3,
                 verbose = 0,
                 batch_size = 300   
                  )


    # In[ ]:


    score = buyModel.evaluate([xTest,xTest], yTest, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    # ## ConvNet for Sell

    # In[ ]:


    # Get values from pandas series as we need a numpy array for our classifier
    sellSeriesVals = sellSeries.values
    trainPercent = 0.999 # first 80% of the data is used for training
    np.random.shuffle(sellSeriesVals)
    #Split into train and test
    trainBegin = int(trainPercent*len(sellSeriesVals)) 
    trains = sellSeriesVals[0:trainBegin]
    train,val = train_test_split(trains)
    test = sellSeriesVals[trainBegin:]
    # np.random.shuffle(train) # shuffle the training dataset

    # Split into x and y
    xTrain,yTrain = train[:,0:-1],train[:,-1] # X is the first lag elements. Y is the lag+1 element
    xVal,yVal = val[:,0:-1],val[:,-1] # Same for Validation
    xTest,yTest = test[:,0:-1],test[:,-1] # Same for testing data

    #scale function to local normalize each row between 0 and 1 so as to amplify any changes
    # standardize = lambda row: skp.normalize(row)
    xTrain =skp.scale(xTrain,axis=1) #np.apply_along_axis(standardize,1,xTrain) #scale to 01
    xTest = skp.scale(xTest,axis=1) #scale to 0 1
    xVal = skp.scale(xVal,axis=1) #scale to 0 1

    #Reshape for keras
    xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1],1)
    xTest = xTest.reshape(xTest.shape[0], xTest.shape[1],1)
    xVal = xVal.reshape(xVal.shape[0],xVal.shape[1],1)


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
    classWeight = class_weight.compute_class_weight('balanced', np.unique(yTrain), yTrain)
    classWeight = dict(enumerate(classWeight))
    xTest.shape
    assert xTrain.shape[0] == yTrain.shape[0]
    assert xTest.shape[0] == yTest.shape[0]
    assert xVal.shape[0] == yVal.shape[0]
    yTrain
    learnRate = 0.5
    batchSize = 10
    totalBatches = (xTrain.shape[0]//batchSize)
    epochs = 5

    nClasses = 2
    nLength = xTrain.shape[1]
    inputShape = (nLength,1)
    # xTrainDataSet = tf.data.Dataset.from_tensors(xTrain)
    # xTrainIter = xTrainDataSet.make_one_shot_iterator()


    # In[ ]:


    # Keras
    #https://arxiv.org/pdf/1709.05206.pdf LSTM-FCN
    sellModelConv = Sequential()
    sellModelConv.add(Conv1D(8,kernel_size= 1, strides=1,
                     input_shape=inputShape,
                     batch_size = None
                       ))
    sellModelConv.add(BatchNormalization())
    sellModelConv.add(Activation('relu'))


    sellModelConv.add(Conv1D(4, kernel_size= 2, strides=1))
    sellModelConv.add(BatchNormalization())
    sellModelConv.add(Activation('relu'))

    sellModelConv.add(Conv1D(8,kernel_size= 2, strides=1))
    sellModelConv.add(BatchNormalization())
    sellModelConv.add(Activation('relu'))

    sellModelConv.add(GlobalAveragePooling1D())
    # convInput = Input(shape=(None,8))
    im = sellModelConv.layers[0].input
    sellConvInput = sellModelConv(im)
     ########################################
    sellModelLSTM = Sequential()
    sellModelLSTM.add(Permute((2, 1), input_shape=inputShape))
    sellModelLSTM.add(LSTM(5))
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
                  optimizer=SGD(lr=learnRate),
                  metrics=['accuracy'])


    # In[ ]:


    sellModel.fit(x=[xTrain,xTrain],
                 y=yTrain, 
                 class_weight=classWeight,
                 validation_data = ([xVal,xVal],yVal),
                 epochs = 3,
                 batch_size = 300,  
                 verbose = 0)


    # In[ ]:


    score = sellModel.evaluate([xTest,xTest], yTest, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    # In[ ]:


    buyModel.save('buyModel.h5')
    sellModel.save('sellModel.h5')


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
            self.i = 0

        def next(self):
            data = self.data.get(size=self.p.period) # get the data
            data = np.array(data) # put it in a numpy array
            # print(data)
            data = skp.scale(data)
            data = data.reshape(1, -1,1) # get it ready for the neural network
            prob = self.p.neuralModel.predict([data,data])[0][0]
    #         print(prob)
            self.lines.Ind[0] = 1 if  prob > 0.6 else 0 # predict and round to 0 for no action and 1 for buy


    # In[ ]:


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
    #         print(type(self.dataclose))

            if self.neuralBuy[0] == 1: 
                buyOrd = self.buy_bracket(limitprice=self.dataclose+0.1,
                                          price=self.dataclose,
                                          stopprice=self.dataclose-0.1,
                                          size = 300,
                                          valid = 0
                                         )




            elif self.neuralSell[0] == 1:
                sellOrd = self.sell_bracket(limitprice=self.dataclose-0.1,
                              price=self.dataclose,
                              stopprice=self.dataclose+0.1,
                              size = 300,
                              valid = 0)


    # In[949]:


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
    printTradeAnalysis(thestrat.analyzers.ta.get_analysis())

    try:
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    except:
        print ("No trades!")
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

    