import backtrader as bt
from keras.models import load_model
from kiteconnect import KiteConnect
import datetime as dt
import pytz
import math
import backtrader.plot as pLaut
import numpy as np
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt

utc = pytz.UTC
backTestStart = utc.localize(dt.datetime(2018,1,13))
backTestEnd = utc.localize(dt.datetime(2018,1,19))
portVals = []
TransVals = []
lag = 30


buyModel = load_model('buyModel.h5')
sellModel = load_model('buyModel.h5')
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
        data = skp.scale(data)
        data = data.reshape(1, -1,1) # get it ready for the neural network
        prob = self.p.neuralModel.predict([data,data])[0][0]
#         print(prob)
        self.lines.Ind[0] = 1 if  prob >= 0.60  else 0 # predict and round to 0 for no action and 1 for buy


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
                                      stopprice=self.dataclose-1,
                                      size = 200,
                                      valid = 0
                                     )




        elif self.neuralSell[0] == 1:
            sellOrd = self.sell_bracket(limitprice=self.dataclose-0.1,
                          price=self.dataclose,
                          stopprice=self.dataclose+1,
                          size = 200,
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


fed = bt.feeds.GenericCSVData(dataname='GMRINFRA.csv',
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

try:
    printTradeAnalysis(thestrat.analyzers.ta.get_analysis())
except:
    print ("No trades!")

print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

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