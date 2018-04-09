
# DEPRECIATED 9th April, 2018

This is depreciated and only up here for references. It should give about 30% precision and 80ish % accuracy on most scrips, so take that sa you will. No guarantees implicitly or explicitly given.
All passwords used in the config files have been changed, all API keys are no longer valid.
You can contact me at varun@alchemy.email for questions.


# kiteGo

A trader built over the Zerodha platform to trade in the Indian Market.

It uses a neural network to learn trading parameters and then predicts buys and sells in the equity market.

NOT ASSOCIATED WITH ZERODHA, WE JUST USE THEIR PLATFORM.


CHECK INSTRUMENT TOKENS BEFORE STARTING

TRAIN NETWORKS BEFORE THE FOLLOWING

To run:

1) Run ./monitor.sh to stay up to date on current price of stock.
2) Start websockClient with python websockClient.py
3) Get new request token and update in config .json
4) Update config.json with new file
5) Open AlgoTrading env with source activate AlgoTrading
6) Check Live Trader parameters
7) Start live trader with ./python LiveTrader.py




