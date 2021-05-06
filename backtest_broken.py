import backtrader as bt
from backtrader.feeds import GenericCSVData
import pandas as pd
import pyfolio as pf
import datetime

# custom datafeed class for importing Hourly Data
class GenericCSV_Hourly(GenericCSVData):
    linesoverride = True,
    # normally backtrader only supports open, high, low, close, datetime, and volume without a custom override
    lines = (
        'datetime' , 'AAL', 'close', 'AAL_return', 'LUV_return', #important
        'LUV_markout_1h','LUV_markout_2h','LUV_markout_4h', #not needed
        'signal' #important
    )

    params = (
        ('datetime', 0), ('AAL',1), ('close',2), ('AAL_return',3), ('LUV_return',4), #important
        ('LUV_markout_1h',5), ('LUV_markout_2h',6), ('LUV_markout_4h',7), #not needed
        ('signal',8) #important
    ) 

class myStrategy(bt.Strategy):
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.signal = self.datas[0].signal

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
        
    def log(self, txt, dt=None):
        ''' 
        Summary: This function allows us to print out whats going on inside the strategy
        Parameters: 
          - txt: Text to print out
          - dt: date that strategy event occured
        '''
        dt = dt or self.datas[0].datetime.datetime()
        print('{}, {}'.format(dt, txt))
        
    def next(self):
        """
        Summary: The next function is the actual logic behind the strategy
        The next function is run on every bar (day) of data
        """
        self.log('close: %.2f' % self.dataclose[0])

        if self.order:
            return

        if not self.position:
            if (self.signal[0] < 0): #buy signal
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy()
                self.log('crash?')

        else:
            if (self.signal[0] > 0): #sell signal
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()
        



if __name__ == '__main__':
    data_filename = './Backtest/LUV_on_AAL.csv'
    data = GenericCSV_Hourly(
            dataname=data_filename,
            headers=True,
            nullvalue=float('NaN'),
            dtformat=('%Y-%m-%d %H:%M:%S%z'),
            compression=60,
            timeframe=bt.TimeFrame.Minutes,
            fromdate=datetime.datetime(2016, 1, 1, 9, 0), # Important: Select you starting day for data here
            todate=datetime.datetime(2021, 4, 30, 23, 0), # Ending day for the data you want here
            sessionstart=datetime.time(8, 0),
            sessionend=datetime.time(23, 0),
            separator=",",
            time=-1,
            open=-1,
            high=-1,
            low=-1,
            volume=-1,
            openinterest=-1,
        )

    cerebro = bt.Cerebro() #stdstats=False)
    #cerebro.broker.setcommission(commission=0.005) Coinbase fee is 0.5%. Other exchanges have lower fees especially if using limit orders and higher volume that could make the strategy most viable
    #cerebro.addobserver(bt.observers.Value)
    #cerebro.addobserver(bt.observers.Trades)
    #cerebro.addobserver(bt.observers.BuySell) a bit too messy I feel, plots the buy sell arrows
    #cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe',timeframe=bt.TimeFrame.Ticks) bug in his library


    data.addfilter(bt.filters.SessionFilter(data))
    cerebro.adddata(data)
    cerebro.addstrategy(myStrategy)
    

    cerebro.broker.setcash(1000.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run() #0 is the strategy (can take list)
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    #cerebro.plot()


