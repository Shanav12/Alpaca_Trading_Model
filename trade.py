'''Import necessary libraries.
   We will be using the Sequential model provided by Keras and added LSTM layers to the neural network'''

from alpaca.trading.client import TradingClient
from alpaca_trade_api import REST, TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream
from alpaca.trading.enums import OrderSide, TimeInForce
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
import numpy as np

class StockTrader:

    '''StockTrader Constructor. Takes in the Alpaca API key, secret key, and tickers to be used to instantiate the object.
       Sets the scaler to be between 0 and 1 which will be used to normalize the data before feeding it into the model
       Initializes the Rest object that will be used to manage our portfolio and receieve stock quotes'''
    def __init__(self, API_KEY, SECRET_KEY, tickers):
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.rest_client = REST(API_KEY, SECRET_KEY)
        self.tickers = tickers
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)


    ''' Function used to create a long position through the Alpaca API for the specified
        ticker. Sets a market order for 100 shares of the specified share '''
    def create_long_trade(self, ticker, qty):
      order_details = MarketOrderRequest(
          symbol=ticker,
          qty=qty,
          side=OrderSide.BUY,
          time_in_force=TimeInForce.DAY
      )
      self.trading_client.submit_order(order_data=order_details)


    ''' Function used to create a short position through the Alpaca API for the specified ticker
        Sets a market order for 100 shares of the specified share'''
    def create_short_trade(self, ticker, qty):
      order_details = MarketOrderRequest(
          symbol=ticker,
          qty=qty,
          side=OrderSide.SELL,
          time_in_force=TimeInForce.DAY
      )
      self.trading_client.submit_order(order_data=order_details)


    '''Gets the last opening price for each ticker
       Which we will use later to determine the quantity
       of stocks to purchase'''
    def get_last_opening_price(self):
      opening_prices = {}
      for ticker in self.tickers:
          opening_prices_df = self.rest_client.get_bars(ticker, TimeFrame.Day, "2023-01-01").df
          data = opening_prices_df['open'].values[-1]
          opening_prices[ticker] = data

      return opening_prices


    def create_datasets(self, dataset, time_step):
      X_data, Y_data = [], []
      for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X_data.append(a)
        Y_data.append(dataset[i + time_step, 0])
      return np.array(X_data, dtype='float32'), np.array(Y_data, dtype='float32')


    '''Receives closing prices on each ticker and creates training and testing data
       Stores the data in dictionaries with the ticker as the key and the data as a 2d array'''
    def get_prices(self):
        predictors_closing_prices = {}
        target_closing_prices = {}
        for ticker in self.tickers:
            predictors_df = self.rest_client.get_bars(ticker, TimeFrame.Day, "2022-01-01", "2023-01-01").df
            target_df = self.rest_client.get_bars(ticker, TimeFrame.Day, "2023-01-01").df

            predictors_scaled = self.scaler.fit_transform(predictors_df['close'].values.reshape(-1, 1))
            targets_scaled = self.scaler.fit_transform(target_df['close'].values.reshape(-1, 1))

            predictors_closing_prices[ticker] = predictors_scaled
            target_closing_prices[ticker] = targets_scaled
        return predictors_closing_prices, target_closing_prices


    '''Instanties the LSTM model. We will be using the sigmoid activation function, adam optimizer function,
       and mean squared error for our loss function. We will also have 4 layers of LSTMs'''
    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units = 100, activation = 'sigmoid', return_sequences = True, input_shape=(100,1)))
        self.model.add(LSTM(units = 100, activation = 'sigmoid', return_sequences = True))
        self.model.add(LSTM(units = 100, activation ='sigmoid', return_sequences = True))
        self.model.add(LSTM(units = 100, activation ='sigmoid', return_sequences = False))
        self.model.add(Dense(1))
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')


    '''Trains the model through 20 epochs and outputs predictions.
       Creating predictions for the next 10 days of the specified ticker
       using closing prices throughout the course of the year
       And returns a list of those outputs'''
    def train_and_predict(self, X_train, X_test, y_train, y_test, ticker):
      self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
      predictors, targets = self.get_prices()
      x_input = targets[ticker][-100:]
      temp_input = list(x_input)
      lst_output = []
      n_steps = 100
      i = 0
      while i < 10:
          if len(temp_input) >= n_steps:
              x_input = np.array(temp_input[-n_steps:]).reshape(1, n_steps, 1)
          else:
              x_input = np.array(temp_input).reshape(1, len(temp_input), 1)

          yhat = self.model.predict(x_input)
          temp_input.append(yhat[0].astype('float32').tolist())
          lst_output.append(yhat[0])
          i=i+1
      return lst_output


    '''Creates the training and testing datasets using normalized prices for each ticker
       Stores the predictions for the next 10 closing prices in output
       Purchases 100 shares of the specified stock If the average of the predicted 10 day
       moving average is higher than the current 10 day moving average'''
    def create_predictions(self):
          predictors, targets = self.get_prices()
          time_step = 100
          for ticker in self.tickers:

            train_data = predictors[ticker]
            test_data = targets[ticker]
            X_train, y_train = self.create_datasets(train_data, time_step)
            X_test, y_test = self.create_datasets(test_data, time_step)

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)

            self.create_model()
            output = self.train_and_predict(X_train, X_test, y_train, y_test, ticker)
            outputs = [item for sublist in output for item in sublist]

            if mean(outputs) > mean(targets[ticker][-10:].flatten()):
              qty = int(50000 / self.get_last_opening_price()[ticker])
              self.create_long_trade(ticker, qty)

            else:
              qty = int(50000 / self.get_last_opening_price()[ticker])
              self.create_short_trade(ticker, qty)