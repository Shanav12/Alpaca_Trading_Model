from trade import StockTrader

API_KEY = #insert here
SECRET_KEY = # insert here


tickers = ['WSM', 'PII', 'GPS', 'SAM', 'VVV', 'AX', 'PFGC', 'AAPL', 
           'DIS', 'MSFT', 'COST', 'HBI', 'JWN', 'GOOGL', 'TSLA', 'KO',
           'ETRN', 'ORCL', 'NVDA', 'AMD', 'KO', 'SBUX', 'JPM', 'WMT']


trader = StockTrader(API_KEY, SECRET_KEY, tickers)
trader.create_predictions()