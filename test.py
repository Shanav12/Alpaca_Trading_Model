from trade import StockTrader

API_KEY = "PKWM3B5LWZFODYFJSVZ7"
SECRET_KEY = "SADYVSptP60GbOVUFcmPQuCnOCCKDynn96JRhzON"


tickers = ['WSM', 'PII', 'GPS', 'SAM', 'VVV', 'AX', 'CEQP', 'PFGC', 'AAPL', 'DIS', 'MSFT', 'COST', 'HBI', 
           'JWN', 'GOOGL', 'TSLA', 'ETRN', 'KO', 'ORCL', 'NVDA', 'AMD', 'KO', 'SBUX', 'JPM', 'WMT']


trader = StockTrader(API_KEY, SECRET_KEY, tickers)
trader.create_model()
trader.create_predictions()