import yfinance as yf
stock = yf.Ticker("AAPL")  # Apple Inc.
data = stock.history(period="5d")
print(data)