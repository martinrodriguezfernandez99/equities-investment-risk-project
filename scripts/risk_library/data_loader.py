import yfinance as yf
import pandas as pd
import numpy as np

class StockDataFetcher:
    """Class to fetch historical stock prices, compute returns, and manage portfolio weights."""

    def __init__(self, tickers, start_date, end_date, weights=None, benchmark_ticker=None):
        """
        :param tickers: List of stock tickers 
        :param start_date: Start date for historical data (YYYY-MM-DD)
        :param end_date: End date for historical data (YYYY-MM-DD)
        :param weights: List of weights corresponding to each stock in the portfolio.
        :param benchmark_ticker: Ticker of the market index 
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_ticker = benchmark_ticker  
        self.data = None  
        self.benchmark_data = None  
        self.benchmark_returns = None  

        if weights:
            if len(weights) != len(tickers):
                raise ValueError("Number of weights must match number of tickers.")
            self.weights = np.array(weights)
        else:
            self.weights = np.ones(len(tickers)) / len(tickers)  

    def fetch_data(self):
        """Fetches adjusted closing prices from Yahoo Finance for portfolio stocks and benchmark."""
        print(f"Fetching data for: {self.tickers} from {self.start_date} to {self.end_date}")

        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)

        if isinstance(self.data.columns, pd.MultiIndex):
            self.data = self.data.xs('Close', axis=1, level=0) 

        if self.benchmark_ticker:
            print(f"Fetching benchmark data for: {self.benchmark_ticker}")
            benchmark_data = yf.download(self.benchmark_ticker, start=self.start_date, end=self.end_date, auto_adjust=True)

            if isinstance(benchmark_data, pd.DataFrame) and isinstance(benchmark_data.columns, pd.MultiIndex):
                benchmark_data = benchmark_data.xs('Close', axis=1, level=0)

            self.benchmark_data = benchmark_data
            self.benchmark_returns = self.benchmark_data.pct_change().dropna()

        return self.data

    def compute_daily_returns(self):
        """Computes daily percentage returns from stock prices."""
        if self.data is None:
            raise ValueError("Stock data is empty. Run fetch_data() first.")
        return self.data.pct_change().dropna()

    def compute_portfolio_returns(self):
        """Computes weighted portfolio returns based on stock returns and assigned weights."""
        daily_returns = self.compute_daily_returns()

        print(f"Available columns in daily_returns: {list(daily_returns.columns)}")
        print(f"Expected tickers: {self.tickers}")

        tickers_in_data = [ticker for ticker in self.tickers if ticker in daily_returns.columns]

        print(f"Tickers found in data: {tickers_in_data}")

        if not tickers_in_data:
            raise ValueError("None of the provided tickers were found in the fetched stock data. Check column names.")

        if len(tickers_in_data) != len(self.weights):
            raise ValueError(f"Mismatch: {len(tickers_in_data)} stocks found in data, but {len(self.weights)} weights provided.")

        portfolio_returns = daily_returns[tickers_in_data].dot(self.weights)

        return portfolio_returns, self.benchmark_returns

    def compute_cumulative_returns(self):
        """Computes cumulative returns for the portfolio."""
        portfolio_returns, _ = self.compute_portfolio_returns()  
        cumulative_returns = (1 + portfolio_returns).cumprod()
        return cumulative_returns

# Example usage:
if __name__ == "__main__":
    fetcher = StockDataFetcher(
        tickers=["AAPL", "MSFT", "GOOGL", "UBS", "RHHBF"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        weights=[0.2, 0.1, 0.3, 0.2, 0.2], 
        benchmark_ticker="^GSPC"
    )

    prices = fetcher.fetch_data()
    portfolio_returns, benchmark_returns = fetcher.compute_portfolio_returns()

    print("Stock Prices: ", prices)
    print("Daily Portfolio Returns: ", portfolio_returns)
    print("Daily Benchmark Returns: ", benchmark_returns)

