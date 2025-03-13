import numpy as np
import pandas as pd

class RiskMetrics:
    """Class for computing equity portfolio risk metrics."""

    def __init__(self, portfolio_returns, benchmark_returns=None, risk_free_rate=0.02):
        """
        :param portfolio_returns: Series of daily portfolio returns.
        :param benchmark_returns: Series of daily benchmark returns 
        :param risk_free_rate: Risk-free rate for Sharpe Ratio calculations.
        """
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate

    def compute_volatility(self):
        """Calculates annualized portfolio volatility."""
        return self.portfolio_returns.std() * np.sqrt(252)

    def compute_var(self, confidence_level=0.95):
        """Computes Value at Risk (VaR) using historical simulation."""
        return np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)

    def compute_sharpe_ratio(self):
        """Calculates the Sharpe Ratio."""
        excess_return = self.portfolio_returns.mean() - self.risk_free_rate / 252
        return excess_return / self.portfolio_returns.std()

    def compute_beta(self):
        """Computes Beta (portfolio sensitivity to market movements)."""
        if self.benchmark_returns is None:
            raise ValueError("Benchmark returns are required to compute Beta.")

        aligned_data = pd.concat([self.portfolio_returns, self.benchmark_returns], axis=1).dropna()

        aligned_portfolio_returns = aligned_data.iloc[:, 0]  
        aligned_benchmark_returns = aligned_data.iloc[:, 1]  

        covariance = np.cov(aligned_portfolio_returns, aligned_benchmark_returns)[0, 1]
        variance = np.var(aligned_benchmark_returns)

        return covariance / variance

    def compute_max_drawdown(self):
        """Calculates the Maximum Drawdown."""
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()

    def compute_tracking_error(self):
        """Computes the Tracking Error (how closely the portfolio follows the benchmark)."""
        if self.benchmark_returns is None:
            raise ValueError("Benchmark returns are required to compute Tracking Error.")

        aligned_data = pd.concat([self.portfolio_returns, self.benchmark_returns], axis=1).dropna()
        aligned_portfolio_returns = aligned_data.iloc[:, 0]
        aligned_benchmark_returns = aligned_data.iloc[:, 1]

        tracking_error = (aligned_portfolio_returns - aligned_benchmark_returns).std() * np.sqrt(252)

        return float(tracking_error)  

    def compute_marginal_var(self, asset_returns_df):
        """
        Computes the Marginal Value at Risk (MVaR) for all assets in the portfolio.
        
        :param asset_returns_df: DataFrame of asset returns (each column is an asset).
        :return: Series with MVaR for each asset.
        """
        if asset_returns_df.shape[0] != len(self.portfolio_returns):
            raise ValueError("Asset returns must have the same length as portfolio returns.")

        aligned_data = pd.concat([asset_returns_df, self.portfolio_returns], axis=1).dropna()
        
        asset_betas = aligned_data.cov().iloc[:-1, -1] / aligned_data.var().iloc[-1]

        portfolio_var = self.compute_var()

        marginal_var = asset_betas * portfolio_var

        return marginal_var

