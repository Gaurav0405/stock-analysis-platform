"""
Monte Carlo Simulation Module
"""
import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """Monte Carlo price path simulation."""
    
    def __init__(self, n_simulations: int = 1000, n_days: int = 30):
        self.n_simulations = n_simulations
        self.n_days = n_days
        self.simulations = None
        self.statistics = None
    
    def run_simulation(self, df: pd.DataFrame) -> dict:
        """Run Monte Carlo simulation based on historical returns."""
        closes = df["Close"].values
        returns = np.diff(closes) / closes[:-1]
        
        # Calculate return statistics
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        last_price = closes[-1]
        
        # Generate random walks
        simulations = np.zeros((self.n_simulations, self.n_days + 1))
        simulations[:, 0] = last_price
        
        for t in range(1, self.n_days + 1):
            random_returns = np.random.normal(mu, sigma, self.n_simulations)
            simulations[:, t] = simulations[:, t-1] * (1 + random_returns)
        
        self.simulations = simulations
        
        # Calculate statistics for each day
        final_prices = simulations[:, -1]
        
        self.statistics = {
            "starting_price": last_price,
            "mean_final": np.mean(final_prices),
            "median_final": np.median(final_prices),
            "std_final": np.std(final_prices),
            "percentile_5": np.percentile(final_prices, 5),
            "percentile_25": np.percentile(final_prices, 25),
            "percentile_75": np.percentile(final_prices, 75),
            "percentile_95": np.percentile(final_prices, 95),
            "prob_up": np.mean(final_prices > last_price) * 100,
            "prob_down": np.mean(final_prices < last_price) * 100,
            "expected_return": (np.mean(final_prices) - last_price) / last_price * 100,
            "max_return": (np.max(final_prices) - last_price) / last_price * 100,
            "max_loss": (np.min(final_prices) - last_price) / last_price * 100
        }
        
        return self.statistics
    
    def get_percentile_paths(self) -> dict:
        """Get percentile paths for visualization."""
        if self.simulations is None:
            return None
        
        return {
            "p5": np.percentile(self.simulations, 5, axis=0),
            "p25": np.percentile(self.simulations, 25, axis=0),
            "p50": np.percentile(self.simulations, 50, axis=0),
            "p75": np.percentile(self.simulations, 75, axis=0),
            "p95": np.percentile(self.simulations, 95, axis=0),
            "mean": np.mean(self.simulations, axis=0)
        }
