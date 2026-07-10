"""
Feature Engineering Module
"""
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame, sentiment_score: float = 0.0, 
                   market_data: dict = None) -> tuple:
    """Build feature set for ML models."""
    d = df.copy()
    
    # Returns
    d["ret_1d"] = d["Close"].pct_change()
    
    # Lagged features
    for lag in [2, 3, 4, 5]:
        d[f"lag_{lag}"] = d["Close"].shift(lag)
    
    # Moving averages and volatility
    for w in [5, 10, 20]:
        d[f"sma_{w}"] = d["Close"].rolling(w).mean()
        d[f"ema_{w}"] = d["Close"].ewm(span=w).mean()
        d[f"vol_{w}"] = d["ret_1d"].rolling(w).std()
    
    # Sentiment
    d["sentiment"] = sentiment_score
    
    # Market context features
    if market_data:
        if "sp500" in market_data:
            d["sp500_ret"] = market_data["sp500"]["ret"].reindex(d.index).fillna(0)
            d["stock_vs_market"] = d["ret_1d"] - d["sp500_ret"]
        if "vix" in market_data:
            d["vix_level"] = market_data["vix"]["close"].reindex(d.index).fillna(method='ffill')
    
    d = d.dropna()
    exclude_cols = ["Target_1", "Target_5", "Target_10", "Target_30", 
                    "Target_High", "Target_Low", "Direction"]
    features = [c for c in d.columns if c not in exclude_cols]
    
    return d, features


def create_multi_horizon_targets(df: pd.DataFrame, horizons: list = [1, 5, 10, 30]) -> pd.DataFrame:
    """Create target variables for multiple prediction horizons."""
    df = df.copy()
    for h in horizons:
        df[f"Target_{h}"] = df["Close"].shift(-h) - df["Close"]
    return df


def create_price_range_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create targets for next-day high and low prediction."""
    df = df.copy()
    df["Target_High"] = df["High"].shift(-1) - df["Close"]
    df["Target_Low"] = df["Low"].shift(-1) - df["Close"]
    return df


def create_direction_target(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Create classification target: UP (>1%), DOWN (<-1%), FLAT."""
    df = df.copy()
    next_ret = df["Close"].pct_change().shift(-1)
    
    df["Direction"] = 0
    df.loc[next_ret > threshold, "Direction"] = 1
    df.loc[next_ret < -threshold, "Direction"] = -1
    
    return df
