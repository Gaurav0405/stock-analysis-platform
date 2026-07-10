"""
Data Download Module
"""
import pandas as pd
import numpy as np
import yfinance as yf


def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download and prepare stock data."""
    print(f"\nDownloading {ticker} data...")
    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    raw = raw.reset_index()
    
    df = pd.DataFrame({
        "Date": raw["Date"].values,
        "Open": np.array(raw["Open"], dtype=float).reshape(-1),
        "High": np.array(raw["High"], dtype=float).reshape(-1),
        "Low": np.array(raw["Low"], dtype=float).reshape(-1),
        "Close": np.array(raw["Close"], dtype=float).reshape(-1),
        "Volume": np.array(raw["Volume"], dtype=float).reshape(-1),
    })
    
    df.set_index("Date", inplace=True)
    df = df.dropna()
    print(f"Downloaded {len(df)} days of data")
    return df


def download_market_context() -> dict:
    """Download S&P 500 and VIX data for market context."""
    market_data = {}
    
    try:
        sp500 = yf.download("^GSPC", period="5y", interval="1d", progress=False)
        if not sp500.empty:
            sp500_close = np.array(sp500["Close"], dtype=float).reshape(-1)
            market_data["sp500"] = {
                "close": pd.Series(sp500_close, index=sp500.index),
                "ret": pd.Series(sp500_close, index=sp500.index).pct_change()
            }
            print("  ✓ S&P 500 data loaded")
    except Exception as e:
        print(f"  ✗ S&P 500 download failed: {e}")
    
    try:
        vix = yf.download("^VIX", period="5y", interval="1d", progress=False)
        if not vix.empty:
            vix_close = np.array(vix["Close"], dtype=float).reshape(-1)
            market_data["vix"] = {
                "close": pd.Series(vix_close, index=vix.index)
            }
            print("  ✓ VIX data loaded")
    except Exception as e:
        print(f"  ✗ VIX download failed: {e}")
    
    return market_data


def get_fundamental_data(ticker: str) -> dict:
    """Get fundamental data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fundamentals = {
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "eps": info.get("trailingEps", None),
            "dividend_yield": info.get("dividendYield", None),
            "market_cap": info.get("marketCap", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", None),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "company_name": info.get("longName", ticker)
        }
        return fundamentals
    except Exception as e:
        print(f"  ✗ Fundamental data error: {e}")
        return {}
