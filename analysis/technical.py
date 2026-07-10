"""
Technical Analysis Module
"""
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    df = df.copy()
    
    # RSI
    df["rsi"] = RSIIndicator(df["Close"]).rsi()
    
    # MACD
    macd = MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(df["Close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
    df["bb_pct"] = (df["Close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])
    
    # ATR
    df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    
    # Stochastic
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    # OBV
    df["obv"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["obv_pct"] = df["obv"].pct_change()

    # ADX (Trend Strength)
    adx = ADXIndicator(df["High"], df["Low"], df["Close"])
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    
    return df


def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """Calculate support and resistance levels using pivot points."""
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    
    # Find local minima and maxima
    supports = []
    resistances = []
    
    for i in range(window, len(closes) - window):
        # Local minimum (support)
        if lows[i] == min(lows[i-window:i+window+1]):
            supports.append(lows[i])
        
        # Local maximum (resistance)
        if highs[i] == max(highs[i-window:i+window+1]):
            resistances.append(highs[i])
    
    # Get strongest levels (most frequently tested)
    current_price = closes[-1]
    
    # Filter to recent relevant levels
    support_levels = sorted([s for s in supports if s < current_price], reverse=True)[:3]
    resistance_levels = sorted([r for r in resistances if r > current_price])[:3]
    
    # Calculate key levels
    pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
    r1 = 2 * pivot - lows[-1]
    s1 = 2 * pivot - highs[-1]
    r2 = pivot + (highs[-1] - lows[-1])
    s2 = pivot - (highs[-1] - lows[-1])
    
    return {
        "current_price": current_price,
        "pivot": pivot,
        "resistance_1": r1,
        "resistance_2": r2,
        "support_1": s1,
        "support_2": s2,
        "detected_supports": support_levels,
        "detected_resistances": resistance_levels
    }


def detect_peaks_troughs(series: np.ndarray, distance: int = 8) -> tuple:
    """Detect peaks and troughs in a price series."""
    # Dynamic prominence based on volatility
    dynamic_prom = np.std(series) * 0.3
    # Use simple peak finding if scipy is available, else manual
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(series, distance=distance, prominence=dynamic_prom)
    troughs, _ = find_peaks(-series, distance=distance, prominence=dynamic_prom)
    return peaks, troughs


def detect_patterns(series: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> list:
    """Detect chart patterns (Double Top/Bottom, Head & Shoulders)."""
    patterns = []
    
    if len(peaks) >= 2:
        if abs(series[peaks[0]] - series[peaks[-1]]) < 0.03 * np.mean(series):
            # Check if they are somewhat recent or dominant
            patterns.append("Double Top")
            
    if len(troughs) >= 2:
        if abs(series[troughs[0]] - series[troughs[-1]]) < 0.03 * np.mean(series):
            patterns.append("Double Bottom")
            
    if len(peaks) >= 3:
        # Check for Head and Shoulders (Middle peak highest, shoulders lower)
        # We need the last 3 peaks: p1(shoulder), p2(head), p3(shoulder)
        # But peaks are indices, not sorted by magnitude.
        # Let's simple check the last 3 detected peaks
        if len(peaks) >= 3:
            p3_idx = peaks[-3:]
            p_vals = series[p3_idx]
            # Head (middle) is highest
            if p_vals[1] > p_vals[0] and p_vals[1] > p_vals[2]:
                # Shoulders are roughly equal (within 5%)
                if abs(p_vals[0] - p_vals[2]) < 0.05 * p_vals[1]:
                     patterns.append("Head and Shoulders")
    
    return patterns


def detect_trend(series: np.ndarray) -> str:
    """Detect trend using linear regression slope."""
    if len(series) < 2:
        return "Neutral"
    x = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]
    return "Up" if slope > 0 else "Down"
