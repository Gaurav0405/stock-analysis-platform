"""
Export Utilities Module
"""
import pandas as pd
import os
from datetime import datetime

from config import OUTPUT_DIR, get_output_path


def export_predictions_to_csv(ticker: str, predictions: dict, 
                               sentiment: dict, backtest: dict,
                               fundamentals: dict = None) -> str:
    """Export all predictions and results to CSV."""
    data = {
        "Ticker": [ticker],
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Sentiment_Score": [sentiment["sentiment_score"]],
        "Sentiment_Label": [sentiment["sentiment_label"]]
    }
    
    for h, pred in predictions.items():
        data[f"Pred_{h}d_Change"] = [pred["mean"]]
        data[f"Pred_{h}d_Std"] = [pred["std"]]
    
    data["Backtest_Return_Pct"] = [backtest["total_return_pct"]]
    data["Backtest_Sharpe"] = [backtest["sharpe_ratio"]]
    data["Backtest_MaxDrawdown"] = [backtest["max_drawdown_pct"]]
    data["Backtest_WinRate"] = [backtest["win_rate_pct"]]
    
    if fundamentals:
        data["PE_Ratio"] = [fundamentals.get("pe_ratio")]
        data["EPS"] = [fundamentals.get("eps")]
        data["Market_Cap"] = [fundamentals.get("market_cap")]
    
    df = pd.DataFrame(data)
    filepath = get_output_path(ticker, "predictions", "csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    return filepath


def export_backtest_trades(ticker: str, trades: list) -> str:
    """Export backtest trade log to CSV."""
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    filepath = get_output_path(ticker, "trades", "csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    return filepath


def export_monte_carlo_results(ticker: str, stats: dict, paths: dict) -> str:
    """Export Monte Carlo summary to CSV."""
    data = {
        "Metric": list(stats.keys()),
        "Value": list(stats.values())
    }
    df = pd.DataFrame(data)
    filepath = get_output_path(ticker, "monte_carlo_stats", "csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    return filepath


def run_multi_ticker_analysis(tickers: list, analyze_func) -> pd.DataFrame:
    """Run analysis on multiple tickers and combine results."""
    results = []
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {ticker}")
        print(f"{'='*60}")
        try:
            result = analyze_func(ticker)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
    
    if results:
        summary_df = pd.DataFrame(results)
        filepath = f"{OUTPUT_DIR}/multi_ticker_summary.csv"
        summary_df.to_csv(filepath, index=False)
        print(f"\n  Multi-ticker summary saved: {filepath}")
        return summary_df
    
    return None
