"""
Backtesting Engine Module
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import OUTPUT_DIR, INITIAL_CAPITAL, get_output_path


class BacktestEngine:
    """Backtest a trading strategy based on model predictions."""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, df: pd.DataFrame, predictions: np.ndarray, 
                     confidence_threshold: float = 0.6, 
                     direction_predictions: np.ndarray = None,
                     direction_confidence: np.ndarray = None) -> dict:
        """Run backtest with prediction-based strategy."""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity = [capital]
        
        closes = df["Close"].values
        
        for i in range(len(predictions)):
            if i >= len(closes) - 1:
                break
                
            current_price = closes[i]
            next_price = closes[i + 1]
            pred = predictions[i]
            
            if direction_predictions is not None and direction_confidence is not None:
                conf = direction_confidence[i] if i < len(direction_confidence) else 0.5
                dir_pred = direction_predictions[i] if i < len(direction_predictions) else 0
                
                if dir_pred == 1 and conf >= confidence_threshold:
                    signal = 1
                elif dir_pred == -1 and conf >= confidence_threshold:
                    signal = -1
                else:
                    signal = 0
            else:
                signal = 1 if pred > 0 else (-1 if pred < 0 else 0)
            
            if signal == 1 and position <= 0:
                if position == -1:
                    pnl = (entry_price - current_price) * (capital / entry_price)
                    capital += pnl
                    trades.append({"type": "close_short", "price": current_price, "pnl": pnl})
                
                position = 1
                entry_price = current_price
                trades.append({"type": "long", "price": current_price, "pnl": 0})
                
            elif signal == -1 and position >= 0:
                if position == 1:
                    pnl = (current_price - entry_price) * (capital / entry_price)
                    capital += pnl
                    trades.append({"type": "close_long", "price": current_price, "pnl": pnl})
                
                position = -1
                entry_price = current_price
                trades.append({"type": "short", "price": current_price, "pnl": 0})
            
            if position == 1:
                unrealized = (next_price - entry_price) * (capital / entry_price)
                equity.append(capital + unrealized)
            elif position == -1:
                unrealized = (entry_price - next_price) * (capital / entry_price)
                equity.append(capital + unrealized)
            else:
                equity.append(capital)
        
        if position == 1:
            final_pnl = (closes[-1] - entry_price) * (capital / entry_price)
            capital += final_pnl
        elif position == -1:
            final_pnl = (entry_price - closes[-1]) * (capital / entry_price)
            capital += final_pnl
        
        self.trades = trades
        self.equity_curve = equity
        
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        std_returns = np.std(returns)
        sharpe = np.mean(returns) / std_returns * np.sqrt(252) if std_returns > 0 else 0
        
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) * 100
        
        closed_trades = [t for t in trades if t["pnl"] != 0]
        wins = sum(1 for t in closed_trades if t["pnl"] > 0)
        win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0
        
        return {
            "final_capital": capital,
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "win_rate_pct": win_rate,
            "total_trades": len(trades),
            "equity_curve": equity
        }
    
    def plot_equity_curve(self, ticker: str):
        """Plot equity curve."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.equity_curve,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title=f'{ticker} Backtest Equity Curve',
            xaxis_title='Trading Days',
            yaxis_title='Portfolio Value ($)',
            height=400
        )
        filepath = get_output_path(ticker, "backtest")
        fig.write_html(filepath)
        print(f"  Saved: {filepath}")
