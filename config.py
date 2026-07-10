"""
Stock Forecaster Configuration
"""
import os
from datetime import datetime

# Output directory for charts
OUTPUT_DIR = "d:/stock/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date format for file naming
DATE_FORMAT = "%Y-%m-%d"

def get_output_path(ticker: str, chart_name: str, extension: str = "html") -> str:
    """Generate date-stamped output path: TICKER_YYYY-MM-DD_chartname.ext"""
    date_str = datetime.now().strftime(DATE_FORMAT)
    return f"{OUTPUT_DIR}/{ticker}_{date_str}_{chart_name}.{extension}"

# Default settings
DEFAULT_PERIOD = "5y"
DEFAULT_HORIZONS = [1, 5, 10, 30]
DIRECTION_THRESHOLD = 0.01  # 1% for UP/DOWN classification

# Backtest settings
INITIAL_CAPITAL = 10000
CONFIDENCE_THRESHOLD = 0.6

# Monte Carlo settings
MC_SIMULATIONS = 1000
MC_DAYS = 30

# Model settings
RANDOM_STATE = 42

