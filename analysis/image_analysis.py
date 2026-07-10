"""
Chart Image Analysis Module
Extracts price data from chart images using computer vision and OCR
"""

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pytesseract
import os
import re

# Set Tesseract path for Windows if default
possible_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    os.path.expanduser(r'~\AppData\Local\Tesseract-OCR\tesseract.exe')
]

for path in possible_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break


def extract_y_axis_prices(img_bgr):
    """
    Extract Y-axis labels using OCR to determine price scale.
    Returns: (min_price, max_price, pixel_min, pixel_max) or None
    """
    try:
        h, w = img_bgr.shape[:2]
        # Crop rightmost 15% for Y-axis labels
        right_margin = int(w * 0.85)
        roi = img_bgr[:, right_margin:]
        
        # Preprocessing for OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Invert if dark background (simple check: mean intensity < 127)
        if np.mean(gray) < 127:
            gray = cv2.bitwise_not(gray)
            
        # Rescale for better OCR
        scale = 2
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Thrombshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # OCR
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
        text = pytesseract.image_to_string(thresh, config=config)
        
        # Parse numbers and their positions
        lines = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        
        found_prices = []
        for i in range(len(lines['text'])):
            txt = lines['text'][i].strip()
            if not txt: continue
            
            # Simple float regex
            if re.match(r'^\d+\.?\d*$', txt):
                try:
                    val = float(txt)
                    # Y-coordinate (center of the box)
                    y = (lines['top'][i] + lines['height'][i]/2) / scale
                    found_prices.append((y, val))
                except:
                    pass
                    
        if len(found_prices) >= 2:
            # Sort by Y (top to bottom)
            # In image coords, usually smaller Y = higher pixel = higher price? 
            # Wait, image Y=0 is TOP. Higher price is usually at TOP.
            # So smaller Y = Higher Price.
            
            # Linear regression: Price = m * y + c
            ys = np.array([p[0] for p in found_prices])
            vals = np.array([p[1] for p in found_prices])
            
            # Fit
            m, c = np.polyfit(ys, vals, 1)
            
            # Start and End of image
            price_top = m * 0 + c
            price_bottom = m * h + c
            
            return price_top, price_bottom
            
    except Exception as e:
        print(f"OCR Failed (this is expected if Tesseract not installed): {e}")
        return None
        
    return None


def extract_prices_from_image(image_path):
    """
    Extract price series from a chart image using color-based line detection.
    Returns: numpy array of price values
    """
    # Read image
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError("Could not read image. Check path / file format.")
    
    h, w = img_bgr.shape[:2]
    
    # Convert to HSV for color-based line detection
    # Convert to HSV for color-based line detection
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # "Ultra" Extraction (User Provided Logic + Robust Fallbacks)
    # 1. TradingView Blue (User preferred)
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([140, 255, 255])
    
    # 2. Green
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])
    
    # 3. Red/Orange
    lower_red = np.array([0, 80, 80])
    upper_red = np.array([20, 255, 255])
    
    color_ranges = [
        (lower_blue, upper_blue),
        (lower_green, upper_green),
        (lower_red, upper_red)
    ]
    
    best_xs, best_ys = [], []
    
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        xs, ys = [], []
        for x in range(mask.shape[1]):
            col = mask[:, x]
            y_indices = np.where(col > 0)[0]
            if len(y_indices) > 0:
                ys.append(np.mean(y_indices))
                xs.append(x)
        
        if len(xs) > len(best_xs):
            best_xs, best_ys = xs, ys

    # Fallback to Canny if color fails
    if len(best_xs) < 20:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        xs, ys = [], []
        for x in range(edges.shape[1]):
            col = edges[:, x]
            y_indices = np.where(col > 0)[0]
            if len(y_indices) > 0:
                ys.append(np.mean(y_indices))
                xs.append(x)
        if len(xs) > len(best_xs):
            best_xs, best_ys = xs, ys

    if len(best_xs) < 20:
         raise RuntimeError("Could not detect price line. Try a higher resolution screenshot.")

    xs = np.array(best_xs)
    ys = np.array(best_ys)

    # Spline smoothing
    try:
        spline = UnivariateSpline(xs, ys, s=len(xs) * 0.5)
        ys_smooth = spline(xs)
    except:
        ys_smooth = ys

    # Normalize to price range
    # Attempt OCR scaling
    price_range = extract_y_axis_prices(img_bgr)
    
    if price_range:
        # OCR success!
        p_top, p_bottom = price_range
        # ys_norm is 0 at min Y (top of line) and 1 at max Y (bottom of line)
        # Wait, ys are pixel coordinates. 0 is top.
        # So we map ys directly using the fitted linear model if possible, 
        # or map min/max ys to price range.
        
        # Let's verify correlation.
        # Image Y=0 -> p_top (High Price)
        # Image Y=h -> p_bottom (Low Price)
        
        # Price = p_top + (y / h) * (p_bottom - p_top)
        prices = p_top + (ys_smooth / h) * (p_bottom - p_top)
        print(f"OCR Scaling Used: {p_bottom:.2f} to {p_top:.2f}")
        
    else:
        # Synthetic fallback
        # Map to 100-150 range
        ys_norm = (ys_smooth - ys_smooth.min()) / (ys_smooth.max() - ys_smooth.min() + 1e-9)
        # 1-ys_norm because min Y (0) is top (high price)
        prices = (1 - ys_norm) * 50 + 100 
    
    full_x = np.arange(xs.min(), xs.max() + 1)
    full_prices = np.interp(full_x, xs, prices)
    
    return full_prices


def compute_technical_indicators(prices):
    """Compute technical indicators from price series."""
    s = pd.Series(prices)
    
    # SMA
    sma20 = s.rolling(window=20, min_periods=1).mean()
    
    # Bollinger Bands
    std20 = s.rolling(window=20, min_periods=1).std().fillna(0)
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # RSI
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # MACD
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    return {
        'sma20': sma20.values,
        'upper_bb': upper_bb.values,
        'lower_bb': lower_bb.values,
        'rsi': rsi.values,
        'macd': macd.values,
        'macd_signal': macd_signal.values,
        'macd_hist': macd_hist.values
    }


def detect_peaks_troughs(prices):
    """Detect peaks and troughs in price series."""
    dynamic_prom = np.std(prices) * 0.3
    peaks, _ = find_peaks(prices, distance=8, prominence=dynamic_prom)
    troughs, _ = find_peaks(-prices, distance=8, prominence=dynamic_prom)
    return peaks, troughs


def detect_patterns(prices, peaks, troughs):
    """Detect chart patterns."""
    patterns = []
    
    if len(peaks) >= 2:
        if abs(prices[peaks[0]] - prices[peaks[1]]) < 0.03 * np.mean(prices):
            patterns.append("Double Top")
    
    if len(troughs) >= 2:
        if abs(prices[troughs[0]] - prices[troughs[1]]) < 0.03 * np.mean(prices):
            patterns.append("Double Bottom")
    
    if len(peaks) >= 3:
        p3 = peaks[np.argsort(prices[peaks])[-3:]]
        if prices[p3[1]] > prices[p3[0]] and prices[p3[1]] > prices[p3[2]]:
            patterns.append("Head and Shoulders (possible)")
    
    return patterns


def calculate_support_resistance(prices, peaks, troughs):
    """Calculate support and resistance levels."""
    P_high = float(np.max(prices))
    P_low = float(np.min(prices))
    P_close = float(prices[-1])
    
    pivot = (P_high + P_low + P_close) / 3.0
    r1 = 2 * pivot - P_low
    s1 = 2 * pivot - P_high
    r2 = pivot + (P_high - P_low)
    s2 = pivot - (P_high - P_low)
    
    # Trendlines
    support_fit = None
    resistance_fit = None
    
    if len(troughs) >= 2:
        xs_tr = np.array(troughs)
        ys_tr = prices[xs_tr]
        support_fit = np.polyfit(xs_tr, ys_tr, 1).tolist()
    
    if len(peaks) >= 2:
        xs_pk = np.array(peaks)
        ys_pk = prices[xs_pk]
        resistance_fit = np.polyfit(xs_pk, ys_pk, 1).tolist()
    
    return {
        'pivot': round(pivot, 2),
        'resistance_1': round(r1, 2),
        'resistance_2': round(r2, 2),
        'support_1': round(s1, 2),
        'support_2': round(s2, 2),
        'support_fit': support_fit,
        'resistance_fit': resistance_fit
    }


def analyze_chart_image(image_path):
    """
    Full analysis pipeline for chart image.
    Returns dict with all analysis results.
    """
    # Extract prices
    prices = extract_prices_from_image(image_path)
    
    
    # Technical indicators
    indicators = compute_technical_indicators(prices)
    
    # Import shared analysis logic
    from analysis.technical import detect_peaks_troughs, detect_patterns, detect_trend
    
    # Peaks and troughs
    peaks, troughs = detect_peaks_troughs(prices)
    
    # Patterns
    patterns = detect_patterns(prices, peaks, troughs)
    
    # Trend
    trend = detect_trend(prices)

    # Support/Resistance (using basic calculation here or shared?)
    # We can use the robust shared one if we had OHLC, but here we only have C.
    # So we use the robust local logic which was updated to match technical.py style or keep as is.
    # The existing calculate_support_resistance in this file is good for single-series.
    sr_levels = calculate_support_resistance(prices, peaks, troughs)
    
    # Current values
    last_price = float(prices[-1])
    last_rsi = float(indicators['rsi'][-1])
    last_macd = float(indicators['macd'][-1])
    
    return {
        'prices': prices.tolist(),
        'price_count': len(prices),
        'last_price': last_price,
        'trend': trend,
        'volatility': float(np.std(prices)),
        'indicators': {
            'rsi': last_rsi,
            'macd': last_macd,
            'sma20': float(indicators['sma20'][-1]),
            'upper_bb': float(indicators['upper_bb'][-1]),
            'lower_bb': float(indicators['lower_bb'][-1])
        },
        'support_resistance': sr_levels,
        'patterns': patterns,
        'peaks_count': len(peaks),
        'troughs_count': len(troughs)
    }
