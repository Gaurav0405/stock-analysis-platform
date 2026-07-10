"""
Sentiment Analysis Module
"""
import numpy as np
import feedparser

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class SentimentAnalyzer:
    """Fetch and analyze news sentiment for a stock ticker."""
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
    
    def fetch_yahoo_rss(self, ticker: str, max_items: int = 20) -> list:
        """Fetch news headlines from Yahoo Finance RSS."""
        headlines = []
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                headlines.append({
                    'title': entry.get('title', ''),
                    'published': entry.get('published', ''),
                    'source': 'Yahoo Finance'
                })
        except Exception as e:
            print(f"Yahoo RSS fetch error: {e}")
        return headlines
    
    def fetch_google_news(self, ticker: str, max_items: int = 10) -> list:
        """Fetch news from Google News RSS."""
        headlines = []
        try:
            query = f"{ticker} stock"
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                headlines.append({
                    'title': entry.get('title', ''),
                    'published': entry.get('published', ''),
                    'source': 'Google News'
                })
        except Exception as e:
            print(f"Google News fetch error: {e}")
        return headlines
    
    def score_headline(self, headline: str) -> float:
        """Score a headline using VADER."""
        if not self.analyzer:
            return 0.0
        return self.analyzer.polarity_scores(headline)['compound']
    
    def analyze_ticker(self, ticker: str) -> dict:
        """Get comprehensive sentiment analysis."""
        yahoo_headlines = self.fetch_yahoo_rss(ticker)
        google_headlines = self.fetch_google_news(ticker)
        all_headlines = yahoo_headlines + google_headlines
        
        if not all_headlines:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'NEUTRAL',
                'headline_count': 0,
                'headlines': [],
                'positive_pct': 0,
                'negative_pct': 0,
                'neutral_pct': 100
            }
        
        scores = []
        scored_headlines = []
        for item in all_headlines:
            score = self.score_headline(item['title'])
            scores.append(score)
            scored_headlines.append({
                **item,
                'sentiment_score': score,
                'sentiment_label': 'POSITIVE' if score > 0.05 else ('NEGATIVE' if score < -0.05 else 'NEUTRAL')
            })
        
        avg_score = np.mean(scores)
        positive_count = sum(1 for s in scores if s > 0.05)
        negative_count = sum(1 for s in scores if s < -0.05)
        neutral_count = len(scores) - positive_count - negative_count
        
        if avg_score > 0.1:
            label = 'BULLISH'
        elif avg_score > 0.05:
            label = 'SLIGHTLY BULLISH'
        elif avg_score < -0.1:
            label = 'BEARISH'
        elif avg_score < -0.05:
            label = 'SLIGHTLY BEARISH'
        else:
            label = 'NEUTRAL'
        
        return {
            'sentiment_score': avg_score,
            'sentiment_label': label,
            'headline_count': len(all_headlines),
            'headlines': scored_headlines[:10],
            'positive_pct': (positive_count / len(scores)) * 100,
            'negative_pct': (negative_count / len(scores)) * 100,
            'neutral_pct': (neutral_count / len(scores)) * 100
        }
