import React, { useState } from 'react';
import { Search, ArrowRight, Loader, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { analyzeTicker } from '../api';
import ChartViewer from './ChartViewer';

const Dashboard = () => {
    const [ticker, setTicker] = useState('');
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!ticker) return;

        setLoading(true);
        setError(null);
        setData(null);

        try {
            const result = await analyzeTicker(ticker);
            setData(result);
        } catch (err) {
            setError(err.response?.data?.error || "Failed to analyze ticker. Please check the symbol and try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-8">
            {/* Search Section */}
            <div className="flex flex-col items-center justify-center space-y-4 py-8">
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
                    Stock Market Forecaster
                </h1>
                <p className="text-gray-400">
                    Advanced AI analysis, multi-horizon predictions, and sentiment scanning.
                </p>

                <form onSubmit={handleSearch} className="w-full max-w-md relative flex items-center">
                    <Search className="absolute left-4 text-gray-500" size={20} />
                    <input
                        type="text"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value.toUpperCase())}
                        placeholder="Enter Information (e.g., AAPL, TSLA)"
                        className="w-full bg-gray-800 border border-gray-700 text-white pl-12 pr-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all shadow-lg"
                    />
                    <button
                        type="submit"
                        disabled={loading}
                        className="absolute right-2 bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-lg transition-colors disabled:opacity-50"
                    >
                        {loading ? <Loader className="animate-spin" size={20} /> : <ArrowRight size={20} />}
                    </button>
                </form>

                {error && (
                    <div className="text-red-400 bg-red-900/20 px-4 py-2 rounded-lg border border-red-800">
                        {error}
                    </div>
                )}
            </div>

            {/* Results Section */}
            {data && (
                <div className="space-y-6 animate-fade-in">
                    {/* Metrics Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <MetricCard
                            label="Current Price"
                            value={`$${data.last_price.toFixed(2)}`}
                            subtext={data.last_date}
                            icon={<TrendingUp className="text-blue-400" />}
                        />
                        <MetricCard
                            label="5-Day Forecast"
                            value={`$${data.prediction_5d.price.toFixed(2)}`}
                            subtext={`${data.prediction_5d.change_pct > 0 ? '+' : ''}${data.prediction_5d.change_pct.toFixed(2)}%`}
                            trend={data.prediction_5d.change_pct > 0 ? 'up' : 'down'}
                        />
                        <MetricCard
                            label="Sentiment"
                            value={data.sentiment.label}
                            subtext={`Score: ${data.sentiment.score.toFixed(2)}`}
                            color={data.sentiment.score > 0 ? 'text-green-400' : 'text-red-400'}
                        />
                        <MetricCard
                            label="Backtest Return"
                            value={`${data.backtest.return_pct.toFixed(1)}%`}
                            subtext={`Win Rate: ${data.backtest.return_pct > 0 ? 'High' : 'Low'}`}
                            trend={data.backtest.return_pct > 0 ? 'up' : 'down'}
                        />
                    </div>

                    {/* Technical Analysis Section */}
                    {data.technical_analysis && (
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {/* Trend & Patterns */}
                            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg">
                                <h3 className="text-gray-400 text-sm font-medium mb-4">Technical Signals</h3>
                                <div className="space-y-4">
                                    <div className="flex justify-between items-center">
                                        <span className="text-gray-300">Trend</span>
                                        <span className={`px-2 py-1 rounded text-xs font-bold ${data.technical_analysis.trend === 'Up' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                                            {data.technical_analysis.trend ? data.technical_analysis.trend.toUpperCase() : 'NEUTRAL'}
                                        </span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-gray-300">Patterns</span>
                                        <span className="text-right text-gray-100 font-medium text-sm">
                                            {data.technical_analysis.patterns && data.technical_analysis.patterns.length > 0 ? data.technical_analysis.patterns.join(", ") : "None Detected"}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Key Levels */}
                            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg col-span-2">
                                <h3 className="text-gray-400 text-sm font-medium mb-4">Key Support & Resistance</h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <span className="text-xs text-gray-500 uppercase">Resistance</span>
                                        <div className="flex flex-wrap gap-2 mt-1">
                                            {data.technical_analysis.resistance_levels && data.technical_analysis.resistance_levels.length > 0 ? (
                                                data.technical_analysis.resistance_levels.map((level, i) => (
                                                    <span key={i} className="bg-red-900/20 text-red-400 px-2 py-1 rounded text-sm font-mono border border-red-900/30">
                                                        {typeof level === 'number' ? level.toFixed(2) : level}
                                                    </span>
                                                ))
                                            ) : <span className="text-gray-500 text-sm">None</span>}
                                        </div>
                                    </div>
                                    <div>
                                        <span className="text-xs text-gray-500 uppercase">Support</span>
                                        <div className="flex flex-wrap gap-2 mt-1">
                                            {data.technical_analysis.support_levels && data.technical_analysis.support_levels.length > 0 ? (
                                                data.technical_analysis.support_levels.map((level, i) => (
                                                    <span key={i} className="bg-green-900/20 text-green-400 px-2 py-1 rounded text-sm font-mono border border-green-900/30">
                                                        {typeof level === 'number' ? level.toFixed(2) : level}
                                                    </span>
                                                ))
                                            ) : <span className="text-gray-500 text-sm">None</span>}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Technical Indicators Grid */}
                    {data.technical_analysis && (
                        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                            {/* RSI */}
                            {data.technical_analysis.rsi !== null && data.technical_analysis.rsi !== undefined && (
                                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                    <span className="text-xs text-gray-500 uppercase">RSI</span>
                                    <div className={`text-xl font-bold mt-1 ${data.technical_analysis.rsi > 70 ? 'text-red-400' : data.technical_analysis.rsi < 30 ? 'text-green-400' : 'text-gray-100'}`}>
                                        {data.technical_analysis.rsi.toFixed(1)}
                                    </div>
                                    <span className="text-xs text-gray-500">{data.technical_analysis.rsi > 70 ? 'Overbought' : data.technical_analysis.rsi < 30 ? 'Oversold' : 'Neutral'}</span>
                                </div>
                            )}
                            {/* MACD */}
                            {data.technical_analysis.macd !== null && data.technical_analysis.macd !== undefined && (
                                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                    <span className="text-xs text-gray-500 uppercase">MACD</span>
                                    <div className={`text-xl font-bold mt-1 ${data.technical_analysis.macd > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        {data.technical_analysis.macd.toFixed(2)}
                                    </div>
                                    <span className="text-xs text-gray-500">Signal: {data.technical_analysis.macd_signal?.toFixed(2) || 'N/A'}</span>
                                </div>
                            )}
                            {/* Bollinger Bands */}
                            {data.technical_analysis.bb_upper !== null && data.technical_analysis.bb_upper !== undefined && (
                                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                    <span className="text-xs text-gray-500 uppercase">Bollinger</span>
                                    <div className="text-sm font-mono mt-1 text-gray-100">
                                        <div>U: {data.technical_analysis.bb_upper.toFixed(2)}</div>
                                        <div>M: {data.technical_analysis.bb_mid?.toFixed(2) || 'N/A'}</div>
                                        <div>L: {data.technical_analysis.bb_lower?.toFixed(2) || 'N/A'}</div>
                                    </div>
                                </div>
                            )}
                            {/* ATR */}
                            {data.technical_analysis.atr !== null && data.technical_analysis.atr !== undefined && (
                                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                    <span className="text-xs text-gray-500 uppercase">ATR</span>
                                    <div className="text-xl font-bold mt-1 text-gray-100">
                                        {data.technical_analysis.atr.toFixed(2)}
                                    </div>
                                    <span className="text-xs text-gray-500">Volatility</span>
                                </div>
                            )}
                            {/* Stochastic */}
                            {data.technical_analysis.stoch_k !== null && data.technical_analysis.stoch_k !== undefined && (
                                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                    <span className="text-xs text-gray-500 uppercase">Stochastic</span>
                                    <div className={`text-xl font-bold mt-1 ${data.technical_analysis.stoch_k > 80 ? 'text-red-400' : data.technical_analysis.stoch_k < 20 ? 'text-green-400' : 'text-gray-100'}`}>
                                        %K: {data.technical_analysis.stoch_k.toFixed(1)}
                                    </div>
                                    <span className="text-xs text-gray-500">%D: {data.technical_analysis.stoch_d?.toFixed(1) || 'N/A'}</span>
                                </div>
                            )}
                            {/* ADX */}
                            {data.technical_analysis.adx !== null && data.technical_analysis.adx !== undefined && (
                                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                    <span className="text-xs text-gray-500 uppercase">ADX</span>
                                    <div className={`text-xl font-bold mt-1 ${data.technical_analysis.adx > 25 ? 'text-blue-400' : 'text-gray-400'}`}>
                                        {data.technical_analysis.adx.toFixed(1)}
                                    </div>
                                    <span className="text-xs text-gray-500">{data.technical_analysis.adx > 25 ? 'Strong Trend' : 'Weak Trend'}</span>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Charts Area */}
                    <ChartViewer files={data.files} ticker={data.ticker} />
                </div>
            )}
        </div>
    );
};

const MetricCard = ({ label, value, subtext, icon, trend, color }) => (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-blue-500/50 transition-all shadow-lg">
        <div className="flex justify-between items-start mb-2">
            <span className="text-gray-400 text-sm font-medium">{label}</span>
            {icon || (trend && (
                trend === 'up' ? <TrendingUp className="text-green-400" size={20} /> : <TrendingDown className="text-red-400" size={20} />
            ))}
        </div>
        <div className={`text-2xl font-bold mb-1 ${color || 'text-white'}`}>
            {value}
        </div>
        <div className={`text-sm ${trend === 'up' ? 'text-green-400' : trend === 'down' ? 'text-red-400' : 'text-gray-500'}`}>
            {subtext}
        </div>
    </div>
);

export default Dashboard;
