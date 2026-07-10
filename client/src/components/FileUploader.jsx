import React, { useState, useCallback } from 'react';
import { Upload, FileText, AlertCircle, Loader, TrendingUp, TrendingDown, Minus, Image as ImageIcon, FileSpreadsheet } from 'lucide-react';
import { uploadFile, uploadImage } from '../api';
import ChartViewer from './ChartViewer';

const FileUploader = () => {
    const [activeTab, setActiveTab] = useState('csv'); // 'csv' or 'image'
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    const [data, setData] = useState(null);

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setError(null);
            setData(null);
        }
    }, []);

    const handleChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError(null);
            setData(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        setError(null);
        setData(null);
        try {
            let result;
            if (activeTab === 'csv') {
                result = await uploadFile(file);
            } else {
                result = await uploadImage(file);
            }
            setData(result);
            setFile(null);
        } catch (err) {
            setError(err.response?.data?.error || "Upload failed. Please check the file format.");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="space-y-8">
            {/* Upload Section */}
            <div className="flex flex-col items-center justify-center space-y-4 py-8">
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
                    Upload Market Data
                </h1>
                <p className="text-gray-400">
                    Upload CSV data or Chart Images for AI-powered analysis.
                </p>

                {/* Tabs */}
                <div className="flex space-x-4 bg-gray-800/50 p-1 rounded-xl border border-gray-700">
                    <button
                        onClick={() => { setActiveTab('csv'); setFile(null); setData(null); setError(null); }}
                        className={`flex items-center px-6 py-2 rounded-lg transition-all ${activeTab === 'csv'
                            ? 'bg-blue-600 text-white shadow-lg'
                            : 'text-gray-400 hover:text-white hover:bg-gray-700'
                            }`}
                    >
                        <FileSpreadsheet size={18} className="mr-2" />
                        CSV Upload
                    </button>
                    <button
                        onClick={() => { setActiveTab('image'); setFile(null); setData(null); setError(null); }}
                        className={`flex items-center px-6 py-2 rounded-lg transition-all ${activeTab === 'image'
                            ? 'bg-purple-600 text-white shadow-lg'
                            : 'text-gray-400 hover:text-white hover:bg-gray-700'
                            }`}
                    >
                        <ImageIcon size={18} className="mr-2" />
                        Chart Image
                    </button>
                </div>

                <div className="w-full max-w-md">
                    <div
                        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 ${dragActive
                            ? 'border-blue-500 bg-blue-500/10'
                            : 'border-gray-700 hover:border-blue-400 hover:bg-gray-800'
                            }`}
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                    >
                        <input
                            type="file"
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                            onChange={handleChange}
                            accept={activeTab === 'csv' ? ".csv" : ".png,.jpg,.jpeg"}
                        />

                        <div className="flex flex-col items-center pointer-events-none">
                            {file ? (
                                activeTab === 'csv' ? (
                                    <FileText className="w-12 h-12 text-blue-400 mb-3" />
                                ) : (
                                    <ImageIcon className="w-12 h-12 text-purple-400 mb-3" />
                                )
                            ) : (
                                <Upload className="w-12 h-12 text-gray-500 mb-3" />
                            )}

                            <p className="text-lg font-medium text-white mb-1">
                                {file ? file.name : `Drag & Drop ${activeTab === 'csv' ? 'CSV' : 'Image'} File`}
                            </p>
                            <p className="text-gray-400 text-sm">
                                {file ? `${(file.size / 1024).toFixed(2)} KB` : "or click to browse"}
                            </p>
                        </div>
                    </div>

                    {file && (
                        <button
                            onClick={handleUpload}
                            disabled={uploading}
                            className={`w-full mt-4 py-3 rounded-xl font-bold text-white transition-all flex items-center justify-center space-x-2 ${uploading
                                ? 'bg-gray-700 cursor-not-allowed'
                                : activeTab === 'csv' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-purple-600 hover:bg-purple-700'
                                }`}
                        >
                            {uploading ? (
                                <>
                                    <Loader className="animate-spin" />
                                    <span>Analyzing...</span>
                                </>
                            ) : (
                                <>
                                    <Upload size={20} />
                                    <span>Generate Analysis</span>
                                </>
                            )}
                        </button>
                    )}

                    {error && (
                        <div className="mt-4 text-red-400 bg-red-900/20 px-4 py-2 rounded-lg border border-red-800 flex items-center">
                            <AlertCircle className="mr-2" size={18} />
                            {error}
                        </div>
                    )}
                </div>
            </div>

            {/* Results Section - Same as Dashboard */}
            {data && (
                <div className="space-y-6 animate-fade-in">
                    {/* Metrics Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <MetricCard
                            label="Current Price"
                            value={`$${data.last_price?.toFixed(2)}`}
                            subtext={data.last_date}
                            icon={<TrendingUp className="text-blue-400" />}
                        />
                        <MetricCard
                            label="5-Day Forecast"
                            value={`$${data.prediction_5d?.price?.toFixed(2)}`}
                            subtext={`${data.prediction_5d?.change_pct > 0 ? '+' : ''}${data.prediction_5d?.change_pct?.toFixed(2)}%`}
                            trend={data.prediction_5d?.change_pct > 0 ? 'up' : 'down'}
                        />
                        <MetricCard
                            label="Sentiment"
                            value={data.sentiment?.sentiment_label || data.sentiment?.label}
                            subtext={`Score: ${(data.sentiment?.sentiment_score ?? data.sentiment?.score ?? 0).toFixed(2)}`}
                            color={(data.sentiment?.sentiment_score ?? data.sentiment?.score ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}
                        />
                        <MetricCard
                            label="Backtest Return"
                            value={`${data.backtest?.return_pct?.toFixed(1)}%`}
                            subtext={`Win Rate: ${data.backtest?.win_rate?.toFixed(1)}%`}
                            trend={data.backtest?.return_pct > 0 ? 'up' : 'down'}
                        />
                    </div>

                    {/* Technical Indicators Grid */}
                    {data.technical_analysis && (
                        <div className="mt-6">
                            <h3 className="text-lg font-semibold text-white mb-4">📊 Technical Indicators</h3>
                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                                {/* RSI */}
                                {data.technical_analysis.rsi != null && (
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <span className="text-xs text-gray-500 uppercase">RSI</span>
                                        <div className={`text-xl font-bold mt-1 ${data.technical_analysis.rsi > 70 ? 'text-red-400' : data.technical_analysis.rsi < 30 ? 'text-green-400' : 'text-gray-100'}`}>
                                            {data.technical_analysis.rsi.toFixed(1)}
                                        </div>
                                        <span className="text-xs text-gray-500">{data.technical_analysis.rsi > 70 ? 'Overbought' : data.technical_analysis.rsi < 30 ? 'Oversold' : 'Neutral'}</span>
                                    </div>
                                )}
                                {/* MACD */}
                                {data.technical_analysis.macd != null && (
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <span className="text-xs text-gray-500 uppercase">MACD</span>
                                        <div className={`text-xl font-bold mt-1 ${data.technical_analysis.macd > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                            {data.technical_analysis.macd.toFixed(2)}
                                        </div>
                                        <span className="text-xs text-gray-500">Signal: {data.technical_analysis.macd_signal?.toFixed(2) || 'N/A'}</span>
                                    </div>
                                )}
                                {/* Bollinger Bands */}
                                {data.technical_analysis.bb_upper != null && (
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <span className="text-xs text-gray-500 uppercase">Bollinger Bands</span>
                                        <div className="text-sm font-mono mt-1 text-gray-100">
                                            <div>U: {data.technical_analysis.bb_upper.toFixed(2)}</div>
                                            <div>M: {data.technical_analysis.bb_mid?.toFixed(2) || 'N/A'}</div>
                                            <div>L: {data.technical_analysis.bb_lower?.toFixed(2) || 'N/A'}</div>
                                        </div>
                                    </div>
                                )}
                                {/* ATR */}
                                {data.technical_analysis.atr != null && (
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <span className="text-xs text-gray-500 uppercase">ATR</span>
                                        <div className="text-xl font-bold mt-1 text-gray-100">
                                            {data.technical_analysis.atr.toFixed(2)}
                                        </div>
                                        <span className="text-xs text-gray-500">Volatility</span>
                                    </div>
                                )}
                                {/* Stochastic */}
                                {data.technical_analysis.stoch_k != null && (
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <span className="text-xs text-gray-500 uppercase">Stochastic</span>
                                        <div className={`text-xl font-bold mt-1 ${data.technical_analysis.stoch_k > 80 ? 'text-red-400' : data.technical_analysis.stoch_k < 20 ? 'text-green-400' : 'text-gray-100'}`}>
                                            %K: {data.technical_analysis.stoch_k.toFixed(1)}
                                        </div>
                                        <span className="text-xs text-gray-500">%D: {data.technical_analysis.stoch_d?.toFixed(1) || 'N/A'}</span>
                                    </div>
                                )}
                                {/* ADX */}
                                {data.technical_analysis.adx != null && (
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <span className="text-xs text-gray-500 uppercase">ADX</span>
                                        <div className={`text-xl font-bold mt-1 ${data.technical_analysis.adx > 25 ? 'text-blue-400' : 'text-gray-400'}`}>
                                            {data.technical_analysis.adx.toFixed(1)}
                                        </div>
                                        <span className="text-xs text-gray-500">{data.technical_analysis.adx > 25 ? 'Strong Trend' : 'Weak Trend'}</span>
                                    </div>
                                )}
                            </div>

                            {/* Trend & Patterns */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                    <span className="text-xs text-gray-500 uppercase">Trend</span>
                                    <div className={`text-xl font-bold mt-1 ${data.technical_analysis.trend === 'Up' ? 'text-green-400' : 'text-red-400'}`}>
                                        {data.technical_analysis.trend || 'N/A'}
                                    </div>
                                </div>
                                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                    <span className="text-xs text-gray-500 uppercase">Patterns Detected</span>
                                    <div className="text-lg font-medium mt-1 text-gray-100">
                                        {data.technical_analysis.patterns?.length > 0
                                            ? data.technical_analysis.patterns.join(', ')
                                            : 'None'
                                        }
                                    </div>
                                </div>
                            </div>

                            {/* Support/Resistance */}
                            {(data.technical_analysis.support_levels?.length > 0 || data.technical_analysis.resistance_levels?.length > 0) && (
                                <div className="grid grid-cols-2 gap-4 mt-4">
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <span className="text-xs text-gray-500 uppercase">Support Levels</span>
                                        <div className="flex flex-wrap gap-2 mt-2">
                                            {data.technical_analysis.support_levels?.map((level, i) => (
                                                <span key={i} className="bg-green-900/30 text-green-400 px-2 py-1 rounded text-sm font-mono">
                                                    {typeof level === 'number' ? level.toFixed(2) : level}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <span className="text-xs text-gray-500 uppercase">Resistance Levels</span>
                                        <div className="flex flex-wrap gap-2 mt-2">
                                            {data.technical_analysis.resistance_levels?.map((level, i) => (
                                                <span key={i} className="bg-red-900/30 text-red-400 px-2 py-1 rounded text-sm font-mono">
                                                    {typeof level === 'number' ? level.toFixed(2) : level}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
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

const MetricCard = ({ label, value, subtext, trend, color, icon }) => (
    <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-4 rounded-xl shadow-lg">
        <div className="flex justify-between items-start">
            <div>
                <p className="text-gray-400 text-sm font-medium">{label}</p>
                <h3 className={`text-2xl font-bold mt-1 ${color || 'text-white'}`}>{value}</h3>
            </div>
            <div className="p-2 bg-gray-700/50 rounded-lg">
                {icon || (trend === 'up' ? <TrendingUp className="text-green-400" /> :
                    trend === 'down' ? <TrendingDown className="text-red-400" /> :
                        <Minus className="text-gray-400" />)}
            </div>
        </div>
        <p className={`text-xs mt-2 ${trend === 'up' ? 'text-green-400' : trend === 'down' ? 'text-red-400' : 'text-gray-500'}`}>
            {subtext}
        </p>
    </div>
);

export default FileUploader;
