import React, { useState } from 'react';

const ChartViewer = ({ files, ticker }) => {
    const [activeChart, setActiveChart] = useState('forecast');

    const charts = [
        { id: 'forecast', label: 'Forecast', path: files.forecast_chart },
        { id: 'price', label: 'Price History', path: files.price_history_chart },
        { id: 'sentiment', label: 'Sentiment', path: files.sentiment_chart },
        { id: 'backtest', label: 'Backtest', path: files.backtest_chart },
        { id: 'monte_carlo', label: 'Monte Carlo', path: files.monte_carlo_chart },
        { id: 'volume', label: 'Volume', path: files.volume_chart },
        { id: 'returns', label: 'Returns', path: files.returns_chart },
        { id: 'support', label: 'Support/Res', path: files.support_resistance_chart },
        { id: 'importance', label: 'Importance', path: files.importance_chart },
        { id: 'comparison', label: 'Model Comp', path: files.model_comparison_chart },
        { id: 'drawdown', label: 'Drawdown', path: files.drawdown_chart },
        { id: 'accuracy', label: 'Accuracy', path: files.accuracy_chart },
        { id: 'correlation', label: 'Correlation', path: files.correlation_chart, type: 'image' },
        { id: 'residuals', label: 'Residuals', path: files.residuals_chart, type: 'image' },
    ];

    // Filter out charts that might be missing (e.g. if file generation failed)
    const availableCharts = charts.filter(c => c.path);

    const getChartUrl = (path) => {
        if (!path) return '';
        const filename = path.split('/').pop().split('\\').pop();
        return `http://localhost:5000/charts/${filename}`;
    };

    return (
        <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl overflow-hidden shadow-2xl flex flex-col h-[700px]">
            <div className="border-b border-gray-700 bg-gray-900/50 px-4 py-3 flex space-x-2 overflow-x-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
                {availableCharts.map((chart) => (
                    <button
                        key={chart.id}
                        onClick={() => setActiveChart(chart.id)}
                        className={`whitespace-nowrap px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeChart === chart.id
                            ? 'bg-blue-600 text-white shadow-lg'
                            : 'text-gray-400 hover:text-white hover:bg-gray-700'
                            }`}
                    >
                        {chart.label}
                    </button>
                ))}
            </div>

            <div className="relative flex-1 bg-white w-full h-full overflow-hidden">
                {availableCharts.map((chart) => (
                    <div
                        key={chart.id}
                        className={`absolute inset-0 w-full h-full transition-opacity duration-300 flex items-center justify-center ${activeChart === chart.id ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'
                            }`}
                    >
                        {chart.type === 'image' ? (
                            <img
                                src={getChartUrl(chart.path)}
                                alt={`${ticker} ${chart.label}`}
                                className="max-w-full max-h-full object-contain p-4"
                            />
                        ) : (
                            <iframe
                                src={getChartUrl(chart.path)}
                                title={`${ticker} ${chart.label}`}
                                className="w-full h-full border-0"
                            />
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ChartViewer;
