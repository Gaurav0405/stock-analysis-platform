import React, { useEffect, useState } from 'react';
import { Calendar, FileText, ArrowRight } from 'lucide-react';
import { getHistory } from '../api';

const RecentHistory = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const data = await getHistory();
                setHistory(data);
            } catch (error) {
                console.error("Failed to load history");
            } finally {
                setLoading(false);
            }
        };
        fetchHistory();
    }, []);

    if (loading) {
        return (
            <div className="flex justify-center items-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto">
            <h2 className="text-2xl font-bold mb-6 flex items-center">
                <Calendar className="mr-2 text-blue-400" />
                Recent Analyses
            </h2>

            <div className="space-y-4">
                {history.length === 0 ? (
                    <div className="text-center py-12 bg-gray-800/50 rounded-xl border border-gray-700 text-gray-400">
                        No history found. Run an analysis to get started.
                    </div>
                ) : (
                    history.map((item, index) => (
                        <div
                            key={index}
                            className="bg-gray-800/50 border border-gray-700 p-6 rounded-xl hover:bg-gray-800 transition-colors group cursor-pointer"
                        >
                            <div className="flex justify-between items-center">
                                <div className="flex items-center space-x-4">
                                    <div className="bg-blue-900/30 p-3 rounded-lg text-blue-400 font-bold text-xl">
                                        {item.ticker}
                                    </div>
                                    <div>
                                        <p className="text-white font-medium">{item.date}</p>
                                        <p className="text-sm text-gray-400">{item.files.length} generated files</p>
                                    </div>
                                </div>
                                <ArrowRight className="text-gray-600 group-hover:text-blue-400 transition-colors" />
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default RecentHistory;
