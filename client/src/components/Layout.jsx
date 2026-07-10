import React from 'react';
import { Home, Upload, History, Activity } from 'lucide-react';

const Layout = ({ children, currentTab, setCurrentTab }) => {
    return (
        <div className="min-h-screen bg-gray-900 text-white font-sans">
            <nav className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">
                        <div className="flex items-center">
                            <Activity className="h-8 w-8 text-blue-500" />
                            <span className="ml-2 text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
                                StockForecaster
                            </span>
                        </div>
                        <div className="flex space-x-4">
                            <NavButton
                                icon={<Home size={18} />}
                                label="Analysis"
                                active={currentTab === 'analysis'}
                                onClick={() => setCurrentTab('analysis')}
                            />
                            <NavButton
                                icon={<Upload size={18} />}
                                label="Upload"
                                active={currentTab === 'upload'}
                                onClick={() => setCurrentTab('upload')}
                            />
                            <NavButton
                                icon={<History size={18} />}
                                label="History"
                                active={currentTab === 'history'}
                                onClick={() => setCurrentTab('history')}
                            />
                        </div>
                    </div>
                </div>
            </nav>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {children}
            </main>

            <footer className="border-t border-gray-800 py-6 mt-12 bg-gray-900">
                <div className="max-w-7xl mx-auto px-4 text-center text-gray-500 text-sm">
                    <p>© 2026 Stock Forecaster. Educational purposes only.</p>
                </div>
            </footer>
        </div>
    );
};

const NavButton = ({ icon, label, active, onClick }) => (
    <button
        onClick={onClick}
        className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${active
            ? 'bg-blue-600/20 text-blue-400'
            : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
    >
        {icon}
        <span className="ml-2">{label}</span>
    </button>
);

export default Layout;
