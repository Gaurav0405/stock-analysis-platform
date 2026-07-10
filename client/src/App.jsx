import React, { useState } from 'react';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import FileUploader from './components/FileUploader';
import RecentHistory from './components/RecentHistory';

function App() {
  const [currentTab, setCurrentTab] = useState('analysis');

  const renderContent = () => {
    switch (currentTab) {
      case 'analysis':
        return <Dashboard />;
      case 'upload':
        return <FileUploader />;
      case 'history':
        return <RecentHistory />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Layout currentTab={currentTab} setCurrentTab={setCurrentTab}>
      {renderContent()}
    </Layout>
  );
}

export default App;
