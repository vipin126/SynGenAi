'use client';

import { useState, useEffect } from 'react';
import { Upload, Activity, Database, CheckCircle, AlertCircle } from 'lucide-react';
import { checkHealth, uploadDataset } from '../../services/api';
import { startTraining, getTrainingStatus, generateSyntheticData } from '../../services/api'; // Or relative path if that's what worked for you


export default function Home() {
  const [status, setStatus] = useState<string>('Checking...');
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [training, setTraining] = useState(false);
const [progress, setProgress] = useState(0);
const [trainMessage, setTrainMessage] = useState('');
const [generatedData, setGeneratedData] = useState<any[]>([]);



const handleStartTrain = async () => {
    if (!uploadResult) return;
    try {
        setTraining(true);
        await startTraining(uploadResult.filename);
        
        // Poll for status
        const interval = setInterval(async () => {
            const status = await getTrainingStatus();
            if (status) {
                setProgress(status.progress);
                setTrainMessage(status.message);
                
                if (status.progress === 100 || status.message.includes('Error')) {
                    clearInterval(interval);
                    setTraining(false);
                }
            }
        }, 1000);
    } catch (error) {
        alert("Failed to start training");
        setTraining(false);
    }
};



const handleGenerate = async () => {
    if (!uploadResult) return;
    try {
        const result = await generateSyntheticData(uploadResult.filename);
        setGeneratedData(result.data); // Save the table data
    } catch (error) {
        alert("Generation failed");
    }
};

  // Check backend health on load
  useEffect(() => {
    checkHealth().then((data) => {
      if (data && data.status === 'ok') {
        setStatus('Online 🟢');
      } else {
        setStatus('Offline 🔴');
      }
    });
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadDataset(file);
      setUploadResult(result);
    } catch (error) {
      alert('Upload failed!');
    } finally {
      setUploading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 p-8">
      {/* Header */}
      <header className="mb-10 flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-slate-900">SynGenAI</h1>
          <p className="text-slate-500">Synthetic Data Generation Platform</p>
        </div>
        <div className="flex items-center gap-2 rounded-full bg-white px-4 py-2 shadow-sm border">
          <Activity className="h-5 w-5 text-blue-500" />
          <span className="font-medium text-sm">System Status: {status}</span>
        </div>
      </header>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Upload Card */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 bg-blue-100 rounded-lg">
              <Database className="h-6 w-6 text-blue-600" />
            </div>
            <h2 className="text-xl font-semibold">Data Ingestion</h2>
          </div>

          <div className="border-2 border-dashed border-slate-200 rounded-lg p-8 text-center hover:bg-slate-50 transition-colors">
            <input 
              type="file" 
              id="file-upload" 
              className="hidden" 
              onChange={handleFileChange}
              accept=".csv,.zip,.png,.jpg"
            />
            <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
              <Upload className="h-10 w-10 text-slate-400 mb-3" />
              <span className="text-slate-600 font-medium">
                {file ? file.name : "Click to upload dataset"}
              </span>
              <span className="text-slate-400 text-sm mt-1">CSV, ZIP, PNG, JPG</span>
            </label>
          </div>

          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className={`w-full mt-4 py-3 rounded-lg font-medium text-white transition-all ${
              !file || uploading ? 'bg-slate-300' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {uploading ? 'Uploading...' : 'Start Processing'}
          </button>
        </div>

        {/* Results Card */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
          <h2 className="text-xl font-semibold mb-6">Processing Status</h2>
          
          {!uploadResult ? (
            <div className="flex flex-col items-center justify-center h-48 text-slate-400">
              <Activity className="h-8 w-8 mb-2 opacity-50" />
              <p>Waiting for data...</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-3 text-green-600 bg-green-50 p-4 rounded-lg">
                <CheckCircle className="h-5 w-5" />
                <div>
                  <p className="font-medium">Upload Successful</p>
                  <p className="text-sm opacity-80">File saved to {uploadResult.path}</p>
                </div>
              </div>
              
              <div className="p-4 bg-slate-50 rounded-lg border border-slate-100">
                <h3 className="font-medium mb-2 text-sm text-slate-500 uppercase">File Stats</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-slate-400">Filename</p>
                    <p className="font-medium">{uploadResult.filename}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-400">Size</p>
                    <p className="font-medium">{(uploadResult.size / 1024).toFixed(2)} KB</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        {/* Training Card */}
<div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 md:col-span-2">
  <div className="flex items-center justify-between mb-6">
    <h2 className="text-xl font-semibold">Model Training</h2>
    {uploadResult && !training && progress !== 100 && (
      <button 
        onClick={handleStartTrain}
        className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
      >
        Start Training GAN
      </button>
    )}
  </div>

  {/* Progress Bar */}
  {(training || progress > 0) && (
    <div className="space-y-3">
      <div className="flex justify-between text-sm font-medium text-slate-600">
        <span>{trainMessage}</span>
        <span>{progress}%</span>
      </div>
      <div className="w-full bg-slate-200 rounded-full h-4 overflow-hidden">
        <div 
          className="bg-indigo-600 h-4 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        ></div>
      </div>
    </div>
  )}
</div>
{/* Generation Card */}
{progress === 100 && (
  <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-100 md:col-span-2">
    <div className="flex items-center justify-between mb-4">
      <h2 className="text-xl font-semibold text-green-700">🚀 Ready to Generate!</h2>
      <button 
        onClick={handleGenerate}
        className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors shadow-sm"
      >
        Generate Synthetic Data
      </button>
    </div>

    {generatedData.length > 0 && (
      <div className="overflow-x-auto mt-4 border rounded-lg">
        <table className="w-full text-sm text-left text-slate-500">
            <thead className="text-xs text-slate-700 uppercase bg-slate-50">
                <tr>
                    {Object.keys(generatedData[0]).map((key) => (
                        <th key={key} className="px-6 py-3">{key}</th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {generatedData.map((row, i) => (
                    <tr key={i} className="bg-white border-b hover:bg-slate-50">
                        {Object.values(row).map((val: any, j) => (
                            <td key={j} className="px-6 py-4">
                                {typeof val === 'number' ? val.toFixed(2) : val}
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
      </div>
    )}
  </div>
)}


      </div>
      
    </main>
  );
}
