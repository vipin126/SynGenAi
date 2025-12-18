import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    return null;
  }
};

export const uploadDataset = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Upload failed:', error);
    throw error;
  }
};



// ... (previous imports and code)

export const startTraining = async (filename: string) => {
  try {
    const response = await api.post(`/train?filename=${filename}&epochs=50`);
    return response.data;
  } catch (error) {
    console.error('Start training failed:', error);
    throw error;
  }
};

export const getTrainingStatus = async () => {
  try {
    const response = await api.get('/status');
    return response.data;
  } catch (error) {
    console.error('Status check failed:', error);
    return null;
  }
};
// Add this at the bottom
export const generateSyntheticData = async (filename: string, count: number = 50) => {
  try {
    const response = await api.post(`/generate?filename=${filename}&num_samples=${count}`);
    return response.data;
  } catch (error) {
    console.error('Generation failed:', error);
    throw error;
  }
};



export default api;
