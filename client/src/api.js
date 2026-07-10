import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

export const analyzeTicker = async (ticker) => {
    try {
        const response = await axios.post(`${API_URL}/analyze`, { ticker });
        return response.data;
    } catch (error) {
        console.error("Analysis error", error);
        throw error;
    }
};

export const getHistory = async () => {
    try {
        const response = await axios.get(`${API_URL}/history`);
        return response.data;
    } catch (error) {
        console.error("History error", error);
        throw error;
    }
};

export const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await axios.post(`${API_URL}/upload`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error("Upload error", error);
        throw error;
    }
};

export const uploadImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await axios.post(`${API_URL}/upload-image`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error("Image upload error", error);
        throw error;
    }
};

