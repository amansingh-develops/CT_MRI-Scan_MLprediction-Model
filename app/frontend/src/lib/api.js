/**
 * ScanSight ML API Service
 * 
 * Communicates with the Python FastAPI backend for ML inference.
 * All requests include the Firebase Auth token for security.
 */

import { auth } from './firebase';

const ML_API_URL = import.meta.env.VITE_ML_API_URL || 'http://localhost:8000';

/**
 * Get the current user's Firebase ID token for authenticated API calls.
 */
async function getAuthToken() {
  const user = auth.currentUser;
  if (!user) throw new Error('User not authenticated');
  return user.getIdToken();
}

/**
 * Upload a CT scan image and get liver/tumor segmentation results.
 * @param {File} file - The CT scan image file
 * @param {string} scanId - The scan document ID (for Firestore update)
 * @param {string} userId - The user's UID
 * @returns {Promise<Object>} - Scan results with masks and metadata
 */
export function predictScan(file, scanId = '', userId = '', onUploadProgress = null) {
  return new Promise(async (resolve, reject) => {
    try {
      const token = await getAuthToken();
      const formData = new FormData();
      formData.append('file', file);
      formData.append('scanId', scanId);
      formData.append('userId', userId);
      formData.append('authorization', `Bearer ${token}`);

      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${ML_API_URL}/api/predict`);
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);

      // Handle upload progress
      if (xhr.upload && onUploadProgress) {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100;
            onUploadProgress(percentComplete);
          }
        };
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            resolve(xhr.responseText);
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.detail || 'Prediction failed'));
          } catch {
            reject(new Error('Prediction failed'));
          }
        }
      };

      xhr.onerror = () => reject(new Error('Network error occurred during prediction'));
      
      xhr.send(formData);
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * Check if the ML server is healthy and ready.
 * @returns {Promise<Object>} - Server health status
 */
export async function checkServerHealth() {
  try {
    const response = await fetch(`${ML_API_URL}/health`, { method: 'GET' });
    if (response.ok) {
      return response.json();
    }
    return { status: 'offline', model_loaded: false };
  } catch {
    return { status: 'offline', model_loaded: false };
  }
}

/**
 * Get ML model info (architecture, parameters, device).
 * @returns {Promise<Object>} - Model information
 */
export async function getModelInfo() {
  try {
    const response = await fetch(`${ML_API_URL}/api/model-info`, { method: 'GET' });
    if (response.ok) {
      return response.json();
    }
    return { loaded: false };
  } catch {
    return { loaded: false };
  }
}

/**
 * Fetch all scans for a user from the FastAPI backend.
 * @param {string} userId - The user's UID
 * @returns {Promise<Array>} - List of scan objects
 */
export async function fetchUserScans(userId) {
  try {
    const token = await getAuthToken();
    const response = await fetch(`${ML_API_URL}/api/scans/${userId}`, {
      method: 'GET',
      headers: { 'Authorization': `Bearer ${token}` },
    });
    if (response.ok) {
      return response.json();
    }
    console.warn('Failed to fetch user scans:', response.status);
    return [];
  } catch (err) {
    console.warn('Failed to fetch user scans:', err.message);
    return [];
  }
}

/**
 * Fetch a single scan's full data (including result) from the FastAPI backend.
 * @param {string} scanId - The scan ID
 * @returns {Promise<Object|null>} - The scan object or null
 */
export async function fetchScanResult(scanId) {
  try {
    const token = await getAuthToken();
    const response = await fetch(`${ML_API_URL}/api/scan/${scanId}`, {
      method: 'GET',
      headers: { 'Authorization': `Bearer ${token}` },
    });
    if (response.ok) {
      return response.json();
    }
    if (response.status === 404) {
      return null; // Scan not found yet (still being created)
    }
    console.warn('Failed to fetch scan result:', response.status);
    return null;
  } catch (err) {
    console.warn('Failed to fetch scan result:', err.message);
    return null;
  }
}
