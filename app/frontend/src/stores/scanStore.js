import { create } from 'zustand';
import { fetchUserScans, fetchScanResult } from '../lib/api';

export const useScanStore = create((set, get) => ({
  scans: [],               // Empty until API loads
  currentScan: null,
  currentSliceIndex: 0,
  analysisState: 'idle',   // 'idle'|'uploading'|'analyzing'|'clear'|'anomaly'
  isLoadingScans: false,
  hasFetched: false,

  setCurrentScan: (scan) => set({ currentScan: scan }),
  setCurrentSliceIndex: (index) => set({ currentSliceIndex: index }),
  setAnalysisState: (state) => set({ analysisState: state }),

  /**
   * Add or update a scan in the local store (optimistic update after upload).
   */
  addScan: (scan) => set((s) => {
    const existingIdx = s.scans.findIndex(ex => ex.scanId === scan.scanId);
    let updatedScans;
    if (existingIdx >= 0) {
      updatedScans = [...s.scans];
      updatedScans[existingIdx] = { ...updatedScans[existingIdx], ...scan };
    } else {
      updatedScans = [scan, ...s.scans];
    }
    return { scans: updatedScans, currentScan: scan, hasFetched: true };
  }),

  /**
   * Fetch all scans for a user from FastAPI (replaces Firestore listener).
   */
  loadScans: async (userId) => {
    if (!userId) return;
    set({ isLoadingScans: true });

    try {
      const scans = await fetchUserScans(userId);
      // Normalize uploadedAt for display
      const normalized = scans.map(s => ({
        ...s,
        uploadedAt: s.uploadedAt
          ? new Date(s.uploadedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
          : 'N/A',
        completedAt: s.completedAt
          ? new Date(s.completedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
          : null,
      }));

      set((state) => {
        const currentId = state.currentScan?.scanId;
        const updatedCurrent = normalized.find(s => s.scanId === currentId) || normalized[0] || null;
        return {
          scans: normalized,
          currentScan: updatedCurrent,
          isLoadingScans: false,
          hasFetched: true,
        };
      });
    } catch (err) {
      console.warn('Failed to load scans:', err.message);
      set({ isLoadingScans: false });
    }
  },

  /**
   * Fetch a single scan by scanId from FastAPI and update it in the store.
   */
  fetchAndUpdateScan: async (scanId) => {
    try {
      const scanData = await fetchScanResult(scanId);
      if (!scanData) return null;

      // Normalize dates
      const normalized = {
        ...scanData,
        uploadedAt: scanData.uploadedAt
          ? new Date(scanData.uploadedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
          : 'N/A',
        completedAt: scanData.completedAt
          ? new Date(scanData.completedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
          : null,
      };

      // Update in the scans array
      set((state) => {
        const existingIdx = state.scans.findIndex(s => s.scanId === scanId);
        const updatedScans = [...state.scans];
        if (existingIdx >= 0) {
          updatedScans[existingIdx] = normalized;
        } else {
          updatedScans.unshift(normalized);
        }
        return {
          scans: updatedScans,
          currentScan: normalized,
        };
      });

      return normalized;
    } catch (err) {
      console.warn('Failed to fetch scan:', err.message);
      return null;
    }
  },
}));
