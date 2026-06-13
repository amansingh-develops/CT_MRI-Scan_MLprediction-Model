export const mockUser = {
  uid: 'mock-uid-001',
  name: 'Rahul Sharma',
  email: 'rahul@example.com',
  createdAt: '2025-01-10',
};

export const mockScans = [
  {
    scanId: 'scan-001',
    label: 'Initial Scan',
    uploadedAt: 'Jan 15, 2025',
    status: 'complete',
    analysisState: 'anomaly',
    totalSlices: 47,
    affectedSlices: 8,
    tumorSize: '2.4 cm²',
    affectedLiverPercent: '6.2%',
    confidence: 87.3,
    estimatedStage: 'Stage II',
  },
  {
    scanId: 'scan-002',
    label: 'Follow-up Scan',
    uploadedAt: 'Mar 20, 2025',
    status: 'complete',
    analysisState: 'clear',
    totalSlices: 43,
    affectedSlices: 0,
    tumorSize: null,
    affectedLiverPercent: null,
    confidence: 94.1,
    estimatedStage: null,
  },
];

export const mockSlices = [
  { index: 1,  hasLiver: true,  hasTumor: false, confidence: 91.2 },
  { index: 12, hasLiver: true,  hasTumor: false, confidence: 89.4 },
  { index: 23, hasLiver: true,  hasTumor: true,  confidence: 91.2 },
  { index: 24, hasLiver: true,  hasTumor: true,  confidence: 88.7 },
  { index: 25, hasLiver: true,  hasTumor: true,  confidence: 85.3 },
  { index: 38, hasLiver: true,  hasTumor: false, confidence: 78.9 },
  { index: 47, hasLiver: false, hasTumor: false, confidence: 45.0 },
];

export const mockChatMessages = [
  { role: 'ai',   text: 'Hello! I am your ScanSight AI Doctor. How can I help you today?' },
  { role: 'user', text: 'What does Stage II mean?' },
  { role: 'ai',   text: 'Stage II means the tumor is localized to the liver and has not spread to nearby lymph nodes or other organs. Early treatment at this stage has good outcomes. Please consult your doctor for clinical confirmation.' },
];
