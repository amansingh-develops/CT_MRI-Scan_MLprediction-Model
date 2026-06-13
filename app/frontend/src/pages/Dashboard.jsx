import React from 'react';
import { useAuthStore } from '../stores/authStore';
import ClinicianDashboard from './ClinicianDashboard';
import PatientDashboard from './PatientDashboard';

const Dashboard = () => {
  const { user } = useAuthStore();

  // If the user's role is 'patient', render the patient-specific dashboard
  if (user?.role === 'patient') {
    return <PatientDashboard />;
  }

  // Default to clinician dashboard for 'clinician' role or fallback
  return <ClinicianDashboard />;
};

export default Dashboard;
