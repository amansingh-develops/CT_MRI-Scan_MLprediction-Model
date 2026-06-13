import React from 'react';
import { createBrowserRouter, RouterProvider, Navigate } from 'react-router-dom';
import { AuthLayout } from '../components/layout/AuthLayout';
import { PageWrapper } from '../components/layout/PageWrapper';
import { ProtectedRoute } from '../components/layout/ProtectedRoute';

// Pages
import Landing from '../pages/Landing';
import SignIn from '../pages/SignIn';
import SignUp from '../pages/SignUp';
import Dashboard from '../pages/Dashboard';
import Upload from '../pages/Upload';
import Analysis from '../pages/Analysis';
import Results from '../pages/Results';
import Compare from '../pages/Compare';
import MyScans from '../pages/MyScans';
import Settings from '../pages/Settings';

const router = createBrowserRouter([
  { path: '/', element: <Landing /> },
  {
    element: <AuthLayout />,
    children: [
      { path: '/signin', element: <SignIn /> },
      { path: '/signup', element: <SignUp /> },
    ],
  },
  {
    element: <ProtectedRoute />,
    children: [
      {
        element: <PageWrapper />,
        children: [
          { path: '/dashboard', element: <Dashboard /> },
          { path: '/upload', element: <Upload /> },
          { path: '/analysis/:scanId', element: <Analysis /> },
          { path: '/results/:scanId', element: <Results /> },
          { path: '/compare', element: <Compare /> },
          { path: '/scans', element: <MyScans /> },
          { path: '/settings', element: <Settings /> },
        ],
      },
    ],
  },
  {
    path: '*',
    element: <Navigate to="/" replace />,
  },
]);

export const AppRouter = () => {
  return <RouterProvider router={router} />;
};
