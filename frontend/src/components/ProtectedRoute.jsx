import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return <div className="py-20 text-center text-gray-500">Loading session...</div>;
  }

  return user ? children : <Navigate to="/login" replace />;
}
