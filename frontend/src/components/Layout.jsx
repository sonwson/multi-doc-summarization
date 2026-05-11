import React from 'react';
import { Link, NavLink } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const navClass = ({ isActive }) =>
  `rounded-full px-4 py-2 text-sm font-medium transition ${
    isActive ? 'bg-brand text-white' : 'text-gray-600 hover:bg-teal-50'
  }`;

export default function Layout({ children }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const isAuthPage = location.pathname === '/login' || location.pathname === '/register';

  return (
    <div className="min-h-screen">
      <header className="flex w-full items-center justify-between px-4 py-6 md:px-6 xl:px-8">
        <Link to="/" className="text-xl font-semibold text-gray-800">
          AI Summarizer
        </Link>
        <nav className="flex items-center gap-2 rounded-full border border-white/70 bg-white/80 p-2 shadow-soft backdrop-blur">
          {user && <NavLink to="/" className={navClass}>Workspace</NavLink>}
          {user && <NavLink to="/history" className={navClass}>History</NavLink>}
          {!user && <NavLink to="/login" className={navClass}>Login</NavLink>}
          {!user && <NavLink to="/register" className={navClass}>Register</NavLink>}
          {user && (
            <button
              onClick={logout}
              className="rounded-full bg-gray-900 px-4 py-2 text-sm font-medium text-white"
            >
              Logout
            </button>
          )}
        </nav>
      </header>
      <main className={`w-full px-4 pb-10 md:px-6 xl:px-8 ${isAuthPage ? 'min-h-[calc(100vh-104px)]' : ''}`}>
        {children}
      </main>
    </div>
  );
}
