import React from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../context/AuthContext';

const profileLinkClass = ({ isActive }) =>
  `block rounded-2xl px-4 py-3 text-sm font-medium transition ${
    isActive ? 'bg-teal-50 text-teal-700' : 'text-gray-700 hover:bg-slate-50'
  }`;

export default function Layout({ children }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const isAuthPage = location.pathname === '/login' || location.pathname === '/register';
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    const handleOutsideClick = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-screen">
      <header className="relative z-50 flex w-full items-center justify-between px-4 py-6 md:px-6 xl:px-8">
        <div className="flex items-center gap-3">
          <Link to="/" className="text-xl font-semibold text-gray-800">
            AI Summarizer
          </Link>
          {user && (
            <Link
              to="/"
              className="rounded-full border border-teal-100 bg-white/80 px-4 py-2 text-sm font-medium text-gray-700 shadow-soft backdrop-blur hover:bg-teal-50"
            >
              Workspace
            </Link>
          )}
        </div>
        <div className="relative z-50" ref={menuRef}>
          {!user ? (
            <div className="flex items-center gap-3">
              <Link
                to="/"
                className="rounded-full border border-teal-100 bg-white/80 px-4 py-2 text-sm font-medium text-gray-700 shadow-soft backdrop-blur hover:bg-teal-50"
              >
                Workspace
              </Link>
              <Link
                to="/login"
                className="rounded-full bg-gray-900 px-4 py-2 text-sm font-medium text-white"
              >
                Login
              </Link>
            </div>
          ) : (
            <>
              <button
                type="button"
                onClick={() => setMenuOpen((current) => !current)}
                className="flex items-center gap-3 rounded-full border border-white/70 bg-white/85 px-3 py-2 shadow-soft backdrop-blur"
              >
                <span className="flex h-10 w-10 items-center justify-center rounded-full bg-teal-100 text-sm font-semibold text-teal-700">
                  {user.name?.slice(0, 1)?.toUpperCase() || 'U'}
                </span>
                <span className="hidden text-left md:block">
                  <span className="block text-sm font-medium text-gray-800">{user.name}</span>
                  <span className="block text-xs text-gray-500">Profile</span>
                </span>
                <span className="text-xs text-gray-500">{menuOpen ? '▲' : '▼'}</span>
              </button>
              {menuOpen && (
                <div className="absolute right-0 top-full z-[100] mt-3 w-72 rounded-3xl border border-white/80 bg-white p-3 shadow-2xl">
                  <div className="border-b border-slate-100 px-3 pb-3">
                    <p className="text-sm font-semibold text-gray-900">{user.name}</p>
                    <p className="mt-1 text-xs text-gray-500">{user.email}</p>
                  </div>
                  <nav className="mt-3 space-y-1">
                    <NavLink to="/history" className={profileLinkClass}>History</NavLink>
                    <NavLink to="/profile" className={profileLinkClass}>Update profile</NavLink>
                    <NavLink to="/change-password" className={profileLinkClass}>Change password</NavLink>
                    <button
                      onClick={logout}
                      className="block w-full rounded-2xl px-4 py-3 text-left text-sm font-medium text-rose-600 transition hover:bg-rose-50"
                    >
                      Logout
                    </button>
                  </nav>
                </div>
              )}
            </>
          )}
        </div>
      </header>
      <main className={`w-full px-4 pb-10 md:px-6 xl:px-8 ${isAuthPage ? 'min-h-[calc(100vh-104px)]' : ''}`}>
        {children}
      </main>
    </div>
  );
}
