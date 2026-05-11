import React from 'react';
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../api/axios';
import { useAuth } from '../context/AuthContext';

export default function AuthForm({ type = 'login' }) {
  const isRegister = type === 'register';
  const navigate = useNavigate();
  const { saveAuth } = useAuth();
  const [form, setForm] = useState({ name: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');

    try {
      const endpoint = isRegister ? '/auth/register' : '/auth/login';
      const { data } = await api.post(endpoint, form);
      saveAuth(data);
      navigate('/');
    } catch (err) {
      setError(err.response?.data?.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-[calc(100vh-144px)] items-center justify-center">
      <div className="w-full max-w-md rounded-3xl bg-white p-8 shadow-soft">
        <h1 className="text-2xl font-semibold text-gray-800">
          {isRegister ? 'Create account' : 'Welcome back'}
        </h1>
        <p className="mt-2 text-sm text-gray-500">
          Sign in to use AI summarization and view your saved history.
        </p>

        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          {isRegister && (
            <input
              className="w-full rounded-2xl border border-gray-200 px-4 py-3 outline-none focus:border-brand"
              placeholder="Full name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          )}
          <input
            className="w-full rounded-2xl border border-gray-200 px-4 py-3 outline-none focus:border-brand"
            placeholder="Email"
            type="email"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
          />
          <input
            className="w-full rounded-2xl border border-gray-200 px-4 py-3 outline-none focus:border-brand"
            placeholder="Password"
            type="password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
          />
          {error && <p className="text-sm text-rose-500">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-2xl bg-brand px-4 py-3 font-semibold text-white transition hover:brightness-95 disabled:opacity-60"
          >
            {loading ? 'Processing...' : isRegister ? 'Register' : 'Login'}
          </button>
        </form>

        <p className="mt-5 text-sm text-gray-500">
          {isRegister ? 'Already have an account?' : 'Need an account?'}{' '}
          <Link className="font-medium text-teal-600" to={isRegister ? '/login' : '/register'}>
            {isRegister ? 'Login' : 'Register'}
          </Link>
        </p>
      </div>
    </div>
  );
}
