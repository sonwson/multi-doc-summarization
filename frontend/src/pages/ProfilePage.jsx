import React from 'react';
import { useState } from 'react';
import api from '../api/axios';
import { useAuth } from '../context/AuthContext';

export default function ProfilePage() {
  const { user, updateUser } = useAuth();
  const [form, setForm] = useState({
    name: user?.name || '',
    email: user?.email || ''
  });
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setMessage('');
    setError('');

    try {
      const { data } = await api.put('/auth/me', form);
      updateUser(data);
      setForm({ name: data.name || '', email: data.email || '' });
      setMessage('Profile updated successfully.');
    } catch (err) {
      setError(err.response?.data?.message || 'Unable to update profile');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl rounded-3xl bg-white p-8 shadow-soft">
      <h1 className="text-2xl font-semibold text-gray-900">Update profile</h1>
      <p className="mt-2 text-sm text-gray-500">Change the profile information attached to your account.</p>

      <form onSubmit={handleSubmit} className="mt-6 space-y-4">
        <input
          className="w-full rounded-2xl border border-gray-200 px-4 py-3 outline-none focus:border-brand"
          placeholder="Full name"
          value={form.name}
          onChange={(e) => setForm((current) => ({ ...current, name: e.target.value }))}
        />
        <input
          className="w-full rounded-2xl border border-gray-200 px-4 py-3 outline-none focus:border-brand"
          placeholder="Email"
          type="email"
          value={form.email}
          onChange={(e) => setForm((current) => ({ ...current, email: e.target.value }))}
        />
        {error && <p className="text-sm text-rose-500">{error}</p>}
        {message && <p className="text-sm text-teal-700">{message}</p>}
        <button
          type="submit"
          disabled={loading}
          className="rounded-2xl bg-brand px-5 py-3 font-semibold text-white disabled:opacity-60"
        >
          {loading ? 'Saving...' : 'Save changes'}
        </button>
      </form>
    </div>
  );
}
