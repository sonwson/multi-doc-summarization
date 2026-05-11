import React from 'react';
import { useState } from 'react';
import api from '../api/axios';

export default function ChangePasswordPage() {
  const [form, setForm] = useState({
    currentPassword: '',
    newPassword: ''
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
      const { data } = await api.put('/auth/change-password', form);
      setMessage(data.message || 'Password updated successfully.');
      setForm({ currentPassword: '', newPassword: '' });
    } catch (err) {
      setError(err.response?.data?.message || 'Unable to change password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl rounded-3xl bg-white p-8 shadow-soft">
      <h1 className="text-2xl font-semibold text-gray-900">Change password</h1>
      <p className="mt-2 text-sm text-gray-500">Use your current password to set a new one.</p>

      <form onSubmit={handleSubmit} className="mt-6 space-y-4">
        <input
          className="w-full rounded-2xl border border-gray-200 px-4 py-3 outline-none focus:border-brand"
          placeholder="Current password"
          type="password"
          value={form.currentPassword}
          onChange={(e) => setForm((current) => ({ ...current, currentPassword: e.target.value }))}
        />
        <input
          className="w-full rounded-2xl border border-gray-200 px-4 py-3 outline-none focus:border-brand"
          placeholder="New password"
          type="password"
          value={form.newPassword}
          onChange={(e) => setForm((current) => ({ ...current, newPassword: e.target.value }))}
        />
        {error && <p className="text-sm text-rose-500">{error}</p>}
        {message && <p className="text-sm text-teal-700">{message}</p>}
        <button
          type="submit"
          disabled={loading}
          className="rounded-2xl bg-brand px-5 py-3 font-semibold text-white disabled:opacity-60"
        >
          {loading ? 'Saving...' : 'Update password'}
        </button>
      </form>
    </div>
  );
}
