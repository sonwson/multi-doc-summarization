import React from 'react';
export default function HistoryList({ items, onDelete }) {
  return (
    <div className="rounded-3xl bg-white p-6 shadow-soft">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Summary history</h2>
          <p className="mt-1 text-sm text-gray-500">Saved summaries are ordered from newest to oldest.</p>
        </div>
        <span className="rounded-full bg-teal-50 px-3 py-1 text-sm font-medium text-teal-700">
          {items.length} records
        </span>
      </div>

      <div className="mt-6 overflow-hidden rounded-2xl border border-gray-100">
        <table className="min-w-full divide-y divide-gray-100 text-left text-sm">
          <thead className="bg-slate-50 text-gray-500">
            <tr>
              <th className="px-4 py-3 font-medium">Inputs</th>
              <th className="px-4 py-3 font-medium">Summary</th>
              <th className="px-4 py-3 font-medium">Date</th>
              <th className="px-4 py-3 font-medium">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white text-gray-700">
            {items.map((item) => (
              <tr key={item._id}>
                <td className="px-4 py-4 align-top">{item.inputs.length} docs</td>
                <td className="max-w-md px-4 py-4 align-top">{item.summary}</td>
                <td className="px-4 py-4 align-top">
                  {new Date(item.createdAt).toLocaleString()}
                </td>
                <td className="px-4 py-4 align-top">
                  <button onClick={() => onDelete(item._id)} className="text-rose-500">
                    Delete
                  </button>
                </td>
              </tr>
            ))}
            {!items.length && (
              <tr>
                <td colSpan="4" className="px-4 py-10 text-center text-gray-400">
                  No history yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
