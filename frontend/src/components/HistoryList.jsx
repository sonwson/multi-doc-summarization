import React from 'react';

function truncateText(text, maxLength = 180) {
  if (!text) {
    return '';
  }

  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

export default function HistoryList({ items, onDelete, onViewDetails, selectedItem, onCloseDetails }) {
  return (
    <>
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
                  <td className="px-4 py-4 align-top">
                    <div className="font-medium text-gray-800">{item.inputs.length} docs</div>
                    {!!item.sourceFiles?.length && (
                      <div className="mt-1 text-xs text-gray-500">
                        Files: {item.sourceFiles.join(', ')}
                      </div>
                    )}
                  </td>
                  <td className="max-w-md px-4 py-4 align-top">{truncateText(item.summary)}</td>
                  <td className="px-4 py-4 align-top">
                    {new Date(item.createdAt).toLocaleString()}
                  </td>
                  <td className="px-4 py-4 align-top">
                    <div className="flex flex-wrap gap-3">
                      <button
                        onClick={() => onViewDetails(item)}
                        className="font-medium text-teal-600"
                      >
                        View details
                      </button>
                      <button onClick={() => onDelete(item._id)} className="text-rose-500">
                        Delete
                      </button>
                    </div>
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

      {selectedItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 px-4 py-8">
          <div className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-3xl bg-white p-6 shadow-2xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-2xl font-semibold text-gray-900">Summary details</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Created at {new Date(selectedItem.createdAt).toLocaleString()}
                </p>
              </div>
              <button
                onClick={onCloseDetails}
                className="rounded-2xl border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700"
              >
                Close
              </button>
            </div>

            {!!selectedItem.sourceFiles?.length && (
              <div className="mt-5 rounded-2xl bg-slate-50 p-4">
                <p className="text-sm font-semibold text-gray-800">Source files</p>
                <p className="mt-2 text-sm text-gray-600">{selectedItem.sourceFiles.join(', ')}</p>
              </div>
            )}

            <div className="mt-6 grid gap-6 xl:grid-cols-[1fr,1fr]">
              <section className="rounded-3xl border border-gray-100 bg-slate-50 p-5">
                <h4 className="text-lg font-semibold text-gray-900">Before summarization</h4>
                <div className="mt-4 space-y-4">
                  {selectedItem.inputs.map((input, index) => (
                    <article key={`${selectedItem._id}-${index}`} className="rounded-2xl bg-white p-4">
                      <p className="text-sm font-medium text-teal-700">Input {index + 1}</p>
                      <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-7 text-gray-700">
                        {input}
                      </p>
                    </article>
                  ))}
                </div>
              </section>

              <section className="rounded-3xl border border-gray-100 bg-gray-900 p-5 text-white">
                <h4 className="text-lg font-semibold">After summarization</h4>
                <div className="mt-4 rounded-2xl bg-white/10 p-4">
                  <p className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-100">
                    {selectedItem.summary}
                  </p>
                </div>
              </section>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
