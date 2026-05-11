import React from 'react';
import { useState } from 'react';
import api from '../api/axios';

const MAX_SOURCES = 10;
const createBlock = () => ({ id: crypto.randomUUID(), value: '' });

export default function SummaryForm({ onCreated }) {
  const [blocks, setBlocks] = useState([createBlock()]);
  const [files, setFiles] = useState([]);
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const totalSources = blocks.length + files.length;
  const remainingSlots = Math.max(0, MAX_SOURCES - totalSources);

  const updateBlock = (id, value) => {
    setBlocks((current) => current.map((block) => (
      block.id === id ? { ...block, value } : block
    )));
  };

  const addBlock = () => {
    if (totalSources >= MAX_SOURCES) {
      setError(`Only ${MAX_SOURCES} text blocks and files combined are allowed.`);
      return;
    }

    setError('');
    setBlocks((current) => [...current, createBlock()]);
  };

  const removeBlock = (id) => {
    const shouldDelete = window.confirm('Delete this text source?');
    if (!shouldDelete) {
      return;
    }

    setBlocks((current) => current.length === 1 ? current : current.filter((block) => block.id !== id));
  };

  const removeFile = (fileName) => {
    const shouldDelete = window.confirm(`Delete file "${fileName}" from the current upload list?`);
    if (!shouldDelete) {
      return;
    }

    setFiles((current) => current.filter((file) => file.name !== fileName));
  };

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files || []);
    const allowedFiles = selectedFiles.slice(0, Math.max(0, MAX_SOURCES - blocks.length));

    if (blocks.length + selectedFiles.length > MAX_SOURCES) {
      setError(`Only ${MAX_SOURCES} text blocks and files combined are allowed.`);
    } else {
      setError('');
    }

    setFiles(allowedFiles);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      const inputs = blocks.map((block) => block.value.trim()).filter(Boolean);

      if (!inputs.length && !files.length) {
        setError('Please enter text or upload up to 10 files.');
        setLoading(false);
        return;
      }

      if (inputs.length + files.length > MAX_SOURCES) {
        setError(`Only ${MAX_SOURCES} text blocks and files combined are allowed.`);
        setLoading(false);
        return;
      }

      formData.append('inputs', JSON.stringify(inputs));
      files.forEach((file) => formData.append('files', file));

      const { data } = await api.post('/summarize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setSummary(data.summary);
      onCreated(data.history);
    } catch (err) {
      setError(err.response?.data?.message || 'Unable to summarize documents');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async () => {
    if (summary) {
      await navigator.clipboard.writeText(summary);
    }
  };

  const handleClear = () => {
    setBlocks([createBlock()]);
    setFiles([]);
    setSummary('');
    setError('');
  };

  return (
    <div className="grid h-full gap-6 xl:grid-cols-[1.15fr,0.85fr]">
      <section className="rounded-3xl bg-white p-6 shadow-soft">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold">Multi-document workspace</h2>
            <p className="mt-1 text-sm text-gray-500">
              Enter text by hand or upload up to 10 .txt/.pdf sources for one summary.
            </p>
          </div>
          <button
            onClick={addBlock}
            disabled={totalSources >= MAX_SOURCES}
            className="rounded-full bg-teal-50 px-4 py-2 text-sm font-medium text-teal-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            + Add block
          </button>
        </div>

        <div className="mt-4 rounded-2xl bg-slate-50 px-4 py-3 text-sm text-gray-600">
          Sources used: {totalSources}/{MAX_SOURCES}. Remaining slots: {remainingSlots}.
        </div>

        <div className="mt-6 space-y-4">
          {blocks.map((block, index) => (
            <div key={block.id} className="rounded-2xl border border-gray-200 p-3">
              <div className="mb-2 flex items-center justify-between text-sm text-gray-500">
                <span>Text source {index + 1}</span>
                {blocks.length > 1 && (
                  <button onClick={() => removeBlock(block.id)} className="text-rose-500">
                    Remove
                  </button>
                )}
              </div>
              <textarea
                rows="5"
                value={block.value}
                onChange={(e) => updateBlock(block.id, e.target.value)}
                placeholder="Paste your content here..."
                className="w-full resize-none rounded-2xl bg-slate-50 px-4 py-3 outline-none focus:ring-2 focus:ring-teal-200"
              />
            </div>
          ))}
        </div>

        <label className="mt-4 flex cursor-pointer items-center justify-center rounded-2xl border border-dashed border-teal-200 bg-teal-50/60 px-4 py-6 text-sm text-teal-700">
          <input
            type="file"
            accept=".txt,.pdf"
            multiple
            className="hidden"
            onChange={handleFileChange}
          />
          Upload .txt or .pdf files ({files.length} selected)
        </label>

        {!!files.length && (
          <div className="mt-4 space-y-2">
            {files.map((file) => (
              <div key={file.name} className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3 text-sm text-gray-600">
                <span className="truncate pr-4">{file.name}</span>
                <button onClick={() => removeFile(file.name)} className="text-rose-500">
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}

        {error && <p className="mt-4 text-sm text-rose-500">{error}</p>}

        <div className="mt-6 flex flex-wrap gap-3">
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="rounded-2xl bg-brand px-5 py-3 font-semibold text-white disabled:opacity-60"
          >
            {loading ? 'Summarizing...' : 'Summarization'}
          </button>
          <button
            onClick={handleCopy}
            className="rounded-2xl border border-gray-200 px-5 py-3 font-medium text-gray-700"
          >
            Copy result
          </button>
          <button
            onClick={handleClear}
            className="rounded-2xl border border-gray-200 px-5 py-3 font-medium text-gray-700"
          >
            Clear
          </button>
        </div>
      </section>

      <section className="rounded-3xl bg-gray-900 p-6 text-white shadow-soft">
        <p className="text-sm uppercase tracking-[0.2em] text-teal-200">AI output</p>
        <h2 className="mt-2 text-2xl font-semibold">Summary result</h2>
        <div className="mt-6 min-h-[320px] rounded-3xl bg-white/10 p-5 text-sm leading-7 text-slate-100 xl:min-h-[calc(100vh-360px)]">
          {loading ? 'Model is analyzing your documents...' : summary || 'Your summary will appear here.'}
        </div>
      </section>
    </div>
  );
}
