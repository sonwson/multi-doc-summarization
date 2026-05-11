import React from 'react';
import { useEffect, useState } from 'react';
import api from '../api/axios';
import SummaryForm from '../components/SummaryForm';

export default function WorkspacePage() {
  const [history, setHistory] = useState([]);
  const [isFullscreen, setIsFullscreen] = useState(Boolean(document.fullscreenElement));

  useEffect(() => {
    api.get('/history').then(({ data }) => setHistory(data)).catch(() => {});
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const toggleFullscreen = async () => {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
      return;
    }

    await document.exitFullscreen();
  };

  return (
    <div className="flex min-h-[calc(100vh-104px)] flex-col space-y-6">
      <section className="rounded-[2rem] bg-white/80 p-8 shadow-soft backdrop-blur">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.25em] text-teal-600">AI Multi-Document Summarization</p>
            <h1 className="mt-3 max-w-3xl text-4xl font-semibold leading-tight text-gray-800 lg:text-5xl">
              Summarize many text sources into one concise answer.
            </h1>
            <p className="mt-3 max-w-3xl text-gray-500">
              Paste multiple documents, attach text files, and keep every result in your private history.
            </p>
          </div>
          <button
            onClick={toggleFullscreen}
            className="rounded-2xl border border-teal-200 bg-teal-50 px-5 py-3 text-sm font-medium text-teal-700 transition hover:bg-teal-100"
          >
            {isFullscreen ? 'Exit fullscreen' : 'Open fullscreen'}
          </button>
        </div>
      </section>

      <div className="flex-1">
        <SummaryForm onCreated={(item) => setHistory((current) => [item, ...current])} />
      </div>

      {history.length > 0 && (
        <div className="rounded-3xl border border-teal-100 bg-teal-50/50 px-5 py-4 text-sm text-teal-800">
          Latest history sync: {history.length} summary record(s) loaded.
        </div>
      )}
    </div>
  );
}
