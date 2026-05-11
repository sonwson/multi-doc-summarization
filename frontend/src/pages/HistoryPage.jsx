import React from 'react';
import { useEffect, useState } from 'react';
import api from '../api/axios';
import HistoryList from '../components/HistoryList';

export default function HistoryPage() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    api.get('/history').then(({ data }) => setItems(data)).catch(() => {});
  }, []);

  const handleDelete = async (id) => {
    const shouldDelete = window.confirm('Delete this summary record?');
    if (!shouldDelete) {
      return;
    }

    await api.delete(`/history/${id}`);
    setItems((current) => current.filter((item) => item._id !== id));
  };

  return <HistoryList items={items} onDelete={handleDelete} />;
}
