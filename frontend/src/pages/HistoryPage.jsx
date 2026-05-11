import React from 'react';
import { useEffect, useState } from 'react';
import api from '../api/axios';
import HistoryList from '../components/HistoryList';

export default function HistoryPage() {
  const [items, setItems] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);

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
    setSelectedItem((current) => (current?._id === id ? null : current));
  };

  return (
    <HistoryList
      items={items}
      onDelete={handleDelete}
      onViewDetails={setSelectedItem}
      selectedItem={selectedItem}
      onCloseDetails={() => setSelectedItem(null)}
    />
  );
}
