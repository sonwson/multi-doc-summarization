import express from 'express';
import { History } from '../models/History.js';
import { protect } from '../middleware/authMiddleware.js';

const router = express.Router();

router.get('/', protect, async (req, res) => {
  const items = await History.find({ user: req.user.id }).sort({ createdAt: -1 });
  return res.json(items);
});

router.delete('/:id', protect, async (req, res) => {
  const item = await History.findOneAndDelete({ _id: req.params.id, user: req.user.id });

  if (!item) {
    return res.status(404).json({ message: 'History not found' });
  }

  return res.json({ message: 'History deleted' });
});

export default router;
