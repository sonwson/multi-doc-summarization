import express from 'express';
import axios from 'axios';
import multer from 'multer';
import { PDFParse } from 'pdf-parse';
import { optionalAuth } from '../middleware/authMiddleware.js';
import { History } from '../models/History.js';

const router = express.Router();
const MAX_SOURCES = 10;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { files: MAX_SOURCES },
  fileFilter: (req, file, callback) => {
    const lowerName = file.originalname.toLowerCase();
    const isTextFile = file.mimetype === 'text/plain' || lowerName.endsWith('.txt');
    const isPdfFile = file.mimetype === 'application/pdf' || lowerName.endsWith('.pdf');

    if (isTextFile || isPdfFile) {
      return callback(null, true);
    }

    return callback(new Error('Only .txt and .pdf files are supported'));
  }
});

const extractFileText = async (file) => {
  if (file.originalname.toLowerCase().endsWith('.pdf')) {
    const parser = new PDFParse({ data: file.buffer });
    const parsed = await parser.getText();
    await parser.destroy();
    return parsed.text || '';
  }

  return file.buffer.toString('utf-8');
};

const handleUpload = (req, res, next) => {
  upload.array('files', MAX_SOURCES)(req, res, (error) => {
    if (error) {
      return res.status(400).json({ message: error.message });
    }

    return next();
  });
};

router.post('/', optionalAuth, handleUpload, async (req, res) => {
  try {
    const rawInputs = req.body.inputs;
    const parsedInputs = Array.isArray(rawInputs)
      ? rawInputs
      : typeof rawInputs === 'string'
        ? JSON.parse(rawInputs || '[]')
        : [];

    const manualInputs = parsedInputs
      .map((item) => `${item}`.trim())
      .filter(Boolean);

    const files = req.files || [];
    if (manualInputs.length + files.length > MAX_SOURCES) {
      return res.status(400).json({ message: `Maximum ${MAX_SOURCES} documents are allowed per summary` });
    }

    const fileTexts = await Promise.all(files.map((file) => extractFileText(file)));
    const fileNames = files.map((file) => file.originalname);
    const inputs = [...manualInputs, ...fileTexts].map((item) => item.trim()).filter(Boolean);

    if (!inputs.length) {
      return res.status(400).json({ message: 'No input text provided' });
    }

    const aiResponse = await axios.post(process.env.AI_SERVER_URL, {
      documents: inputs
    });

    const summary = aiResponse.data.summary || 'No summary returned from AI service.';

    const history = req.user
      ? await History.create({
          user: req.user.id,
          inputs,
          summary,
          sourceFiles: fileNames
        })
      : null;

    return res.status(201).json({ summary, history, savedToHistory: Boolean(history) });
  } catch (error) {
    const detail = error.response?.data || error.message;
    return res.status(500).json({ message: 'Summarization failed', detail });
  }
});

export default router;
