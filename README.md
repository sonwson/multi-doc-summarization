# Multi-Documents Summarization Template

## Folder structure

```text
multi-documents_summarization/
|-- ai-service/
|   |-- app.py
|   `-- requirements.txt
|-- backend/
|   |-- src/
|   |   |-- config/
|   |   |-- middleware/
|   |   |-- models/
|   |   |-- routes/
|   |   |-- utils/
|   |   |-- app.js
|   |   `-- server.js
|   |-- .env.example
|   `-- package.json
|-- frontend/
|   |-- src/
|   |   |-- api/
|   |   |-- components/
|   |   |-- context/
|   |   `-- pages/
|   |-- .env.example
|   |-- index.html
|   |-- package.json
|   |-- postcss.config.js
|   `-- tailwind.config.js
|-- results/
`-- README.md
```

## Quick start

1. Create `.env` files from each `.env.example`.
2. Install dependencies:
   - `cd backend && npm install`
   - `cd frontend && npm install`
   - `cd ai-service && pip install -r requirements.txt`
3. Run AI service: `uvicorn app:app --reload --host 127.0.0.1 --port 8000`
4. Run backend: `npm run dev`
5. Run frontend: `npm run dev`

## API overview

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/summarize`
- `GET /api/history`
- `DELETE /api/history/:id`

## AI server contract

Backend expects your AI service to accept:

```json
{
  "documents": ["doc 1", "doc 2", "doc 3"]
}
```

And return:

```json
{
  "summary": "Combined summary text"
}
```

## Integrated trained model

The web app is now wired to the trained checkpoint at:

- `results/checkpoints_extractive/best_extractive_sentence_model.bin`
- `results/checkpoints_extractive/tokenizer/`

`ai-service/app.py` loads this checkpoint directly with PhoBERT, scores candidate sentences, and returns an extractive summary through `POST /api/summarize`.
