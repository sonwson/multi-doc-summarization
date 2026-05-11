# Multi-Document Summarization

Ứng dụng web tóm tắt nhiều tài liệu trong một lần nhập, gồm 3 phần:

- `frontend/`: giao diện React + Vite
- `backend/`: API Express, auth và history
- `ai-service/`: FastAPI chạy model tóm tắt extractive

## Cấu trúc thư mục

```text
multi-documents_summarization/
|-- ai-service/
|   |-- app.py
|   `-- requirements.txt
|-- backend/
|   |-- src/
|   |-- .env.example
|   `-- package.json
|-- frontend/
|   |-- src/
|   |-- .env.example
|   `-- package.json
|-- results/
|   `-- checkpoints_extractive/
`-- README.md
```

## Yêu cầu

- Node.js 18+
- Python 3.10+
- `pip`

## Chuẩn bị biến môi trường

Tạo file `.env` từ file mẫu:

```powershell
Copy-Item backend\.env.example backend\.env
Copy-Item frontend\.env.example frontend\.env
```

Giá trị mẫu hiện tại:

`backend/.env`


`frontend/.env`

```env
VITE_API_URL=http://localhost:5000/api
```

## Cài dependencies

### Frontend

```powershell
cd frontend
npm install
```

### Backend

```powershell
cd backend
npm install
```

### AI service

```powershell
cd ai-service
pip install -r requirements.txt
```

## Cách chạy local

Mở 3 terminal riêng.

### 1. Chạy AI service

```powershell
cd D:\multi-documents_summarization\ai-service
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 2. Chạy backend

```powershell
cd D:\multi-documents_summarization\backend
npm run dev
```

Backend mặc định chạy ở `http://localhost:5000`.

### 3. Chạy frontend

```powershell
cd D:\multi-documents_summarization\frontend
npm run dev
```

Frontend mặc định chạy ở `http://localhost:5173`.

## Thứ tự khởi động nên dùng

1. `ai-service`
2. `backend`
3. `frontend`

Nếu frontend mở được nhưng bấm tóm tắt không chạy, kiểm tra lại `AI_SERVER_URL` trong `backend/.env` và chắc chắn AI service đang chạy.

## API chính

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/summarize`
- `GET /api/history`
- `DELETE /api/history/:id`

## Model đang dùng

`ai-service/app.py` đang load checkpoint từ:

- `results/checkpoints_extractive/best_extractive_sentence_model.bin`
- `results/checkpoints_extractive/tokenizer/`

Đây là dữ liệu model/runtime, không nên đẩy lên GitHub repo thường.

## Lưu ý khi push GitHub

Không push các thư mục và file sau:

- `.env`
- `node_modules/`
- `frontend/dist/`
- `results/`
- `*.log`

Repo đã có `.gitignore` để bỏ qua các file này.
