from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
import os
from typing import List

app = FastAPI()

MODEL_NAME = "jinaai/jina-embeddings-v3"

model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Model and Tokenizer loaded successfully for {MODEL_NAME}")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise RuntimeError(f"Failed to load model {MODEL_NAME}. Error: {e}")

class EmbeddingsRequest(BaseModel):
    texts: list[str]

@app.post("/embed/")
async def create_embeddings(request: EmbeddingsRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    try:
        inputs = tokenizer(request.texts, return_tensors="pt", padding=True, truncation=True)
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist() 

        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {e}")


## Chỉ đọc file và trả về Embeddings
@app.post("/embed-file/")
async def embed_file(file: UploadFile = File(...)):
    """
    Endpoint để tải lên một file văn bản (ví dụ: .txt, .md).
    Nội dung file sẽ được đọc, tạo embeddings và trả về cho client.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded yet. Please try again later.")

    file_content = ""
    try:
        # Đọc nội dung file
        # Lưu ý: Chỉ đọc toàn bộ file. Đối với file rất lớn, bạn có thể cần đọc từng phần.
        content_bytes = await file.read()
        file_content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode file content as UTF-8. Please ensure it's a valid text file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Kiểm tra nội dung rỗng
    if not file_content.strip():
        raise HTTPException(status_code=400, detail="File content is empty or contains only whitespace.")

    try:
        inputs = tokenizer(file_content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()

        # Trả về embedding của file dưới dạng một danh sách (vì nó là một embedding cho toàn bộ file)
        return {"file_name": file.filename, "embeddings": embeddings[0] if embeddings else None}
        # Sử dụng [0] vì outputs.last_hidden_state.mean(dim=1).tolist() sẽ là một list chứa một list embedding nếu đầu vào là một chuỗi duy nhất.

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings for file content: {e}")

@app.get("/health/")
async def health_check():
    if model is not None and tokenizer is not None:
        return {"status": "ok", "message": "Model and tokenizer loaded oki nhén."}
    else:
        return {"status": "loading", "message": "Model and tokenizer are still loading or failed to load."}