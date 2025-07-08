# Sử dụng một base image Python nhẹ và phù hợp. python:3.11-slim-bookworm là một lựa chọn tốt.
FROM python:3.11-slim-bookworm

# Đặt biến môi trường cho việc không tạo file .pyc và không buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Đặt biến môi trường cho thư mục cache của Hugging Face bên trong container
# Model sẽ được tải vào đường dẫn này
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy file requirements.txt và cài đặt các dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Cài đặt PyTorch (CPU-only) ---
# Kiểm tra: https://pytorch.org/get-started/locally/
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# --- TẢI MODEL TRỰC TIẾP TRONG CONTAINER KHI BUILD ---
ARG JINA_EMBEDDING_HASH="f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"
ARG XLM_ROBERTA_FLASH_HASH="2b6bc3f30750b3a9648fe9b63448c09920efe9be"

# Sử dụng Python script để tải jina-embeddings-v3
# Model sẽ tự động được lưu vào $HF_HOME đã định nghĩa.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jinaai/jina-embeddings-v3', revision='${JINA_EMBEDDING_HASH}', local_dir_use_symlinks=False)"

# Sử dụng Python script để tải xlm-roberta-flash-implementation
# Model sẽ tự động được lưu vào $HF_HOME đã định nghĩa.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jinaai/xlm-roberta-flash-implementation', revision='${XLM_ROBERTA_FLASH_HASH}', local_dir_use_symlinks=False)"

# Copy mã nguồn ứng dụng FastAPI
COPY main.py .

# Khai báo cổng mà ứng dụng lắng nghe
EXPOSE 8000

# Lệnh để khởi chạy ứng dụng bằng Uvicorn
# --host 0.0.0.0 để lắng nghe trên tất cả các interface
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]