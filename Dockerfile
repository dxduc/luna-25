# Sử dụng Python 3.10
FROM python:3.10-slim

# Tắt cache py để giảm dung lượng
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài đặt các thư viện cần thiết cho SimpleITK, numpy, v.v.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app
WORKDIR /app

# Copy file requirements trước để tận dụng cache layer
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ project vào container
COPY . .

# Tạo thư mục lưu upload và output
RUN mkdir -p uploads/input
RUN mkdir -p results

# Expose port uvicorn
EXPOSE 8000

# Lệnh chạy API
CMD ["python", "api.py"]