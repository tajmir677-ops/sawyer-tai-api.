FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    mesa-common-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgles2 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    && ldconfig \
    && test -n "$(find /usr/lib -name 'libGLESv2.so.2' -print -quit)" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN ldconfig -p | grep -F 'libGLESv2.so.2' && python -c "import mediapipe as mp; print('MediaPipe import OK:', mp.__version__)"

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
