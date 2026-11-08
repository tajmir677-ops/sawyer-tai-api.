from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SawyerTai API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_SIZE = 8 * 1024 * 1024

@app.get("/")
async def home():
    return {"service": "SawyerTai API", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_photo(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Разрешены только JPEG, PNG и WEBP")

    data = await file.read(MAX_SIZE + 1)

    if len(data) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Максимальный размер файла — 8 МБ")

    valid_image = (
        data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"\x89PNG\r\n\x1a\n")
        or (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )

    if not valid_image:
        raise HTTPException(status_code=400, detail="Файл не прошёл проверку изображения")

    return {
        "received": True,
        "size_bytes": len(data),
        "message": "Фотография принята. Модуль анализа пока не подключён."
    }
