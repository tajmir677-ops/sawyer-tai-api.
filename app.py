import io
import math

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SawyerTai API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_SIZE = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

@app.get("/")
async def home():
    return {"service": "SawyerTai API", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

def distance(a, b):
    return math.dist(a, b)

def analyze_landmarks(image_bytes):
    # Load the computer-vision stack only when an image is analyzed.
    # This lets the web server bind its port without waiting for MediaPipe startup.
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        face_mesh = mp.solutions.face_mesh
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Модуль анализа временно недоступен: {type(exc).__name__}: {exc}",
        )

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    height, width = image.shape[:2]
    if width < 160 or height < 160:
        raise HTTPException(status_code=400, detail="Изображение слишком маленькое")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as detector:
        result = detector.process(rgb)

    if not result.multi_face_landmarks:
        raise HTTPException(status_code=422, detail="Лицо не обнаружено. Используй чёткое фото анфас.")

    if len(result.multi_face_landmarks) > 1:
        raise HTTPException(status_code=422, detail="На фото найдено несколько лиц. Загрузи фото одного человека.")

    points = [(p.x * width, p.y * height) for p in result.multi_face_landmarks[0].landmark]

    # MediaPipe Face Mesh landmark indices. Ratios are geometric observations,
    # not attractiveness scores or medical measurements.
    face_top = points[10]
    chin = points[152]
    left_face = points[234]
    right_face = points[454]
    left_eye_outer = points[33]
    right_eye_outer = points[263]
    left_eye_inner = points[133]
    right_eye_inner = points[362]
    nose_tip = points[1]
    left_mouth = points[61]
    right_mouth = points[291]

    face_height = distance(face_top, chin)
    face_width = distance(left_face, right_face)
    eye_distance = distance(left_eye_inner, right_eye_inner)
    mouth_width = distance(left_mouth, right_mouth)
    eye_span = distance(left_eye_outer, right_eye_outer)

    if face_height <= 0 or face_width <= 0:
        raise HTTPException(status_code=422, detail="Не удалось определить геометрию лица")

    # Approximate left/right landmark correspondence deviation, normalized
    # by face width. This is sensitive to pose, expression and image quality.
    pairs = [(33, 263), (133, 362), (159, 386), (61, 291), (234, 454)]
    deviations = []
    center_x = (left_face[0] + right_face[0]) / 2
    for left_idx, right_idx in pairs:
        left = points[left_idx]
        right = points[right_idx]
        mirrored_left_x = 2 * center_x - left[0]
        deviations.append(math.hypot(mirrored_left_x - right[0], left[1] - right[1]) / face_width)
    symmetry_deviation = sum(deviations) / len(deviations)

    return {
        "face_detected": True,
        "face_count": 1,
        "measurements": {
            "face_width_height_ratio": round(face_width / face_height, 3),
            "eye_distance_face_width_ratio": round(eye_distance / face_width, 3),
            "eye_span_face_width_ratio": round(eye_span / face_width, 3),
            "mouth_width_face_width_ratio": round(mouth_width / face_width, 3),
            "nose_tip_face_center_offset": round((nose_tip[0] - center_x) / face_width, 3),
            "left_right_landmark_deviation": round(symmetry_deviation, 3),
        },
        "categories": {
            "harmony": "Измерения пропорций получены; интерпретация и валидированная шкала пока не реализованы.",
            "symmetry": "Получено приблизительное отклонение пар ориентиров; результат зависит от ракурса.",
            "angularity": "Пока не оценивается: для этого нужна отдельная проверенная методика контура челюсти.",
            "dimorphism": "Не оценивается: система не делает выводов о поле или гормональных характеристиках по фото.",
            "features": "Получены базовые геометрические отношения; отдельная оценка глаз, носа, губ и челюсти ещё не реализована.",
        },
        "limitations": [
            "Это экспериментальные геометрические измерения, а не оценка привлекательности или здоровья.",
            "Поворот головы, мимика, освещение, объектив и качество фото влияют на результат.",
            "Не используй результат для медицинских или иных важных решений.",
        ],
        "message": "Лицо обнаружено. Получены предварительные геометрические измерения."
    }

@app.post("/upload")
async def upload_photo(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
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

    return analyze_landmarks(data)
