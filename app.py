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
    # MediaPipe 0.10.30+ no longer ships the legacy mp.solutions API.
    # Use the supported Tasks FaceLandmarker API and cache its model in /tmp.
    try:
        import os
        import urllib.request
        import cv2
        import mediapipe as mp
        import numpy as np

        model_path = "/tmp/face_landmarker.task"
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 100000:
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            temporary_path = model_path + ".download"
            with urllib.request.urlopen(model_url, timeout=45) as response:
                model_data = response.read()
            if len(model_data) < 100000:
                raise RuntimeError("Downloaded face model is unexpectedly small")
            with open(temporary_path, "wb") as model_file:
                model_file.write(model_data)
            os.replace(temporary_path, model_path)

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

        height, width = image_bgr.shape[:2]
        if width < 160 or height < 160:
            raise HTTPException(status_code=400, detail="Изображение слишком маленькое")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            result = detector.detect(mp_image)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Модуль анализа временно недоступен: {type(exc).__name__}: {exc}",
        )

    if not result.face_landmarks:
        raise HTTPException(status_code=422, detail="Лицо не обнаружено. Используй чёткое фото анфас.")

    if len(result.face_landmarks) > 1:
        raise HTTPException(status_code=422, detail="На фото найдено несколько лиц. Загрузи фото одного человека.")

    points = [(p.x * width, p.y * height) for p in result.face_landmarks[0]]

    # Preliminary framing checks only; this is not a validated image-quality score.
    landmark_x = [p[0] for p in points]
    landmark_y = [p[1] for p in points]
    face_left, face_right = min(landmark_x), max(landmark_x)
    face_top, face_bottom = min(landmark_y), max(landmark_y)
    face_box_width = face_right - face_left
    face_box_height = face_bottom - face_top
    face_width_image_ratio = face_box_width / width
    face_height_image_ratio = face_box_height / height
    edge_margin_ratio = min(
        face_left / width, face_top / height,
        (width - face_right) / width, (height - face_bottom) / height,
    )
    quality_warnings = []
    if width < 640 or height < 640:
        quality_warnings.append("Низкое разрешение снимка может снизить точность ориентиров.")
    if face_width_image_ratio < 0.20:
        quality_warnings.append("Лицо занимает небольшую часть кадра; лучше загрузить более крупный портрет.")
    if edge_margin_ratio < 0.02:
        quality_warnings.append("Лицевая область расположена близко к краю кадра.")
    photo_quality = {
        "status": "check_warnings" if quality_warnings else "basic_checks_passed",
        "image_width": width,
        "image_height": height,
        "face_width_image_ratio": round(face_width_image_ratio, 3),
        "face_height_image_ratio": round(face_height_image_ratio, 3),
        "minimum_edge_margin_ratio": round(edge_margin_ratio, 3),
        "warnings": quality_warnings,
        "not_assessed": ["Размытие и резкость", "Освещение", "Поворот головы и перспектива", "Мимика"],
        "note": "Предварительные проверки кадрирования; они не подтверждают точность антропометрических измерений."
    }


    # Face Landmarker mesh indices. Ratios are geometric observations,
    # not attractiveness scores or medical measurements.
    face_top = points[10]
    chin = points[152]
    left_face = points[234]
    right_face = points[454]
    # Additional approximate contour landmarks from the Face Mesh.
    left_jaw = points[172]
    right_jaw = points[397]
    left_cheek = points[234]
    right_cheek = points[454]
    left_brow = points[105]
    right_brow = points[334]
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
    jaw_width = distance(left_jaw, right_jaw)
    cheek_width = distance(left_cheek, right_cheek)
    brow_span = distance(left_brow, right_brow)
    eye_line_tilt = abs(left_eye_outer[1] - right_eye_outer[1]) / face_width
    mouth_line_tilt = abs(left_mouth[1] - right_mouth[1]) / face_width
    lower_face_height = distance(points[168], chin)

    if face_height <= 0 or face_width <= 0:
        raise HTTPException(status_code=422, detail="Не удалось определить геометрию лица")

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
        "photo_quality": photo_quality,
        "measurements": {
            "face_width_height_ratio": round(face_width / face_height, 3),
            "eye_distance_face_width_ratio": round(eye_distance / face_width, 3),
            "eye_span_face_width_ratio": round(eye_span / face_width, 3),
            "mouth_width_face_width_ratio": round(mouth_width / face_width, 3),
            "nose_tip_face_center_offset": round((nose_tip[0] - center_x) / face_width, 3),
            "left_right_landmark_deviation": round(symmetry_deviation, 3),
            "jaw_cheek_width_ratio": round(jaw_width / cheek_width, 3) if cheek_width else None,
            "brow_span_face_width_ratio": round(brow_span / face_width, 3),
            "eye_line_tilt_ratio": round(eye_line_tilt, 3),
            "mouth_line_tilt_ratio": round(mouth_line_tilt, 3),
            "lower_face_face_height_ratio": round(lower_face_height / face_height, 3),
        },
        "categories": {
            "harmony": "Измерены общие пропорции лица, положение глаз и нижняя треть. Балл пока не выставляется: нужны проверенные референсные диапазоны.",
            "symmetry": "Измерено среднее отклонение выбранных пар ориентиров. Это приблизительная 2D-оценка, чувствительная к повороту головы и мимике.",
            "angularity": "Добавлено отношение ширины нижней челюсти к ширине скул. Оно описывает ширину контура, но само по себе не измеряет углы или выраженность костных структур.",
            "dimorphism": "Не оценивается по фотографии: система не определяет пол, гормональные характеристики или гендерную идентичность.",
            "features": "Добавлены геометрические показатели глазной линии, бровей, рта, носа и нижней части лица. Это не оценка привлекательности отдельных черт.",
        },
        "category_details": {
            "harmony": {"status": "measurements_available", "score": None, "score_status": "requires_reference_ranges", "metrics": ["face_width_height_ratio", "eye_distance_face_width_ratio", "eye_span_face_width_ratio", "lower_face_face_height_ratio"]},
            "symmetry": {"status": "experimental_measurement", "score": None, "score_status": "requires_validation", "metrics": ["left_right_landmark_deviation", "eye_line_tilt_ratio", "mouth_line_tilt_ratio"]},
            "angularity": {"status": "partial_measurement", "score": None, "score_status": "requires_validated_contour_method", "metrics": ["jaw_cheek_width_ratio"]},
            "dimorphism": {"status": "not_assessed", "score": None, "score_status": "not_available", "metrics": []},
            "features": {"status": "descriptive_measurements", "score": None, "score_status": "requires_feature_specific_validation", "metrics": ["eye_distance_face_width_ratio", "eye_span_face_width_ratio", "brow_span_face_width_ratio", "mouth_width_face_width_ratio", "nose_tip_face_center_offset"]}
        },
        "limitations": [
            "Это экспериментальные геометрические измерения, а не оценка привлекательности или здоровья.",
            "Поворот головы, мимика, освещение, объектив и качество фото влияют на результат.",
            "Оценки по шкале 1–10 пока не выдаются: для них необходимы заранее определённые и проверенные критерии и референсные данные.",
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
