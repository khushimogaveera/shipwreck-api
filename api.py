import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from model import UNet
from inference import predict_full_image


app = FastAPI(title="SONAR AI API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet().to(device)

checkpoint_path = (
    BASE_DIR
    / "checkpoints"
    / "best_unet_shipwreck.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "SONAR AI API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.6
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("L")

    image_array = np.array(image)

    prediction_mask, probability_map = predict_full_image(
        image_array,
        model,
        device,
        threshold=threshold
    )

    predicted_pixels = int(
        prediction_mask.sum()
    )

    total_pixels = prediction_mask.size

    predicted_area_percent = (
        predicted_pixels / total_pixels
    ) * 100

    # Mask PNG
    _, mask_buffer = cv2.imencode(
        ".png",
        prediction_mask * 255
    )

    mask_base64 = base64.b64encode(
        mask_buffer.tobytes()
    ).decode("utf-8")

    # Overlay
    base = np.stack(
        [image_array] * 3,
        axis=-1
    )

    overlay = base.copy()

    overlay[prediction_mask == 1] = [
        255,
        0,
        0
    ]

    blended = (
        0.65 * base +
        0.35 * overlay
    ).astype(np.uint8)

    _, overlay_buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(
            blended,
            cv2.COLOR_RGB2BGR
        )
    )

    overlay_base64 = base64.b64encode(
        overlay_buffer.tobytes()
    ).decode("utf-8")

    return {
        "filename": file.filename,
        "width": int(image_array.shape[1]),
        "height": int(image_array.shape[0]),
        "predicted_pixels": predicted_pixels,
        "predicted_area_percent": round(
            predicted_area_percent,
            2
        ),
        "max_probability": float(
            probability_map.max()
        ),
        "mean_probability": float(
            probability_map.mean()
        ),
        "threshold": threshold,
        "mask_base64": mask_base64,
        "overlay_base64": overlay_base64,
        "device": str(device)
    }