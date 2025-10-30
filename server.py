import json

from fastapi import FastAPI, File, UploadFile
import numpy as np
import pandas as pd
import cv2
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from utils import normalize_bgr_img
from predicting_model import predict_match_img, distances

reduced_data = pd.read_csv("reduced_data.csv")
model_params = {}
with open("model_params.json", "r") as file:
    model_params = json.load(file)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/", tags=["Match Model"])
async def upload_file(file: UploadFile = File(...)):
    """Upload an image file and predict matched file in existing data."""

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

    norm_img = normalize_bgr_img(img)
    match_img_index = predict_match_img(
        norm_img, model_params["components"], model_params["avg_data"], reduced_data, distances["euclid"])

    return {
        "title": reduced_data.loc[match_img_index, "title"]
    }
