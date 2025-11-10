import json
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
import pandas as pd
import cv2
from pymongo import MongoClient
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

# MongoDB connection
client = MongoClient("mongodb://mongodb:27017/")
db = client['predicted_images_db']
collection = db['predicted_images']

@app.get("/predict/")
async def get_predicted_files(page: int = 1, limit: int = 10):
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be greater than 0")
    
    skip = (page - 1) * limit
    images = list(collection.find().skip(skip).limit(limit))

    for img in images:
        img['_id'] = str(img['_id'])  # Convert ObjectId to string

    total_images = collection.count_documents({})
    total_pages = (total_images + limit - 1) // limit  # Calculate total pages

    return {
        "images": images,
        "total_images": total_images,
        "total_pages": total_pages,
        "current_page": page,
        "per_page": limit,
    }


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

    file_location = f"predicted/{file.filename}" 
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)  

    match_img_title = reduced_data.loc[match_img_index, "title"]
    collection.insert_one({"match_img_title": match_img_title, "predicted_img_path": file.filename})  
    
    return {
        "title": match_img_title
    }
