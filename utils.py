import os
import cv2
import numpy as np
import pandas as pd


def normalize_bgr_img(img):
    """Normalize img: make img gray and put pixels to value in range [0, 1]"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_scaled_img = gray_img.astype("float32") / 255.0
    gray_scaled_flat_img = gray_scaled_img.flatten()

    return gray_scaled_flat_img


def read_normalize_img(folder_path, filename):
    """Read and normalize img by folder_path and filename"""
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(str(img_path))

    if img is None:
        raise Exception("img is None")

    norm_img = normalize_bgr_img(img)
    return norm_img


def read_normalize_imgs(folder_path):
    """Read and normalize imgs by folder_path"""
    pixels_list = []
    titles_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            norm_img = read_normalize_img(folder_path, filename)
            pixels_list.append(norm_img)
            titles_list.append(filename)

    pixels_arr = np.array(pixels_list, dtype="object")
    df = pd.DataFrame(pixels_arr)
    return df, titles_list
