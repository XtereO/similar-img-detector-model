import os
import cv2
import numpy as np
import pandas as pd


def normalize_bgr_img(img):
    '''Normalize img: make it gray and put pixels to value [0,1]'''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_norm = gray_img.astype('float32') / 255.0

    return gray_img_norm


def read_normalize_img(folder_path, filename):
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(str(img_path))

    if img is None:
        raise Exception("img is None")

    norm_img = normalize_bgr_img(img)
    flat_img = norm_img.flatten()
    return flat_img


def read_normalize_imgs(folder_path):
    pixels_list = []
    titles_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
            flat_img = read_normalize_img(folder_path, filename)
            pixels_list.append(flat_img)
            titles_list.append(filename)

    pixels_arr = np.array(pixels_list, dtype="object")
    df = pd.DataFrame(pixels_arr)
    return df, titles_list
