## About this project

This project contains the module with math model that is able to predict the most similar image (blueprint/development, 998x1409 pixels)
among existed images (in folder A4). Also this project contains the module with server where we can test our model by uploading test images.

![alt text](doc_imgs/about_project.png)

## Idea of the model

The model consists of two steps: preparation images and calculating distances.

### Preparation of images

First we make an image gray.
Then we scale it to values in range from 0 to 1.  
After that we make them flat.
Image is normalized if it passed this preparation. Such image will be adopted to our model
because it has data within range from 0 to 1 (it will be important for calculating distances).
Then we use PCA (principal components analysis) to find the most valuable components of our images
so we can remove noises and focus on the most important information.

| Step | Data type | Shape |
| ---- | --------- | ----- |
| Loading img | BGR [0-255, 0-255, 0-255] | (998, 1409) |
| Convert to gray | gray 0-255 | (998, 1409) |
| Scale img | gray 0-1 | (998, 1409) |
| Flatten img| gray 0-1 | (998 $\cdot$ 1409) |
| Use PCA | float | n components |

### Calculating the most closed image

We calculate distance between normalized existing and uploaded images.
The distance is a symmetric always-non-negative function (In our project we only use Minkowski distances).
Among all existing images we pick the one image that has min distance with uploaded one.

So if $X$ is a set of normalized images and $y$ is an uploaded image: 

matched_img = $arg_{x \in X} min_{x \in X} d(x, y)$
$\underset{x \in X}{\arg}$
$a \; b$ 
$ \min $
$ \arg $
$ \text{matched\_img} = {\arg}_{x \in X} \; {\min}_{x \in X} \; d(x, y) $
$ \text{matched\_img} = \underset{x \in X}{\arg} \; \underset{x \in X}{\min} \; d(x, y) $

## Possible improvements

- This project can be extended to using other formats of developments/blueprints (A3, A2, ...) by using the same
  number of main components so prepared data will have the same dimensions (but we should use different PCA model for each format,
  it might not work because the components may be too different).
- We can create a logger instead of usual printing in the function `predict_match_img` so we can store training results more easily (e.g. put
  the result in logs.txt).
- This algo can be improved if developers/engineers will make kind of additional standardization about placing detail on the blueprint (make only one position and exactly coordinates + defining the same view for all details).
- We can extract first only informative tables and panels in developments (e.g. a table with material, plane of detail, ...).
- We can actually hire an engineer that can say what is the most "similar" imgs so we can setup feedback for our model
  (+it allows us to define the best distance function).
- The blueprints/developments should be from one sphere (e.g. only molds, engines, ...) so our model will work better for some specific blueprints/developments.

## Mini-usage documentation

This project has a few useful functions: predict_match_img,
normalize_bgr_image.

### `predict_match_img`

#### Inputs

- `normalized_img: List[float]` is a list that consists of gray pixels that are coded in the range 0-1.
- `components: List[List[float]]` is a list that consists of main components from PCA.
- `avg_data: List[float]` is a list that consists of mean values of properties of initial data.
- `reduced_data: List[List[float]]` is a list (dataframe) that consists of transformed initial data by PCA.
- `dist_func: (function(p1: List[float], p2: List[float], dim: int)->float` is a function that satisfies distances requirements (>=0, symmetric, =0 if points are the same).

#### Output

- `match_index: (int>=-1)` is an index of matched img in initial/reduced data (if -1 -> not found).

#### Example

```
import read_normalize_img from utils
import distances, predict_match_img from predicting_model

reduced_data = pd.read_csv("reduced_data.csv")
model_params = {}
with open("model_params.json", "r") as file:
    model_params = json.load(file)
test_img = read_normalize_img("./", "./squirrel.png")

match_index = predict_match_img(test_img, model_params["components"],model_params["avg_data"], reduced_data, distances["euclid"])
print(reduced_data.iloc[match_index, "title"])
```

### `normalize_bgr_img`

#### Input

- `img: file` is an img that is read as cv2.imread("./img_path").

#### Output

- `normalized_img: List[float]` is a list of gray scaled pixels.

#### Example

```
import os
import cv2
import normalize_bgr_img from utils

img_path = os.path.join("./A4", "cap.png")
img = cv2.imread(str(img_path))
norm_img = normalize_bgr_img(img)

"""
actually there are functions: read_normalize_img and read_normalize_imgs
first one accepts folder_path and folder_name, second one accepts folder_path
they read images with cv2 and then normalize them by function normalize_bgr_img
they return normalized img(s).
""" 
```

## Setup launching

First you should define a virtual environment (basic module venv of python is used), in the root directory of the project run these commands in your terminal to create and activate the virtual environment:

`python -m venv venv`

`.\venv\Scripts\activate`

After that run the following command to install required packages for this projects:

`pip install -r req.txt`

After activating virtual environment (`.\venv\Scripts\activate`), we must run the following command to train our model by
the files(developments/blueprints) in our folder A4:

`python training_model.py`

It should print us `"model params and reduced data are saved."` in the terminal. If it does then we can launch our server:

`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

The terminal should tell us when it's launched then we can go to `http://localhost:8000/docs` where we can test our model
with our test images.
