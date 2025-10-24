from sklearn.decomposition import PCA
from utils import read_normalize_imgs, read_normalize_img

# function which will find the similar img


def img_match(normalized_img):
    pass


if __name__ == "__main__":

    folder_path = "./A4"
    data, titles = read_normalize_imgs(folder_path)

    pca = PCA(svd_solver="full").fit(data)

    # taking enough components to describe main data and remove noises
    n = 0
    explained_variance = 0
    for v in pca.explained_variance_ratio_:
        if explained_variance > 0.9:
            break
        explained_variance += v
        n += 1

    components = pca.components_[:n]
    avg_data = data.mean()
    reduced_data = (data - avg_data) @ components.transpose()

    print(
        f"explained_variance {explained_variance},\ncomponents {components.shape} {components},\nreduced data {reduced_data.shape} {reduced_data}")

    # this code should be done as a function so I can use it in backend (arg: img file itself; take other args by once read config with model params at the start of server)
    test_img = read_normalize_img("./", "./squirrel.png")
    reduced_test_img = (test_img-avg_data) @ components.transpose()
    min_err, index = float("Inf"), -1
    rows, cols = reduced_data.shape
    for i in range(rows):
        err = 0
        for j in range(cols):
            # actually here can be any loss function (e.g. std, mae, ...) - we just want to find min value for this err - it can be as arg in function (loss function)
            err += (reduced_data.iloc[i, j] - reduced_test_img[j])**2
        if min_err > err:
            min_err = err
            index = i
        print(err, titles[i])
    print(f"the best match: {titles[index]}, {min_err}")


# in algo we should first transform img: normalize, extract A rows, use F (F*(Y-A) = U); then use kNN (k=1; metric choice the best appropriate or just Euclid)

# I should export it in some file: avg, components, transformed data

# then do mini-fastapi for this (uploading img to find the most similar in some database)

# make a mini-doc with launching(server)/using(functions) this project with describing this algo (so we can improve it in future in case we want)
# if I have enough wish then I can create also logger to understand how my function calculates

# this algo can be improved if developers will make kind of additional standartization about placing detail on the blueprint (make only one position and exactly coordinates + defining the same view for all details) - add it to Readme in Improvement section (or if we extract exactly detail and its param info first, then my algo with reducing)
# to improve we can actually hire engineer that can say what is the most "similar" imgs so we can setup feedback for our model (+understand what is the best lost function)
