import json

from sklearn.decomposition import PCA

from utils import read_normalize_imgs, read_normalize_img
from predicting_model import predict_match_img, distances


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
    reduced_data.insert(0, 'title', titles)

    print(
        f"explained_variance {explained_variance},\ncomponents {components.shape} {components},\nreduced data {reduced_data.shape} {reduced_data}")

    # testing model
    test_img = read_normalize_img("./", "./squirrel.png")
    predict_match_img(test_img, components, avg_data,
                      reduced_data, distances["euclid"])

    # saving model params and reduced data
    with open('model_params.json', 'w') as f:
        model_params = {
            "components": list(map(lambda c: list(c), components)),
            "avg_data": list(avg_data)
        }
        json.dump(model_params, f)
    reduced_data.to_csv('reduced_data.csv', index=False,
                        header=True, na_rep='N/A', sep=',')
    print("model params and reduced data are saved.")
    