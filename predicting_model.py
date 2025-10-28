import numpy as np


def minkov_distance(p1, p2, dim, k):
    d = 0
    for i in range(dim):
        d += (p1[i] - p2[i])**k
    return d


distances = {
    "minkov": lambda k: (lambda p1, p2, dim: minkov_distance(p1, p2, dim, k)),
    "manhattan": lambda p1, p2, dim: minkov_distance(p1, p2, dim, 1),
    "euclid": lambda p1, p2, dim: minkov_distance(p1, p2, dim, 2),
}


def predict_match_img(normalized_img, components, avg_data, reduced_data, dist_func):
    reduced_img = (normalized_img-np.array(avg_data)
                   ) @ np.array(components).transpose()
    min_dist, index = float("Inf"), -1
    X = reduced_data.drop("title", axis=1)
    y = reduced_data["title"]
    rows, cols = X.shape
    for i in range(rows):
        dist = dist_func(X.iloc[i], reduced_img, cols)
        if min_dist > dist:
            min_dist = dist
            index = i
        print(dist, y[i])
    print(f"the best match: {y[index]}, {min_dist}")
    return index
