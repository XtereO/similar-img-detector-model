import numpy as np


def minkowski_distance(p1, p2, dim, k):
    """Function calculates Minkowski distance for two points in some dimension with k param.

    Args:
        p1 (list of floats): A list that consists of coordinates of first point.
        p2 (list of floats): The same as p1 but for second point.
        dim (int>0): A number of dimensions of points/space.
        k (int>0): A param of Minkowski distance that indicates the extent of differences points.

    Returns:
        distance (d): A number that shows the distance between points p1 and p2.
    
    """
    d = 0
    for i in range(dim):
        d += (abs(p1[i] - p2[i]))**k
    return d


distances = {
    "minkov": lambda k: (lambda p1, p2, dim: minkowski_distance(p1, p2, dim, k)),
    "manhattan": lambda p1, p2, dim: minkowski_distance(p1, p2, dim, 1),
    "euclid": lambda p1, p2, dim: minkowski_distance(p1, p2, dim, 2),
}


def predict_match_img(normalized_img, components, avg_data, reduced_data, dist_func):
    """Function predicts an index of img that is matched to normalized_img arg.

    Args:
        normalized_img (list of floats): A list that consists of gray pixels that are coded in the range 0-1.
        components (list of lits with floats): A list that consists of main components from PCA.
        avg_data (list of floats): A list that consists of mean values of properties of initial data.
        reduced_data (list of lists with floats): A list (dataframe) that consists of transformed initial data by PCA.
        dist_func (function(p1: List[float], p2: List[float], dim: int)->float): A function that satisfies distances requirements (>=0, symmetric, =0 if points are the same). 

    Returns:
        match_index (int>=-1): An index of matched img in initial/reduced data (if -1 -> not found).
    """
    reduced_img = (normalized_img-np.array(avg_data)
                   ) @ np.array(components).transpose()
    X = reduced_data.drop("title", axis=1)
    y = reduced_data["title"]

    rows, cols = X.shape
    min_dist, match_index = float("Inf"), -1
    for i in range(rows):
        dist = dist_func(X.iloc[i], reduced_img, cols)
        if min_dist > dist:
            min_dist = dist
            match_index = i
        print(dist, y[i])

    print(f"the best match: {y[match_index]}, {min_dist}")
    return match_index
