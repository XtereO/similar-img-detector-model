from sklearn.decomposition import PCA
from utils import read_imgs

folder_path = "./A4"
data = read_imgs(folder_path)

print(data.describe)

pca = PCA(svd_solver="full").fit(data)
print(pca.components_.shape, pca.explained_variance_.shape)

reduced_data = pca.transform(data)
print(reduced_data, pca.explained_variance_ratio_)


# create PCA model by sklearn
# take components enough to explain 0.90? (I should think about the border number)
# extract transform functions from PCA (enough components, mean values A for a difference: F*(X - A) = Z - so I need matrix F, Z and A)

# in algo we should first transform img: normalize, extract A rows, use F (F*(Y-A) = U); then use kNN (k=1; metric choice the best appropriate or just Euclid)  


