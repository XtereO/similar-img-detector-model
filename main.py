# do it for the first time only for A4 developments 
# init venv

def read_imgs(folder_path: str) -> list[list]:
    pass

folder_path = "./some_path"
data = read_imgs(folder_path)

# normalize imgs somehow: e.g. rgb -> to black shades (by calculating mean of color e.g. (r+g+b)/3 -> so instead of list [r,g,b]->one value;
# and then make this value related e.g. x/255)?

# create PCA model by sklearn
# take components enough to explain 0.90? (I should think about the border number)
# extract transform functions from PCA (enough components, mean values A for a difference: F*(X - A) = Z - so I need matrix F, Z and A)

# in algo we should first transform img: normalize, extract A rows, use F (F*(Y-A) = U); then use kNN (k=1; metric choice the best appropriate or just Euclid)  


