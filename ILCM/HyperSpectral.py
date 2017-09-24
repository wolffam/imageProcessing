import numpy as np
from PIL import Image
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

im = Image.open('person.jpg')
width, height = im.size
print(width,height)

def pixel_matrix(processed_image: Image) -> np.ndarray:
    width, height = processed_image.size
    bands = len(processed_image.getbands())
    pix_matrix = np.zeros([height, width, bands])
    for x in range(width):
        for y in range(height):
            for z in range(bands):
                pix_matrix[y, x, z] = processed_image.getpixel((x, y))[z]
    return pix_matrix


def pixel_matrix_opt(processed_image):
    data_list = list(processed_image.getdata())
    test = np.array(data_list)
    print(test.shape)
    test2 = test.reshape(600,463,3)
    print(test2.shape)
    print(test2[0,0,0])
    print(data_list[0])
    print(processed_image.getdata().shape)
#pix_mat = pixel_matrix(im)
pixel_matrix_opt(im)


pca = PCA(n_components=1)
pca_mat = np.zeros([height, width])
for x in range(width):
    for y in range(height):
        pca.fit(pix_mat[y, x, :].reshape(-1,1))
        PCA(copy=True, iterated_power='auto', n_components=1, random_state=None,
          svd_solver='auto', tol=0.0, whiten=False)
        pca_mat[y,x] = pca.singular_values_
        if x % 100 == 0 and y % 100 == 0:
            print(x, y)

def plot_heatmap(matrix: np.ndarray):
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()

plot_heatmap(pca_mat)

