import numpy as np
from PIL import Image
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pre_process(image_file: str) -> Image:
    im_raw = Image.open(image_file)
    width, height = im_raw.size
    sub = 100
    im_crop = im_raw.crop((0+sub, 0+sub, width-sub, height-sub))
    return im_crop


def pixel_matrix(processed_image: Image) -> np.ndarray:
    width, height = processed_image.size
    bands = len(processed_image.getbands())
    pix_matrix = np.array(processed_image.getdata(), ndmin=3)
    pix_matrix = pix_matrix.reshape(height, width, bands)
    # TODO CLOSE! transpose just the first 2 axes
    print(pix_matrix.transpose((0)).shape)
    print(pix_matrix[0:5, 0:5, 0])
    pix_matrix = np.zeros([width, height, bands])
    for x in range(width):
        # TODO Find new operation to make quicker
        for y in range(height):
            if bands == 1:
                pix_matrix[x, y] = processed_image.getpixel((x, y))
            else:
                for z in range(bands):
                    pix_matrix[x, y, z] = processed_image.getpixel((x, y))[z]
    print(pix_matrix[0:5,0:5,0])
    print(pix_matrix.shape)
    return pix_matrix


def average_bands(pix_matrix: np.ndarray) -> np.ndarray:
    return pix_matrix.mean(axis=2)


def process_windows(window_mat: np.ndarray, target_wid: int) -> (np.ndarray, np.ndarray):

    m_matrix = np.multiply(1/(target_wid ** 2), window_mat.sum(axis=(2, 3)))
    l_matrix = np.amax(window_mat, axis=(2, 3))

    return m_matrix, l_matrix


def calc_ilcm(padded: np.ndarray, m_matrix: np.ndarray, l_matrix: np.ndarray) -> np.ndarray:

    ilcm_win = view_as_windows(padded, (3,3), step=1)
    max_m_mat = np.amax(ilcm_win, axis=(2,3))
    ilcm_mat = np.divide(np.multiply(l_matrix, m_matrix), max_m_mat)

    return ilcm_mat


def calc_thres(matrix: np.ndarray, k: int) -> int:

    mean = matrix.mean()
    mu = matrix.std()
    thres = mu + k*mean
    print(np.amax(matrix))
    print(thres)

    return thres


def plot_heatmap(matrix: np.ndarray):
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()


def plot_3d(matrix: np.ndarray):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    # Create an X-Y mesh of the same dimension as the 2D data
    x_data, y_data = np.meshgrid( np.arange(matrix.shape[1]),
                                  np.arange(matrix.shape[0]))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = matrix.flatten()
    ax.bar3d( x_data,
              y_data,
              np.zeros(len(z_data)),
              1, 1, z_data)
    plt.show()


def main():
    # Input settings
    # Set to width of target in # of pixels
    target_width = 10
    # Sets stride of windows -- always half of the target width
    stride = target_width // 2

    # PreProcess and crop image
    im = pre_process('person.jpg')

    # Returns a matrix of pixel values
    p_matrix = pixel_matrix(im)

    # Only necessary when bands > 1
    ave_mat = average_bands(p_matrix)

    # Break image into windows
    windowed = view_as_windows(ave_mat,(target_width,target_width), step=stride)

    # Return m and l matrices for ilcm calculation
    m_mat, l_mat = process_windows(windowed, target_width)

    # Add a pad of 0's to aid calculation
    padded_m_mat = np.pad(m_mat, 1, 'constant', constant_values=0)

    # Return ilcm matrix for plotting
    ilcm_mat = calc_ilcm(padded_m_mat, m_mat, l_mat)

    # TODO
    #threshold = calc_thres(ilcm_mat, k=1.25)

    # Visualization
    plot_heatmap(ilcm_mat.transpose()) # Unsure why transpose is necessary?
    #plot_3d(ilcm_mat)

if __name__ == '__main__':
    main()