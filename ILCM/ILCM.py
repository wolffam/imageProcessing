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
    pix_matrix = np.zeros([height, width])
    for x in range(width):
        for y in range(height):
            # '0' to just get the 'R' value
            pix_matrix[y, x] = processed_image.getpixel((x, y))#[0]
    return pix_matrix


def process_windows(window_mat: np.ndarray, target_wid: int) -> (np.ndarray, np.ndarray):
    x_win_dim = window_mat.shape[0]
    y_win_dim = window_mat.shape[1]
    m_matrix = np.zeros([x_win_dim,y_win_dim])
    l_matrix = np.zeros([x_win_dim,y_win_dim])

    for y_win in range(y_win_dim):
        for x_win in range(x_win_dim):
            m_matrix[x_win, y_win] = 1 / (target_wid ** 2) * window_mat[x_win,y_win].sum()
            l_matrix[x_win, y_win] = np.amax(window_mat[x_win,y_win])

    return m_matrix, l_matrix


def calc_ilcm(padded: np.ndarray, m_matrix: np.ndarray, l_matrix: np.ndarray) -> np.ndarray:
    x_win_dim = m_matrix.shape[0]
    y_win_dim = m_matrix.shape[1]
    ilcm_mat = np.zeros([x_win_dim,y_win_dim])
    ilcm_win = view_as_windows(padded,(3,3),step=1)
    for y_win in range(y_win_dim):
        for x_win in range(x_win_dim):
            max_m = np.amax(ilcm_win[x_win,y_win])
            ilcm_mat[x_win,y_win] = l_matrix[x_win,y_win]*m_matrix[x_win, y_win]/max_m

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
    target_width = 150
    # Sets stride of windows -- always half of the target width
    stride = target_width // 2

    # PreProcess and crop image
    im = pre_process('building.png')

    # Returns a matrix of pixel values
    p_matrix = pixel_matrix(im)

    # Break image into windows
    windowed = view_as_windows(p_matrix,(target_width,target_width), step=stride)

    # Return m and l matrices for ilcm calculation
    m_mat, l_mat = process_windows(windowed, target_width)

    # Add a pad of 0's to aid calculation
    padded_m_mat = np.pad(m_mat, 1, 'constant', constant_values=0)

    # Return ilcm matrix for plotting
    ilcm_mat = calc_ilcm(padded_m_mat, m_mat, l_mat)

    # TODO
    #threshold = calc_thres(ilcm_mat, k=1.25)

    # Visualization
    plot_heatmap(ilcm_mat)
    #plot_3d(ilcm_mat)

if __name__ == '__main__':
    main()