import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# matrix is a 2d numpy array
# save file could br like foo.pdf

def get_data_matrix_from_file(filename):
    X = np.loadtxt(filename, delimiter=',')
    return X

def get_2d_matrix_tsne(matrix):
    X_2_dimensional = TSNE(n_components=2).fit_transform(matrix)
    return X_2_dimensional

def get_2d_matrix_pca(matrix):
    pca = PCA(n_components=2)
    X_2_dimensional = pca.fit_transform(matrix)
    return X_2_dimensional

def visualize_2d_matrix_without_labels(matrix_2d, plot_saving_file):
    plt.plot(matrix_2d[:,0], matrix_2d[:,1], 'ko')
    plt.show()
    plt.savefig(plot_saving_file)


def visualize_2d_matrix_with_labels(matrix_2d, matrix_2d_labels, plot_saving_file):
    colors = ['red', 'green', 'blue', 'purple', 'cyan', 'magenta', 'yellow', 'black']
    plt.scatter(matrix_2d[:,0], matrix_2d[:,1], c=matrix_2d_labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()
    plt.savefig(plot_saving_file)
