
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF, FastICA

class Problem5():
    def __init__(self, tif_small):
        self.tif_small = tif_small
        self.pixels_by_time = self.tif_small.reshape(500, 500*500).T

    def part_A(self):
        pixels_by_time_scaled = StandardScaler().fit_transform(self.pixels_by_time)
        num_of_components = 6
        pca2 = PCA(n_components=num_of_components)
        X_pca = pca2.fit_transform(pixels_by_time_scaled)
        
        plt.figure(figsize=(12,3))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("PCA Components", fontsize=18, y=0.95)

        # loop through the length of tickers and keep track of index
        for n in np.arange(num_of_components):
            # add a new subplot iteratively
            plt.subplot(1, num_of_components, n + 1)
            img_pca = X_pca[:,n].reshape(500,500)

            # filter df and plot ticker on the new subplot axis
            plt.imshow(img_pca)
            plt.title("PCA "+str(n))

    def part_B(self, num_of_components = 6):
        model = NMF(n_components=num_of_components, init='random', random_state=0)
        W = model.fit_transform(self.pixels_by_time)

        plt.figure(figsize=(12,3))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("NMF", fontsize=18, y=0.95)
        for n in np.arange(num_of_components):
            # add a new subplot iteratively
            plt.subplot(1, num_of_components, n + 1)
            img_nmf = W[:,n].reshape(500,500)

            # filter df and plot ticker on the new subplot axis
            plt.imshow(img_nmf)
            plt.title("NMF "+str(n))

    def part_C(self, num_of_components = 6):
        # train the model
        ica = FastICA(n_components=num_of_components)
        S_ = ica.fit_transform(self.pixels_by_time)  # estimated independent sources

        plt.figure(figsize=(12,3))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("ICA Components", fontsize=18, y=0.95)

        for n in np.arange(num_of_components):
            # add a new subplot iteratively
            plt.subplot(1, num_of_components, n + 1)
            img_ica = S_[:,n].reshape(500,500)

            # filter df and plot ticker on the new subplot axis
            plt.imshow(img_ica)
            plt.title("ICA "+str(n))

