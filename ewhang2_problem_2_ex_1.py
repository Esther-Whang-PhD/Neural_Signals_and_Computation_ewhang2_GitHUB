from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from scipy.signal import correlate


class Problem2():
    def __init__(self, file_name = "TEST_MOVIE_00001-small.tif"):
        self.tif_small = self.load_tif(file_name)

    def part_A(self):
        plt.figure(figsize=(10,4))
        plt.suptitle("Summary Images")
        plt.subplot(1,3,1)
        tif_small_mean = self.tif_small.mean(axis=0)
        plt.title("Mean")
        plt.imshow(tif_small_mean)
        plt.subplot(1,3,2)
        tif_small_median = np.median(self.tif_small,axis=0)
        plt.title("Median")
        plt.imshow(tif_small_median)
        plt.subplot(1,3,3)
        tif_small_var = np.var(self.tif_small,axis=0)
        plt.title("Variance")
        plt.imshow(tif_small_var)
        return (tif_small_var, tif_small_median, tif_small_mean)
    
    def part_B(self):
        plt.figure(figsize=(12,7))
        plt.suptitle("Summary Images: Additional Statistics")
        plt.subplot(2,3,1)
        tif_small_mean = self.tif_small.mean(axis=0)
        plt.title("Mean")
        plt.imshow(tif_small_mean)
        plt.subplot(2,3,2)
        tif_small_median = np.median(self.tif_small,axis=0)
        plt.title("Median")
        plt.imshow(tif_small_median)
        plt.subplot(2,3,3)
        tif_small_var = np.var(self.tif_small,axis=0)
        plt.title("Variance")
        plt.imshow(tif_small_var)
        plt.subplot(2,3,4)
        tif_small_max= self.tif_small.max(axis=0)
        plt.title("Max")
        plt.imshow(tif_small_max)
        plt.subplot(2,3,5)
        tif_small_max= self.tif_small.min(axis=0)
        plt.title("Min")
        plt.imshow(tif_small_max)
    def load_tif(self, file_name):
        tif  = io.imread(file_name)
        return tif