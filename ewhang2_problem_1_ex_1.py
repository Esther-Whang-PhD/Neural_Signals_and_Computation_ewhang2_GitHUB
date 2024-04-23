from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from scipy.signal import correlate

class Problem1():
    def __init__(self, file_name = "TEST_MOVIE_00001-small-motion.tif"):
        self.tif_motion = self.load_tif(file_name)
        print(self.tif_motion.shape)

    def part_A(self,num_of_frames = 500):
        print("Problem 1.A")
        self.fake_animation(self.tif_motion, num_of_frames)

    def part_B(self,num_frame_a, num_frame_b, num_frame_c):
        print("Problem 1.B")
        frame_a = self.tif_motion[num_frame_a,:,:]
        frame_b = self.tif_motion[num_frame_b,:,:]
        frame_c = self.tif_motion[num_frame_c,:,:]

        print("Biggest Shift")
        self.visualize_shift(frame_a, frame_b, num_frame_a, num_frame_b)
        self.image_corr(frame_a, frame_b)

        print("Different Shift")
        self.visualize_shift(frame_a, frame_c, num_frame_a, num_frame_c)
        self.image_corr(frame_a, frame_c)

    # load tif
    def load_tif(self, file_name):
        tif  = io.imread(file_name)
        return tif

    def fake_animation(self, tif_motion, num_of_frames):
        # some issues with plotly running in colabs, this is the workaround
        for i in range(0,num_of_frames):
            clear_output(wait=True)
            time.sleep(0.001)
            plt.figure(figsize=(4,4))
            plt.title(str(i))
            plt.imshow(tif_motion[i,:,:])
            plt.show()

    def visualize_shift(self, frame_a, frame_b, num_frame_a, num_frame_b):
        plt.figure(figsize=(5,3))
        plt.suptitle("Comparing Two Frames")
        plt.subplot(1,2, 1)
        plt.title("Frame " + str(num_frame_a))
        plt.imshow(frame_a)
        plt.subplot(1,2,2)
        plt.title("Frame " + str(num_frame_b))
        plt.imshow(frame_b)
        plt.tight_layout()
        plt.show()


    def image_corr(self, frame_a, frame_b):
        # get autocorrelation peak
        corr = correlate(frame_a , frame_a, mode='same')
        ssr = np.max(corr)
        snd = np.argmax(corr)
        ij_self, ji_self = np.unravel_index(snd, corr.shape)
        print(ij_self, ji_self)

        # get cross-correlation peak
        corr = correlate(frame_a , frame_b, mode='same')
        ssr = np.max(corr)
        snd = np.argmax(corr)
        ij_7, ji_7 = np.unravel_index(snd, corr.shape)
        print("Correlation Peak is at")
        print(ij_7, ji_7)
        print("X-Shift of", ij_7-ij_self)
        print("Y-Shift of", ji_7-ji_self)