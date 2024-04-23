
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from scipy.signal import correlate
import cv2


class Problem3():
    def __init__(self, tif_small):
        self.tif_small = tif_small
        self.centers = np.zeros((5,2))

    def part_A(self):
        blurred_circle_template = self.create_template()
        tif_small_var = np.var(self.tif_small,axis=0)
        processed_summary_image = self.process_summary_image(tif_small_var)
        mask = self.threshold_summary_image(processed_summary_image, blurred_circle_template)
        self.centers, labeled_image = self.find_centers(mask, processed_summary_image)
        # show circle template, processed summary image, mask, and centers
        if self.centers.shape == (5,2):
            print("Congratulations! You've found the 5 ROI required for this homework.")
            print("Your ROI are located at: ")
            print(self.centers)
            self.plot_part_A(processed_summary_image, mask, labeled_image)
        else:
            print("Five ROI were not found.")
            print("Despair")

    def part_B(self):
        self.plot_summary_and_ROI()

    def create_template(self):
        radius = 8
        image_size = 16
        # Generate x and y coordinates for the circle
        x = np.linspace(-radius, radius, image_size)
        y = np.linspace(-radius, radius, image_size)
        X, Y = np.meshgrid(x, y)

        # Create a mask for the circular shape
        circle_mask1 = (X**2 + Y**2) <= radius**2
        circle_mask2 = (X**2 + Y**2) >= (radius-3)**2
        circle_mask = circle_mask1 == circle_mask2
        # Convert the mask to an integer image (0s and 1s)
        circle_image = circle_mask.astype(float)
        kernel_size = (7,7)
        blurred_circle_image = cv2.GaussianBlur(circle_image, kernel_size, 0)
        return blurred_circle_image
    
    def process_summary_image(self, summary_img):
        kernel_size = (21,21)
        blurred_summary_img= cv2.GaussianBlur(summary_img, kernel_size, 0)
        return blurred_summary_img
    
    def threshold_summary_image(self, blurred_tif_small_var, circle_template):
        image_corr = correlate(blurred_tif_small_var , circle_template, mode='same')
        thres = self.threshold_mean_std(image_corr)
        mask = image_corr > thres
        return mask

    def threshold_mean_std(self,image, k=4):
        return image.mean() + k * image.std()
    
    def find_centers(self, mask, img, visualize =False):
        contours, hierarchy = cv2.findContours(mask.astype(int), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        viz_img = img.copy()
        min_value = np.min(viz_img)
        max_value = np.max(viz_img)
        normalized_image = ((viz_img - min_value) / (max_value - min_value)) * 255

        centers = np.zeros((10,2))
        # weird bug where the contour is spotted twice
        i=0
        for c in contours:

            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            #cv2.circle(normalized_image, (cx, cy), 7, (255, 255, 255), 2)
            cv2.rectangle(normalized_image, (cx-9, cy-9), (cx+9, cy+9), (255, 255, 255), 1)
            cv2.putText(normalized_image, str(cx) + ", " + str(cy), (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            centers[i,:] = (cx, cy)
            i +=1
        centers = np.unique(centers,axis=0).astype(int)
        return centers, normalized_image
    
    def plot_part_A(self, blurred_summary_image, mask, labeled_image):
        plt.figure(figsize=(8,3))
        plt.suptitle("Finding Region of Interest")
        plt.subplot(1,3,1)
        plt.title("Blurred Summary Image")
        plt.imshow(blurred_summary_image)
        plt.subplot(1,3,2)
        plt.title("Threshold")
        plt.imshow(mask)
        plt.subplot(1,3,3)
        plt.title("Region of Interest")
        plt.imshow(labeled_image)

    def plot_summary_and_ROI(self):
        plt.figure(figsize=(12,7))
        plt.suptitle("Checking ROI with Summary Images")
        plt.subplot(2,3,1)
        plt.title("Mean")
        plt.imshow(self.label_ROI(self.tif_small.mean(axis=0)))
        plt.subplot(2,3,2)
        plt.title("Median")
        plt.imshow(self.label_ROI(np.median(self.tif_small,axis=0)))
        plt.subplot(2,3,3)
        plt.title("Variance")
        plt.imshow(self.label_ROI(np.var(self.tif_small,axis=0)))
        plt.subplot(2,3,4)
        plt.title("Max")
        plt.imshow(self.label_ROI(self.tif_small.max(axis=0)))
        plt.subplot(2,3,5)
        plt.title("Min")
        plt.imshow(self.label_ROI(self.tif_small.min(axis=0)))

    def label_ROI(self,viz_img):
      min_value = np.min(viz_img)
      max_value = np.max(viz_img)
      normalized_image = ((viz_img - min_value) / (max_value - min_value)) * 255
      for i in np.arange(5):
        cv2.rectangle(normalized_image, (self.centers[i, 0]-9, self.centers[i, 1]-9), (self.centers[i, 0]+9, self.centers[i, 1]+9), (255, 255, 255), 1)
        cv2.putText(normalized_image, str(self.centers[i, 0]) + ", " + str(self.centers[i, 1]), (self.centers[i, 0], self.centers[i, 1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      return normalized_image

          