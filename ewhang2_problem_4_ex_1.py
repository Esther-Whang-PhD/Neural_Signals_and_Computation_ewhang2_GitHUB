
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Problem4():
    def __init__(self, tif_small, centers):
        self.tif_small = tif_small
        self.centers = centers
        self.time_traces = np.zeros((1,1))

    def part_A(self):
        print("Problem 4.A")
        # take the centers, make patches, then take the mean of the patches
        self.time_traces = self.extract_time_traces(self.centers)
        # plot the time traces
        plt.figure(figsize=(12,3))
        for i in np.arange(5):
            x, y = self.centers[i,:]
            plt.plot(self.time_traces[:,i] +5-i-0.5,label= str(x) + ", " + str(y))
        plt.legend(title = "Center Position")
        plt.title("Problem 4.A: Time Traces for ROIs in Normalized Video")            

    def part_B(self):
        print("Problem 4.B")
        plt.figure(figsize=(18,4))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("Maximum Value")
        for i in np.arange(5):
          trace_argmax = np.argmax(self.time_traces[:, i])
          labeled_max_img = self.visual_inspection(trace_argmax)
          plt.subplot(1,5,i+1)
          plt.title("Trace " + str(self.centers[i,:]) + ", Frame " + str(trace_argmax)) 
          plt.imshow(labeled_max_img)

    def extract_time_traces(self, centers):
        time_traces = np.zeros((500,5))
        for i in np.arange(5):
          x, y = centers[i,:]
          patch = self.tif_small[:,y-5:y+5,x-5:x+5]
          patch = np.sum(patch, axis=(1,2))
          patch = (patch -patch.min())/(patch.max() - patch.min())
          time_traces[:,i] = patch
        return time_traces
    
    def visual_inspection(self, num_sample_frame):
        frame = self.tif_small[int(num_sample_frame),:,:]
        viz_img = frame.copy()
        min_value = np.min(viz_img)
        max_value = np.max(viz_img)
        normalized_image = ((viz_img - min_value) / (max_value - min_value)) * 255

        for i in np.arange(5):
            cv2.rectangle(normalized_image, (self.centers[i, 0]-9, self.centers[i, 1]-9), (self.centers[i, 0]+9, self.centers[i, 1]+9), (255, 255, 255), 1)
            cv2.putText(normalized_image, str(self.centers[i, 0]) + ", " + str(self.centers[i, 1]), (self.centers[i, 0], self.centers[i, 1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return normalized_image
        


   