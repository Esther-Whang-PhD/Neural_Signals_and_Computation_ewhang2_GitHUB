from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Problem2():
    def __init__(self):
      self.mat_data = loadmat('sample_dat.mat')
      self.sample_data = self.mat_data['dat'] 
      self.sample_data_py = self.reformat_mat()
      self.psth_per_neuron = self.calc_psth_per_neuron()
      self.smoothed_psth_per_neuron = self.tried_gaussian_process()

    def partA(self):
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.title("Spike Rate (Averaged Across Trials)")
        plt.plot(self.psth_per_neuron.T)
        plt.subplot(1,2,2)
        plt.title("Spike Rate after Smoothing (Averaged Across Trials)")
        plt.plot(self.smoothed_psth_per_neuron.T)

    def partB(self):
        psth_scaled = StandardScaler().fit_transform(self.psth_per_neuron)
        num_of_components = 3
        pca2 = PCA(n_components=num_of_components)
        psth_pca = pca2.fit_transform(psth_scaled.T)
        print(np.shape(psth_pca))
        # Create 3D figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plotting the 3D scatter plot
        ax.plot(psth_pca[:,0],psth_pca[:,1], psth_pca[:,2])

        # Set labels and title
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('PSTH')

        plt.show()

        psth_scaled = StandardScaler().fit_transform(self.smoothed_psth_per_neuron)
        num_of_components = 3
        pca2 = PCA(n_components=num_of_components)
        psth_pca = pca2.fit_transform(psth_scaled.T)
        print(np.shape(psth_pca))
        # Create 3D figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plotting the 3D scatter plot
        ax.plot(psth_pca[:,0],psth_pca[:,1], psth_pca[:,2])

        # Set labels and title
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('Smooth PSTH')

        plt.show()


    def reformat_mat(self):
        sample_data_py = np.zeros((53, 400, 56))
        for i in np.arange(53):
            test = self.sample_data[:,i]
            trial = test[0][1]
            sample_data_py[:,:,i] = trial
        return sample_data_py
    def calc_psth_per_neuron(self):
        # get a histogram for every neuron?
        psth_per_neuron = np.zeros((53, 400))
        avg_firing_rate = np.mean(self.sample_data_py, axis=2)
        for i in np.arange(53):
            psth_per_neuron[i,:] = np.histogram(np.squeeze(avg_firing_rate[i,:]), bins=400)[0]
        return psth_per_neuron
    
    def tried_gaussian_process(self):
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 2.0))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)

        # Plot original and smoothed PSTHs for each neuron
        smoothed_psth_per_neuron = np.zeros((53, 400))
        num_neurons = self.sample_data_py.shape[0]  # Number of neurons
        for neuron_idx in range(num_neurons):
            # Extract PSTH data for the current neuron
            psth_data = self.psth_per_neuron[neuron_idx,:]

            # Flatten PSTH data (combine time and trials)
            time_points = np.arange(self.psth_per_neuron.shape[1])
            flattened_data = np.column_stack((time_points.reshape(-1, 1), psth_data.reshape(-1, 1)))
        
            # Fit Gaussian Process model to the flattened data
            gp.fit(flattened_data[:, 0][:, np.newaxis], flattened_data[:, 1])

            # Generate new time points for prediction
            new_time_points = np.linspace(0, self.sample_data_py.shape[1], 400)[:, None]

            # Predict the smoothed PSTH using the GP model
            smoothed_psth, _ = gp.predict(new_time_points, return_std=True)
            smoothed_psth_per_neuron[neuron_idx,:] = smoothed_psth
            # Plot original and smoothed PSTHs
            # plt.subplot(num_neurons, 1, neuron_idx + 1)
            # plt.plot(time_points, psth_per_neuron[i,:], label='Original PSTH', color='blue', alpha=0.7)
            # plt.plot(new_time_points, smoothed_psth, label='Smoothed PSTH', color='red')
            # plt.xlabel('Time')
            # plt.ylabel('Firing Rate')
            # plt.title(f'Neuron {neuron_idx + 1} PSTH')
            # plt.legend()
        return smoothed_psth_per_neuron
        # plt.tight_layout()
        # plt.show()