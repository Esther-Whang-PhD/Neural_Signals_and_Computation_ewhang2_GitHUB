from scipy.special import factorial
import math
from scipy.optimize import minimize
import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

class Problem1():
    def __init__(self):
      self.N = 0
      self.M = 0
      self.X = 0

    def partA(self, N):
      gaussian_window =  windows.gaussian(N, std=5)
      t = np.arange(N)
      cosine_part = np.cos(2 * np.pi * t / 10)
      g = gaussian_window * cosine_part
      g = g.reshape(-1, 1)
      X = 2 * np.random.rand(N, 1)
      lam = np.exp(np.dot(g.ravel(), X.ravel()))
      r_values = np.arange(0, 50) # range of possible values
      pmf_values = (lam**r_values * np.exp(-lam)) / factorial(r_values)
      probabilities = pmf_values / np.sum(pmf_values)
      samples = np.random.choice(r_values, size=1000, p=probabilities)

      plt.figure(figsize=(4,4))
      plt.hist(samples)

      X = 2 * np.random.rand(N, 1)
      lam = np.exp(np.dot(g.ravel(), X.ravel()))
      r_values = np.arange(0, 50) # range of possible values
      pmf_values = (lam**r_values * np.exp(-lam)) / factorial(r_values)
      probabilities = pmf_values / np.sum(pmf_values)
      samples = np.random.choice(r_values, size=1000, p=probabilities)

      plt.figure(figsize=(4,4))
      plt.hist(samples)
      X = 2 * np.random.rand(N, 1)
      lam = np.exp(np.dot(g.ravel(), X.ravel()))
      r_values = np.arange(0, 20) # range of possible values
      pmf_values = (lam**r_values * np.exp(-lam)) / factorial(r_values)
      probabilities = pmf_values / np.sum(pmf_values)
      samples = np.random.choice(r_values, size=1000, p=probabilities)

      plt.figure(figsize=(4,4))
      plt.hist(samples)

    def partB(self, N, M):
      X_b, g_b, r_b = self.generate_response(N, M)
      # Often in higher dimensional settings we linearize models, in this
      # case assuming that r ≈ Xg with added independent, identically distributed (i.i.d.) Gaussian
      # noise. Set up a probabilistic relationship (likelihood) of r conditioned on g under a Gaussian
      # noise assumption. Specifically, assume that r = Xg + ε where ε is a mean-zero, Gaussian noise
      # vector with variance σ
      # I^2. Use the likelihood derived to set up a maximum likelihood inference of
      # g given the responses and stimuli (hint: think least-squares). How close is the estimate of g to
      # the real g?

      sigma = 0.1
      eps = np.random.normal(0, sigma, M)
      r_estimate = X_b.T@g_b + eps.reshape(-1,1) 
      p_inv_X = np.linalg.pinv(X_b.T)
      g_estimate = p_inv_X@r_estimate
      mse  = np.linalg.norm((g_estimate-g_b)**2)
      print("MSE is")
      print(np.linalg.norm((g_estimate-g_b)**2))
      return

    def partC(self,N):
      g_ML_MN, g_estimate_MN = self.tuning_curve_ML(N, N)
      g_ML_M2N, g_estimate_M2N = self.tuning_curve_ML(N, 2*N)
      g_ML_MN2, g_estimate_MN2 = self.tuning_curve_ML(N, int(N/2))
      plt.figure(figsize=(12,5))
      plt.suptitle("Maximum Likelihood")
      plt.subplot(1,3,1)
      plt.title("M=N")
      plt.plot(g_estimate_MN, label="Pseudoinverse Tuning Curve")
      plt.plot(g_ML_MN, label="M=N")
      plt.legend()
      plt.subplot(1,3,2)
      plt.title("M=2N")
      plt.plot(g_estimate_M2N, label="Pseudoinverse Tuning Curve")
      plt.plot(g_ML_M2N, label="M=2N")
      plt.legend()
      plt.subplot(1,3,3)
      plt.title("M=N/2")
      plt.plot(g_estimate_MN2, label="Pseudoinverse Tuning Curve")
      plt.plot(g_ML_MN2, label="M=N/2")
      plt.legend()

    def partD(self,N):
      g_MAP_MN, g_estimate_MN = self.tuning_curve_MAP_gauss(N, N)
      g_MAP_M2N, g_estimate_M2N = self.tuning_curve_MAP_gauss(N, 2*N)
      g_MAP_MN2, g_estimate_MN2 = self.tuning_curve_MAP_gauss(N, int(N/2))
      plt.figure(figsize=(12,5))
      plt.suptitle("MAP: Gaussian")
      plt.subplot(1,3,1)
      plt.title("M=N")
      plt.plot(g_estimate_MN, label="Pseudoinverse Tuning Curve")
      plt.plot(g_MAP_MN, label="M=N")
      plt.legend()
      plt.subplot(1,3,2)
      plt.title("M=2N")
      plt.plot(g_estimate_M2N, label="Pseudoinverse Tuning Curve")
      plt.plot(g_MAP_M2N, label="M=2N")
      plt.legend()
      plt.subplot(1,3,3)
      plt.title("M=N/2")
      plt.plot(g_estimate_MN2, label="Pseudoinverse Tuning Curve")
      plt.plot(g_MAP_MN2, label="M=N/2")
      plt.legend()

      g_MAP_MN, g_estimate_MN = self.tuning_curve_MAP_poisson(N, N)
      g_MAP_M2N, g_estimate_M2N = self.tuning_curve_MAP_poisson(N, 2*N)
      g_MAP_MN2, g_estimate_MN2 = self.tuning_curve_MAP_poisson(N, int(N/2))
      plt.figure(figsize=(12,5))
      plt.subplot(1,3,1)
      plt.title("M=N")
      plt.suptitle("MAP: Poisson")
      plt.plot(g_estimate_MN, label="Pseudoinverse Tuning Curve")
      plt.plot(g_MAP_MN, label="M=N")
      plt.legend()
      plt.subplot(1,3,2)
      plt.title("M=2N")
      plt.plot(g_estimate_M2N, label="Pseudoinverse Tuning Curve")
      plt.plot(g_MAP_M2N, label="M=2N")
      plt.legend()
      plt.subplot(1,3,3)
      plt.title("M=N/2")
      plt.plot(g_estimate_MN2, label="Pseudoinverse Tuning Curve")
      plt.plot(g_MAP_MN2, label="M=N/2")
      plt.legend()

    def partE(self, N,A, regularization, noise):
      g_MAP_MN, g_estimate_MN =  self.tuning_curve_MAP_poisson(N, 1000*N, A=A, regularization=regularization, noise = noise)
      g_MAP_M2N, g_estimate_M2N =  self.tuning_curve_MAP_poisson(N, 5000*N, A=A,regularization=regularization, noise = noise)
      g_MAP_MN2, g_estimate_MN2 =  self.tuning_curve_MAP_poisson(N, 7500*N, A=A,regularization=regularization, noise = noise)
      plt.figure(figsize=(12,5))
      plt.subplot(1,3,1)
      plt.title("M=1000*N")
      plt.plot(g_estimate_MN, label="Pseudoinverse Tuning Curve")
      plt.plot(g_MAP_MN, label="M=1000*N")
      plt.legend()
      plt.subplot(1,3,2)
      plt.title("M=5000*N")
      plt.plot(g_estimate_M2N, label="Pseudoinverse Tuning Curve")
      plt.plot(g_MAP_M2N, label="M=5000*N")
      plt.legend()
      plt.subplot(1,3,3)
      plt.title("M=7500*N")
      plt.plot(g_estimate_MN2, label="Pseudoinverse Tuning Curve")
      plt.plot(g_MAP_MN2, label="M=7500*N")
      plt.legend()

    def generate_response(self, N, M, A=1):
      response = np.zeros((M,1))
      X = 2 * np.random.rand(N, M)*A
      t = np.arange(N)
      cosine_part = np.cos(2 * np.pi * t / 10)
      gaussian_window =  windows.gaussian(N, std=5)
      g = gaussian_window * cosine_part
      g = g.reshape(-1, 1)
      lam = np.exp(X.T@g)
      r_values = np.arange(0, 50) # range of possible values
      pmf_values = (lam**r_values * np.exp(-lam)) / factorial(r_values)
      
      
      for i in np.arange(M):
        probabilities = pmf_values[i,:] / np.sum(pmf_values[i,:])
        response[i] = np.random.choice(r_values, size=1, p=probabilities)
      return X, g, response

    def log_likelihood_gauss(self,g, X, r_values):
        predicted_values = X.T @ g
        squared_diff = (r_values - predicted_values) ** 2
        log_likelihood = 0.5 * np.sum(squared_diff)
        return log_likelihood

    def log_likelihood_poisson(self,g, X, r_values):
        lam = np.exp(X.T@g)
        pmf_values = (lam**r_values * np.exp(-lam)) / factorial(r_values)
        log_pmf_values = np.log(pmf_values)
        return -np.sum(log_pmf_values)  # Minimize negative log likelihood

    def tuning_curve_ML(self,N, M, A=1, noise = 0.5):
        X, g, r = self.generate_response(N, M, A=A)

        result = minimize(self.log_likelihood_poisson, g.ravel(),  args=(X, r.ravel()))
        optimized_g = result.x

        eps = np.random.normal(0, noise, M)
        r_linearmodel = X.T@g + eps.reshape(-1,1)

        p_inv_X = np.linalg.pinv(X.T)
        g_estimate = p_inv_X@r_linearmodel 
        return optimized_g, g_estimate

    def log_prior_gauss(self,g, mu, sigma):
        prior = 0.5 * np.sum(((g - mu) ** 2) / (sigma ** 2) + np.log(2 * np.pi * (sigma ** 2)))
        return prior

    def log_posterior_gauss(self,g, X, r_values, regularization):
        mu = 0
        sigma = regularization
        return self.log_prior_gauss(g, mu, sigma) + self.log_likelihood_gauss(g, X, r_values)

    def log_posterior_poisson(self,g, X, r_values, regularization):
        mu = 0
        sigma = regularization
        smoothing = self.log_prior_gauss(g, mu, sigma)
        return self.log_likelihood_poisson(g, X, r_values) + smoothing


    def tuning_curve_MAP_gauss(self,N, M, A=1, regularization=1, noise = 0.5):

        X_map, g, r = self.generate_response(N, M, A=A)
        result = minimize(self.log_posterior_gauss, g.ravel(),  args=(X_map, r.ravel(), regularization))
        optimized_g = result.x

        eps = np.random.normal(0, noise, M)
        r_linearmodel = X_map.T@g + eps.reshape(-1,1)

        p_inv_X = np.linalg.pinv(X_map.T)
        g_estimate = p_inv_X@r_linearmodel 
        return optimized_g, g_estimate

    def tuning_curve_MAP_poisson(self,N, M,A=1, regularization=1, noise = 0.5):

      X_map, g, r = self.generate_response(N, M, A=A)
      result = minimize(self.log_posterior_poisson, g.ravel(),  args=(X_map, r.ravel(), regularization))
      optimized_g = result.x

      eps = np.random.normal(0,noise, M)
      r_linearmodel = X_map.T@g + eps.reshape(-1,1)

      p_inv_X = np.linalg.pinv(X_map.T)
      g_estimate = p_inv_X@r_linearmodel 

      return optimized_g, g_estimate