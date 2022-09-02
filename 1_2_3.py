from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
# Defining the values of mean vector and variance-covariance matrix
mu = np.array([5, 8])
A = [-0.5, 0, 0.5, 1]
a = A[0]
sigma = np.array([[1, 2*a],[2*a, 4]])

# Creating figure for plotting
fig1 = plt.figure(figsize=(10,10))
fig2 = plt.figure(figsize=(10,10))
fig1.suptitle("Histogram of generated sample")
fig2.suptitle("Contour Plots of actual density")

k=1
for a in A:
    seq = []
    # Generating the multivariate distribution
    for i in range(10000):
        z1 = np.random.normal(0,1)
        z2 = np.random.normal(0,2)
        z = np.array([z1, z2])
        x = mu+np.dot(sigma, z)
        seq.append(x)
    seq = np.array(seq)
    print("The value of mean vector: ",np.array([np.mean(seq[:,0]), np.mean(seq[:,1])]))
    
    # Plotting them on 2d Histogram
    ax = fig1.add_subplot(2,2,k)
    ax.hist2d(seq[:,0], seq[:,1], bins=50)
    ax.set_title(f"Plot for a = {a}")

    # Plotting the contour plots
    ax1 = fig2.add_subplot(2,2,k)
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mu, sigma)
    ax1.contourf(x, y, rv.pdf(pos))

    k+=1

plt.show()