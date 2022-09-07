from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Function to generate random sample
def genSample(m):
    # List to store the sample
    seq = []
    for i in range(m):
        # generate U from U(0,1)
        u = np.random.random()
        # generate Y_i
        y = np.exp(np.sqrt(u))
        # append in the sequence
        seq.append(y)

    return seq

# Function to calculate the confidence interval
def confInt(confidence, seq):
    m = len(seq)
    mean = np.sum(seq)/m
    # Compute value of delta
    delta = norm.ppf((confidence+1)/2.)
    # Compute value of unbiased variance
    s = np.sqrt(np.sum(np.square(seq-mean))/(m-1))
    # Return the confidence interval
    return ( mean - delta*s/np.sqrt(m), mean + delta*s/np.sqrt(m))

M = [100, 1000, 10000, 100000]

for m in M:
    seq = genSample(m)
    mean = np.sum(seq)/m
    left, right = confInt(0.95, seq)
    # print(seq)
    print(f"Expected mean for sample size of {m} = ", mean)
    print(f"95% Confidence interval for the sample = ({left}, {right})")