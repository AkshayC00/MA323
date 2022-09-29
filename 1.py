
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Defining the values of M
M = [100,1000,10000,100000]

# helper function
def f(u):
    return np.exp(np.sqrt(u))   

def gen(m):
    # Define mean and var to calculate the expected value and confidence intervals
    mean=0
    var=0
    for i in range(m):
        # Generate u from U(0,1)
        y = f(np.random.uniform(0,1))
        mean+=y
        var+=y**2
    mean /= float(m)
    var /= float(m)
    var -= mean**2

    return (mean, var)



# Lists to store the expected value and confidence intervals
seq = []
confint = []
for m in M:
    I, s = gen(m) 
    seq.append(I)
    delta = norm.ppf(1.95/2.0)
    confint.append((I-delta*s/np.sqrt(m), I+delta*s/np.sqrt(m)))



# Print for each value of M
for i in range(4):
    print(f"Expected value for M = {M[i]} is {seq[i]} with 95% Confidence Interval {confint[i]}")





