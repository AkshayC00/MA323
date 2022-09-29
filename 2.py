import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Defining the values of M
M = [100,1000,10000,100000]

# helper function
def f(u):
    return np.exp(np.sqrt(u))

def gen(m):
    # Define mean and var to calculate the expected value
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

def genAntithetic(m):
    # Define mean and var to calculate the expected value 
    mean=0
    var=0
    for i in range(m):
        # Generate u from U(0,1)
        u = np.random.uniform(0,1)
        y = (f(u)+f(1-u))/2.
        mean+=y
        var+=y**2
    mean /= float(m)
    var /= float(m)
    var -= mean**2

    return (mean, var)


# Lists to store the expected value
seq = []
confint = []
for m in M:
    I1, s1 = gen(m)
    I2, s2 = genAntithetic(m)
    print(f"Percentage reduction in variance = ", (s1-s2)/s1 * 100)



