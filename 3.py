
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

# Lists to store the expected value and confidence intervals
seq = []
confint = []
for m in M:
    # List of generated values of I
    I = []
    for i in range(100):
        I1, _ = gen(m)
        I.append(I1)
    var1 = np.var(I)
    
    # List of generated values of control random variable Y
    Y = np.sqrt(np.random.uniform(0,1,size=np.shape(I)))

    # Calculation of c
    c = - np.cov(I, Y)[0][1] / np.var(Y)


    # New value of I
    I1 = I + c * (Y-np.mean(Y))
    var2 = np.var(I1)
    
    print(f"Percentage reduction in variance for M = {m} is ", (var1-var2)/var1 * 100)
    print("Square of Corelation Coeff = ",np.square(np.cov(I, Y)[0][1]/np.sqrt(np.var(I)*np.var(Y))))
    print("")



