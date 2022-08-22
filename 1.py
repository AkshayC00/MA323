import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N = 10000

# Creating a list to store sample
seq = []

# defining the function f(x)
def f(x):
    return 20*x*(1-x)**3

# defining the function g(x)
def g(x):
    return 2*x

# defining the inverse of CDF G(x)
def G_in(x):
    return np.sqrt(x)

# list to store num of iterations to generate a variable from distribution f(x)
iter = [0]*N

# Function for generating 10000 random variables
def generate_rv(c):
    seq = []
    for i in range(N):
        while 1:
            # generate u from U(0,1)
            u = np.random.uniform(0,1)

            # generate random variable X
            x = G_in(u)

            # Compute probability of acceptance
            p = f(x)/(c*g(x))
            p_x = np.random.uniform(0,1) 
            if p_x<=p:
                iter[i]+=1
                seq.append(x)
                break
            else:
                iter[i]+=1
                
    # computing sample mean
    sample_mean = np.mean(seq)
    print("Sample mean = ", sample_mean)

    print("Expectation of f(x) = ", 1/3)

    # Calculating approx value of P(0.25<=X<=0.75)
    p1 = np.sum([x<=0.75 and x>=0.25 for x in seq])/len(seq)
    print("Estimated value of P(0.25<=X<=0.75): ", p1)

    # Exact value of probability
    print("Exact value of P(0.25<=X<=0.75): ", 79/128)

    # Estimated average number of iterations required
    avg_it = np.mean(iter)
    print("Average number of iterations = ", avg_it)

    # Average number of iterations
    print("Actual Average number of iterations = ", c)

    return seq

# Generating seq using the smallest possible value of c
seq = generate_rv(c=10)

fig, ax = plt.subplots(1,2, figsize=(20,5))

print("Plotting graphs....")

ax[0].hist(seq, bins=20)
ax[0].set_xlabel('Range')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Histogram of sample')

x = np.arange(0, 1, 0.01)
y = f(x)

ax[1].plot(x, y, '-b')
ax[1].set_xlabel('Range')
ax[1].set_ylabel('Frequency')
ax[1].set_title('Plot of PDF')

# generate sample for c=20,100
print()
seq1 = generate_rv(20)
print()
seq2 = generate_rv(100)

fig, ax = plt.subplots(1,3, figsize=(30,5))


ax[0].hist(seq, bins=100, color='r')
ax[0].set_xlabel('Range')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Histogram for c=10')

ax[1].hist(seq1, bins=100, color='orange')
ax[1].set_xlabel('Range')
ax[1].set_ylabel('Frequency')
ax[1].set_title('Histogram for c=20')

ax[2].hist(seq2, bins=100, color='lime')
ax[2].set_xlabel('Range')
ax[2].set_ylabel('Frequency')
ax[2].set_title('Histogram for c=100')


plt.show()