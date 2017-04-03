# coding:utf8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import *

# Part 1: Basic Function
A = np.eye(5)
print "5x5 Identity Matrix: \n", A


# Part 2: Plotting
print "\n Plotting Data ...\n"
X = []
y = []
data = open('ex1data1.txt')
for line in data.readlines():
    lineArr = line.strip().split(',')
    X.append([float(lineArr[0])])
    y.append([float(lineArr[1])])
m = np.shape(y)[0]  # number of training examples

fig = plt.figure()
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
ax = fig.add_subplot(111)
ax.scatter(X, y, s=30, c='red', marker='x')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Figure 1: Scatter plot of training data', fontsize=20)
plt.xlim([4, 24])
plt.ylim([-5, 25])
plt.xticks([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
plt.yticks([-5, 0, 5, 10, 15, 20, 25])
# plt.show()


# Part 3: Cost and Gradient descent
a = np.ones((m, 1))
X = np.hstack((a, X))     # Add a column of ones to x
theta = np.zeros((2, 1))  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print '\n Testing the cost function ...\n'
# compute and display initial cost


def computeCost(X, y, theta):
    h = X * mat(theta)
    result = sum(power((h-y), 2)/(2*m))
    return result


def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = zeros((num_iters, 1))
    for iter in xrange(num_iters):
        h = X * mat(theta)
        theta[0] = theta[0] - alpha*sum(h-y)/m
        theta[1] = theta[1] - alpha*sum(h-y)*X[:, 1]/m
        J_history[iter] = computeCost(X, y, theta)    # Save the cost J in every iteration
    return theta, J_history


J = computeCost(X, y, theta)
print 'With theta = [0 ; 0]\nCost computed = ', J
print 'Expected cost value (approx) 32.07\n'

# further testing of the cost function
test = mat([[-1], [2]])
J = computeCost(X, y, test)
print '\nWith theta = [-1 ; 2]\nCost computed = ', J
print 'Expected cost value (approx) 54.24\n'

print '\nRunning Gradient Descent ...\n'
# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)


# print theta to screen
print 'Theta found by gradient descent:\n%f\n', theta
print 'Expected theta values (approx)\n'
print ' -3.6303\n  1.1664\n\n'

# Plot the linear fit
# hold on; % keep previous plot visible
plt.plot(X[:, 1], X*theta, '-')
plt.legend('Training data', 'Linear regression')
plt.show()
# don't overlay any more plots on this figure
