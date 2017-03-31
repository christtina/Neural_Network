"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
import math

class LogisticRegression:
    """
    This class implements a Logistic Regression Classifier.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        
        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, output_dim))
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average cost per data sample
        """
        #TODO:
        z = np.dot(X,self.theta) + self.bias # z = w T x + b
        exp_z = np.exp(z) # e^z
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True) # this is h(z)

        for i in range(0, len(X)):
            if y[i] == 0:
                one_hot_y = np.array([1, 0])
            else:
                one_hot_y = np.array([0, 1])

            cost_for_sample = - np.sum(one_hot_y * np.log(softmax_scores[i]))

        cost = np.average(cost_for_sample)
        return cost

    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------

    def gradientDecent(self, X, y):
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        alpha = 0.1
        sub = np.subtract(softmax_scores, y)
        mul = np.multiply(sub, x)
        CFD = np.sum(mul)*(1/len(y))
        new_theta = np.subtract(theta, CFD)
        return new_theta

    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.

        1) For number_of_iterations (i.e. the number of times you want to update your weights/biases):

            for i in range(0, 40):

          i) Do forward propagation (which simply means compute the softmax scores for X)

                z = np.dot(X,self.theta) + self.bias
                exp_z = np.exp(z)
                softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

         ii) Do backward propagation (which means compute the gradient of the cost w.r.t. your weights/biases and update them)

                   - For multinomial logistic regression, gradient w.r.t the weight matrix theta is:
                   dot product of input X with the difference between your predictions (softmax_scores) and the ground truth (one_hot_y).

                  difference = np.subtract(softmox_scores, one_hot_y)
                  grad_weight = np.dot(X, np.absolute(difference))

                   - Similarly the gradient w.r.t. the biases is:
                   dot product of a column of ones (same length as X) with the difference between your predictions (softmax_scores) and the ground truth (one_hot_y)

                   col_ones = np.ones(len(X), 1)
                   grad_bias = np.dot(col_ones, np.absolute(difference))
                   
                   # diff = np.subtract(softmax_scores, y)

        iii) Update model parameters: After you've computed the gradients, update your model parameters using the gradient descent equations:
                               -> w = w - learning_rate * gradient of cost w.r.t weights
                                  self.theta = self.theta - (0.1 * grad_weight)
                               -> b = b - learning_rate * gradient of cost w.r.t. biases
                                  self.bias = self.bias - (0.1 * grad_bias)
        """  
        #TODO:
        for i in range(0, 40):
             z = np.dot(X,self.theta) + self.bias
             exp_z = np.exp(z)
             softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

             difference = np.subtract(softmax_scores, one_hot_y)
             grad_weight = np.dot(X, np.absolute(difference))

             col_ones = np.ones(len(X), 1)
             grad_bias = np.dot(col_ones, np.absolute(difference))

             self.theta = self.theta - (0.1 * grad_weight)
             self.bias = self.bias - (0.1 * grad_bias)


        return self.theta, self.bias

#--------------------------------------------------------------------------

#--------------------------------------------------------------------------

def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """
    
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()


################################################################################    
# # Linear 
# X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',') #https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
# y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')

# test = LogisticRegression(2,2)

# plot_decision_boundary(test,X,y)
# c = test.compute_cost(X,y)
# print c

# fit = test.backprop(X,y)
# c = test.compute_cost(X,y)
# plot_decision_boundary(test,X,y)
# print c

# # Non-Linear
X = np.genfromtxt('DATA/NonLinear/X.csv', delimiter=',') #https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
y = np.genfromtxt('DATA/NonLinear/y.csv', delimiter=',')

test = LogisticRegression(2,2)

plot_decision_boundary(test,X,y)
c = test.compute_cost(X,y)

fit = test.fit(X,y)
c = test.compute_cost(X,y)
plot_decision_boundary(test,X,y)
print c


# # # Digit Classification
# X = np.genfromtxt('DATA/Digits/X_train.csv', delimiter=',') #https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
# y = np.genfromtxt('DATA/Digits/y_train.csv', delimiter=',')

# test = NeuralNet(64,30,10)

# # c = test.compute_cost(X,y)
# # print c

# fit = test.backprop(X, y)
# # p = test.predict(X)
# # c = test.compute_cost(X,y)
# # print p
# # print c

# X_test = np.genfromtxt('DATA/Digits/X_test.csv', delimiter=',') #https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
# y_test = np.genfromtxt('DATA/Digits/y_test.csv', delimiter=',')

# #fitTest = test.backprop(X_test, y_test)
# p = test.predict(X_test)

# for each in p:
#      print each
    
# for each in p:
#   if each == 0:
#       lr_0 = 
            
    
