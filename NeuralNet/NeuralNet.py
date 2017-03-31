import numpy as np 
import matplotlib.pyplot as plt 
import math


class NeuralNet:
	"""
	This class implements a Neural Network Classifier
	"""

	def __init__(self, input_dim, hidden_dim, output_dim):
		"""
		Initializes the parameters of the neural network classifer to
        random values.

        args:
            input_dim: Number of dimensions of the input data
            hidden_dim: Number of hidden layer nodes
            output_dim: Number of classes

		"""
		# save hidden_dim & output_dim for ease
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		# input -> hidden layer
		self.theta_hidden = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
		self.bias_hidden = np.zeros((1, hidden_dim))

		# hidden layer -> output
		self.theta_output = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
		self.bias_output = np.zeros((1, output_dim))

	def zed(self, X, theta, bias):
		"""
		Computes the dot product of activation value and weights with the bias added
		"""
		zed = np.dot(X,theta) + bias # z = X * W + 2
		return zed

	def sigmoid(self, z):
		"""
		Computes the sigmoid function of the input

		input is the activation value of the input layer
		"""
		sigmoid_scores = 1.0/(1.0+np.exp(-z))
		return sigmoid_scores

	def softmax(self, z):
		"""
		Computes the softmax function of the input

		input is the activation value of the hidden layer
		"""
		exp_y = np.exp(z)
		softmax_scores = exp_y/np.sum(exp_y, axis = 1, keepdims = True)
		return softmax_scores

	def sigmoid_deriv(self, z):
		"""Derivative of the sigmoid function."""
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def one_hot_y(self, y, dim): 
		"""
		Computes the y label matrix

		args: 
			y: Labels corresponding to input data
			output_dim: the number of column in the matrix that we make

		returns: 
			hot_array: size -> (len(y), output_dim) of ones for element y[i]

		"""
		index = 0
		hot_array = np.zeros((len(y), dim))
		for value in y:
			array = [0] * dim
			array[int(value)] = 1
			hot_array[index] = np.array(array)
			index += 1
		return hot_array

	def predict(self, X):
		"""
		Makes a prediction based on current model parameters.

		args:
			X: Data array

		returns:
			predictions: array of predicted labels
		""" 
		z_one = self.zed(X, self.theta_hidden, self.bias_hidden)
		sigmoid_scores = self.sigmoid(z_one)
		z_two = self.zed(sigmoid_scores, self.theta_output, self.bias_output)
		softmax_scores = self.softmax(z_two)
		predictions = np.argmax(softmax_scores, axis = 1)
		return predictions

	def compute_cost(self, X, y):
		"""
		Computes the total cost on the dataset.

		args:
			X: Data array
			y: Labels corresponding to input data

		returns:
			cost: average cost per data sample
		"""
		# 1. compute sigmoid_scores
		z_hidden = self.zed(X, self.theta_hidden, self.bias_hidden)
		sigmoid_scores = self.sigmoid(z_hidden)
		
		# 2. compute softmax_scores(sigmoid_scores)
		z_outer = self.zed(sigmoid_scores, self.theta_output, self.bias_output)
		softmax_scores = self.softmax(z_outer)

		# 3. compute one_hot_y *but neural network version* *PLEASE ASK!*
		hot_y = self.one_hot_y(y, self.output_dim)

		# 4. multiply one_hot_y with the log of softmax_scores
		mul = np.multiply(hot_y, np.log(softmax_scores))

		# 5. cost_per_samplek = np.sum(step4, axis = 1, keepdims = True)
		cost_per_samplek = -np.sum(mul, axis = 1, keepdims = True)

		# 7. cost = np.average(cost_per_samplem)
		cost = np.average(cost_per_samplek)

		return cost

	def backprop(self, X, y):
		"""
		Builds a model that finds the best weights and biases to make better predictions

		args:
			X: Data Array
			y: Labels corresponding to input data
			output_dim: mainly used just to pass to the one_hot_y function        
		"""
		for i in range(0,1000):
			# Forward propagation
			z1 = self.zed(X, self.theta_hidden, self.bias_hidden)
			a1 = self.sigmoid(z1)
			z2 = self.zed(a1, self.theta_output, self.bias_output)
			a2 = self.softmax(z2)
			hot_array = self.one_hot_y(y, self.output_dim)

			# Backpropagation
			delta3 = np.subtract(a2, hot_array)
			a = self.sigmoid(z1)*(1-self.sigmoid(z1))
			delta2 = delta3.dot(self.theta_output.T) * a
			dW2 = np.dot(np.transpose(a1),delta3)
			db2 = np.sum(delta3, axis = 0, keepdims = True)
			dW1 = np.dot(np.transpose(X), delta2)
			db1 = np.sum(delta2, axis = 0)

			#update parameters
			self.theta_hidden = np.subtract(self.theta_hidden, (0.001 * dW1))
			self.theta_output = np.subtract(self.theta_output, (0.001 * dW2))

			self.bias_hidden = np.subtract(self.bias_hidden, np.multiply( 0.001, db1))
			self.bias_output = np.subtract(self.bias_output, np.multiply( 0.001, db2))

			# cost = self.compute_cost(X, y)
			# print cost




###############################################################################
def plot_decision_boundary(model, X, y):
	"""
	Function to print the decision boundary given by model.

	args:
	    model: model, whose parameters are used to plot the decision boundary. has theta and bais
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

# # Linear 
# X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',') #https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
# y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')

# test = NeuralNet(2,3,2)

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

test = NeuralNet(2,30,2)

plot_decision_boundary(test,X,y)
c = test.compute_cost(X,y)

fit = test.backprop(X,y)
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
# 	if each == 0:
# 		lr_0 = 
