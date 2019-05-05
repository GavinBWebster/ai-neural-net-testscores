# 2 inputs, 3 hidden + 1 output layer

import numpy as np

# X=Hrs sleep,Hrs study y=Score on test
X = np.array(([3,5],[5,1],[10,2]), dtype=float)
y = np.array(([75],[82],[93]), dtype=float)

# Scaling to 0-1
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100

class Neural_Network(object):
	def __init__(self):
		#Define HyperParameters
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		#Weights do the learning (Parameters)
		self.W1 = np.random.randn(self.inputLayerSize, \
								  self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, \
								  self.outputLayerSize)

	def forward(self, X):
		#Propogate inputs through network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		#Apply sigmoid activation function to scalar, vector, or
		return 1/(1+np.exp(-z))

NN = Neural_Network()
yHat = NN.forward(X)
print(yHat)
print(y)