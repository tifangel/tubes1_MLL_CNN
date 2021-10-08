import numpy as np
from Layer import Layer

class Detector:
    def __init__(self, idlayer, activation_type):
        # self.input = np.array(matrix)
        self.activation_type = activation_type

        # invoking the __init__ of the parent class 
        Layer.__init__(self, idlayer)

    def forward(self, inputMatrix):
        self.input = inputMatrix
        result = []
        for rearrange in self.input:
            if self.activation_type == 0:
                result.append(list(sigmoid(rearrange)))
            elif self.activation_type == 1:
                result.append(list(relu(rearrange)))
            elif self.activation_type == 2:
                result.append(list(softmax(rearrange)))
            else:
                result.append(list(x))
        self.output = result
        return result

    def backward(self, inputmatrix):
        result = []
        for rearrange in inputmatrix:
            if self.activation_type == 0:
                result.append(list(dSigmoid(rearrange)))
            elif self.activation_type == 1:
                result.append(list(dRelu(rearrange)))
            else:
                result.append(list(rearrange))
        return result
    
def sigmoid(X):
    return 1/(1+np.exp(-X))

def dSigmoid(X):
    return sigmoid(X)*(1-sigmoid(X))

def relu(X):
    return np.maximum(0,X)

def dRelu(X):
    return np.where(X < 0, 0, 1)

def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))
