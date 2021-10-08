import numpy as np

class Detector:
    def __init__(self, matrix, activation_type):
        self.input = np.array(matrix)
        self.activation_type = activation_type

    def forward(self):
        result = []
        for rearrange in self.input:
            if self.activation_type == 'sigmoid':
                result.append(list(sigmoid(rearrange)))
            elif self.activation_type == 'relu':
                result.append(list(relu(rearrange)))
            elif self.activation_type == 'softmax':
                result.append(list(softmax(rearrange)))
            else:
                result.append(list(rearrange))
        return result

    def backward(self):
        result = []
        for rearrange in self.input:
            if self.activation_type == 'sigmoid':
                result.append(list(dSigmoid(rearrange)))
            elif self.activation_type == 'relu':
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
