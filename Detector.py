import numpy as np
from Layer import Layer

class Detector(Layer):
    def __init__(self, idlayer, activfunc):
        self.activation_type = activfunc

        # invoking the __init__ of the parent class 
        Layer.__init__(self, idlayer)

    def activate(self):
        result = []
        for x in self.input:
            # Sigmoid
            if self.activation_type == 0:
                result.append(list(self.sigmoid(x)))
            # Relu
            elif self.activation_type == 1: 
                result.append(list(self.relu(x)))
            # Softmax
            elif self.activation_type == 2:
                result.append(list(self.softmax(x)))
            # Linear
            else:
                result.append(list(x))
        self.output = result
        return result
    
    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def relu(self, X):
        return np.maximum(0,X)
    
    def softmax(self, X):
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X))
        return expo/expo_sum
