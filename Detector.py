import numpy as np

class Detector:
    def __init__(self):
        self.input = [] 
        self.activation_type = ''

    def activate(self, input_matrix, activation_type='sigmoid'):
        self.input = input_matrix
        result = []
        for x in self.input:
            if activation_type == 'sigmoid':
                result.append(list(self.sigmoid(x)))
            elif activation_type == 'relu':
                result.append(list(self.relu(x)))
            elif activation_type == 'softmax':
                result.append(list(self.softmax(x)))
            else:
                result.append(list(x))
        return result
    
    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def relu(self, X):
        return np.maximum(0,X)
    
    def softmax(self, X):
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X))
        return expo/expo_sum
