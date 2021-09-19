import math
import numpy as np

# Input berupa matriks setelah convolution 
# Output berupa matriks yang telah diaktivasi
class Detector:
    def __init__(self):
        self.input = [] 
        self.activation_type = 'sigmoid'

    def activate(self, rearrange_output, activation_type='sigmoid'):
        self.input = rearrange_output
        result = []
        for rearrange in rearrange_output:
            if activation_type == 'sigmoid':
                result.append(list(self.sigmoid(rearrange)))
            elif activation_type == 'relu':
                result.append(list(self.relu(rearrange)))
            elif activation_type == 'softmax':
                result.append(list(self.softmax(rearrange)))
            else:
                result.append(list(rearrange))
        return result
    
    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def relu(self, X):
        return np.maximum(0,X)
    
    def softmax(self, X):
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X))
        return expo/expo_sum
