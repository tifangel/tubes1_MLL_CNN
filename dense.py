import random
import numpy as np
import activation as actv
from Layer import Layer

class Dense:
    def __init__(self, units, activation = None):
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f'Invalid value for Units, expected a positive integer. Received: units={units}')
        self.activation = actv.get(activation)

        # invoking the __init__ of the parent class 
        # Layer.__init__(self, idlayer)

    def build(self, input_size, weight_range = None):
        if weight_range is not None:
            x, y = weight_range
            if x > y:
                raise ValueError(f'Invalid range for weight, expected a < b in (a,b). Received: range (a,b)={weight_range}')
        else:
            x = 0 
            y = 1
        self.weight = [[random.uniform(x,y) for i in range(self.units)] for j in range(input_size)]
        
    def compute_dot(self, input_matriks):
        result = np.dot(input_matriks, self.weight)
        return result

    def compute_output(self, dot):
        row = []
        result = []
        for i in dot:
            result.append(self.activation(i))
        return result

# TEST
# dense = Dense(2, 'sigmoid')
# dense.build(3, (-3,3))
# print(dense.weight)
# dot = dense.compute_dot([[3,2,1],[1,2,3]])
# print('DOT : ', dot)
# output = dense.compute_output(dot)
# print('OUTPUT : ', output)
