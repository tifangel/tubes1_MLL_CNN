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
        self.output = result
        return result

    def predict(self, input_matriks):
        dot = self.compute_dot(input_matriks)
        output = self.compute_output(dot)
        return output

    def train(self, inputs, target, epoch, l_rate):
        for iteration in range(epoch):
            # Pass the training set through the network.
            output = self.predict(inputs)
            # Calculate the error
            error = []
            for batch_idx, batch in enumerate(output):
                batch_error = []
                for value_idx, x in enumerate(batch):
                    batch_error.append(((target[batch_idx][value_idx] - x)**2)/2)
                error.append(sum(batch_error))
                        
            # Update weight
            for i, batch in enumerate(self.output):
                for j, x in enumerate(batch):
                    t = target[i][j]
                    y = self.output[i][j]
                    delta = self.sigmoid_derivative(y) * (t - y) * -1 * y
                    new_weight = []
                    y = self.weight
                    for weights in y:
                        x = weights
                        w = x[j] - delta*l_rate
                        x[j] = w
                        new_weight.append(x)
                    self.weight = new_weight
            return error

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    

#TEST
# inputs = [[3,2,1]]
# target = [[0,1]]
# dense = Dense(2, 'sigmoid')
# dense.build(3, (-3,3))
# error = dense.train(inputs, target, 30, 0.5)
# output = dense.predict([[3,2,1]])
# print(output)
#dense.train([[1, 0], [0, 1]], 10)