import numpy as np
from numpy.lib.stride_tricks import as_strided
from Layer import Layer

class Pooling(Layer):
    def __init__(self, idlayer, stride, padding, pool_mode, kernel_size=(2,2)):
        self.kernel_size = kernel_size
        self.stride= stride
        self.padding = padding
        self.pool_mode = pool_mode

        # invoking the __init__ of the parent class 
        Layer.__init__(self, idlayer)

    def apply(self, input_matrix):
        self.input = np.pad(input_matrix, self.padding, mode='constant')

        output_shape = (
            (self.input.shape[0] - self.kernel_size[0])//self.stride + 1,
            (self.input.shape[1] - self.kernel_size[1])//self.stride + 1
        )

        matrix_w = as_strided(
            self.input,
            shape = output_shape + self.kernel_size,
            strides = (self.stride*self.input.strides[0], self.stride*self.input.strides[1]) + self.input.strides
        )
        matrix_w = matrix_w.reshape(-1, *self.kernel_size)

        if self.pool_mode == 'max':
            return matrix_w.max(axis=(1,2)).reshape(output_shape)
        elif self.pool_mode == 'avg':
            return matrix_w.mean(axis=(1,2)).reshape(output_shape)
        else:
            raise ValueError(pool_mode, 'is not valid!')
