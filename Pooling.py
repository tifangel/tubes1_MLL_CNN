import numpy as np
from numpy.lib.stride_tricks import as_strided

class Pooling:
    def __init__(self):
        self.matrix = []
        self.kernel_size = (0,0)
        self.stride= 0
        self.padding = 0
        self.pool_mode = 'max'

    def apply(self, input_matrix, kernel_size=(2,2), stride=2, padding=0, pool_mode='max'):
        self.kernel_size = kernel_size
        self.stride= stride
        self.padding = padding
        self.pool_mode = pool_mode
        self.matrix = np.pad(input_matrix, self.padding, mode='constant')

        output_shape = (
            (self.matrix.shape[0] - self.kernel_size[0])//self.stride + 1,
            (self.matrix.shape[1] - self.kernel_size[1])//self.stride + 1
        )

        matrix_w = as_strided(
            self.matrix,
            shape = output_shape + self.kernel_size,
            strides = (self.stride*self.matrix.strides[0], self.stride*self.matrix.strides[1]) + self.matrix.strides
        )
        matrix_w = matrix_w.reshape(-1, *self.kernel_size)

        if self.pool_mode == 'max':
            return matrix_w.max(axis=(1,2)).reshape(output_shape)
        elif self.pool_mode == 'avg':
            return matrix_w.mean(axis=(1,2)).reshape(output_shape)
        else:
            raise ValueError(pool_mode, 'is not valid!')
