import numpy as np
from numpy.lib.stride_tricks import as_strided
from Detector import Detector

class Pooling:
    def __init__(self):
        self.matrix = []
        self.kernel_size = 0
        self.stride= 0
        self.padding = 0
        self.pool_mode = 'max'

    def apply(self, matrix, kernel_size=2, stride=2, padding=0, pool_mode='max'):
        self.matrix = matrix
        self.kernel_size = kernel_size
        self.stride= stride
        self.padding = padding
        self.pool_mode = pool_mode

        self.matrix = np.pad(self.matrix, self.padding, mode='constant')

        output_shape = (
            (self.matrix.shape[0] - self.kernel_size)//self.stride + 1,
            (self.matrix.shape[1] - self.kernel_size)//self.stride + 1
        )

        self.kernel_size = (self.kernel_size, self.kernel_size)

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

    def infoLayer(self):
        print(self.filter_size[0], self.filter_size[1])
        print(self.stride_size)
        print(self.mode)
    
matrix = np.array([
    [1, 1, 2, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 0],
    [1, 2, -3, 4]
])

detector = Detector()
output_detector = detector.activate(matrix, '')
print("output detector:", output_detector)
pooling = Pooling()
output_pooling = pooling.apply(output_detector, kernel_size=2, stride=2, padding=0, pool_mode='max')
print("output_pooling:", output_pooling)
