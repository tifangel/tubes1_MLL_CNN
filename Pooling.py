import numpy as np
from numpy.lib.stride_tricks import as_strided
from Layer import Layer

class Pooling:
    def __init__(self, matrix, kernel_size, stride=2, padding=0, pool_mode='max'):
        self.input = matrix
        self.kernel_size = kernel_size
        self.stride= stride
        self.padding = padding
        self.pool_mode = pool_mode

        self.output_shape = (0, 0)
        self.input_reshape = []

        self.forward_output = []
        self.backward_output = []

    def forward(self):
        self.input = np.pad(self.input, self.padding, mode='constant')

        output_shape = (
            (self.input.shape[0] - self.kernel_size[0] + 2*self.padding)//self.stride + 1,
            (self.input.shape[1] - self.kernel_size[1] + 2*self.padding)//self.stride + 1
        )
        self.output_shape = output_shape

        matrix_w = as_strided(
            self.input,
            shape = output_shape + self.kernel_size,
            strides = (self.stride*self.input.strides[0], self.stride*self.input.strides[1]) + self.input.strides
        )
        matrix_w = matrix_w.reshape(-1, *self.kernel_size)

        self.input_reshape = matrix_w

        if self.pool_mode == 'max':
            self.forward_output = matrix_w.max(axis=(1,2)).reshape(output_shape)
        elif self.pool_mode == 'avg':
            self.forward_output = matrix_w.mean(axis=(1,2)).reshape(output_shape)
        return self.forward_output

    def backward(self):
        reshape_forward_output = self.forward_output.flatten()
        mask = []
        for i in range (0, len(self.input_reshape)):
            mask.append(self.input_reshape[i] == reshape_forward_output[i])
        mask = np.array(mask)
        
        output = []
        if self.pool_mode == 'max':
            for i in range (len(mask)):
                temp_output = []
                for j in range (len(mask[i])):
                    temp_row = []
                    for k in range (len(mask[i][j])):
                        temp_row.append(self.input_reshape[i][j][k]) if mask[i][j][k] else temp_row.append(-999)
                    temp_output.append(temp_row)
                output.append(temp_output)
        elif self.pool_mode == 'avg':
            for i in range (len(self.input_reshape)):
                temp_output = []
                for j in range (len(self.input_reshape[i])):
                    temp_row = []
                    for k in range (len(self.input_reshape[i][j])):
                        temp_row.append(reshape_forward_output[i] / len(self.input_reshape))
                    temp_output.append(temp_row)
                output.append(temp_output)
        self.backward_output = output
        return self.backward_output

matrix = [
    [1, 1, 2, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 0],
    [1, 2, -3, 4]
]

matrices =[
    [
        [-85, 76, 64],
        [109, -1, 10],
        [118, 71, 67]
    ],
    [
        [-144, -291, 66],
        [-347, 102, -192],
        [-239, -52, -162]
    ]
]

# FORWARD
for matrix in matrices:
    detector = Detector(matrix, 'relu')
    output_detector = detector.forward()
    print(detector.input)
    print("FORWARD DETECTOR :", np.array(output_detector))
    # pooling = Pooling(output_detector, kernel_size=(2,2), stride=2, padding=0, pool_mode='max')
    pooling = Pooling(output_detector, kernel_size=(3,3), stride=1, padding=0, pool_mode='max')
    forward_pooling = pooling.forward()
    print("FORWARD POOL :", np.array(forward_pooling))

    # BACKWARD
    backward_pooling = pooling.backward()
    backward_pooling = np.array(backward_pooling).reshape(pooling.input.shape)
    print("BACKWARD POOL:", backward_pooling)
    dDetector = Detector(backward_pooling, 'relu')
    print("BACKWARD POOL ACTIVATION:", np.array(dDetector.backward()))
    print("BACKWARD DETECTOR:", np.array(detector.backward()))
    print()
