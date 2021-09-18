import numpy as np

class Pooling:
    def __init__(self, filter_size=(2,2), stride_size=2, mode='max'):
        self.input = []
        self.filter_size = filter_size
        self.stride_size= stride_size
        self.mode = mode

    def apply(self, matrix):
        '''
        TO DO: 
            - nambahin stride
        '''
        mat = np.array(matrix)
        x, y = mat.shape
        feat_x = x // self.filter_size[0]
        feat_y = y // self.filter_size[1]

        if self.mode == 'max': 
            return(
                mat[:feat_x*self.filter_size[0], :feat_y*self.filter_size[1]].reshape(
                    feat_x, self.filter_size[0], feat_y, self.filter_size[1]
                    ).max(axis=(1, 3)))
        elif self.mode == 'avg':
            return(
                mat[:feat_x*self.filter_size[0], :feat_y*self.filter_size[1]].reshape(
                    feat_x, self.filter_size[0], feat_y, self.filter_size[1]
                    ).mean(axis=(1, 3)))
        else:
            raise ValueError("Mode pooling harus 'max' atau 'avg'")

    def infoLayer(self):
        print(self.filter_size[0], self.filter_size[1])
        print(self.stride_size)
        print(self.mode)
    
pooling = Pooling((2,2), 2, 'max')
output = pooling.apply([
    [1, 1, 2, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 0],
    [1, 2, 3, 4]
])
print(output)