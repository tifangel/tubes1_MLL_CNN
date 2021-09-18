import math

class Pooling:
    def __init__(self, filter_size=(2,2), stride_size=2, mode='max'):
        self.input = []
        self.filter_size = filter_size
        self.stride_size= stride_size
        self.mode = mode

    def apply():
        '''
        TO DO: 
            - mecah input berdasarkan filter & stride
            - operasi melakukan pooling berdasarkan mode
        '''
        return []

    def infoLayer(self):
        print(self.filter_size[0], self.filter_size[1])
        print(self.stride_size)
        print(self.mode)
    
pooling = Pooling((3,4), 3, 'avg')
pooling.infoLayer()