import math

# Input berupa matriks setelah convolution 
# Output berupa matriks yang telah diaktivasi
class Detector:
    def __init__(self):
        self.input = [] 
        self.activation_type = 'sigmoid'

    def activate(self, rearrange_output, activation_type='sigmoid'):
        result = []
        for rearrange in rearrange_output:
            if activation_type == 'sigmoid':
                result.append(self.sigmoid(rearrange))
            elif activation_type == 'relu':
                result.append(self.relu(rearrange))
            elif activation_type == 'softmax':
                result.append(self.softmax (rearrange))
            else:
                result.append(rearrange)
        return result
    
    def sigmoid(self, arr):
        res = []
        for x in arr:
            x = round((1/(1 + math.exp(0-x))),3)
            
            res.append(x)
        return res

    def relu(self, arr):
        res = []
        for x in arr:
            if (x >= 0) :
                res.append(max(0, x))
            else :
                res.append(x * 0.0001)
        return res
    
    def softmax(self, arr):
        
        e = []
        p = []
        for i in arr:
            e.append(math.exp(i))
        c = sum(e)
        for j in e:
            p.append(j / c)
        return p

# Contoh
matriks_input = [
    [
        [21, 9, 7, 21, 27],
        [-24, 5, 25, -29, 18],
        [22, 1, 6, 25, 11],
        [29, -24, 16, 10, 10],
        [23, 26, 4, 11, -18]
    ],
    [
        [7, 30, 13, 8, 17],
        [13, -20, 8, 25, 24],
        [7, 22, 29, 25, 25],
        [18, -29, 27, 18, 12],
        [20, 7, 8, 11, 13]
    ]
]

matriks_output = []
for matrik in matriks_input:
    detector_layer = Detector()
    matriks_output.append(detector_layer.activate(matrik, 'sigmoid'))

for matrix in matriks_output:
    print(matrix)
