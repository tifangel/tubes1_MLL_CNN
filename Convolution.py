class Convolution:
    def __init__(self, size_input, input, n_filter, size_filter, kernels, size_padding, size_stride, isSharing):
        self.input = input
        self.kernels = kernels
        self.size_padding = size_padding
        self.size_stride = size_stride
        self.isSharing = isSharing
        self.size_input = size_input
        self.n_filter = n_filter
        self.size_filter = size_filter
        self.feature_map = []

    def getFeatureMap(self):
        return self.feature_map

    def doConvolution(self):
        # Add padding to input
        new_w = self.size_input[0] + 2 * self.size_padding
        add_px = [0 for l in range(self.size_padding)]
        add_py = [0 for l in range(new_w)]
        if (self.size_padding > 0):
            for i in range(self.size_input[2]):
                for j in range(self.size_input[0]):
                    self.input[i][j] = add_px + self.input[i][j] + add_px
                self.input[i].insert(0, add_py)
                self.input[i].append(add_py)

        # Initialize feature map
        v = (self.size_input[0] - self.size_filter[0] + 2 * self.size_padding) // self.size_stride + 1
        self.feature_map = [[[0 for i in range(v)] for j in range(v)] for m in range(self.n_filter)]

        # Convolution stage
        for m in range(self.n_filter):
            for n in range(self.size_input[2]):
                input_matrix = self.input[n]
                if self.isSharing == 0:
                    filter_matrix = self.kernels[m][n]
                else:
                    filter_matrix = self.kernels[m]
                for i in range(0, new_w, self.size_stride):
                    if i + self.size_filter[0] >= new_w + 1:
                        continue
                    for j in range(0, new_w, self.size_stride):
                        if j + self.size_filter[0] >= new_w + 1:
                            continue
                        s = 0
                        for p in range(self.size_filter[0]):
                            for q in range(self.size_filter[0]):
                                s += input_matrix[i+p][j+q] * filter_matrix[p][q]
                        self.feature_map[m][i//self.size_stride][j//self.size_stride] += s
    
    # def print3D(self, m):
    #     for i in range (len(m)):
    #         for j in range(len(m[i])):
    #             print(m[i][j])
    #         print("\n")

# input = [
#     [
#         [16, 24, 32],
#         [47, 18, 26],
#         [68, 12, 9]
#     ],
#     [
#         [26, 57, 43],
#         [24, 21, 12],
#         [2, 11, 19]
#     ],
#     [
#         [18, 47, 21],
#         [4, 6, 12],
#         [81, 22, 13]
#     ]
# ]
# kernel = [
#     [
#         [
#             [0, -1],
#             [1, 0]
#         ],
#         [
#             [5, 4],
#             [3, 2]
#         ],
#         [
#             [16, 24],
#             [68, -2]
#         ]
#     ],
#     [
#         [
#             [60, 22],
#             [32, 18]
#         ],
#         [
#             [35, 46],
#             [7, 23]
#         ],
#         [
#             [78, 81],
#             [20, 42]
#         ]
#     ]
# ]

# c = Convolution([3,3,3], input, 2, [2,2,3], kernel, 0, 1, 0)
# c.doConvolution()
# c.print3D(c.getFeatureMap())