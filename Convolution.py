import numpy as np

class Convolution:
    def __init__(self, input, kernels, size_padding, size_stride, isSharing):
        self.input = input.tolist()
        self.kernels = kernels.tolist()
        self.size_padding = size_padding
        self.size_stride = size_stride
        self.isSharing = isSharing
        self.d_input = len(input)
        self.w_input = len(input[0])
        self.h_input = len(input[0][0])
        self.n_filter = len(kernels)
        self.f_filter = len(kernels[0])
        if (isSharing == 0):
            self.f_filter = len(kernels[0][0])  
        self.feature_map = []

    def doConvolution(self):
        # Add padding to input
        new_w = self.w_input + 2 * self.size_padding
        new_h = self.h_input + 2 * self.size_padding
        add_px = [0 for l in range(self.size_padding)]
        add_py = [0 for l in range(new_w)]
        if (self.size_padding > 0):
            for i in range(self.d_input):
                for j in range(self.w_input):
                    self.input[i][j] = add_px + self.input[i][j] + add_px
                for k in range(self.size_padding):
                    self.input[i].insert(0, add_py)
                    self.input[i].append(add_py)

        # Initialize feature map
        vw = (self.w_input - self.f_filter + 2 * self.size_padding) // self.size_stride + 1
        vh = (self.h_input - self.f_filter + 2 * self.size_padding) // self.size_stride + 1
        self.feature_map = [[[0 for i in range(vh)] for j in range(vw)] for m in range(self.n_filter)]

        # self.print3D(self.input)
        # print ("new:", new_w, new_h, vw, vh)
        # Calculate feature map value 
        for m in range(self.n_filter):
            for n in range(self.d_input):
                input_matrix = self.input[n]
                if self.isSharing == 0:
                    filter_matrix = self.kernels[m][n]
                else:
                    filter_matrix = self.kernels[m]
                for i in range(0, new_w, self.size_stride):
                    if i + self.f_filter >= new_w + 1:
                        continue
                    for j in range(0, new_h, self.size_stride):
                        if j + self.f_filter >= new_h + 1:
                            continue
                        s = 0
                        for p in range(self.f_filter):
                            for q in range(self.f_filter):
                                # print(i,j,p,q)
                                s += input_matrix[i+p][j+q] * filter_matrix[p][q]
                        self.feature_map[m][i//self.size_stride][j//self.size_stride] += s
        return self.feature_map

    # def print3D(self, m):
    #     for i in range (len(m)):
    #         for j in range(len(m[i])):
    #             print(m[i][j])
    #         print("\n")

    # def printInfo(self):
    #     print("CONVOLUTION")
    #     print("input", self.input)
    #     print("kernels", self.kernels)
    #     print("size_padding", self.size_padding)
    #     print("size_stride", self.size_stride)
    #     print("isSharing", self.isSharing)
    #     print("d_input", self.d_input)
    #     print("w_input", self.w_input)
    #     print("n_filter", self.n_filter)
    #     print("f_filter", self.f_filter)
    #     print("feature_map", self.feature_map)

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

# input = [[
#     [1, 1, 2, 4],
#     [5, 6, 7, 8],
#     [3, 2, 1, 0],
#     [1, 2, -3, 4]
# ]]
# kernel = [
#     [
#         [0, -1],
#         [1, 0]
#     ]
# ]
# c = Convolution(input, kernel, 0, 2, 1)
# c.doConvolution()
# c.print3D(c.getFeatureMap())