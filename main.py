from mlxtend.data import loadlocal_mnist
import random
import re
import math
import csv
import sys
import numpy as np
from Layer import Layer
from Convolution import Convolution
from Pooling import Pooling
from Detector import Detector
from dense import Dense
from BackConv import BackConv

class CNNClassifier:
    def __init__(self):
        self.size_input = [] ## Ukuran Input
        self.size_padding = 0 ## Ukuran Padding
        self.n_filter = 0 ## Jumlah Filter
        self.size_filter = [] ## Ukuran Filter
        self.size_stride = 0 ## Ukuran Stride
        self.n_layer_konvolusi = 0 ## Jumlah Konvolusi Layer
        self.n_layer = 0
        self.isSharing = 0 ## 1 = sharing, 0 = not sharing
        self.layers = []
        self.kernels = []
        self.input = []
        self.target = []
        self.epoch = 0
        self.learning_rate = 0
        self.momentum = 0
    
    def setTypeCNN(self, isSharing):
        self.isSharing = isSharing

    def getLayer(self,idx):
        return self.layers[idx]

    def getInput(self):
        return self.input

    def getKernels(self):
        return self.kernels
    
    # Fungsi buat convert
    def convertInt(self, arr): 
        numlist = [0 for i in arr]
        for i in range(len(arr)):
            numlist[i] = int(arr[i])
        return numlist

    # Fungsi buat convert
    def convertFloat(self, arr):
        numlist = [0 for i in arr]
        for i in range(len(arr)):
            numlist[i] = float(arr[i])
        return numlist
    
    def loadInputMNist(self, input_path, target_path):
        # Pilih Dataset untuk input
        X, y = loadlocal_mnist(
            images_path=input_path, 
            labels_path=target_path)
        self.input = X
        self.target = y

    def load(self,filename):
        f = open(filename, "r")
        numbers = re.split('\n',f.read())
        # Ukuran Input
        self.size_input = self.convertInt(re.split(' ',numbers[0]))
        # Ukuran Padding
        self.size_padding = int(numbers[1])
        # Jumlah Filter
        self.n_filter = int(numbers[2])
        # Ukuran filter
        self.size_filter = self.convertInt(re.split(' ',numbers[3]))
        # Ukuran Stride
        self.size_stride = int(numbers[4])
        # Jumlah Konvolusi Layer
        self.n_layer_konvolusi = int(numbers[5])
        # Jumlah layer
        self.n_layer = self.n_layer_konvolusi + 2
        # CNN Sharing Parameter or not
        self.isSharing = int(numbers[6])

        print('Ukuran Input', self.size_input)
        print('Ukuran Padding', self.size_padding)
        print('Jumlah Filter', self.n_filter)
        print('Ukuran Filter', self.size_filter)
        print('Ukuran Stride', self.size_stride)
        print('Jumlah konvolusi layer', self.n_layer_konvolusi)
        print('CNN Sharing', self.isSharing)
        
        # Inisialisasi Kernel / Filter / Bobot secara random
        kernels = []
        if (self.isSharing == 1):
            for i in range(self.n_filter):
                kernel = []
                for j in range(self.size_filter[0]):
                    row = []
                    for k in range(self.size_filter[1]):
                        row.append(random.randint(1,30))
                    kernel.append(row)
                kernels.append(kernel)
        elif (self.isSharing == 0):
            for i in range(self.n_filter):
                kernel = []
                for j in range(self.size_filter[2]):
                    matriks = []
                    for k in range(self.size_filter[0]):
                        row = []
                        for l in range(self.size_filter[1]):
                            row.append(random.randint(1,30))
                        matriks.append(row)
                    kernel.append(matriks)
                kernels.append(kernel)
        self.kernels = kernels

        # Inisialisasi Layer
        layers = []
        # Konvolusi Layer
        for i in range(self.n_layer_konvolusi):
            newLayer = Convolution(i, self.kernels, self.size_padding, self.size_stride, self.isSharing)
            newLayer.printKernels()
            layers.append(newLayer)
        # Detector Layer
        detectorLayer = Detector(self.n_layer_konvolusi, int(numbers[7]))
        layers.append(detectorLayer)
        # Pooling Layer
        poolingLayer = Pooling(self.n_layer_konvolusi + 1, self.size_stride, self.size_padding, 'max', (3,3))
        layers.append(poolingLayer)
        # Dense Layer
        denseLayer = Dense(1, 'RELU')
        layers.append(denseLayer)
        # Set Layer to CNN
        self.layers = layers

        self.epoch = int(numbers[9])
        self.learning_rate = float(numbers[10])
        self.momentum = int(numbers[11])

    def saveModel(self):
        print("Save")
        array = []
        array.append(self.size_input)
        array.append(self.n_layer)
        array.append(self.n_layer_konvolusi)
        array.append(self.size_padding)
        array.append(self.size_stride)
        array.append(self.n_filter)
        array.append(self.size_filter)
        array.append(self.isSharing)
        array.append(self.kernels)
        print(array)
        file = open("model.txt", "w")
        last_line = len(array) - 1
        for i,line in enumerate(array):
            if (not isinstance(line, list)):
                file.write(str(line))
            else:
                last_num = len(line) - 1
                for j,num in enumerate(line):
                    if (not isinstance(num, list)):
                        file.write(str(num))
                        if (j != last_num):
                            file.write(" ")
                    else:
                        last_elm = len(num) - 1
                        for k, elm in enumerate(num):
                            if (not isinstance(elm, list)):
                                file.write(str(elm))
                                if (k != last_elm):
                                    file.write(" ")
                            else:
                                last_str = len(elm) - 1
                                for l, elm_str in enumerate(elm):
                                    if (not isinstance(elm_str, list)):
                                        file.write(str(elm_str))
                                        if (l != last_str):
                                            file.write(" ")
                                    else:
                                        last_str_elm = len(elm_str) - 1
                                        for m, element in enumerate(elm_str):
                                            file.write(str(element))
                                            if (m != last_str_elm):
                                                file.write(" ")
                                        if (l != last_str):
                                            file.write("\n")

                                if (k != last_elm):
                                    file.write("\n")    
                        if (j != last_num):
                            file.write("\n")
            if (i != last_line):
                file.write("\n")
        file.close()

    def readModel(self):
        f = open("model.txt", "r")
        numbers = re.split('\n',f.read())
        print(numbers)
        # Ukuran Input
        self.size_input = self.convertInt(re.split(' ',numbers[0]))
        # Jumlah layer
        self.n_layer = int(numbers[1])
        # Jumlah Konvolusi Layer
        self.n_layer_konvolusi = int(numbers[2])
        # Ukuran Padding
        self.size_padding = int(numbers[3])
        # Ukuran Stride
        self.size_stride = int(numbers[4])
        # Jumlah Filter
        self.n_filter = int(numbers[5])
        # Ukuran filter
        self.size_filter = self.convertInt(re.split(' ',numbers[6]))
        # CNN Sharing Parameter or not
        self.isSharing = int(numbers[7])
        kernels = []
        indexrow = 8
        if (self.isSharing == 1):
            for i in range(self.n_filter):
                kernel = []
                for j in range(self.size_filter[0]):
                    row = []
                    for k in range(self.size_filter[1]):
                        row.append(random.randint(1,30))
                    kernel.append(row)
                kernels.append(kernel)
        elif (self.isSharing == 0):
            for i in range(self.n_filter):
                kernel = []
                for j in range(self.size_filter[2]):
                    matriks = []
                    for k in range(self.size_filter[0]):
                        row = []
                        for l in range(self.size_filter[1]):
                            row.append(random.randint(1,30))
                        matriks.append(row)
                    kernel.append(matriks)
                kernels.append(kernel)
        self.kernels = kernels

    def feedFoward(self):
        inputLayer = self.input
        # inputLayer = [
        #     [1, 1, 2, 4],
        #     [5, 6, 7, 8],
        #     [3, 2, 1, 0],
        #     [1, 2, -3, 4]
        # ]
        # Convolution
        for i in range(self.n_layer_konvolusi):
            self.layers[i].setInput([inputLayer])
            inputLayer = self.layers[i].doConvolution()[0]
            print("output convolution:", inputLayer)
        # Detector
        inputLayer = self.layers[self.n_layer_konvolusi].forward(inputLayer)
        print("output detector:", inputLayer)
        # Pooling
        outputLayer = self.layers[self.n_layer_konvolusi + 1].forward(inputLayer)
        print("output_pooling:", outputLayer)

        input_dense = []
        for i in range(len(outputLayer)):
            input_dense.append(outputLayer[i])

        self.layers[self.n_layer_konvolusi + 2].build(1, (-1,1))
        print(self.layers[self.n_layer_konvolusi + 2].weight)
        dot = self.layers[self.n_layer_konvolusi + 2].compute_dot(input_dense)
        print('DOT : ', dot)
        output = self.layers[self.n_layer_konvolusi + 2].compute_output(dot)
        print('OUTPUT : ', output)

    def backprop(self):
        print("Backward Propagation")
        last_idx = len(self.layers) - 1
        
        # for idx in range(len(self.layers)-2, -1, -1): ## Ini belum ada dense layer
        backward_pooling = self.layers[last_idx - 1].backward()
        backward_pooling = np.array(backward_pooling).reshape(self.layers[last_idx - 1].input.shape)
        backward_detector = self.layers[last_idx - 2].backward(backward_pooling)
        print(backward_detector)
        # loss = [ [0, -1],
        #         [1, 0] ]
        # for idx in range(self.n_layer_konvolusi-1, -1, -1):
        #     back_conv = BackConv(backward_detector,self.kernels,loss)
        #     f_update, loss_grad = bc.back_conv()

CNN = CNNClassifier()

CNN.load("text.txt")

# Pilih Dataset untuk input
CNN.loadInputMNist('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')

CNN.feedFoward()
CNN.backprop()
# CNN.saveModel()
# CNN.readModel()

# matrix = np.array([
#     [1, 1, 2, 4],
#     [5, 6, 7, 8],
#     [3, 2, 1, 0],
#     [1, 2, -3, 4]
# ])
# kernel = np.array([
#     [
#         [1, 0],
#         [0, 1]
#     ]
# ])

# print("input", matrix)
# print("kernel", kernel)


# convolution = Convolution([CNN.getInput()], CNN.getKernels(), 0, 100, 1)
# output_convolution = convolution.doConvolution()[0]
# print("output convolution:", output_convolution)
# detector = Detector()
# output_detector = detector.activate(input_matrix=output_convolution, activation_type='')
# print("output detector:", output_detector)
# pooling = Pooling()
# output_pooling = pooling.apply(input_matrix=output_detector, kernel_size=(2,2), stride=2, padding=0, pool_mode='max')
# print("output_pooling:", output_pooling)