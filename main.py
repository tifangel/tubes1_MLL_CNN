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

        # Pilih Dataset untuk input
        X, y = loadlocal_mnist(
            images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')
        self.input = X
        self.target = y

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
        poolingLayer = Pooling(self.n_layer_konvolusi + 1, self.size_stride, self.size_padding, 'max')
        layers.append(poolingLayer)
        # Set Layer to CNN
        self.layers = layers

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
        self.layers[self.n_layer_konvolusi].setInput(inputLayer)
        inputLayer = self.layers[self.n_layer_konvolusi].activate()
        print("output detector:", inputLayer)
        # Pooling
        outputLayer = self.layers[self.n_layer_konvolusi + 1].apply(inputLayer)
        print("output_pooling:", outputLayer)

CNN = CNNClassifier()

CNN.load("text.txt")

CNN.feedFoward()

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