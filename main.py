from mlxtend.data import loadlocal_mnist
import random
import re
import math
import csv
import sys
import numpy as np
from Convolution import Convolution
from Pooling import Pooling
from Detector import Detector

class Layer:
    def __init__(self, idlayer):
        self.id_layer = idlayer
        self.activfunc = -1
        self.inputs = []
        self.kernels = []

    def setActivFunc(self, activfunc):
        self.activfunc = activfunc
    
    def setKernels(self, kernels):
        self.kernels = kernels
    
    def setInput(self, inputs):
        self.inputs = inputs

    def printKernels(self, isSharing):
        print("Kernel Layer ", self.id_layer)
        if (isSharing == 0):
            for i, kernel in enumerate(self.kernels):
                print("Kernel ", i)
                for j, elmt in enumerate(kernel):
                    print(elmt)
        else:
            for i, kernel in enumerate(self.kernels):
                print(kernel)

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
            newLayer = Layer(i)
            newLayer.setKernels(self.kernels[i])
            newLayer.printKernels(self.isSharing)
            layers.append(newLayer)
        # Detector Layer
        detectorLayer = Layer(self.n_layer_konvolusi + 1)
        detectorLayer.setActivFunc(int(numbers[7]))
        layers.append(detectorLayer)
        # Pooling Layer
        poolingLayer = Layer(self.n_layer_konvolusi + 2)
        poolingLayer.setActivFunc(int(numbers[8]))
        layers.append(poolingLayer)
        # Set Layer to CNN
        self.layers = layers

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
    
    def doActivFunc(self, f, arr):
        # Activation function: linear
        if (f == 1):
            return arr
        # Activation function: sigmoid
        elif (f == 2):
            return self.sigmoid(arr)
        # Activation function: reLU
        elif (f == 3):
            return self.relu(arr)
        # Activation function: softmax
        elif (f == 4):
            return self.softmax(arr)

CNN = CNNClassifier()

CNN.load("text.txt")

matrix = np.array([
    [1, 1, 2, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 0],
    [1, 2, -3, 4]
])
kernel = np.array([
    [
        [1, 0],
        [0, 1]
    ]
])
print("input", matrix)
print("kernel", kernel)
convolution = Convolution(np.array([matrix]), kernel, 2, 2, 1)
output_convolution = convolution.doConvolution()[0]
print("output convolution:", output_convolution)
detector = Detector()
output_detector = detector.activate(input_matrix=output_convolution, activation_type='')
print("output detector:", output_detector)
pooling = Pooling()
output_pooling = pooling.apply(input_matrix=output_detector, kernel_size=(2,2), stride=2, padding=0, pool_mode='max')
print("output_pooling:", output_pooling)