class Layer:
    def __init__(self, idlayer):
        self.id_layer = idlayer
        self.input = []
        self.output = []
    
    def setInput(self, input):
        self.input = input
    
    def setOutput(self, output):
        self.output = output