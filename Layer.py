class Layer:
    def __init__(self, idlayer):
        self.id_layer = idlayer
        self.inputs = []
        self.output = []
    
    def setInput(self, inputs):
        self.inputs = inputs
    
    def setOutput(self, outputs):
        self.output = outputs