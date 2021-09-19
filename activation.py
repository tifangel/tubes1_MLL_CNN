import math

def linear(arr):
    result = []
    for x in arr:
        result.append(x)
    return result

def sigmoid(arr):
    res = []
    for x in arr:
        x = round((1/(1 + math.exp(0-x))),3)
        
        res.append(x)
    return res

def relu(arr):
    res = []
    for x in arr:
        if (x >= 0) :
            res.append(max(0, x))
        else :
            res.append(x * 0.0001)
    return res

def softmax(arr):
    e = []
    p = []
    for i in arr:
        e.append(math.exp(i))
    c = sum(e)
    for j in e:
        p.append(j / c)
    return p

def get(identifier = None):
    if identifier is None:
        return linear
    identifier = identifier.upper()
    if(identifier == 'RELU'):
        return relu
    elif(identifier == 'SIGMOID'):
        return sigmoid
    elif(identifier == 'SOFTMAX'):
        return softmax
    else:
        return linear