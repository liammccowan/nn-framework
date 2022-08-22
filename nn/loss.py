import numpy as np
from nn.layers import Layer

class MeanSquaredError(Layer):
    def __init__(self, predicted, real):
        self.predicted = predicted
        self.real = real
        self.type = 'Mean Squared Error'

    def forward(self):
        return np.power(self.predicted - self.real, 2).mean()

    def backward(self):
        return 2 * (self.predicted - self.real).mean()

class BinaryCrossEntropy(Layer):
    def __init__(self):
        self.type = 'Binary Cross Entropy'

    def forward(self, predicted, real):
        self.real = real
        self.predicted = predicted
        n = len(self.real)
        loss = np.nansum(-self.real @ np.log(self.predicted) - (1 - self.real) @ np.log(1 - self.predicted)) / n
        return np.squeeze(loss)

    def backward(self):
        n = len(self.real)
        return (-(self.real / self.predicted) + ((1 - self.real) / (1 - self.predicted))) / n
        