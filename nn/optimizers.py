import numpy as np

class Optimizer(object):
    # optimizer abstract class
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError

    def zeroGradient(self):
        for p in self.parameters:
            p.grad = 0.0

    def clipGradient(self, max_value = 5):
        for p in self.parameters:
            p.grad = np.clip(p.grad, -max_value, max_value, out = p.grad)

class SGD(Optimizer):
    def __init__(self, parameters, learning_rate = 0.001, weight_decay = 0.0, momentum = 0.9):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocity = []
        for p in parameters:
            self.velocity.append(np.zeros_like(p.grad))
        
    def step(self):
        for p, v in zip(self.parameters, self.velocity):
            v = (self.momentum * v) + p.grad + (self.weight_decay * p.data)
            p.data = p.data - (self.learning_rate * v)

class Adam(Optimizer):
    def __init__(self, parameters, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.v = []
        self.s = []
        self.t = 1
        for p in parameters:
            self.v.append(np.zeros_like(p.grad))
            self.s.append(np.zeros_like(p.grad))
    
    def step(self):
        for p, v, s in zip(self.parameters, self.v, self.s):
            v = ((self.beta_1 * v) + ((1 - self.beta_1) * p.grad)) / (1 - (self.beta_1**self.t))
            s = ((self.beta_2 * s) + ((1 - self.beta_2) * (p.grad**2))) / (1 - (self.beta_2**self.t))
            p.data = p.data - (self.learning_rate * (v / (np.sqrt(s) + self.epsilon)))
            self.t += 1


        
