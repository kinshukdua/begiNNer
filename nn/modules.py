import numpy as np
from .activations import sigmoid
class Linear:
    def __init__(self, in_feat, out_feat):
        self.size = (in_feat, out_feat)
        self.weights = np.random.rand(in_feat, out_feat)
        self.bias = np.random.rand(out_feat)

    def forward(self, x):
        return x@self.weights+self.bias

    def __call__(self,x):
        return self.forward(x)
    
    def __repr__(self):
        return f"Linear Layer {self.size}"
        
    def backward(self):
        pass

class MLP:
    def __init__(self, input_size, output_size, hidden_size, activation=sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation
        self._create_model()

    def _create_model(self):
        self.layers = []
        layer = Linear(self.input_size, self.hidden_size[0])
        self.layers.append(layer)
        for i in range(1,len(self.hidden_size)-1):
            layer = Linear(self.hidden_size[i], self.hidden_size[i+1])
            self.layers.append(layer)
        layer = Linear(self.hidden_size[-1],  self.output_size)
        self.layers.append(layer)

    def forward(self, x_inp):
        a = x_inp
        for layer in self.layers:
            z = layer(a)
            a = self.activation(z)
        return a

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"MLP of size: {self.input_size}x{self.hidden_size}x{self.output_size}"
        
        
            