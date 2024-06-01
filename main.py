import numpy as np
from nn.modules import MLP

input_size = 2
output_size = 1
hidden_size = [3,3,32,1,2]
batch = 100

model = MLP(input_size, output_size, hidden_size)

x = np.random.rand(batch, input_size)
print(model(x).shape)
# assert model(x).shape == (batch, output_size)
