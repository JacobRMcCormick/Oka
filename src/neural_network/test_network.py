import numpy as np
from model import Network

rng = np.random.default_rng(0)
x = rng.uniform(-2, 2, (500, 1))
y = x**2

net = Network(1)
net.add_layer(32, "relu")
net.add_layer(1, "linear")
hist = net.train(x[:400], x[400:], y[:400], y[400:], epochs=600, lr=0.001, batch_size=32)