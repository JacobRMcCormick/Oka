import numpy as np

def activate(z, activation):
        if activation == "relu":
            return np.maximum(0, z)
        elif activation == "gelu":
            return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
        else:
            return z

def activation_derivative(z, activation):
    if activation == "relu":
        return (z > 0).astype(z.dtype)
    elif activation == "gelu":
        c = np.sqrt(2/np.pi)
        t = np.tanh(c * (z + 0.044715 * z**3))
        return 0.5 * (1 + t) + 0.5 * z * (1 - t**2) * c * (1 + 3 * 0.044715 * z**2)
    else:
        return np.ones_like(z)

def mse(pred, target):
    return np.mean((pred - target) ** 2), 2 * (pred - target) / target.size

class Layer:
    def __init__(self, num_input, num_output, activation="linear"):
        rng = np.random.default_rng()
        self.W = rng.standard_normal((num_input, num_output)) * np.sqrt(2 / num_input)
        self.b = np.zeros(num_output)
        self.activation = activation
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        return activate(self.z, self.activation)

    def backward(self, da):
        dz = da * activation_derivative(self.z, self.activation)
        x2d = self.x.reshape(-1, self.x.shape[-1])
        dz2d = dz.reshape(-1, dz.shape[-1])
        self.dW = x2d.T @ dz2d
        self.db = dz2d.sum(axis=0)
        return dz @ self.W.T
        

        


class Network:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []

    def add_layer(self, num_output, activation="linear"):
        self.layers.append(Layer(self.input_size, num_output, activation))
        self.input_size = num_output

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, da):
        for layer in reversed(self.layers):
            da = layer.backward(da)
        return da

    def step(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def train(self, x_train, x_test, y_train, y_test, loss_fn="mse", epochs=100, lr=0.01, batch_size=None, shuffle=True, log_every=10):
        if batch_size is None:
            batch_size = x_train.shape[0]
        rng = np.random.default_rng()
        history = {"train_loss": [], "test_loss": []}
        for epoch in range(epochs):
            idx = rng.permutation(x_train.shape[0]) if shuffle else np.arange(x_train.shape[0])
            epoch_loss = 0.0
            for start in range(0, x_train.shape[0], batch_size):
                batch = idx[start:start + batch_size]
                xb, yb = x_train[batch], y_train[batch]
                pred = self.forward(xb)
                if loss_fn == "mse":
                    loss, grad = mse(pred, yb)
                else:
                    print("Unknown loss function")
                    return 0
                self.backward(grad)
                self.step(lr)
                epoch_loss += loss * len(batch)
            train_loss = epoch_loss / x_train.shape[0]
            if loss_fn == "mse":
                test_loss, _ = mse(self.forward(x_test), y_test)
            else:
                print("Unknown loss function")
                return 0
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            if log_every and epoch % log_every == 0:
                print(f"epoch {epoch}: train {train_loss:.4f}  test {test_loss:.4f}")
        return history

                

    