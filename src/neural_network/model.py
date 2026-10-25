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

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params):
        if self.m is None: 
            self.m = [np.zeros_like(p) for p, _ in params]
            self.v = [np.zeros_like(p) for p, _ in params]
        self.t += 1
        for i, (p, g) in enumerate(params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)   
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

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

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]

        


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

    def train(self, x_train, x_test, y_train, y_test, loss_fn="mse", epochs=100, lr=0.01, batch_size=None, shuffle=True, log_every=10):
        if batch_size is None:
            batch_size = x_train.shape[0]
        rng = np.random.default_rng()
        history = {"train_loss": [], "test_loss": []}
        optimizer = Adam(lr=lr)
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
                optimizer.step(self.params())
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

    def params(self):
        return [p for layer in self.layers for p in layer.params()]       

    