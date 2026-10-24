import numpy as np

def process_layer(x, W, B, activation):
    z = x @ W + B
    if activation == "relu":
        return np.maximum(0, z), z
    elif activation == "linear":
        return z, z

def activation_derivative(z, activation):
    if activation == "relu":
        return (z > 0).astype(z.dtype)
    elif activation == "linear":
        return np.ones_like(z)

def mse(pred, target):
    return np.mean((pred - target) ** 2), 2 * (pred - target) / pred.size

class Network:
    def __init__(self, input_size, load_from=None):
        self.input_size = input_size
        self.weight_list = []
        self.bias_list = []
        self.activation_list = []
        self.inputs = []
        self.z_vals = []

    def add_layer(self, num_output, activation):
        rng = np.random.default_rng()
        self.weight_list.append(rng.standard_normal((self.input_size, num_output)) * np.sqrt(2 / self.input_size))
        self.bias_list.append(np.zeros((1, num_output)))
        self.activation_list.append(activation)
        self.input_size = num_output

    def forward(self, x):
        self.inputs = []
        self.z_vals = []
        for i in range(len(self.weight_list)):
            self.inputs.append(x)
            x, z = process_layer(x, self.weight_list[i], self.bias_list[i], self.activation_list[i])
            self.z_vals.append(z)
        return x

    def backward(self, da, lr=0.01):
        for i in reversed(range(len(self.weight_list))):
            dz = da * activation_derivative(self.z_vals[i], self.activation_list[i])
            dW = self.inputs[i].T @ dz
            db = dz.sum(axis=0, keepdims=True)

            da = dz @ self.weight_list[i].T

            self.weight_list[i] -= lr * dW
            self.bias_list[i]   -= lr * db

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
                self.backward(grad, lr)
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

                

    