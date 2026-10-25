import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(z.dtype)

def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

def gelu_derivative(z):
    c = np.sqrt(2/np.pi)
    t = np.tanh(c * (z + 0.044715 * z**3))
    return 0.5 * (1 + t) + 0.5 * z * (1 - t**2) * c * (1 + 3 * 0.044715 * z**2)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def softmax_backward(s, da, axis=-1):
    return s * (da - np.sum(da * s, axis=axis, keepdims=True))

def mse(pred, target):
    return np.mean((pred - target) ** 2), 2 * (pred - target) / target.size

def cross_entropy(logits, targets):
    V = logits.shape[-1]
    l2d = logits.reshape(-1, V)
    t = targets.reshape(-1)
    N = t.shape[0]

    shifted = l2d - l2d.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(N), t].mean()
    grad = np.exp(log_probs)          
    grad[np.arange(N), t] -= 1        
    grad /= N
    return loss, grad.reshape(logits.shape)

ACTIVATIONS = {
    "relu":   (relu, relu_derivative),
    "gelu":   (gelu, gelu_derivative),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}

LOSSES = {
    "mse": mse,
    "cross_entropy": cross_entropy,
}


def build_transformer(*, input_type, out_dim, d_model=64, n_heads=4, n_layers=2, max_len=64, vocab_size=None, in_features=None, causal=True, output="per_position"):
    if (input_type == "tokens"):
        layers = [TokenPosEmbedding(vocab_size, max_len, d_model)]
    else:
        layers = [Dense(in_features, d_model, "linear"), PosEmbedding(max_len, d_model)]
    layers += [TransformerBlock(d_model, n_heads, causal) for _ in range(n_layers)]
    layers.append(LayerNorm(d_model))
    if (output == "mean"):
        layers.append(MeanPool())
    elif output == "last":
        layers.append(SelectToken(-1))
    layers.append(Dense(d_model, out_dim, "linear"))
    return Network(layers)


class MeanPool:
    def forward(self, x):
        self.T = x.shape[1]
        return x.mean(axis=1)

    def backward(self, da):
        return np.repeat(da[:, None, :], self.T, axis=1) / self.T


class SelectToken:
    def __init__(self, pos=-1):
        self.pos = pos

    def forward(self, x):
        self.shape = x.shape
        return x[:, self.pos] 

    def backward(self, da):
        dx = np.zeros(self.shape)
        dx[:, self.pos] = da
        return dx


class Embedding:
    def __init__(self, num_embeddings, dim):
        rng = np.random.default_rng()
        self.W = rng.standard_normal((num_embeddings, dim)) * 0.02
        self.dW = np.zeros_like(self.W)

    def forward(self, idx):
        self.idx = idx
        return self.W[idx]

    def backward(self, da):
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.idx, da)
        return None

    def params(self):
        return [(self.W, self.dW)]


class PosEmbedding:
    def __init__(self, max_len, dim):
        self.pos = Embedding(max_len, dim)

    def forward(self, x):
        self.T = x.shape[1]
        return x + self.pos.forward(np.arange(self.T))

    def backward(self, da):
        self.pos.backward(da.sum(axis=0))
        return da

    def params(self):
        return self.pos.params()


class TokenPosEmbedding:
    def __init__(self, vocab_size, max_len, dim):
        self.tok = Embedding(vocab_size, dim)
        self.pos = Embedding(max_len, dim)

    def forward(self, idx):
        T = idx.shape[1]
        self.T = T
        return self.tok.forward(idx) + self.pos.forward(np.arange(T))

    def backward(self, da):
        self.tok.backward(da)
        self.pos.backward(da.sum(axis=0))  
        return None

    def params(self):
        return self.tok.params() + self.pos.params()

def split_heads(x, H):
    B, T, D = x.shape
    return x.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)

def merge_heads(x):
    B, H, T, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, causal=True):
        assert d_model % n_heads == 0
        self.H = n_heads
        self.dh = d_model // n_heads
        self.Wq = Dense(d_model, d_model, "linear")
        self.Wk = Dense(d_model, d_model, "linear")
        self.Wv = Dense(d_model, d_model, "linear")
        self.Wo = Dense(d_model, d_model, "linear")
        self.causal = causal

    def forward(self, x):
        B, T, D = x.shape
        self.q = split_heads(self.Wq.forward(x), self.H)
        self.k = split_heads(self.Wk.forward(x), self.H)
        self.v = split_heads(self.Wv.forward(x), self.H)

        scores = self.q @ self.k.transpose(0, 1, 3, 2) / np.sqrt(self.dh)
        if self.causal:
            mask = np.tril(np.ones((T, T), dtype=bool))
            scores = np.where(mask, scores, -1e9)

        self.attn = softmax(scores)
        out = merge_heads(self.attn @ self.v)
        return self.Wo.forward(out)

    def backward(self, da):
        dout = split_heads(self.Wo.backward(da), self.H)

        dattn = dout @ self.v.transpose(0, 1, 3, 2)
        dv = self.attn.transpose(0, 1, 3, 2) @ dout

        dscores = softmax_backward(self.attn, dattn) / np.sqrt(self.dh)
        dq = dscores @ self.k
        dk = dscores.transpose(0, 1, 3, 2) @ self.q

        dx = self.Wq.backward(merge_heads(dq))
        dx = dx + self.Wk.backward(merge_heads(dk))
        dx = dx + self.Wv.backward(merge_heads(dv))
        return dx

    def params(self):
        return (self.Wq.params() + self.Wk.params() + self.Wv.params() + self.Wo.params())
        


class FeedForward:
    def __init__(self, d_model, d_hidden=None):
        d_hidden = d_hidden or 4 * d_model
        self.fc1 = Dense(d_model, d_hidden, "gelu")
        self.fc2 = Dense(d_hidden, d_model, "linear")

    def forward(self, x):
        return self.fc2.forward(self.fc1.forward(x))

    def backward(self, da):
        return self.fc1.backward(self.fc2.backward(da))

    def params(self):
        return self.fc1.params() + self.fc2.params()


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.dgamma = np.zeros(dim)
        self.dbeta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mean) * self.inv_std
        return self.gamma * self.xhat + self.beta

    def backward(self, da):
        flat = lambda a: a.reshape(-1, a.shape[-1])
        self.dgamma = (flat(da) * flat(self.xhat)).sum(axis=0)
        self.dbeta = flat(da).sum(axis=0)

        dxhat = da * self.gamma
        return self.inv_std * (dxhat - dxhat.mean(axis=-1, keepdims=True) - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True))

    def params(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


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


class TransformerBlock:
    def __init__(self, d_model, n_heads, causal=True):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        h = x + self.attn.forward(self.ln1.forward(x))
        return h + self.ffn.forward(self.ln2.forward(h))

    def backward(self, da):
        dh = da + self.ln2.backward(self.ffn.backward(da))
        return dh + self.ln1.backward(self.attn.backward(dh))

    def params(self):
        return (self.ln1.params() + self.attn.params() + self.ln2.params() + self.ffn.params())


class Dense:
    def __init__(self, num_input, num_output, activation="linear"):
        rng = np.random.default_rng()
        self.W = rng.standard_normal((num_input, num_output)) * np.sqrt(2 / num_input)
        self.b = np.zeros(num_output)
        self.act, self.act_deriv = ACTIVATIONS[activation]
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        return self.act(self.z)

    def backward(self, da):
        dz = da * self.act_deriv(self.z)
        x2d = self.x.reshape(-1, self.x.shape[-1])
        dz2d = dz.reshape(-1, dz.shape[-1])
        self.dW = x2d.T @ dz2d
        self.db = dz2d.sum(axis=0)
        return dz @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]

        


class Network:
    def __init__(self, layers=None):
        self.layers = layers or []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, da):
        for layer in reversed(self.layers):
            da = layer.backward(da)
        return da

    def train(self, x_train, x_test, y_train, y_test, loss_fn="mse", epochs=100, lr=1e-3, batch_size=None, shuffle=True, log_every=10):
        if loss_fn not in LOSSES:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        loss_func = LOSSES[loss_fn]

        n = x_train.shape[0]
        if batch_size is None:
            batch_size = n

        rng = np.random.default_rng()
        optimizer = Adam(lr=lr)
        history = {"train_loss": [], "test_loss": []}

        for epoch in range(epochs):
            idx = rng.permutation(n) if shuffle else np.arange(n)
            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                xb, yb = x_train[batch], y_train[batch]

                pred = self.forward(xb)
                loss, grad = loss_func(pred, yb)
                self.backward(grad)
                optimizer.step(self.params())

                epoch_loss += loss * len(batch)

            train_loss = epoch_loss / n
            test_loss, _ = loss_func(self.forward(x_test), y_test)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)

            if log_every and epoch % log_every == 0:
                print(f"epoch {epoch}: train {train_loss:.4f}  test {test_loss:.4f}")
        return history

    def params(self):
        return [p for layer in self.layers for p in layer.params()]       

    