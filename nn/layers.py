import numpy as np

class Tensor():
    # tensor class
    def __init__(self, shape):
        self.data = np.ndarray(shape, np.float32)
        self.grad = np.ndarray(shape, np.float32)

class Layer(object):
    # layer abstract class
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def getParameters(self):
        return []

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = Tensor((input_dim, output_dim))
        self.biases = Tensor((1, output_dim))
        self.type = 'Linear'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = input_val
        return (input_val @ self.weights.data) + self.biases.data

    def backward(self, dA):
        self.weights.grad += (self._prev_acti.T @ dA)
        self.biases.grad += np.sum(dA, axis = 0, keepdims = True)
        
        grad_input = (dA @ self.weights.data.T)

        return grad_input

    def getParameters(self):
        return [self.weights, self.biases]

class Convolutional(Layer):
    def __init__(self, input_channels, num_filters, filter_dims, padding = 0, stride = 1):
        self.weights = Tensor((num_filters, input_channels, filter_dims[0], filter_dims[1]))
        self.biases = Tensor((num_filters))
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.filter_dims = filter_dims
        self.padding = padding
        self.stride = stride
        self.type = 'Convolutional'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self.input = input_val

        N, _, H, W = self.input.shape

        out_height = int((H + (2 * self.padding) - self.filter_dims[0]) / self.stride) + 1
        out_width = int((W + (2 * self.padding) - self.filter_dims[1]) / self.stride) + 1
        output = np.zeros((N, self.num_filters, out_height, out_width), dtype = self.input.dtype)

        self.input_cols = self.im2col(self.input, self.filter_dims, self.padding, self.stride)

        weights_rows = self.weights.data.reshape(self.weights.data.shape[0], -1)
        
        result = (weights_rows @ self.input_cols) + self.biases.data.reshape(-1, 1)
        
        output = result.reshape(self.weights.data.shape[0], output.shape[2], output.shape[3], self.input.shape[0])

        return output.transpose(3, 0, 1, 2)

    def backward(self, dA):
        self.biases.grad = np.sum(dA, axis = (0, 2, 3))
        
        dA_reshaped = dA.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        self.weights.grad = (dA_reshaped @ self.input_cols.T).reshape(self.weights.data.shape)

        d_input_cols = self.weights.data.reshape(self.num_filters, -1).T @ dA_reshaped
        
        return self.col2im(d_input_cols, self.input.shape, self.filter_dims, self.padding, self.stride)

    def getIndices(self, x_shape, filter_dims, padding, stride):
        _, C, H, _ = x_shape

        output_size = int((H + (2 * padding) - filter_dims[0]) / stride) + 1

        c = np.repeat(np.arange(C), filter_dims[0] * filter_dims[1]).reshape(-1, 1)

        i0 = np.repeat(np.arange(filter_dims[0]), filter_dims[1])
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(output_size), output_size)
        j0 = np.tile(np.arange(filter_dims[0]), filter_dims[1] * C)
        j1 = stride * np.tile(np.arange(output_size), output_size)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        return (c, i, j)

    def im2col(self, x, filter_dims, padding, stride):
        p = padding
        input_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = 'constant')

        c, i, j = self.getIndices(x.shape, filter_dims, padding, stride)

        col = input_padded[:, c, i, j]
        col = col.transpose(1, 2, 0)

        return col.reshape(filter_dims[0] * filter_dims[1] * x.shape[1], -1)

    def col2im(self, cols, x_shape, filter_dims, padding, stride):
        N, C, H, _ = x_shape
        padded_shape = H + (2 * padding)
        input_padded = np.zeros((N, C, padded_shape, padded_shape), dtype = cols.dtype)

        col_reshaped = cols.reshape(C * filter_dims[0] * filter_dims[1], -1, N)
        col_reshaped = col_reshaped.transpose(2, 0, 1)

        c, i, j = self.getIndices(x_shape, filter_dims, padding, stride)
        np.add.at(input_padded, (slice(None), c, i, j), col_reshaped)

        if padding == 0:
            return input_padded
        return input_padded[:, :, padding:-padding, padding:-padding]

    def getParameters(self):
        return [self.weights, self.biases]

class MaxPool(Layer):
    def __init__(self, pool_dims, padding = 0, stride = 2):
        self.pool_dims = pool_dims
        self.padding = padding
        self.stride = stride
        self.type = 'Max Pool'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self.input = input_val

        N, C, H, W = self.input.shape

        assert (H - self.pool_dims[0]) % self.stride == 0, 'Invalid height'
        assert (W - self.pool_dims[1]) % self.stride == 0, 'Invalid width'

        out_height = int((H - self.pool_dims[0]) / self.stride + 1)
        out_width = int((W - self.pool_dims[1]) / self.stride + 1)

        input_split = self.input.reshape(N * C, 1, H, W)
        self.input_cols = self.im2col(input_split, self.pool_dims, self.padding, self.stride)
        self.input_cols_argmax = np.argmax(self.input_cols, axis = 0)
        input_cols_max = self.input_cols[self.input_cols_argmax, np.arange(self.input_cols.shape[1])]

        return input_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    def backward(self, dA):
        N, C, H, W = self.input.shape

        dA_reshaped = dA.transpose(2, 3, 0, 1).flatten()
        d_input_cols = np.zeros_like(self.input_cols)
        d_input_cols[self.input_cols_argmax, np.arange(d_input_cols.shape[1])] = dA_reshaped
        d_input = self.col2im(d_input_cols, (N * C, 1, H, W), self.pool_dims, self.padding, self.stride)

        return d_input.reshape(self.input.shape)

    def getIndices(self, x_shape, filter_dims, padding, stride):
        _, C, H, _ = x_shape

        output_size = int((H + (2 * padding) - filter_dims[0]) / stride) + 1

        c = np.repeat(np.arange(C), filter_dims[0] * filter_dims[1]).reshape(-1, 1)

        i0 = np.repeat(np.arange(filter_dims[0]), filter_dims[1])
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(output_size), output_size)
        j0 = np.tile(np.arange(filter_dims[0]), filter_dims[1] * C)
        j1 = stride * np.tile(np.arange(output_size), output_size)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        return (c, i, j)

    def im2col(self, x, filter_dims, padding, stride):
        p = padding
        input_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = 'constant')

        c, i, j = self.getIndices(x.shape, filter_dims, padding, stride)

        col = input_padded[:, c, i, j]
        col = col.transpose(1, 2, 0)

        return col.reshape(filter_dims[0] * filter_dims[1] * x.shape[1], -1)

    def col2im(self, cols, x_shape, filter_dims, padding, stride):
        N, C, H, _ = x_shape
        padded_shape = H + (2 * padding)
        input_padded = np.zeros((N, C, padded_shape, padded_shape), dtype = cols.dtype)

        col_reshaped = cols.reshape(C * filter_dims[0] * filter_dims[1], -1, N)
        col_reshaped = col_reshaped.transpose(2, 0, 1)

        c, i, j = self.getIndices(x_shape, filter_dims, padding, stride)
        np.add.at(input_padded, (slice(None), c, i, j), col_reshaped)

        if padding == 0:
            return input_padded
        return input_padded[:, :, padding:-padding, padding:-padding]

class Flatten(Layer):
    def __init__(self):
        self.type = 'Flatten'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self.input = input_val
        return self.input.reshape(self.input.shape[0], -1)

    def backward(self, dA):
        return dA.reshape(self.input.shape)

class ReLU(Layer):
    def __init__(self):
        self.type = 'ReLU'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = np.maximum(0, input_val)
        return self._prev_acti

    def backward(self, dJ):
        return dJ * np.heaviside(self._prev_acti, 0)

class Sigmoid(Layer):
    def __init__(self):
        self.type = 'Sigmoid'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self._prev_acti = 1 / (1 + np.exp(-input_val))
        return self._prev_acti

    def backward(self, dJ):
        sig = self._prev_acti
        return dJ * sig * (1 - sig)

class SoftmaxWithLoss(Layer):
    def __init__(self):
        self.type = 'Softmax With Loss'

    def forward(self, input_val, Y):
        unnormalized_probability = np.exp(input_val - np.max(input_val, axis = 1, keepdims = True))
        self.probability = unnormalized_probability / np.sum(unnormalized_probability, axis = 1, keepdims = True)

        self.target = Y

        eps = np.finfo(float).eps
        return -(1.0/self.target.shape[0]) * np.sum(self.target * np.log(self.probability + eps))

    def backward(self):
        return self.probability - self.target

class Dropout(Layer):
    def __init__(self, prob):
        self.prob = prob
        self.type = 'Dropout'

    def forward(self, input_val):
        self.mask = np.random.rand(*input_val.shape)
        self.mask = (self.mask >= self.prob).astype(int)
        return (input_val * self.mask) / (1 - self.prob)

    def backward(self, dA):
        return (dA * self.mask) / (1 - self.prob)

class BatchNorm(Layer):
    def __init__(self, num_features, num_dims = 2, eps = 1e-8, momentum = 0.9):
        if num_dims == 2:
            self.gamma = Tensor((1, num_features))
            self.beta = Tensor((1, num_features))
        else:
            self.gamma = Tensor((1, num_features, 1, 1))
            self.beta = Tensor((1, num_features, 1, 1))            
        self.eps = eps
        self.momentum = momentum
        self.mean_avg = 0.0
        self.var_avg = 0.0
        self.type = 'Batch Normalization'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self.x = input_val
        if len(self.x.shape) == 2:
            self.mean = np.mean(self.x, axis = 0)
            self.var = np.var(self.x, axis = 0)
        else:
            self.mean = np.mean(self.x, axis = (0, 2, 3), keepdims = True)
            self.var = np.var(self.x, axis = (0, 2, 3), keepdims = True)           
        self.inv_std = 1.0 / np.sqrt(self.var + self.eps)
        self.x_norm = (self.x - self.mean) * self.inv_std

        self.mean_avg = (self.momentum * self.mean_avg) + ((1 - self.momentum) * self.mean)
        self.var_avg = (self.momentum * self.var_avg) + ((1 - self.momentum) * self.var)
        
        return (self.x_norm * self.gamma.data) + self.beta.data

    def forward_predict(self, input_val):
        self.x_p = input_val
        self.inv_std_avg = 1.0 / np.sqrt(self.var_avg + self.eps)
        self.x_norm_p = (self.x_p - self.mean_avg) * self.inv_std_avg
        
        return (self.x_norm_p * self.gamma.data) + self.beta.data

    def backward(self, dA):
        inv_N = 1.0 / np.prod(self.mean.shape)

        self.gamma.grad = np.sum((dA * self.x_norm), axis = 0)
        self.beta.grad = np.sum(dA, axis = 0)

        self.delta = dA * self.gamma.data
        self.mean_delta = np.mean((self.delta * (-self.inv_std)), axis = 0)
        self.var_delta = np.sum((self.delta * (self.x - self.mean)), axis = 0) * (-0.5 * self.inv_std**3)

        return (self.delta * self.inv_std + self.var_delta * 2 *(self.x - self.mean) * inv_N + self.mean_delta * inv_N)

    def getParameters(self):
        return [self.gamma, self.beta]

class LSTM(Layer):
    def __init__(self, input_size, hidden_size, final_output_only = False):
        self.Wf = Tensor((hidden_size, hidden_size + input_size))
        self.bf = Tensor((hidden_size, 1))
        self.Wi = Tensor((hidden_size, hidden_size + input_size))
        self.bi = Tensor((hidden_size, 1))
        self.Wc = Tensor((hidden_size, hidden_size + input_size))
        self.bc = Tensor((hidden_size, 1))
        self.Wo = Tensor((hidden_size, hidden_size + input_size))
        self.bo = Tensor((hidden_size, 1))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.final_output_only = final_output_only
        self.type = 'LSTM'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val, a_prev, c_prev):
        self.input_val = input_val
        x, self.z, self.ft, self.it, self.cct, self.c, self.ot, self.a = {}, {}, {}, {}, {}, {}, {}, {}

        self.a[-1] = a_prev
        self.c[-1] = c_prev

        for t in range(len(self.input_val)):
            self.z[t], self.ft[t], self.it[t], self.cct[t], self.c[t], self.ot[t], self.a[t] = self.cell_forward(self.input_val[t], self.a[t-1], self.c[t-1])

        del self.a[-1]

        if self.final_output_only:
            return self.a[t]
        else:
            return self.a
   
    def backward(self, da):
        da_next = np.zeros_like(self.a[0])
        dc_next = np.zeros_like(self.c[0])

        dx = {}

        if self.final_output_only:
            da_final = da.T
            da = {}
            for t in reversed(range(len(self.input_val))):
                da[t] = np.zeros_like(da_final)
            da[len(self.input_val) - 1] = da_final

        for t in reversed(range(len(self.input_val))):
            dx_next, da_next, dc_next = self.cell_backward(da[t], da_next, dc_next, self.c[t-1], self.z[t], self.ft[t], self.it[t], self.cct[t], self.c[t], self.ot[t], self.a[t])
            dx[t] = dx_next

        return dx, self.a[len(self.input_val) - 1], self.c[len(self.input_val) - 1]

    def cell_forward(self, xt, a_prev, c_prev):
        z = np.concatenate((a_prev, xt), axis = 0)

        ft = self.sigmoid((self.Wf.data @ z) + self.bf.data)
        it = self.sigmoid((self.Wi.data @ z) + self.bi.data)
        cct = np.tanh((self.Wc.data @ z) + self.bc.data)
        c = (ft * c_prev) + (it * cct)
        ot = self.sigmoid((self.Wo.data @ z) + self.bo.data)
        a = ot * np.tanh(c)

        return z, ft, it, cct, c, ot, a

    def cell_backward(self, da, da_next, dc_next, c_prev, z, ft, it, cct, c, ot, a):
        n_a, _ = a.shape

        da += da_next

        dot = da * np.tanh(c)
        da_ot = dot * ot * (1 - ot)

        dc = da * ot * (1 - np.tanh(c)**2)
        dc += dc_next

        dcct = dc * it
        da_c = dcct * (1 - cct**2)

        dit = dc * cct
        da_it = dit * it * (1 - it)

        dft = dc * c_prev
        da_ft = dft * ft * (1 - ft)

        self.Wf.grad += da_ft @ z.T
        self.bf.grad += da_ft
        self.Wi.grad += da_it @ z.T
        self.bi.grad += da_it
        self.Wc.grad += da_c @ z.T
        self.bc.grad += da_c
        self.Wo.grad += da_ot @ z.T
        self.bo.grad += da_ot

        dz = (self.Wf.data.T @ da_ft) + (self.Wi.data.T @ da_it) + (self.Wc.data.T @ da_c) + (self.Wo.data.T @ da_ot)

        da_prev = dz[: n_a, :]
        dc_prev = ft * dc
        dx = dz[n_a :, :]

        return dx, da_prev, dc_prev

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def getParameters(self):
        return [self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo]

class LSTMLoss(Layer):
    def __init__(self):
        self.type = 'LSTM Loss'

    def forward(self, input, target):
        self.target = target
        loss = 0

        self.y_hat = {}

        for t in range(len(self.target)):
            self.y_hat[t] = self.softmax(input[t])
            loss += -np.log(self.y_hat[t][self.target[t], 0])
        return loss

    def backward(self):
        dy = {}
        for t in reversed(range(len(self.target))):
            dy[t] = np.copy(self.y_hat[t])
            dy[t][self.target[t]] -= 1   
        return dy

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

class LSTMDense(Layer):
    def __init__(self, input_size, hidden_size):
        self.Wy = Tensor((hidden_size, input_size))
        self.by = Tensor((hidden_size, 1))
        self.type = 'LSTM Dense'

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self, input_val):
        self.input_val = input_val
        y = {}
        for t in range(len(self.input_val)):
            y[t] = self.cell_forward(input_val[t])
        return y

    def cell_forward(self, input_val):
            return (self.Wy.data @ input_val) + self.by.data

    def backward(self, dy):
        da = {}
        for t in reversed(range(len(self.input_val))):
            self.Wy.grad += dy[t] @ self.input_val[t].T
            self.by.grad += dy[t]
            da[t] = self.Wy.data.T @ dy[t]
        return da
    
    def getParameters(self):
        return [self.Wy, self.by]




    







            
        