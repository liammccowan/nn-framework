import numpy as np
import time

class Model():
    def __init__(self):
        self.layers = []
        self.parameters = []

    def add(self, layer):
        self.layers.append(layer)
        self.parameters += layer.getParameters()

    def __initializeNetwork(self):
        for l in self.layers:
            if l.type == 'Linear':
                weights, biases = l.getParameters()
                weights.data = np.random.randn(weights.data.shape[0], weights.data.shape[1]) * np.sqrt(2.0 / weights.data.shape[0])
                biases.data = np.zeros((1, biases.data.shape[1]))
            if l.type == 'Convolutional':
                weights, biases = l.getParameters()
                weights.data = np.random.randn(weights.data.shape[0], weights.data.shape[1], weights.data.shape[2], weights.data.shape[3]) * np.sqrt(2.0 / (weights.data.shape[0] * weights.data.shape[2] * weights.data.shape[3]))
                biases.data = np.zeros((biases.data.shape[0]))
            if l.type == 'Batch Normalization':
                gamma, beta = l.getParameters()
                if len(gamma.data.shape) == 2:
                    gamma.data = np.ones((1, gamma.data.shape[1]))
                    beta.data = np.zeros((1, beta.data.shape[1]))
                else:
                    gamma.data = np.ones((1, gamma.data.shape[1], 1, 1))
                    beta.data = np.zeros((1, beta.data.shape[1], 1, 1))
            if l.type == 'LSTM':
                Wf, bf, Wi, bi, Wc, bc, Wo, bo = l.getParameters()
                n_h = Wo.data.shape[0]
                n_z = Wo.data.shape[1]
                std = np.sqrt(1.0 / Wo.data.shape[1])
                Wf.data = np.random.randn(n_h, n_z) * std
                bf.data = np.zeros((n_h, 1))
                Wi.data = np.random.randn(n_h, n_z) * std
                bi.data = np.zeros((n_h, 1))
                Wc.data = np.random.randn(n_h, n_z) * std
                bc.data = np.zeros((n_h, 1))
                Wo.data = np.random.randn(n_h, n_z) * std
                bo.data = np.zeros((n_h, 1))
            if l.type == 'LSTM Dense':
                Wy, by = l.getParameters()
                Wy.data = np.random.randn(Wy.data.shape[0], Wy.data.shape[1]) * np.sqrt(2.0 / Wy.data.shape[0])
                by.data = np.zeros((by.data.shape[0], 1))

    def predict(self, X, batch_size = 1):
        output = []
        for b in range(0, len(X), batch_size):
            x = X[b:b + batch_size]
            for i, _ in enumerate(self.layers):
                if self.layers[i].type == 'Dropout':
                    pass
                if self.layers[i].type == 'Batch Normalization':
                    forward = self.layers[i].forward_predict(x)
                    x = forward
                else:
                    forward = self.layers[i].forward(x)
                    x = forward
            output.append(x)
        return np.concatenate(output)

    def train(self, X_train, Y_train, batch_size, epochs, optimizer, loss_function, verbose = False):
        self.training_loss = []
        self.__initializeNetwork()
        data_gen = MinibatchGenerator(X_train, Y_train, batch_size)
        for epoch in range(epochs):
            start = time.time()
            loss_total = 0.0
            itr = 0
            for X, Y in data_gen:
                optimizer.zeroGradient()

                # forward pass
                for i, _ in enumerate(self.layers):
                    X = self.layers[i].forward(X)

                loss = loss_function.forward(X, Y)
                grad = loss_function.backward()
                loss_total += loss

                # backpropagation
                for i, _ in reversed(list(enumerate(self.layers))):
                    grad = self.layers[i].backward(grad)
                
                itr += 1

                optimizer.step()

            loss_average = loss_total / itr
            end = time.time()

            if verbose:
                print("Epoch: {}/{} | Average Loss: {:.4f} | Time: {:.2f} seconds".format(epoch + 1, epochs, loss_average, (end - start)))
            
            self.training_loss.append(loss_average)

    def trainLSTMsequence(self, data, ix_to_char, char_to_ix, optimizer, loss_function, epochs = 10, n_a = 50, seq_len = 100, vocab_size = 27, verbose = False, sample = False):
        self.training_loss = []
        self.vocab_size = vocab_size
        self.n_a = n_a
        self.seq_len = seq_len
        self.ix_to_char = ix_to_char
        self.char_to_ix = char_to_ix
        self.__initializeNetwork()

        loss = -np.log(1.0 / self.vocab_size) * self.seq_len
        num_batches = len(data) // self.seq_len
        data_trimmed = data[: num_batches * self.seq_len]

        a_prev = []
        c_prev = []

        for epoch in range(epochs):
            start = time.time()

            for i in range(len(self.n_a)):
                a_prev.append(np.zeros((self.n_a[i], 1)))
                c_prev.append(np.zeros((self.n_a[i], 1)))

            for j in range(0, len(data_trimmed) - self.seq_len, self.seq_len):
                optimizer.zeroGradient()

                x_batch = [self.char_to_ix[ch] for ch in data_trimmed[j: j + self.seq_len]]
                y_batch = [self.char_to_ix[ch] for ch in data_trimmed[j + 1: j + self.seq_len + 1]]

                # one-hot encode x_batch
                input = {}
                for t in range(self.seq_len):
                    input[t] = np.zeros((self.vocab_size, 1))
                    if (x_batch[t] != None):
                        input[t][x_batch[t]] = 1

                for i, _ in enumerate(self.layers):
                    if self.layers[i].type == 'LSTM Dense':
                        x = self.layers[i].forward(input)
                    else:
                        a_next_layer = self.layers[i].forward(input, a_prev[i], c_prev[i])
                        input = a_next_layer

                current_loss = loss_function.forward(x, y_batch)
                loss = self.smooth_loss(loss, current_loss)
                self.training_loss.append(loss)
                grad = loss_function.backward()

                for i, _ in reversed(list(enumerate(self.layers))):
                    if self.layers[i].type == 'LSTM Dense':
                        grad = self.layers[i].backward(grad)
                    else:
                        grad, a_prev[i], c_prev[i] = self.layers[i].backward(grad)
                        
                optimizer.clipGradient(5)
                optimizer.step()

            end = time.time()

            if verbose:
                print("Epoch: {}/{} | Loss: {:.4f} | Time: {:.2f}".format((epoch + 1), epochs, loss, (end - start)))
                if sample:
                    self.sampleLSTM(prompt = False)
        return

    def sampleLSTM(self, max_sample_size = 500, prompt = False):
        if prompt:
            input_text = input("Write the beginning of your text: ")
            input_text = input_text.lower()
            input_idx = [self.char_to_ix[ch] for ch in input_text]
            user_input = {}
        else:
            input_val = np.zeros((self.vocab_size, 1))
        
        a = []
        c = []
        idx = -1
        counter = 0
        sample_string = ""

        newline_character = self.char_to_ix['\n']

        for i in range(len(self.n_a)):
                a.append(np.zeros((self.n_a[i], 1)))
                c.append(np.zeros((self.n_a[i], 1)))

        if prompt:
            for t in range(len(input_idx)):
                user_input[t] = np.zeros((self.vocab_size, 1))
                if input_idx[t] != None:
                    user_input[t][input_idx[t]] = 1

                for i, _ in enumerate(self.layers):
                    if self.layers[i].type == 'LSTM Dense':
                        input_val = self.layers[i].cell_forward(user_input[t])
                    else:
                        _, _, _, _, c[i], _, a[i] = self.layers[i].cell_forward(user_input[t], a[i], c[i])
                        user_input[t] = a[i]

                idx = input_idx[t]
                char = self.ix_to_char[idx]
                sample_string += char

            e_x = np.exp(input_val - np.max(input_val))
            y_hat = e_x / np.sum(e_x)
            idx = np.argmax(y_hat)
            input_val = np.zeros((self.vocab_size, 1))
            input_val[idx] = 1

        while (idx != newline_character and counter != max_sample_size):  
            for i, _ in enumerate(self.layers):
                if self.layers[i].type == 'LSTM Dense':
                    input_val = self.layers[i].cell_forward(input_val)
                else:
                    _, _, _, _, c[i], _, a[i] = self.layers[i].cell_forward(input_val, a[i], c[i])
                    input_val = a[i]

                e_x = np.exp(input_val - np.max(input_val))
                y_hat = e_x / np.sum(e_x)

            idx = np.random.choice(range(self.vocab_size), p = y_hat.ravel())
            input_val = np.zeros((self.vocab_size, 1))
            input_val[idx] = 1

            char = self.ix_to_char[idx]
            sample_string += char
            counter += 1

        print(sample_string)
        
        return

    def trainLSTMpredict(self, x_train, y_train, optimizer, loss_function, epochs = 10, n_a = 50, seq_len = 100, vocab_size = 27, verbose = False):
        self.training_loss = []
        self.vocab_size = vocab_size
        self.n_a = n_a
        self.seq_len = seq_len
        self.__initializeNetwork()

        loss = -np.log(1.0 / self.vocab_size) * self.seq_len

        a_prev = []
        c_prev = []

        for epoch in range(epochs):
            start = time.time()

            for i in range(len(self.n_a)):
                a_prev.append(np.zeros((self.n_a[i], 1)))
                c_prev.append(np.zeros((self.n_a[i], 1)))

            for j in range(0, len(x_train)):
                optimizer.zeroGradient()

                x_batch = x_train[j].squeeze(axis = 0)
                y_batch = y_train[j]

                input = {}
                for t in range(len(x_batch)):
                    input[t] = np.expand_dims(x_batch[t, :], axis = 1)

                y = np.expand_dims(y_batch, axis = 1)

                for i, _ in enumerate(self.layers):
                    if self.layers[i].type == 'Linear':
                        x = self.layers[i].forward(input.T)
                    else:
                        a_next_layer = self.layers[i].forward(input, a_prev[i], c_prev[i])
                        input = a_next_layer

                current_loss = loss_function.forward(x, y.T)
                loss = self.smooth_loss(loss, current_loss)
                self.training_loss.append(loss)
                grad = loss_function.backward()

                for i, _ in reversed(list(enumerate(self.layers))):
                    if self.layers[i].type == 'Linear':
                        grad = self.layers[i].backward(grad)
                    else:
                        grad, a_prev[i], c_prev[i] = self.layers[i].backward(grad)
                        
                optimizer.clipGradient(5)
                optimizer.step()

            end = time.time()

            if verbose:
                print("Epoch: {}/{} | Loss: {:.4f} | Time: {:.2f}".format((epoch + 1), epochs, loss, (end - start)))

        return
        
    def LSTMpredict(self, X):
        output = []
        a_prev = []
        c_prev = []

        for i in range(len(self.n_a)):
            a_prev.append(np.zeros((self.n_a[i], 1)))
            c_prev.append(np.zeros((self.n_a[i], 1)))

        for j in range(0, len(X)):

            x_batch = X[j].squeeze(axis = 0)

            input = {}
            for t in range(len(x_batch)):
                input[t] = np.expand_dims(x_batch[t, :], axis = 1)

            for i, _ in enumerate(self.layers):
                if self.layers[i].type == 'Linear':
                    x = self.layers[i].forward(input.T)
                else:
                    a_next_layer = self.layers[i].forward(input, a_prev[i], c_prev[i])
                    input = a_next_layer

            e_x = np.exp(x - np.max(x))
            y_hat = e_x / np.sum(e_x)

            output.append(y_hat)

        return np.concatenate(output)

    def smooth_loss(self, loss, current_loss):
        return loss * 0.999 + current_loss * 0.001

class MinibatchGenerator():
    def __init__(self, data, target, batch_size, shuffle = True):
        self.shuffle = shuffle
        if shuffle:
            shuffled_indices = np.random.permutation(len(data))
        else:
            shuffled_indices = range(len(data))

        self.data = data[shuffled_indices]
        self.target = target[shuffled_indices]
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(data.shape[0] / batch_size))
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.num_batches:
            batch_data = self.data[self.counter * self.batch_size:(self.counter + 1) * self.batch_size]
            batch_target = self.target[self.counter * self.batch_size:(self.counter + 1) * self.batch_size]
            self.counter += 1
            return batch_data, batch_target
        else:
            if self.shuffle:
                shuffled_indices = np.random.permutation(len(self.target))
            else:
                shuffled_indices = range(len(self.target))

            self.data = self.data[shuffled_indices]
            self.target = self.target[shuffled_indices]

            self.counter = 0
            raise StopIteration
