"""
CS7015: A course on deep learning, assignment 3Authos: Souvik
"""
import numpy as np
from sklearn.metrics import log_loss

class NeuralNetwork(object):
    layers = 0
    neurons = []
    weights = np.array([])
    biases = []
    activations = []
    a = []
    h = []
    loss_function = ''
    layerInput = []
    layerOutput = []
    grad_a = []
    grad_h = []
    grad_w = []
    grad_b = []
    def __init__(self, Input, output, size, g, loss_function):
        self.neurons = [Input] + size + [output]
        self.layers = len(self.neurons) - 1
        self.activations = [g] * (self.layers + 2)
        self.weights = [np.random.randn(y, x)* 2/np.sqrt(x+y) for x,y in zip(self.neurons[:-1], self.neurons[1:])]
        self.biases = [np.zeros((x, 1)) for x in self.neurons[1:]]
        self.loss_function = loss_function
        self.grad_a = [np.zeros((self.neurons[i], 1)) for i in range(len(self.neurons))]
        self.grad_h = [np.zeros((n, 1)) for n in self.neurons]
        self.grad_w = [np.zeros((y, x)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
        self.grad_b = [np.zeros((x, 1)) for x in self.neurons[1:]]


    def sigmoid(self, x, derivative = False):
        x = np.clip(x, -500, 500)
        if not derivative:
            return 1 / (1 + np.exp(-x))
        else:
            sigma = self.sigmoid(x)
            return sigma * (1- sigma)


    def tanh(self, x, derivative = False):
        if not derivative:
            return np.tanh(x)
        else:
            tanh = self.tanh(x)
            return 1 - np.power(tanh,2)

    def loss(self, x, y):
        if self.loss_function.lower() == 'sq':
            return np.sum(np.square(x - y))
        elif self.loss_function.lower() == 'ce':
            return -np.sum(y * np.log(x + 1e-9))
        return None

    def activate(self, z, activation):
        if activation == 'sigmoid':
            return self.sigmoid(z)
        elif activation == 'tanh':
            return self.tanh(z)
        return None

    def softmax(self, x):
        x = x - np.max(x)
        numer = np.exp(x)
        denom = np.sum(numer, axis = 0)
        return numer / denom

    def forward_propagation(self, x):
        self.layerInput = []
        self.layerOutput = []
        W = self.weights
        b = self.biases
        # n = x.shape[0]
        # dim = x.shape[1]
        h = []
        a = []
        # a.append(x[0:1, :].T)
        x = x.reshape(-1, 1)
        a.append(x)
        h.append(x)
        # layer = 3 -> layer + 2 = 5 -> loop 1, 2, 3, 4
        for k in range(self.layers):
            # Pre-activation
            # print(h[k - 1].shape)
            assert W[k].shape[1] == h[k].shape[0]
            a.append(np.matmul(W[k], h[k]) + b[k])
            #Storing Zs
            self.layerInput.append(a[k])
            # print(a[k - 1].shape)
            # Activation
            if k != self.layers - 1:
                h.append(self.activate(a[k + 1], self.activations[k]))
            else:
                h.append(self.softmax(a[k + 1]))
            # print(h[k - 1].shape)
            #Storing As
            # print("Actual h = %s as per layer = %s \n", h[k - 1], k)
            self.layerOutput.append(h[k + 1])
        yhat = self.layerOutput[-1]
        # print(len(yhat))
        return yhat, a, h

    def back_propagation(self, h, a, loss_function, y, yhat, W, activation):
        error = 0
        L = len(self.neurons) - 1
        if loss_function.lower() == 'ce':
            y = y.reshape(-1, 1)
            self.grad_a[L] =  yhat - y
            error = self.loss(yhat, y)
        # h and a is following 1 indexing
        for layer in reversed(range(L)):
            self.grad_w[layer] = np.dot(self.grad_a[layer + 1], h[layer].T)
            self.grad_b[layer] = self.grad_a[layer + 1]
            self.grad_h[layer] = np.dot(W[layer].T, self.grad_a[layer + 1])
            gdash = 0
            if activation == 'sigmoid':
                gdash = self.sigmoid(a[layer], True)
            elif activation == 'tanh':
                gdash = self.tanh(a[layer], True)
            self.grad_a[layer] = self.grad_h[layer] * gdash
        return self.grad_w, self.grad_b

    def gradient_descent(self, X, Y, X_val, Y_val, eta, loss_function, batch):
        epoch = 0
        w = self.weights
        b = self.biases
        max_iter = 100
        loss = 0
        while epoch < max_iter:
            grad_ws = [np.zeros((y, x)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
            grad_bs = [np.zeros((x, 1)) for x in self.neurons[1:]]
            current_batch = 0
            accuracy = 0
            for x, y in zip(X, Y):
                yhat, a, h = self.forward_propagation(x)
                grad_w, grad_b = self.back_propagation(h, a, loss_function, y, yhat, w, activation = 'sigmoid')
                current_batch += 1
                for i in range(len(grad_ws)):
                    grad_ws[i] += grad_w[i]
                    grad_bs[i] += grad_b[i]
                if current_batch == batch:
                    current_batch = 0
                    for i in range(len(self.weights)):
                        self.weights[i] -= eta * grad_ws[i]/batch
                        self.biases[i] -= eta * grad_bs[i]/batch
                    grad_ws = [np.zeros((y, x)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
                    grad_bs = [np.zeros((x, 1)) for x in self.neurons[1:]]
            loss = 0
            correct = 0
            for x_val, y_val in zip(X_val, Y_val):
                yhat_val, a_val, h_val = self.forward_propagation(x_val)
                loss += log_loss(y_val, yhat_val)
                if np.argmax(yhat_val) == np.argmax(y_val):
                    correct += 1
            accuracy = correct / float(len(X_val)) * 100
            loss /= float(len(X_val))
            print("Update for epoch %s completed with loss = %s with accurecy = %s" %(epoch + 1, loss, accuracy))
            epoch = epoch + 1

    def momemtum_gradient_descent(self, X, Y, X_val, Y_val, eta, gamma, loss_function, batch):
        epoch = 0
        w = self.weights
        b = self.biases
        max_iter = 10
        w_update = [np.zeros((y, x)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
        b_update = [np.zeros((x, 1)) for x in self.neurons[1:]]
        while epoch < max_iter:
            grad_ws = [np.zeros((y, x)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
            grad_bs = [np.zeros((x, 1)) for x in self.neurons[1:]]
            current_batch = 0
            accuracy = 0
            for x, y in zip(X, Y):
                yhat, a, h = self.forward_propagation(x)
                grad_w, grad_b = self.back_propagation(h, a, loss_function, y, yhat, w, activation = 'sigmoid')
                current_batch += 1
                for i in range(len(grad_ws)):
                    grad_ws[i] += grad_w[i]
                    grad_bs[i] += grad_b[i]
                if current_batch == batch:
                    current_batch = 0
                    for i in range(len(self.weights)):
                        w_update[i] = gamma * w_update[i] + eta * grad_ws[i] / batch
                        b_update[i] = gamma * b_update[i] + eta * grad_bs[i] / batch
                        self.weights[i] = self.weights[i] - w_update[i]
                        self.biases[i] = self.biases[i] - b_update[i]
                    grad_ws = [np.zeros((y, x)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
                    grad_bs = [np.zeros((x, 1)) for x in self.neurons[1:]]
            loss = 0
            correct = 0
            for x_val, y_val in zip(X_val, Y_val):
                yhat_val, a_val, h_val = self.forward_propagation(x_val)
                loss += log_loss(y_val, yhat_val)
                if np.argmax(yhat_val) == np.argmax(y_val):
                    correct += 1
            accuracy = correct / float(len(X_val)) * 100
            loss /= float(len(X_val))
            print("Update for epoch %s completed with loss = %s with accurecy = %s" %(epoch + 1, loss, accuracy))
            epoch = epoch + 1

    def nag(self, X, Y, X_val, Y_val, eta, gamma, loss_function, batch):
        epoch = 0
        w = self.weights
        b = self.biases
        max_iter = 100
        w_update = [np.zeros((y, x)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
        b_update = [np.zeros((x, 1)) for x in self.neurons[1:]]
        while epoch < max_iter:
            grad_ws = [np.zeros((y, x)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
            grad_bs = [np.zeros((x, 1)) for x in self.neurons[1:]]
            current_batch = 0
            accuracy = 0
            for x, y in zip(X, Y):
                original_w = [w.copy() for w in self.weights]
                original_b = [b.copy() for b in self.biases]
                for i in range(len(original_w)):
                    self.weights[i] = self.weights[i] - gamma * w_update[i]
                    self.biases[i] = self.biases[i] - gamma * b_update[i]
                yhat, a, h = self.forward_propagation(x)
                grad_w, grad_b = self.back_propagation(h, a, loss_function, y, yhat, w, activation = 'sigmoid')
                current_batch += 1
                for i in range(len(grad_ws)):
                    grad_ws[i] += grad_w[i]
                    grad_bs[i] += grad_b[i]
                self.weights = original_w
                self.biases = original_b
                if current_batch == batch:
                    current_batch = 0
                    for i in range(len(self.weights)):
                        w_update[i] = gamma * w_update[i] + eta * grad_ws[i] / batch
                        b_update[i] = gamma * b_update[i] + eta * grad_bs[i] / batch
                        self.weights[i] = self.weights[i] - w_update[i]
                        self.biases[i] = self.biases[i] - b_update[i]
                    grad_ws = [np.zeros((y, x)) for x, y in zip(self.neurons[:-1], self.neurons[1:])]
                    grad_bs = [np.zeros((x, 1)) for x in self.neurons[1:]]
            loss = 0
            correct = 0
            for x_val, y_val in zip(X_val, Y_val):
                yhat_val, a_val, h_val = self.forward_propagation(x_val)
                loss += log_loss(y_val, yhat_val)
                if np.argmax(yhat_val) == np.argmax(y_val):
                    correct += 1
            accuracy = correct / float(len(X_val)) * 100
            loss /= float(len(X_val))
            print("Update for epoch %s completed with loss = %s with accurecy = %s" %(epoch + 1, loss, accuracy))
            epoch = epoch + 1

    def predictionlable(self, y):
        return y.index(max(y))
    def test(self, X , Y):
        W = self.weights
        b = self.biases
        correct = 0
        for x, y in zip(X, Y):
            h = []
            a = []
            layerInput = []
            layerOutput = []
            x = x.reshape(-1, 1)
            h.append(x)
            # layer = 3 -> layer + 2 = 5 -> loop 1, 2, 3, 4
            for k in range(self.layers):
                # Pre-activation
                assert W[k].shape[1] == h[k].shape[0]
                a.append(np.matmul(W[k], h[k]) + b[k])
                # Storing Zs
                layerInput.append(a[k])
                # Activation
                h.append(self.activate(a[k], self.activations[k]))
                layerOutput.append(h[k])
            yhat = layerOutput[-1]
            if self.predictionlable(yhat) == self.predictionlable(y):
                correct += 1

        return correct / float(len(X))