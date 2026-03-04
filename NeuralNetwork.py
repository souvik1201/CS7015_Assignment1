"""
CS7015: A course on deep learning, assignment 3Authos: Souvik
"""
import numpy as np

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
    def __init__(self, Input, output, size, g, loss_function):
        self.layers = len(size)
        self.neurons = [Input] + size + [output]
        self.activations = [g] * (self.layers + 2)
        self.weights = [np.random.randn(y, x)* 2/np.sqrt(x+y) for x,y in zip(self.neurons[:-1], self.neurons[1:])]
        self.biases = [np.zeros((x, 1)) for x in self.neurons[1:]]
        self.loss_function = loss_function


    def sigmoid(self, x, derivative = False):
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
            return -1 * y * np.log(x)
        return None

    def activate(self, z, activation):
        if activation == 'sigmoid':
            return self.sigmoid(z)
        elif activation == 'tanh':
            return self.tanh(z)
        return None

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
        x = x.reshape(1, -1)
        h.append(x.T)
        # layer = 3 -> layer + 2 = 5 -> loop 1, 2, 3, 4
        for k in range(1, self.layers + 2):
            # Pre-activation
            # print(h[k - 1].shape)
            assert W[k - 1].shape[1] == h[k - 1].shape[0]
            a.append(np.matmul(W[k - 1], h[k - 1]) + b[k - 1])
            #Storing Zs
            self.layerInput.append(a[k - 1])
            # print(a[k - 1].shape)
            # Activation
            h.append(self.activate(a[k - 1], self.activations[k - 1]))
            # print(h[k - 1].shape)
            #Storing As
            # print("Actual h = %s as per layer = %s \n", h[k - 1], k)
            self.layerOutput.append(h[k])
        yhat = self.layerOutput[-1]
        # print(len(yhat))
        return yhat, a, h

    def back_propagation(self, h, a, loss_function, y, yhat, W, activation):
        grad_a = [np.zeros((self.neurons[i], 1)) for i in range(len(self.neurons))]
        grad_h = [np.zeros((self.neurons[i], 1)) for i in range(self.layers + 1)]
        grad_w = [np.zeros((y, x)) for x,y in zip(self.neurons[:-1], self.neurons[1:])]
        grad_b = [np.zeros((x, 1)) for x in self.neurons[1:]]
        error = 0
        L = len(self.neurons) - 1
        if loss_function.lower() == 'sq':
            y = y.reshape(-1, 1)
            grad_a[L] =  yhat - y
            error = self.loss(y, yhat)
        for layer in reversed(range(L)):
            grad_w[layer] = np.dot(grad_a[layer + 1], h[layer].T)
            grad_b[layer] = grad_a[layer + 1]
            grad_h[layer] = np.dot(W[layer].T, grad_a[layer + 1])
            gdash = 0
            if layer > 0:
                if activation == 'sigmoid':
                    gdash = self.sigmoid(a[layer - 1], True)
                elif activation == 'tanh':
                    gdash = self.tanh(a[layer - 1], True)
                grad_a[layer] = grad_h[layer] * gdash
        return grad_w, grad_b, error

    def gradient_descent(self, X, Y, eta, loss_function):
        epoch = 0
        w = self.weights
        b = self.biases
        w_old = w
        b_old = b
        grad_ws = []
        grad_bs = []
        max_iter = 1000
        while epoch < max_iter:
            loss = 0
            for x, y in zip(X, Y):
                yhat, a, h = self.forward_propagation(x)
                grad_ws, grad_bs, loss = self.back_propagation(h, a, loss_function, y, yhat, w, activation = 'sigmoid')
            for layer, (grad_w, grad_b) in enumerate(zip(grad_ws, grad_bs)):
                w[layer] = w[layer] - eta * grad_w
                b[layer] = b[layer] - eta * grad_b
                w_old[layer] = w[layer]
                b_old[layer] = b[layer]
            print("Update for epoch %s completed with loss = %s" %(epoch, loss))
            epoch = epoch + 1