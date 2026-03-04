"""
CS7015: A course on deep learning, assignment 3
Authos: Souvik
"""
import numpy as np

import NeuralNetwork
import NeuralNetwork as nn
import LoadData as ld
import argparse

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read command line argument.")
    parser.add_argument("--lr", help = "Leartning rate", type = float, default = 0.001)
    parser.add_argument("--momentum", help = "Value of gamma", type = float, default = 0.5)
    parser.add_argument("--num_hidden", help = "Total number of hidden layer", type = int, default = 3)
    parser.add_argument("--sizes", help = "Size of the each hidden layes")
    parser.add_argument("--activation", help = "Activation function", default = "relu")
    parser.add_argument("--loss", help = "Loss function", default = "sq")
    parser.add_argument("--opt", help = "Optimization algorithm", default = "gd")
    parser.add_argument("--batch_size", help = "Batch size", type = int, default = 20)
    parser.add_argument("--epochs", help = "number of epochs", type = int, default = 10)
    parser.add_argument("--anneal", help = "if true the algorithm should halve the learning rate if at any epoch the validation loss decreases and then restart that epoch")
    parser.add_argument("--save_dir", help = "the directory in which the pickled model should be saved")
    parser.add_argument("--expt_dir", help = "the directory in which the log files will be saved")
    parser.add_argument("--train", help = "path to the Training dataset")
    parser.add_argument("--test", help = "path to the Test dataset")
    arg = parser.parse_args()
    n_classes = 10
    train = r"C:\Users\souvi\PycharmProjects\Data\train.csv"
    data = ld.LoadData(train, n_classes)
    NN = NeuralNetwork.NeuralNetwork(784, 10, [100, 100, 100], "sigmoid", "sq")
    NN.gradient_descent(data.x, data.y, 0.5, "sq")
    print(data.xshape())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
