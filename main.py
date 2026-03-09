"""
CS7015: A course on deep learning, assignment 3
Authos: Souvik
"""
import numpy as np
import pandas as pd
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
    parser.add_argument("--activation", help = "Activation function", default = "sigmoid")
    parser.add_argument("--loss", help = "Loss function", default = "sq")
    parser.add_argument("--opt", help = "Optimization algorithm", default = "gd")
    parser.add_argument("--batch_size", help = "Batch size", type = int, default = 20)
    parser.add_argument("--epochs", help = "number of epochs", type = int, default = 10)
    parser.add_argument("--anneal", help = "if true the algorithm should halve the learning rate if at any epoch the validation loss decreases and then restart that epoch")
    parser.add_argument("--save_dir", help = "the directory in which the pickled model should be saved")
    parser.add_argument("--expt_dir", help = "the directory in which the log files will be saved")
    parser.add_argument("--train", help = "path to the Training dataset")
    parser.add_argument("--test", help = "path to the Test dataset")
    args = vars(parser.parse_args())
    print(args)
    n_classes = 10
    Traindata = ld.LoadData(args['train'], n_classes)
    Valdata = ld.LoadData(r"C:\Users\souvi\PycharmProjects\Data\val.csv", n_classes)
    sizes = args['sizes'].split(',')
    size = list(map(int, sizes))
    NN = NeuralNetwork.NeuralNetwork(784, 10, size, args['activation'], args['loss'])
    #NN.gradient_descent(Traindata.x, Traindata.y, Valdata.x, Valdata.y, args["lr"], args['loss'], args['batch_size'])
    NN.momemtum_gradient_descent(Traindata.x, Traindata.y, Valdata.x, Valdata.y, args["lr"], args['momentum'], args['loss'], args['batch_size'])
    Testdata = pd.read_csv(args['test'])
    Testdata = Testdata.to_numpy()
    image_id = Testdata[:, 0:1]
    Testdata = Testdata[:, 1:]
    TestPrediction = []
    for image, x_test in zip(image_id, Testdata):
        yhat, a, h = NN.forward_propagation(x_test)
        TestPrediction.append([np.argmax(yhat)])
    submit = pd.DataFrame(columns = ["id", "label"])
    submit['id'] = image_id
    submit['label'] = TestPrediction
    result = submit.sort_values(by=['label'], ascending=False)
    result.to_csv(args['expt_dir'] + 'result.csv', index = False, sep=',')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
