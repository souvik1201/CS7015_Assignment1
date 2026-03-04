import pandas as pd
import numpy as np


def normalize(x, mean = 0, std = 1):
    x = x - mean
    x = x / std
    return x


def one_hot_encoding(y, n_classes):
    arr = np.zeros((len(y), n_classes))
    for i in range(len(y)):
        arr[i][y[i]] = 1.0
    return arr


class LoadData(object):
    x = []
    y = []
    n_classes = 0
    def __init__(self, file, n_classes):
        self.n_classes = n_classes
        df = pd.read_csv(file)
        df = df.to_numpy()
        x, y = df[:, 1 : -1], df[:, -1]
        y = y.reshape(-1, 1)
        y = one_hot_encoding(y, n_classes)
        self.x = normalize(x)
        self.y = y

    def xshape(self):
        return self.x.shape
    def yshape(self):
        return self.y.shape