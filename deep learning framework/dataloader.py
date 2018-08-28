# encoding=utf-8
import numpy as np

train_eval_rate = 0.8

class DataMaster(object):
    def __init__(self):
        self.datasets = np.load('./data/protein_matrix.npy')
        self.dataembs = np.load('./data/protein_emb.npy')
        self.datalocs = np.load('./data/protein_loc.npy')
        self.datalabels = np.load('./data/protein_label.npy')
        self.training_size = int(train_eval_rate * len(self.datasets))
        self.train_X = self.datasets[:self.training_size]
        self.train_E = self.dataembs[:self.training_size]
        self.train_L = self.datalocs[:self.training_size]
        self.train_Y = self.datalabels[:self.training_size]
        print("training data numbers:%d" % self.training_size)
        self.test_X = self.datasets[self.training_size:]
        self.test_E = self.dataembs[self.training_size:]
        self.test_L = self.datalocs[self.training_size:]
        self.test_Y = self.datalabels[self.training_size:]
        self.test_size = len(self.test_X)
        print("test data numbers:%d" % self.test_size)

    def shuffle(self):
        mark = list(range(self.training_size))
        np.random.shuffle(mark)
        self.train_X = self.train_X[mark]
        self.train_E = self.train_E[mark]
        self.train_L = self.train_L[mark]
        self.train_Y = self.train_Y[mark]


if __name__ == '__main__':
    DataMaster()
