import os
import torch
import matplotlib.pyplot as plt


class GRNN():
    def __init__(self, trainx, trainy, sigma=1):
        """
        Parameters
        ----------
        Z : tensor, shape (T, N)
            The measurement matrix which is used to calculate the
            models, where N means the feature dimensions of the
            measurement. In BCI application, it should be the X's
            corresponding neural data.
        X : tensor, shape (T, M)
            The state matrix which is used to calculate the models,
            where T means the time steps and M means the feature
            dimensions. In the BCI application, it can be the
            movement or other behavior data.
        """
        self.trainx = trainx
        self.trainy = trainy
        self.sigma = sigma

    def predict(self, testx):
        hdn = self.hdn_layer(self.trainx, testx, sigma=self.sigma)
        sum = self.sum_layer(hdn, self.trainy)
        out = self.out_layer(sum)
        return out

    def hdn_layer(self, trainx, testx, sigma=1):
        sum_testx2 = torch.sum(testx ** 2, dim=1, keepdim=True)
        sum_trainx2 = torch.sum(trainx ** 2, dim=1, keepdim=True)
        dist = torch.sqrt(sum_testx2 + sum_trainx2.T -
                          2 * testx.matmul(trainx.T))
        out = torch.exp(-dist / (2 * sigma ** 2))
        return out

    def sum_layer(self, out_hdn, trainy):
        sum1 = torch.sum(out_hdn, dim=-1, keepdim=True)
        sum2 = torch.matmul(out_hdn, trainy)
        return torch.cat((sum1, sum2), dim=-1)

    def out_layer(self, out_sum):
        return out_sum[:, 1:] / out_sum[:, :1]


if __name__ == '__main__':
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(
        os.path.join(cur_path, '..', 'Test/fitting/grnn.txt')
    )
    with open(data_path) as f:
        data = []
        for line in f.readlines():
            lines = [float(i) for i in line.strip().split('\t')]
            data.append(lines)
        data = torch.tensor(data)

    trainx, testx = data[:190, [0]], data[190:, [0]]
    trainy, testy = data[:190, [1]], data[190:, [1]]
    grnn = GRNN(trainx, trainy, sigma=0.1)
    out = grnn.predict(testx)

    plt.plot(out)
    plt.plot(testy)
    plt.legend(['prediction', 'actual'])
    plt.show()
