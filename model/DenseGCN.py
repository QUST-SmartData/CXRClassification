import math
import numpy as np
import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn

import config


def gen_A(num_classes, t):
    _adj = np.zeros((len(config.class_labels), len(config.class_labels)))
    labels = np.array(config.train_data.labels)
    for i in range(labels.shape[1]):
        for j in range(i + 1, labels.shape[1]):
            # 两个类别共现次数：两列同时为1的行数
            _adj[i, j] = len(np.where(labels[:, i] + labels[:, j] == 2)[0])
            _adj[j, i] = _adj[i, j]

    _nums = np.sum(config.train_data.labels, 0)
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int64)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, t=0.2):
        super(GCNResnet, self).__init__()
        self.features = model.features
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(7, 7)

        self.gc1 = GraphConvolution(300, 512)
        self.relu1 = nn.LeakyReLU(0.2)
        self.gc2 = GraphConvolution(512, 1024)
        self.relu2 = nn.LeakyReLU(0.2)
        self.gc3 = GraphConvolution(1024, 2048)
        self.relu3 = nn.LeakyReLU(0.2)
        self.gc4 = GraphConvolution(2048, 4096)
        self.relu4 = nn.LeakyReLU(0.2)
        self.gc5 = GraphConvolution(4096, 1920)

        # 初始化标签共现矩阵
        _adj = gen_A(num_classes, t)
        self.A = Parameter(torch.from_numpy(_adj).float())

        self.word2vec = torch.Tensor(np.loadtxt(f'./w2v/{config.datasets}.txt'))

    def forward(self, feature):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        # 计算邻接矩阵
        adj = gen_adj(self.A).detach()
        x = self.gc1(self.word2vec.to(adj.device), adj)
        x = self.relu1(x)
        x = self.gc2(x, adj)
        x = self.relu2(x)
        x = self.gc3(x, adj)
        x = self.relu3(x)
        x = self.gc4(x, adj)
        x = self.relu4(x)
        x = self.gc5(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x


def gcn_densenet201(pretrained=False, in_chans=3, num_classes=5, t=0.2):
    model = models.densenet201(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t)


if __name__ == '__main__':
    m = gcn_densenet201(num_classes=len(config.class_labels))
    x = torch.rand((2, 3, 224, 224))
    pred = m(x).sigmoid()
    print(pred)
