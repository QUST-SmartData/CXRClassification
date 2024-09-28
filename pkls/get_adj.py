import pickle

import numpy as np

import config

if __name__ == '__main__':
    _adj = np.zeros((len(config.class_labels), len(config.class_labels)), dtype=np.int64)
    labels = np.array(config.train_data.labels)
    for i in range(labels.shape[1]):
        for j in range(i + 1, labels.shape[1]):
            # 两个类别共现次数：两列同时为1的行数
            _adj[i, j] = len(np.where(labels[:, i] + labels[:, j] == 2)[0])
            _adj[j, i] = _adj[i, j]

    _nums = np.squeeze(np.sum(config.train_data.labels, 0)).astype(np.int64)

    result = {'nums': _nums, 'adj': _adj}

    pickle.dump(result, open(f'./pkls/{config.datasets}_adj.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
