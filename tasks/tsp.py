"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

class TSPDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=None):
        super(TSPDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(1234)

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset = torch.rand((num_samples, 2, size))  # 坐标
        # self.dataset[:, :, -1] = self.dataset[:, :, 0]
        self.dynamic = torch.zeros(num_samples, 1, size)  # 需求？（0）
        self.num_nodes = size
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    # 将mask的第chosen_idx.unsqueeze(1)参数改成0，即将已经访问过的城市的mask设为0，
    # 这样mask.log()负无穷，在进行softmax后为0，即选择概率为0
    return mask

def update_dynamic(dis_matrix, ptr ):
    """动态更新dynamic：当前节点下其余节点与本节点的距离"""
    batch_size, sequence_size, _ = dis_matrix.shape
    # 用矩阵运算，不要用for
    Index = torch.from_numpy(
        sequence_size * sequence_size * np.arange(batch_size).reshape((batch_size, 1)) * np.ones(
            [1, sequence_size]) + sequence_size * ptr.cpu().numpy().reshape(
            (batch_size, 1)) * np.ones([1, sequence_size]) + np.ones([batch_size, sequence_size]) * np.arange(sequence_size)).long()
    temp = torch.take(dis_matrix.cpu(), Index)
    temp_max = torch.max(temp, 1).values  # 按行求max
    temp_min = torch.min(temp, 1).values  # 按行求min

    # 最大最小归一化
    dynamic = ((temp_max.unsqueeze(1)*torch.ones(1, sequence_size)-temp)/(temp_max.unsqueeze(1)*torch.ones(1, sequence_size)-temp_min.unsqueeze(1)*torch.ones(1, sequence_size))).unsqueeze(1).to('cuda').float()

    return dynamic

def reward(static, tour_indices):

    #计算给定策略的reward，此处是欧氏距离总和
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour

    idx = tour_indices.unsqueeze(1).expand_as(static[:, :, :tour_indices.shape[-1]])
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)  # 在路径后再加入起点
    # y = tour

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()


def render(static, tour_indices, save_path):
    """Plots the found tours."""
    # 画出结果路线图

    plt.close('all')

    num_plots = 1 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    # location = static[0, :, :].cpu().numpy()
    # plt.scatter(location[0], location[1], s=10)

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)  # 拓展一维
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()  # 坐标

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], color='red', linewidth=0.5)  # 画出点

        # for j in range(data.shape[1]-1):
        #     ax.text(data[0, j], data[1, j], '%.0f' % idx[0, j].tolist(), fontdict={'fontsize': 8})
        # ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        # ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)
        ax.scatter(static[0, 0, :], static[0, 1, :], marker='o', color='green')
        # ax.scatter(data[0], data[1], marker='o', color='green')
        ax.text(data[0,0], data[1,0], 'Depot')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


    plt.tight_layout()
    if os.path.exists(save_path):
        plt.savefig(save_path.format(int(time.time())))
    else:
        plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close('all')

