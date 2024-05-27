"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None 这个是更新mask(掩码，用于按人的意愿改变每次选择节点的偏好)的函数
* update_fn: can be None 这是每次更新动态参数（dynamic）的函数
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import os.path
from pathlib import Path
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tasks import tsp
from model import DRL4TSP, Encoder
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StateCritic(nn.Module):
    """Estimates the problem complexity.
    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    # 这个函数直接将原始输入数据encoder后的向量做critic了？
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()
        self.static_encoder = Encoder(static_size, hidden_size)  # embedding层输出静态输入数据的的一维数组
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)  # embedding层输出动态输入数据的的一维数组
        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        # nn.Conv1d一维卷积，只对宽度进行卷积，对高度不卷积
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):
        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        hidden = torch.cat((static_hidden, dynamic_hidden), 1)
        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.
    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()
        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = F.relu(self.fc1(input.unsqueeze(1)))
        # unsqueeze()用来增加1维度；squeeze()减少1维度
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5, GAgen=None, return_mean=True, if_test=False):
    """Used to monitor progress on a validation set & optionally plot solution."""
    actor.eval()
    rewards = []
    for batch_idx, batch in enumerate(data_loader):
        static, dynamic, x0 = batch
        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None
        with torch.no_grad():
            tour_indices, tour_logp = actor(static, dynamic, x0, if_test)
        reward_no_mean = reward_fn(static, tour_indices)
        reward_mean = reward_no_mean.mean().item()
        rewards.append(reward_mean)
    # actor.train()
    if return_mean:
        return np.mean(rewards), tour_indices
    else:
        return reward_no_mean, tour_indices

def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, save_dir=None,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    epoch_num = 10
    save_dir = os.path.join(task, '%d' % num_nodes,
                            'seed1234_Bacth' + str(batch_size) + '_Ts' + str(int(train_data.size / 10000)) + 'w_e' + str(
                                epoch_num)) if save_dir is None else save_dir
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)  # 将100,000,0个样本划分为3097个256的batch
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)
    best_reward = np.inf

    for epoch in range(epoch_num):
        print('epoch:', epoch, '\n')
        actor.train()
        critic.train()
        times, losses, rewards, critic_rewards = [], [], [], []
        epoch_start = time.time()
        start = epoch_start
        for batch_idx, batch in enumerate(train_loader):
            static, dynamic, x0 = batch
            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None
            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)
            reward = reward_fn(static, tour_indices)
            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)  # critic估算的reward
            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            # with torch.autograd.detect_anomaly():
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()
            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end
                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))
        # 写入excel
        dic_log = {'loss_list': pd.Series(losses),
                   'reward_list': pd.Series(rewards)}
        dic_log_df = pd.DataFrame(dic_log)
        data_dir = os.path.join(save_dir, 'train_process.xlsx')
        my_file = Path(data_dir)
        if my_file.is_file():
            df = pd.read_excel(data_dir)
            df_new = df.append(dic_log_df, ignore_index=True)  # 添加新的epoch数据
            df_new.to_excel(data_dir, index=False, engine="openpyxl")
        else:
            dic_log_df.to_excel(data_dir, index=False)

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)
        [mean_valid, _] = validate(valid_loader, actor, reward_fn, render_fn,
                                   valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:
            best_reward = mean_valid
            save_path = os.path.join(save_dir,
                                     'actor.pt')
            torch.save(actor.state_dict(), save_path)
            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)
        temp = np.mean(times) if len(times) > 0 else 0.0
        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs ' \
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
               temp))



def train_tsp(args):
    from tasks import tsp
    from tasks.tsp import TSPDataset
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 1  # dummy for compatibility
    train_data = TSPDataset(args.num_nodes, args.train_size, args.seed)
    valid_data = TSPDataset(args.num_nodes, args.valid_size, args.seed + 1)
    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    tsp.update_dynamic,
                    tsp.update_mask,
                    args.num_layers,
                    args.dropout)
    actor = actor.to(device)
    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)  # 一个statecritic对象
    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tsp.reward
    kwargs['render_fn'] = tsp.render
    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))
        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))
    if not args.test:
        train(actor, critic, **kwargs)

    # ----下面是validation部分----
    test_data = TSPDataset(args.num_nodes, args.train_size, args.seed + 2)
    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    # 将test_data数据分为batch
    [out, tour_indices] = validate(test_loader, actor, tsp.reward, tsp.render, test_dir, num_plot=5)
    # 上面的输入参数分别是：测试数据，训练好的actor模型，tsp.py中的reward函数，tsp.py中的render函数，test结果的储存路径，在该路径下画图的数量
    print('Average tour length: ', out, '\ntour_indices: ', tour_indices)


def my_test(args, test_data, GAgen=None):
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 1  # dummy for compatibility
    update_fn = tsp.update_dynamic
    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    update_fn,
                    tsp.update_mask,  # 这里定义了mask的更新函数，直接调用了tsp.py中的update_mask函数
                    args.num_layers,
                    args.dropout).to(device)
    path = os.path.join(args.checkpoint, 'actor.pt')
    actor.load_state_dict(torch.load(path, device))

    test_dir = 'my_test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out, tour_indices = validate(test_loader, actor, tsp.reward, tsp.render, test_dir, num_plot=5, GAgen=GAgen, return_mean=False, if_test=True)
    return out, tour_indices


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='tsp')
    parser.add_argument('--nodes', dest='num_nodes', default=50, type=int)
    parser.add_argument('--actor_lr', default=5e-3, type=float)
    parser.add_argument('--critic_lr', default=5e-3, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.05, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=1280000, type=int)
    parser.add_argument('--valid-size', default=10000, type=int)
    args = parser.parse_args()
    train_tsp(args)