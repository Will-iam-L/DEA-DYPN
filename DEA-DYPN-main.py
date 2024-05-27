import numpy as np
import pandas as pd
import geatpy as ea
import torch
from tasks.tsp import TSPDataset
import argparse
from trainer import my_test
import os
import time
import scipy.spatial.distance as dis
import torch.nn.functional as F
from pathlib import Path
from numpy import mean
import copy
from load_testdata import load_testdata
from DEA_templet import DEA_templet
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 参数设置
ablation_value_list = [None]  # Conduct ablation study on the following three methods: 1:greedy initialization 2: restart 3: fitness sharing selection


class MyProblem(ea.Problem):  # Inherited from Problem class.
    def __init__(self, M, args, N, maxtime, case_index):  # M is the number of objects.
        name = 'KP'  # Problem's name.
        maxormins = [-1]  # -1是最大化目标
        self.args = args
        if N == 21:
            data = load_testdata(filename='DATASET_rand/op_unif20_test_seed1234.pkl', offset=0, num_samples=128)
        elif N == 51:
            data = load_testdata(filename='DATASET_rand/op_unif50_test_seed1234.pkl', offset=0, num_samples=128)
        elif N == 101:
            data = load_testdata(filename='DATASET_rand/op_unif100_test_seed1234.pkl', offset=0, num_samples=128)
        elif N == 201:
            data = load_testdata(filename='DATASET_rand/op_unif200_test_seed1234.pkl', offset=0, num_samples=128)
        elif N == 501:
            data = load_testdata(filename='DATASET_rand/op_unif500_test_seed1234.pkl', offset=0, num_samples=128)

        data_i = case_index
        num_nodes = len(data[data_i]['prize']) + 1
        maxtime_dealt = int(data[data_i]['max_length'].item())
        maxtime = maxtime_dealt
        score = data[data_i]['prize'].numpy()
        # 在score第一位加0
        score = np.insert(score, 0, 0)
        location = data[data_i]['loc'].numpy()
        # 将data[data_i]['depot']作为起点
        location = np.row_stack((data[data_i]['depot'].numpy(), location))
        self.value = score
        self.Capacity = maxtime_dealt
        self.OriCapacity = maxtime
        Dim = num_nodes  # Set the dimension of decision variables.
        varTypes = [1] * Dim  # Set the types of decision variables. 0 means continuous while 1 means discrete.
        lb = [0] * Dim  # The lower bound of each decision variable.
        ub = [1] * Dim  # The upper bound of each decision variable.
        lbin = [1] * Dim  # Whether the lower boundary is included.
        ubin = [1] * Dim  # Whether the upper boundary is included.
        # Call the superclass's constructor to complete the instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.location = torch.from_numpy(location).T  # 坐标
        self.location = self.location.reshape([1, 2, num_nodes])

    def aimFunc(self, pop, GAgen=None):  # Write the aim function here, pop is an object of Population class.
        Vars = pop.Phen.astype(int)  # Get the decision variables
        Vars[:, 0] = 1  # 起点必须选
        # 目标函数是总价值最大
        f = np.sum(self.value * Vars, 1)
        # 下面是约束条件，即总资源不超标
        pop_num = Vars.shape[0]
        # 生成一个大小为[pop_num,]的array，每个元素都是maxtime
        constrain = np.ones([pop_num, ]) * self.Capacity
        test_data = TSPDataset(Vars.shape[-1], pop_num, 4321)  # 1*2*num_nodes
        for i in range(pop_num):
            # 这里是test_data的构建，即将01编码序列转换成深度强化学习中的tspdata类的test数据
            num_nodes = np.count_nonzero(Vars[i, :])
            temp = np.where(Vars[i, :])[0]  # 背包选取物品的编号
            test_data.dataset[i, :, :num_nodes] = self.location[:, :, temp]
            # 在test_data.dataset[i, :, :]后面补全为num_nodes个点
            test_data.dataset[i, :, num_nodes:] = self.location[:, :, :1]
            test_data.dataset = test_data.dataset.type(torch.float32)  # 这里不改成float32会在深度强化学习的模型中报错
        route_len, tour_indices = my_test(args, test_data, GAgen=GAgen)  # 但这里的tour_indices是[0：Vars中非零数量]的一个排列
        tour_indices = tour_indices.cpu()
        pop.CV = route_len.cpu().numpy() - constrain
        # pop.CV增加一个维度
        pop.CV = np.vstack(pop.CV)
        # pop.tour_indices = tour_indices
        # 目标函数值
        pop.ObjV = np.vstack(f)


# 下面两个函数是初始化用的
def count_k_nearest_matrix(problem):
    # 距离矩阵
    dis_matrix = dis.cdist(problem.location[0, :, :].cpu().T, problem.location[0, :, :].cpu().T, metric='euclidean')
    # 价值密度
    value_dis_matrix = np.ones([problem.value.shape[0], 1]) * problem.value.reshape(1,
                                                                                    problem.value.shape[0]) / dis_matrix
    return dis_matrix, value_dis_matrix


def get_init_prophetPop(self, dis_matrix, value_dis_matrix, problem):  # 生成满足约束的初始种群
    self.Chrom = np.zeros([self.sizes, self.Lind])
    self.CV = np.zeros([self.sizes, 1])
    self.ObjV = np.zeros([self.sizes, 1])
    self.Chrom[:, 0] = 1
    for i in range(self.sizes):
        dis_now = 0
        point_now = 0  # 当前选点
        times = 0
        index = np.where(np.arange(self.Lind) * (1 - self.Chrom[i, :]))[0]
        while times < self.Lind and len(index) > 1:
            # 注意，已经选过的点不纳入考虑范围
            point_now_vd = value_dis_matrix[point_now, :]  # 当前点下，各个点的价值密度（价值/当前点到该点的距离）
            index = np.where(np.arange(self.Lind) * (1 - self.Chrom[i, :]))[0]  # 选出非0的索引，即除开已经选的点，index是可选点的索引
            point_now_vd_aval = point_now_vd[index]
            probs_temp = torch.from_numpy(point_now_vd_aval / sum(point_now_vd_aval)) * point_now_vd_aval.shape[0]
            if i <= int(1 * self.sizes / 2):
                probs = F.softmax(probs_temp, dim=0).clone()  # softmax方法
            else:
                probs = probs_temp * 0 + 1 / point_now_vd_aval.size
            try:
                m = torch.distributions.Categorical(probs)
            except:
                print('i:', i)
                print('times:', times)
                print('probs:', probs)
                print('point_now_vd_aval:', point_now_vd_aval)
                print('self.Chrom[i, :]:', self.Chrom[i, :])
                print('point_now:', point_now)
            point_next = index[int(m.sample())]
            if dis_now + dis_matrix[point_now, point_next] + dis_matrix[point_next, 0] <= problem.Capacity:
                dis_now += dis_matrix[point_now, point_next]
                self.Chrom[i, point_next] = 1
                point_now = point_next
                times = 0
            else:
                times += 1
        dis_now += dis_matrix[point_now, 0]
        self.CV[i] = dis_now - problem.Capacity
        self.ObjV[i, 0] = np.round(np.sum(problem.value[np.where(self.Chrom[i, :])[0]]), 3)
    return self


if __name__ == '__main__':
    for ablation_value in ablation_value_list:  # 1:greedy initialization 2: restart 3: fitness sharing selection
        """=========================Instantiate your problem=========================="""
        M = 1  # Set the number of objects.
        parser = argparse.ArgumentParser(description='Combinatorial Optimization')
        parser.add_argument('--seed', default=12345, type=int)
        parser.add_argument('--checkpoint', default=None)  # 这个可以使用预训练模型，如果取none就是重新训练，否则用pt的路径
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
        parser.add_argument('--valid-size', default=1000, type=int)

        args = parser.parse_args()
        N_set = [21, 51, 101, 201, 501]
        for N in N_set:
            if N == 21:
                Capacity = 2
            elif N == 51:
                Capacity = 3
            elif N == 101:
                Capacity = 4
            elif N == 201:
                Capacity = 5
            elif N == 501:
                Capacity = 6
            args.checkpoint = os.path.join('tsp', '50', 'pre-trained' + os.path.sep)

            # 使用预训练模型
            print('NOTE: SETTTING CHECKPOINT: ')
            print(args.checkpoint)
            time_list = []
            reward_list = []
            for run_times in range(10):
                case_index = 0
                print('====================N:', N, '===tmax:', Capacity, '=======================')
                problem = MyProblem(M, args, N, Capacity, case_index)  # Instantiate MyProblem class
                args.num_nodes = problem.Dim
                """===============================Population set=============================="""
                Encoding = 'BG'  # Encoding type.
                NIND = 256  # Set the number of individuals.
                Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
                                  problem.borders)  # Create the field descriptor.
                population = ea.Population(Encoding, Field,
                                           NIND)  # Instantiate Population class(Just instantiate, not initialize the population yet.)
                """================================Algorithm set==============================="""
                myAlgorithm = DEA_templet(problem, population)
                myAlgorithm.MAXGEN = 40  # Set the max times of iteration.
                myAlgorithm.logTras = 1  # Set the frequency of logging. If it is zero, it would not log.
                myAlgorithm.verbose = True  # Set if we want to print the log during the evolution or not.
                myAlgorithm.drawing = 0  # 1 means draw the figure of the result.
                """===============================Start evolution============================="""
                prophetPop = ea.Population(Encoding, Field, NIND)
                prophetPop.Lind = problem.Dim
                dis_matrix, value_dis_matrix = count_k_nearest_matrix(problem)
                save_dir = 'EADRL_result'
                if ablation_value == 1:
                    save_dir = 'EADRL_result/DEA-greedy'
                elif ablation_value == 2:
                    save_dir = 'EADRL_result/DEA-restart'
                elif ablation_value == 3:
                    save_dir = 'EADRL_result/DEA-fitness-sharing'

                # -------进化----------
                folder_name = ''.join(["N=" + str(problem.Dim - 1) + "Tmax=" + str(problem.Capacity)])
                if os.path.isdir(save_dir):
                    if os.path.isdir(os.path.join(save_dir, folder_name)):
                        save_dir = os.path.join(save_dir, folder_name)
                    else:
                        os.mkdir(os.path.join(save_dir, folder_name))
                        save_dir = os.path.join(save_dir, folder_name)

                start = time.time()
                print('第', run_times + 1, "次run：")
                prophetPop = get_init_prophetPop(copy.deepcopy(prophetPop), dis_matrix, value_dis_matrix, problem)
                [NDSet, population] = myAlgorithm.run(prophetPop=prophetPop,
                                                      ablation_value=ablation_value)  # Run the algorithm templet.
                # myAlgorithm.call_aimFunc(NDSet)  # 生成具体的路径tour_indices
                end = time.time()
                # 储存myAlgorithm.log为
                path_logsave = os.path.join(save_dir, 'result_log_' + str(run_times) + '.pkl')
                # 储存dict类型的变量myAlgorithm.log
                with open(path_logsave, 'wb') as f:
                    pickle.dump(myAlgorithm.log, f)
                print('time=', end - start)
                time_list.append(end - start)
                reward_list.append(NDSet.ObjV[0][0])
                """=============================Analyze the result============================"""

            mean_reward = mean(reward_list)
            mean_time = mean(time_list)
            # 输出time_list和reward_list
            dic_log = {'reward_list': pd.Series(reward_list),
                       'time_list': pd.Series(time_list),
                       'mean_reward': pd.Series(mean_reward),
                       'mean_time': pd.Series(mean_time)}
            dic_log_df = pd.DataFrame(dic_log)
            data_dir = os.path.join(save_dir, 'TIME_REWARD.xlsx')
            my_file = Path(data_dir)
            dic_log_df.to_excel(data_dir, index=False)
