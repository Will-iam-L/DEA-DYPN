# -*- coding: utf-8 -*-
import geatpy as ea  # 导入geatpy库
from sys import path as paths
from os import path
import numpy as np
import time  # 在文件顶部导入time模块


paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


def fitness_sharing(population, alpha=1, sigma=1):
    """
    实现健康度共享，使用矩阵计算加快速度。
    population: 种群对象
    alpha: 共享函数的指数参数
    sigma: 共享函数的距离参数
    """
    # 获取种群中所有个体的染色体信息
    chroms = population.Chrom
    # 计算所有个体之间的欧氏距离
    diff_matrix = np.expand_dims(chroms, axis=1) - np.expand_dims(chroms, axis=0)
    distances = np.sqrt(np.sum(diff_matrix**2, axis=2))
    # 应用共享函数
    shared_function = np.maximum(0, sigma - distances) ** alpha
    sh = np.sum(shared_function, axis=1)
    # 计算共享适应度
    shared_fitness = population.FitnV.flatten() / sh
    return shared_fitness

# 在选择过程中使用 fitness_sharing 函数
def select_with_diversity(population, NIND):
    """
    结合多样性的选择过程。
    population: 种群对象
    NIND: 选择的个体数
    """
    shared_fitness = fitness_sharing(population)
    selected_indices = np.argsort(-shared_fitness)[:NIND]  # 根据共享适应度选择
    return population[selected_indices]

# 在这里定义种群重启机制相关的函数
def is_stagnant(population, last_best_fit, stagnant_gen_threshold, gen, last_update_gen):
    """
    检查种群是否停滞。
    population: 当前种群
    last_best_fit: 上一次记录的最优适应度
    stagnant_gen_threshold: 停滞的代数阈值
    gen: 当前的代数
    """
    current_best_fit = np.max(population.FitnV)
    if current_best_fit > last_best_fit:
        return False, current_best_fit
    elif gen - last_update_gen >= stagnant_gen_threshold:
        return True, last_best_fit
    else:
        return False, last_best_fit

class DEA_templet(ea.SoeaAlgorithm):

    def __init__(self, problem, population):
        ea.SoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'SEGA'
        self.selFunc = 'tour'  # 锦标赛选择算子
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=0.7)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=0.5)  # 生成逆转变异算子对象
        else:
            self.recOper = ea.Xovud(XOVR=0.8)  # 生成单点交叉算子对象
            if population.Encoding == 'BG':
                self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
            elif population.Encoding == 'RI':
                self.mutOper = ea.Mutbga(Pm=1 / self.problem.Dim, MutShrink=0.5, Gradient=20)  # 生成breeder GA变异算子对象
            else:
                raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')


    def run(self, prophetPop, ablation_value=None):  # prophetPop为先知种群（即包含先验知识的种群）


        start_time = time.time()  # 记录算法开始的时间
        last_population = None  # 初始化上一个generation的种群变量

        # assert检测ablation_value必须为1,2,3中的一个或者None
        if ablation_value is not None:
            assert ablation_value in [1, 2, 3], 'ablation_value必须为1,2,3中的一个或者None'

        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        population.initChrom(NIND)  # 初始化种群染色体矩阵

        if ablation_value == 1:  # 针对种群初始化方法的消融实验
            prophetPop = None
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        self.call_aimFunc(population)  # 计算种群的目标函数值

        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度

        # 初始化种群重启机制的参数
        last_best_fit = -np.inf  # 记录上一次最优适应度
        stagnant_gen_threshold = 20  # 设置停滞代数阈值
        last_update_gen = 0  # 记录上次更新的代数

        # ===========================开始进化============================
        gen = 0
        while self.terminated(population) == False:
            current_time = time.time()  # 获取当前时间
            if current_time - start_time > 300:  # 检查是否超过300秒
                print("运行时间超过300秒，终止迭代。")
                population = last_population  # 如果有必要，可以在这里处理last_population为None的情况
                break  # 跳出循环
            gen += 1
            # 选择
            offspring = population[ea.selecting('urs', population.FitnV, NIND)]
            # 进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
            offspring.Chrom[:, 0] = 1
            self.call_aimFunc(offspring)  # 计算目标函数值
            population = population + offspring  # 父子合并
            population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度

            # 使用多样性选择
            if ablation_value == 3:
                population = population[ea.selecting(
                    'dup', population.FitnV, NIND)]  # 采用基于适应度排序的直接复制选择生成新一代种群
            else:
                population = select_with_diversity(population, NIND)

            if ablation_value != 2:
                # 检测种群是否停滞
                stagnant, last_best_fit = is_stagnant(population, last_best_fit, stagnant_gen_threshold, gen, last_update_gen)
                if stagnant:
                    print('种群已经停滞，正在重启种群...')
                    elite_ratio = 0.5  # 精英保留比例
                    num_elite = int(NIND * elite_ratio)  # 保留的精英个体数

                    # 选择精英个体
                    sorted_indices = np.argsort(-population.FitnV[:, 0])  # 按适应度降序排序
                    elite_indices = sorted_indices[:num_elite]
                    elite = population[elite_indices]

                    # 生成新的个体
                    new_population = population.copy()
                    new_population.initChrom(int(NIND*0.5))

                    # 合并精英个体和新个体
                    population = elite + new_population
                    self.call_aimFunc(population)  # 重新计算目标函数值
                    population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度

                    last_update_gen = gen  # 更新最后一次更新代数
                    last_best_fit = np.max(population.FitnV)  # 更新最优适应度记录

            # 记录当前种群为上一个generation的种群
            last_population = population.copy()

        CV_neg_index = np.where(population.CV <= 0)[0]
        population = population[CV_neg_index]
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
