# # -*- coding: utf-8 -*-
# """
# Created on Fri May  6 16:30:19 2022
#
# @author: 工作站1
# """
#
# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# from sklearn.svm import SVC
# import numpy as np
# import random
# import math
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from vmdpy import VMD
# from scipy.fftpack import hilbert,fft,ifft
# from math import log
# import pandas as pd
#
#
# # data = np.loadtxt('ball18.txt')
# # x = data[0:2048]
#
#
# df=pd.read_csv("wind_speed_1.csv",)
# data=df.to_numpy()
# f = data[:,0]
# x = f[0:2200]
#
# tau = 0.  # noise-tolerance (no strict fidelity enforcement)
# DC = 0  # no DC part imposed
# init = 1  # initialize omegas uniformly
# tol = 1e-7
#
#
#
# # 3.初始化参数
# W = 0.5                                # 惯性因子
# c1 = 0.2                                # 学习因子
# c2 = 0.5                                # 学习因子
# n_iterations = 50                       # 迭代次数
# n_particles = 30                       # 种群规模
# low = [3, 100]
# up = [10, 3000]
# var_num = 2
# bound = (low,up)
#
#
# # 4.设置适应度值
# def fitness_function(position):
#
#     K = int(position[0])
#     alpha = position[1]
#     if K < bound[0][0]:
#         K = bound[0][0]
#     if K > bound[1][0]:
#         K = bound[1][0]
#
#     if alpha < bound[0][1]:
#         alpha = bound[0][1]
#     if alpha > bound[1][1]:
#         alpha =bound[1][1]
#
#
#     u, u_hat, omega = VMD(x, alpha, tau, K, DC, init, tol)
#     #
#     EP = []
#     for i in range(K):
#         H = np.abs(hilbert(u[i,:]))  # 每个分量的包络信号
#         e1 = []
#         for j in range(len(H)):
#             p = H[j]/np.sum(H)
#             e = -p*log(p,2)
#             e1.append(e)
#         E = np.sum(e1)
#         EP.append(E)
#     s = np.sum(EP)/K
#     return s
#
#
#
# ## 5.粒子图
# # def plot(position):
# #     x = []
# #     y = []
# #     for i in range(0, len(particle_position_vector)):
# #         x.append(particle_position_vector[i][0])
# #         y.append(particle_position_vector[i][1])
# #     colors = (0, 0, 0)
# #     plt.scatter(x, y, c=colors, alpha=0.1)
# #     # 设置横纵坐标的名称以及对应字体格式
# #     #font2 = {'family': 'Times New Roman','weight': 'normal', 'size': 20,}
# #     plt.xlabel('gamma')
# #     plt.ylabel('C')
# #     plt.axis([0, 10, 0, 10],)
# #     plt.gca().set_aspect('equal', adjustable='box')
# #     return plt.show()
#
# # 6.初始化粒子位置，进行迭代
# pop_x = np.zeros((n_particles,var_num))
# g_best = np.zeros(var_num)
# temp = -1
# for i in range(n_particles):
#     for j in range(var_num):
#         pop_x[i][j] = np.random.rand()*(bound[1][j]-bound[0][j])+bound[0][j]
#     fit = fitness_function(pop_x[i])
#
#     if fit > temp:
#         g_best = pop_x[i]
#         temp = fit
# # particle_position_vector = np.array([np.array([random.random() * 100, random.random() * 100]) for _ in range(n_particles)])
# # print('zzz',particle_position_vector)
# pbest_position = pop_x
# pbest_fitness_value = np.zeros(n_particles)
# # print(pbest_fitness_value)
# gbest_fitness_value = np.zeros(var_num)
# # print(gbest_fitness_value[1])
# gbest_position = g_best
# velocity_vector = ([np.array([0, 0]) for _ in range(n_particles)])
# iteration = 0
#
# while iteration < n_iterations:
#     # plot(particle_position_vector)
#     for i in range(n_particles):
#         # print(pop_x[i])
#         fitness_cadidate = fitness_function(pop_x[i])
#         print("error of particle-", i, "is (training, test)", fitness_cadidate)
#         print(" At (K, alpha): ",int(pop_x[i][0]),pop_x[i][1])
#
#         if (pbest_fitness_value[i] > fitness_cadidate):
#             pbest_fitness_value[i] = fitness_cadidate
#             pbest_position[i] = pop_x[i]
#
#         elif (gbest_fitness_value[1] > fitness_cadidate):
#             gbest_fitness_value[1] = fitness_cadidate
#             gbest_position = pop_x[i]
#
#         elif (gbest_fitness_value[0] < fitness_cadidate):
#             gbest_fitness_value[0] = fitness_cadidate
#             gbest_position = pop_x[i]
#
#     for i in range(n_particles):
#         new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
#                     pbest_position[i] - pop_x[i]) + (c2 * random.random()) * (
#                                    gbest_position - pop_x[i])
#         new_position = new_velocity + pop_x[i]
#         pop_x[i] = new_position
#
#     iteration = iteration + 1
#
# plt.plot()
# # 7.输出最终结果
# print("The best position is ", int(gbest_position[0]),gbest_position[1], "in iteration number", iteration, "with error (train, test):",
#       fitness_function(gbest_position))





# # 用矩池云跑，电脑带不起来
# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# from sklearn.svm import SVC
# import numpy as np
# import random
# import math
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from vmdpy import VMD
# from scipy.fftpack import hilbert, fft, ifft
# from math import log
# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")
#
# # data = np.loadtxt('ball18.txt')
# # x = data[0:2048]
#
#
# df = pd.read_csv("wind_speed_1.csv", )
# data = df.to_numpy()
# f = data[:, 0]
# date = f[0:2200]
#
# tau = 0.  # noise-tolerance (no strict fidelity enforcement)
# DC = 0  # no DC part imposed
# init = 1  # initialize omegas uniformly
# tol = 1e-7
#
# low = [3, 100]
# up = [10, 3000]
# bound = (low,up)
#
# # 计算每个IMF分量的包络熵
# def calculate_entropy(imf):
#     # 每个分量的包络信号为env
#     env = np.abs(hilbert(imf))
#     # 将每个包络信号归一化到 [0, 1] 区间内
#     env_norm = env / np.max(env)  # 在计算包络熵的过程中，需要对包络信号进行归一化处理，以确保不同幅度的包络信号具有可比性。
#     # 根据信息熵的定义，可以通过将包络信号的概率分布进行估计，
#     # 并计算该概率分布的熵值来度量其不确定性。因此，在这段代码中，将归一化后的包络信号作为概率分布，通过p = env_norm / np.sum(env_norm)计算其概率分布。
#     p = env_norm / np.sum(env_norm)
#     return -np.sum(p * np.log2(p))
#
#
# # 定义适应度函数，即最大包络熵
# def fitness_func(x):
#     # 确定搜索范围
#     K = int(x[1])
#     alpha = x[0]
#     if K < bound[0][1]:
#         K = bound[0][1]
#     if K > bound[1][1]:
#         K = bound[1][1]
#
#     if alpha < bound[0][0]:
#         alpha = bound[0][0]
#     if alpha > bound[1][0]:
#         alpha = bound[1][0]
#
#     u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
#     num_modes = u.shape[0]
#     entropy = np.zeros(num_modes)
#     for i in range(num_modes):
#         entropy[i] = calculate_entropy(u[i, :])
#
#     # 找到最小的包络熵对应的模态
#     min_entropy_index = np.argmin(entropy)
#     min_entropy_mode = u[min_entropy_index]
#
#     #     print("最小包络熵对应的模态：", min_entropy_index)
#     # x为VMD参数向量
#     # signal为要分解的信号
#     # 分解信号并计算最大包络熵
#     # 返回最小包络熵值
#     return entropy[min_entropy_index]
#
#
# # 定义PSO算法
# def pso_optimization(num_particles, num_dimensions, fitness_func, max_iter):
#     # 初始化粒子位置和速度
#     particles_pos = np.zeros((num_particles, num_dimensions))
#
#     for i in range(num_particles):
#         particles_pos[i, 0] = np.random.uniform(500, 3000) # 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
#         particles_pos[i, 1] = np.random.uniform(3, 12)
#     particles_vel = np.zeros((num_particles, num_dimensions))
#     # 记录每个粒子的最佳位置和适应度值
#     particles_best_pos = np.copy(particles_pos)
#     particles_best_fit = np.zeros(num_particles)
#     # 记录整个群体的最佳位置和适应度值
#     global_best_pos = np.zeros(num_dimensions)
#     global_best_fit = float('inf')
#
#     # 迭代更新
#     for i in range(max_iter):
#         # 计算每个粒子的适应度值
#         particles_fitness = np.array([fitness_func(p) for p in particles_pos])
#
#         # 更新每个粒子的最佳位置和适应度值
#         for j in range(num_particles):
#             if particles_fitness[j] < particles_best_fit[j]:
#                 particles_best_fit[j] = particles_fitness[j]
#                 particles_best_pos[j] = np.copy(particles_pos[j])
#
#         # 更新整个群体的最佳位置和适应度值
#         global_best_idx = np.argmin(particles_fitness)
#         if particles_fitness[global_best_idx] < global_best_fit:
#             global_best_fit = particles_fitness[global_best_idx]
#             global_best_pos = np.copy(particles_pos[global_best_idx])
#
#         # 更新每个粒子的速度和位置
#         for j in range(num_particles):
#             # 计算新速度
#             r1 = np.random.rand(num_dimensions)
#             r2 = np.random.rand(num_dimensions)
#             cognitive_vel = 2.0 * r1 * (particles_best_pos[j] - particles_pos[j])
#             social_vel = 2.0 * r2 * (global_best_pos - particles_pos[j])
#             particles_vel[j] = particles_vel[j] + cognitive_vel + social_vel
#             # 更新位置
#             particles_pos[j] = particles_pos[j] + particles_vel[j]
#
#         # 记录每次迭代的global_best_pos和global_best_fit
#         global_best_pos_list.append(global_best_pos)
#         global_best_fit_list.append(global_best_fit)
#         print("第：" + str(i+1) + '次迭代：' + str(global_best_fit) )   # 每次迭代最好适应度：最小包络
#         print(" At (K,alpha): ", int(global_best_pos[1]),global_best_pos[0])
#
#
#
#     # 返回全局最优位置和适应度值
#     return global_best_pos, global_best_fit
#
#
# # 初始化空列表用于存储每次迭代的global_best_pos和global_best_fit
# global_best_pos_list = []
# global_best_fit_list = []
# # 使用PSO算法优化VMD参数
# num_particles = 2  # 种群规模
# num_dimensions = 2  # VMD参数个数
# max_iter = 50
# best_pos, best_fit = pso_optimization(num_particles, num_dimensions, fitness_func, max_iter)
#
# # 输出结果
# print("Best VMD parameters:", best_pos)
# print("Best fitness value:", best_fit)


# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# from sklearn.svm import SVC
# import numpy as np
# import random
# import math
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from vmdpy import VMD
# from scipy.fftpack import hilbert, fft, ifft
# from math import log
# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")
#
# # data = np.loadtxt('ball18.txt')
# # x = data[0:2048]
#
#
# df = pd.read_csv("wind_speed_1.csv", )
# data = df.to_numpy()
# f = data[:, 0]
# date = f[0:2200]
#
# tau = 0.  # noise-tolerance (no strict fidelity enforcement)
# DC = 0  # no DC part imposed
# init = 1  # initialize omegas uniformly
# tol = 1e-7
#
# def calculate_entropy(imf):
#     # 每个分量的包络信号为env
#     env = np.abs(hilbert(imf))
#     # 将每个包络信号归一化到 [0, 1] 区间内
#     env_norm = env / np.max(env)  # 在计算包络熵的过程中，需要对包络信号进行归一化处理，以确保不同幅度的包络信号具有可比性。
#     # 根据信息熵的定义，可以通过将包络信号的概率分布进行估计，
#     # 并计算该概率分布的熵值来度量其不确定性。因此，在这段代码中，将归一化后的包络信号作为概率分布，通过p = env_norm / np.sum(env_norm)计算其概率分布。
#     p = env_norm / np.sum(env_norm)
#     return -np.sum(p * np.log2(p))
#
# # 定义适应度函数，即最小包络熵
# def fitness_func(x):
#     alpha = x[0]
#     K = int(x[1])
#     # 进行VMD分解并计算包络熵
#     u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
#     entropy = calculate_entropy(u)
#     return entropy
#
# # 定义PSO算法
# def pso_optimization(num_particles, num_dimensions, fitness_func, max_iter, low, up):
#     # 初始化粒子位置和速度
#     particles_pos = np.random.uniform(low, up, (num_particles, num_dimensions))
#     particles_vel = np.zeros((num_particles, num_dimensions))
#
#     # 记录每个粒子的最佳位置和适应度值
#     particles_best_pos = np.copy(particles_pos)
#     particles_best_fit = np.zeros(num_particles)
#
#     # 记录整个群体的最佳位置和适应度值
#     global_best_pos = np.zeros(num_dimensions)
#     global_best_fit = float('inf')
#
#     # 迭代更新
#     for i in range(max_iter):
#         # 计算每个粒子的适应度值
#         particles_fitness = np.array([fitness_func(p) for p in particles_pos])
#
#         # 更新每个粒子的最佳位置和适应度值
#         for j in range(num_particles):
#             if particles_fitness[j] < particles_best_fit[j]:
#                 particles_best_fit[j] = particles_fitness[j]
#                 particles_best_pos[j] = np.copy(particles_pos[j])
#
#         # 更新整个群体的最佳位置和适应度值
#         global_best_idx = np.argmin(particles_fitness)
#         if particles_fitness[global_best_idx] < global_best_fit:
#             global_best_fit = particles_fitness[global_best_idx]
#             global_best_pos = np.copy(particles_pos[global_best_idx])
#
#         # 更新每个粒子的速度和位置
#         for j in range(num_particles):
#             # 计算新速度
#             r1 = np.random.rand(num_dimensions)
#             r2 = np.random.rand(num_dimensions)
#             cognitive_vel = 2.0 * r1 * (particles_best_pos[j] - particles_pos[j])
#             social_vel = 2.0 * r2 * (global_best_pos - particles_pos[j])
#             particles_vel[j] = particles_vel[j] + cognitive_vel + social_vel
#
#             # 更新位置
#             particles_pos[j] = particles_pos[j] + particles_vel[j]
#
#     # 返回全局最优位置和适应度值
#     return global_best_pos, global_best_fit
#
#
# # 设置VMD参数范围
# low = [100, 2]
# up = [4000, 12]
#
# # 设置PSO算法参数
# num_particles = 20  # 种群规模
# num_dimensions = 2  # VMD参数个数
# max_iter = 50
#
# # 使用PSO算法优化VMD参数
# best_pos, best_fit = pso_optimization(num_particles, num_dimensions, fitness_func, max_iter, low, up)
#
# # 输出结果
# print("Best VMD parameters:", best_pos)
# print("Best fitness value:", best_fit)

# 最优程序
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.svm import SVC
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vmdpy import VMD
from scipy.fftpack import hilbert, fft, ifft
from math import log
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# data = np.loadtxt('ball18.txt')
# x = data[0:2048]


df = pd.read_csv("speed_sin.csv", )
data = df.to_numpy()
f = data[:, 0]
date = f[0:2050]

tau = 0.  # noise-tolerance (no strict fidelity enforcement)
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7

k_low = 2
k_up = 12
alpha_low = 100
alpha_up = 4000

# 计算每个IMF分量的包络熵
def calculate_entropy(imf):
    # 每个分量的包络信号为env
    env = np.abs(hilbert(imf))
    # 将每个包络信号归一化到 [0, 1] 区间内
    env_norm = env / np.max(env)  # 在计算包络熵的过程中，需要对包络信号进行归一化处理，以确保不同幅度的包络信号具有可比性。
    # 根据信息熵的定义，可以通过将包络信号的概率分布进行估计，
    # 并计算该概率分布的熵值来度量其不确定性。因此，在这段代码中，将归一化后的包络信号作为概率分布，通过p = env_norm / np.sum(env_norm)计算其概率分布。
    p = env_norm / np.sum(env_norm)
    return -np.sum(p * np.log2(p))


# 定义适应度函数
def fitness_func(x):
    K = int(x[1])
    alpha = x[0]
    if K < k_low:
        K = int(np.random.uniform(k_low, k_up))
    if K > k_up:
        K = int(np.random.uniform(k_low, k_up))

    if alpha < alpha_low:
        alpha = np.random.uniform(alpha_low, alpha_up)
    if alpha > alpha_up:
        alpha = np.random.uniform(alpha_low, alpha_up)

    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    num_modes = u.shape[0]
    entropy = np.zeros(num_modes)
    for i in range(num_modes):
        entropy[i] = calculate_entropy(u[i, :])

    # 找到最小的包络熵对应的模态
    min_entropy_index = np.argmin(entropy)
    min_entropy_mode = u[min_entropy_index]

    #     print("最小包络熵对应的模态：", min_entropy_index)
    # x为VMD参数向量
    # signal为要分解的信号
    # 分解信号并计算最大包络熵
    # 返回最小包络熵值
    return entropy[min_entropy_index]


# 定义PSO算法
def pso_optimization(num_particles, num_dimensions, fitness_func, max_iter):
    # 初始化粒子位置和速度
    particles_pos = np.zeros((num_particles, num_dimensions))

    for i in range(num_particles):
        particles_pos[i, 0] = np.random.uniform(alpha_low, alpha_up)
        particles_pos[i, 1] = np.random.uniform(k_low, k_up)
    particles_vel = np.zeros((num_particles, num_dimensions))
    # 记录每个粒子的最佳位置和适应度值
    particles_best_pos = np.copy(particles_pos)
    particles_best_fit = np.zeros(num_particles)
    # 记录整个群体的最佳位置和适应度值
    global_best_pos = np.zeros(num_dimensions)
    global_best_fit = float('inf')

    # 迭代更新
    for i in range(max_iter):
        # 计算每个粒子的适应度值
        particles_fitness = np.array([fitness_func(p) for p in particles_pos])

        # 更新每个粒子的最佳位置和适应度值
        for j in range(num_particles):
            if particles_fitness[j] < particles_best_fit[j]:
                particles_best_fit[j] = particles_fitness[j]
                particles_best_pos[j] = np.copy(particles_pos[j])

        # 更新整个群体的最佳位置和适应度值
        global_best_idx = np.argmin(particles_fitness)
        if particles_fitness[global_best_idx] < global_best_fit:
            global_best_fit = particles_fitness[global_best_idx]
            global_best_pos = np.copy(particles_pos[global_best_idx])

        # 更新每个粒子的速度和位置
        for j in range(num_particles):
            # 计算新速度
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            cognitive_vel = r1 * (particles_best_pos[j] - particles_pos[j])
            social_vel = r2 * (global_best_pos - particles_pos[j])
            particles_vel[j] = particles_vel[j] + cognitive_vel + social_vel
            # 更新位置
            temp = particles_pos[j] + particles_vel[j]
            if((k_low<temp[1]<k_up)&(alpha_low<temp[0]<alpha_up)):
                particles_pos[j] = temp
            else:
                particles_pos[j] = particles_pos[j]

        # 记录每次迭代的global_best_pos和global_best_fit
        global_best_pos_list.append(global_best_pos)
        global_best_fit_list.append(global_best_fit)
        print("第" + str(i+1) + '次迭代：' + str(global_best_fit) )   # 每次迭代最好适应度：最小包络
        print(" At (K,alpha): ", global_best_pos[1],global_best_pos[0])



    # 返回全局最优位置和适应度值
    return global_best_pos, global_best_fit


# 初始化空列表用于存储每次迭代的global_best_pos和global_best_fit
global_best_pos_list = []
global_best_fit_list = []
# 使用PSO算法优化VMD参数
num_particles = 10  # 种群规模
num_dimensions = 2  # VMD参数个数
max_iter = 200
best_pos, best_fit = pso_optimization(num_particles, num_dimensions, fitness_func, max_iter)

# 输出结果
print("Best VMD parameters:", best_pos)
print("Best fitness value:", best_fit)
