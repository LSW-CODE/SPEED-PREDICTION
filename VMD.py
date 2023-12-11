# import numpy as np
# import matplotlib.pyplot as plt
# from vmdpy import VMD  # 导入定义的VMD函数
# import pandas as pd
#
# T = 1990
# fs = 1/T
# t = np.arange(0,T)/T  # 对0-2200进行归一化
# # freqs = 2*np.pi*(t-0.5-fs)/(fs)  # 造一个输入，这是造的那个输入
#
# df = pd.read_csv("speed_sin.csv")
# # print(df)
# # print(type(df))  # 数据类型为pandas.core.frame.DataFrame
# data = df.to_numpy()
# # print(data)
# f = data[:,0]  # 取全部行，第0列
# f=f[0:1990]  # 取f中0:2200的数据
# # print(f)
# # print(f.shape)
#
#
# # some sample parameters for VMD
# # function [u, u_hat, omega] = VMD(signal, alpha, tau, K, DC, init, tol)
# alpha = 1902.83  # 适度的带宽约束/惩罚因子
# tau = 0  # 噪声容限
# DC = 0  # 无直流部分
# init = 1  # omegas的均匀初始化
# tol = 1e-7
# K = 7  # 分解的模态数
# print(f.shape)
# # u表示分解模式IMF的集合，u_hat表示模式的光谱范围，omega表示估计模态的中心频率
# u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
# print(u.shape)
#
# # # Run actual VMD code
# # # real_data里面有五列，每一列是一组
# # real_data = np.genfromtxt('real.csv', delimiter=',')  # 读取文件，分割符选择 ‘,’
# # # 分割数据并存储到ndarray类型的列表中
# # result_list = [real_data[:,i] for i in range(5)]  # 分别读取每一列
#
# # 打印结果
# np.savetxt('u.csv', u.T, delimiter=',')  # 保存u的转置
# # fsub = {1:u[0,:],2:u[1,:],3:u[2,:],4:u[3,:],5:u[4,:],6:u[5,:],7:u[6,:]}
# fsub = {1:u[0,:],2:u[1,:],3:u[2,:],4:u[3,:]}
#
# # Simple Visualization of decomposed modes可视化
# plt.figure()
# plt.plot(f)  # 绘图f
# plt.show()
#
# plt.figure()
# plt.plot(u.T)  # 绘图u，总共分解了五个
# plt.title('Decomposed modes')
#
# sortIndex = np.argsort(omega[-1,:])
# omega = omega[:,sortIndex]
# u_hat = u_hat[:,sortIndex]
# u = u[sortIndex,:]
# linestyles = ['b', 'g', 'm', 'c', 'k', 'r', 'k']
# #
#
# fig1 = plt.figure()
#
#
# plt.subplot(611)
#
# plt.plot(t,f,'purple')  # x=t;y=f
# plt.xlim((0,1))
# plt.xticks(color='g')
#
# for key, value in fsub.items():
#     plt.subplot(6,1,key+1)  # 确定绘制位置
#     plt.plot(t,value,'orangered')
#
#     plt.subplots_adjust(wspace =0, hspace =0.1)  # 调整位置
#     plt.xticks(color='g')  # 调整刻度颜色
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pandas as pd
import csv

# # alpha 惩罚系数；带宽限制经验取值为抽样点长度1.5-2.0倍.
# # 惩罚系数越小，各IMF分量的带宽越大，过大的带宽会使得某些分量包含其他分量言号;
# # a值越大，各IMF分量的带宽越小，过小的带宽是使得被分解的信号中某些信号丢失该系数常见取值范围为1000~3000
# alpha=2000
# tau=0 # tau 噪声容限，即允许重构后的信号与原始信号有差别。
# K=3 # K 分解模态（IMF）个数
# DC=0 # DC 若为0则让第一个IMF为直流分量/趋势向量
# init=1 # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
# tol=1e-7 # 控制误差大小常量，决定精度与迭代次数
# u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol) # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率


T = 2000
fs = 1/T
fre_axis=np.linspace(0,T/2,int(T/2))
# t = np.arange(0,T)/T  # 对0-2200进行归一化
t = np.arange(0,T)  # 对0-2200进行归一化
# freqs = 2*np.pi*(t-0.5-fs)/(fs)  # 造一个输入，这是造的那个输入

df=pd.read_csv("speed.csv") # 忽略第一行，即列名
data=df.to_numpy()
f = data[:,0]  # 取全部行，第0列
f=f[0:2000]  # 取f中0:2200的数据


alpha = 1000  # 适度的带宽约束/惩罚因子
tau = 0  # 噪声容限
DC = 0  # 无直流部分
init = 1  # omegas的均匀初始化
tol = 1e-7
K = 3  # 分解的模态数
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

plt.plot(t,f)
plt.show()
# np.savetxt('omega.csv', omega.T, delimiter=',')


# # 1 画原始信号和它的各成分
# plt.figure(figsize=(10,7));plt.subplot(K+1, 1, 1);plt.plot(t,f)
# for i,y in enumerate(v):
#     plt.subplot(K+1, 1, i+2);plt.plot(t,y)
# plt.suptitle('Original input signal and its components');plt.show()

# # 2 分解出来的各IMF分量
# plt.figure(figsize=(10,7))
# plt.plot(t,u.T);plt.title('all Decomposed modes');plt.show()  # u.T是对u的转置
#
#
# # 3 各IMF分量的fft幅频图
# plt.figure(figsize=(10, 7), dpi=80)
# for i in range(K):
#     plt.subplot(K, 1, i + 1)
#     fft_res=np.fft.fft(u[i, :])
#     plt.plot(fre_axis,abs(fft_res[:int(T/2)])/(T/2))
#     plt.title('(FFT) amplitude frequency of IMF {}'.format(i + 1))
# plt.show()
#
# # 4 分解出来的各IMF分量的频谱
# # print(u_hat.shape,t.shape,omega.shape)
# plt.figure(figsize=(10, 7), dpi=80)
# for i in range(K):
#     plt.subplot(K, 1, i + 1)
#     plt.plot(fre_axis,abs(u_hat[:, i][int(T/2):])/(T/2))
#     plt.title('（VMD）amplitude frequency of the modes{}'.format(i + 1))
# plt.tight_layout();plt.show()

# # 5 各IMF的中心频率
# plt.figure(figsize=(12, 7), dpi=80)
# for i in range(K):
#     # plt.subplot(K, 1, i + 1)
#     plt.plot(omega[:,i]) # X轴为迭代次数，y轴为中心频率
#     plt.title('mode center-frequencies{}'.format(i + 1))
# plt.tight_layout();plt.show()

# plt.figure(figsize=(10,7))
# plt.plot(t,np.sum(u,axis=0))
# plt.plot(t,f)
# plt.title('reconstructed signal')
# plt.show()


# 保存数据值
np.savetxt('vmd_nopso.csv', u.T, delimiter=',')  # 保存u的转置,即IMF序列
# with open('vmd.csv', mode='w', newline='') as file:  # 确保文件操作完成后自动关闭文件
#     writer = csv.writer(file)
#     writer.writerow(['IMF',])
#     for i in range(220):
#         writer.writerow([np.sum(u.T[:][i])])


# with open('result.csv', mode='w', newline='') as file:  # 确保文件操作完成后自动关闭文件
#     writer = csv.writer(file)
#     writer.writerow(['IMF',])
#     for i in range(len(u[0])):
#         writer.writerow([u[0][i]])
