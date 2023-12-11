import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv('temp.csv',)  # 忽略第一行，即列名
scaler = MinMaxScaler()
data = scaler.fit_transform(data.values)
data = data[:2000]
print(data.shape)
# print(data)
# 生成数据集
def create_dataset(dataset, look_back=10):
    X, y = [], []
    for i in range(len(dataset)-look_back-1): #这里不需要-1.下一篇论文记得改正这个程序，现在数据集的数据只有1999个了！！
        X.append(dataset[i:(i+look_back), 0])
        y.append(dataset[i + look_back, 0])
    return torch.tensor(X), torch.tensor(y)

look_back = 10
X, y = create_dataset(data, look_back=look_back)
# print(X.shape,y.shape)
# print(X,y)
# 划分训练集和测试集
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]
# print(train_X.shape,test_X.shape)
# print(train_y.shape,test_y.shape)
# print(train_y,test_y)
# 定义模型
class BiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=1):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_size*2, 1)  # 注意力层
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        attn_weights = torch.softmax(self.attn(out), dim=1)  # 注意力权重
        out = torch.sum(attn_weights * out, dim=1)  # 注意力加权求和

        out = self.fc(out)

        return out

# 创建一个BiLSTM模型对象model，输入大小为1，隐藏层大小为64，输出大小为1
model = BiLSTM(input_size=1, hidden_size=64, output_size=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 加载模型
model = BiLSTM(input_size=1, hidden_size=64, output_size=1)
model.load_state_dict(torch.load('IMF1_pre.pth'))

# 模型预测
with torch.no_grad():
    train_predict = model(train_X.unsqueeze(-1).float()).squeeze().tolist()
    test_predict = model(test_X.unsqueeze(-1).float()).squeeze().tolist()

train_actual = scaler.inverse_transform(train_y.numpy().reshape(-1, 1)).flatten()
test_actual = scaler.inverse_transform(test_y.numpy().reshape(-1, 1)).flatten()
train_predict = scaler.inverse_transform(np.array(train_predict).reshape(-1, 1)).flatten()
test_predict = scaler.inverse_transform(np.array(test_predict).reshape(-1, 1)).flatten()

train_rmse = ((sum([(train_actual[i]-train_predict[i])**2 for i in range(len(train_actual))]))/len(train_actual))**0.5
test_rmse = ((sum([(test_actual[i]-test_predict[i])**2 for i in range(len(test_actual))]))/len(test_actual))**0.5

print('Train RMSE: {:.4f}'.format(train_rmse))
print('Test RMSE: {:.4f}'.format(test_rmse))

# 保存test_predict值
with open('IMF1_pre.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['test_actual',])
    for i in range(len(test_actual)):
        writer.writerow([test_predict[i]])

# 可视化结果
# 训练集风速对比曲线
plt.figure(figsize=(10, 6))
plt.plot(train_actual, label='The original curve of IMF1')
plt.plot(train_predict, label='The predicted curve of IMF1')
plt.xlabel('Time (hours)')
plt.ylabel('Wind Speed (m/s)')

plt.legend()
plt.show()

# 测试集风速对比曲线
plt.figure(figsize=(14, 3))
plt.plot(test_actual,label='The original curve of IMF1')
plt.plot(test_predict,label='The predicted curve of IMF1')
plt.show()

