from email import message
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time 
import numpy as np
import util

# --------------------------------（常量定义）------------------------------
time_steps = 24 # ~ 2小时
epoch = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
log_every = 20 
patience = 200 
min_val_loss = float('inf')
# ------------------------------- （LSTM 模型）---------------------------------------

# 加载数据集
def load_data(path, np_type=True):
    data_csv = pd.read_csv(path, usecols=[2])
    data_csv = data_csv.dropna()
    dataset = data_csv.values
    dataset = dataset.astype('float32')
    x, y = split_data(dataset,np_type)
    return x, y

def split_data(dataset,np_type=True):
    dataX, dataY = [], []
    lens = len(dataset)    
    for i in range(lens - time_steps):
        dataX.append(dataset[i: i + time_steps])
        dataY.append(dataset[i + time_steps])
    dataX = np.array(dataX).reshape(-1, time_steps)
    dataY = np.array(dataY)
    return dataX, dataY

class TrafficDataset(Dataset):
    def __init__(self,filepath):
        train_x, train_y = load_data(filepath)
        var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
        var_y = torch.tensor(train_y, dtype=torch.float32, device=device)
        self.x = var_x
        self.y = var_y
        self.len = train_y.shape[0]
        print(self.len)

    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

# 模型定义
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.reg = nn.Linear(hidden_size * time_steps, 1024) # 回归
        self.reg2 = nn.Linear(1024, output_size) # 回归
        self.activate = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(b, s*h) # 转换成线性层的输入格式 
        x = self.reg(x)
        x = self.activate(x)
        x = self.reg2(x)
        return x

# ------------------------------- （LSTM训练）---------------------------------------

def test_net(net):
    net.eval()
    test_x, test_y = load_data('~/16-5m.csv')
    test_x = test_x[0,:]
    print(test_x)
    result_y = []
    
    for i in range(0, test_y.shape[0]):
        var_x = torch.tensor(test_x, dtype=torch.float32, device=device)
        var_x = util.get_batchs_for_lstm(var_x, time_steps)
        pred_y = net(var_x).cpu()
        pred_y_num = pred_y.view(-1).data.numpy().tolist()[0]
        test_x = test_x.tolist()
        test_x.append(pred_y_num)
        result_y.append(pred_y_num)
        test_x = test_x[1:] #后移一格，24步预测后一步
        test_x = np.array(test_x)
    print('begin draw...')
    plt.plot(result_y, 'r', label='prediction')
    plt.plot(test_y, 'b', label='real')
    plt.legend(loc='best')
    plt.show()
    
def train_lstm():
    train_dataset = TrafficDataset('~/15-5m.csv')
    val_dataset = TrafficDataset('~/16-5m.csv')

    print(torch.cuda.is_available())
    net = lstm_reg(1, 64).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)

    val_loader = DataLoader(dataset=val_dataset, 
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)

    for epoch_num in range(epoch):
        losses = []
        for i, data in enumerate(train_loader, 0):
            # 前向传播
            var_x, var_y = data
            var_x = util.get_batchs_for_lstm(var_x, time_steps)            
            out = net(var_x)
            loss = criterion(out, var_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_num = loss.data.item()
            losses.append(loss_num)

        print('Epoch: {}'.format(epoch_num))
        if (epoch_num % log_every) == log_every - 1:
            val_loss = util.compute_val_loss(net, val_loader, criterion, epoch_num, device, time_steps)
            message = 'Epoch: {}, train_mae: {:.5f}, val_mae: {:.6f} '.format(epoch_num, np.mean(losses), val_loss)
            print(message)
    return net

result_net = train_lstm()
test_net(result_net)
