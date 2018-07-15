import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

epochs = 600
device = torch.device('cuda')

class Sine(Dataset):
    def __init__(self, test=False):
        super().__init__()
        self.x = np.arange(0, 20, 0.01)
        if test:
            self.x = np.arange(0, 40, 0.01)
        self.raw_data = np.sin(self.x**2)*self.x + 2*self.x

    def __len__(self):
        return len(self.raw_data)-10

    def __getitem__(self, item):
        return self.raw_data[item:item+10], self.raw_data[item+10]

class SineLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 60)
        torch.nn.init.xavier_normal_(self.linear.weight)
        self.lstm = nn.LSTM(60, 60, batch_first=True)
        self.conv1 = nn.Conv1d(10, 30, 1)
        self.conv2 = nn.Conv1d(30, 60, 1)
        self.predict = nn.Linear(120, 60)
        self.predict2 = nn.Linear(60, 1)

    def init_zeros(self):
        init = torch.zeros(1, self.batch_size, 60).to(device)
        self.hidden = (init, init)

    def forward(self, input):
        self.batch_size = input.shape[0]
        self.init_zeros()
        input = input[:,:,None]
        out = self.linear(input)
        out, self.hidden = self.lstm(out, self.hidden)
        out = out[:, -1]
        conv = self.conv2(self.conv1(input)).view(self.batch_size, -1)
        pred = self.predict2(F.elu(self.predict(torch.cat((out, conv), dim=1))))
        return pred

sine = Sine()
sine_test = Sine(test=True)
loader = DataLoader(sine, batch_size=50, shuffle=True, pin_memory=True)
test_loader = DataLoader(sine_test, batch_size=1, pin_memory=True)
net = SineLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.1)

start = time.time()
for epoch in range(epochs):
    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = net(x.float())
        loss = criterion(pred.view(-1), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(time.time()-start, loss.item())
    test_predictions = np.array([])
    print(epoch, time.time()-start, loss.item())
for test_batch in test_loader:
    x_t, y_t = test_batch
    x_t, y_t = x_t.to(device), y_t.to(device)
    pred_t = net(x_t.float())
    test_predictions = np.append(test_predictions, pred_t.detach().cpu().numpy())
plt.plot(test_predictions)
plt.plot(sine.raw_data[10:])
plt.show()
