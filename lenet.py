import torch
import torch.nn as nn

class lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self,x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = self.tanh(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

#input (batch, 1, 32, 32) --> output (batch, 10)

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.randn(4,1,32,32).to(device)
    net = lenet5().to(device)
    y = net(tensor)
    print(y.size())

if __name__ == '__main__':
    test()