import torch
import torch.nn as nn

class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,stride=4)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm = nn.LocalResponseNorm(2)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size =3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size = 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.norm(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.norm(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))    
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
#input (batch, 3, 227, 227) --> output(batch, 1000)

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.randn(4,3,227,227).to(device)
    net = alexnet().to(device)
    y = net(tensor)
    print(y.size())

if __name__ == '__main__':
    test()