import torch
import torch.nn as nn

#format [1x1, 3x3 reduce, 3x3, 5x5 reduce, 5x5, pool proj]
dict = {'3a':[64, 96,128,16,32,32],
        '3b':[128,128,192,32,96,64],
        '4a':[192,96,208,16,48,64],
        '4b':[160,112,224,24,64,64],
        '4c':[128,128,256,24,64,64],
        '4d':[112,144,288,32,64,64],
        '4e':[256,160,320,32,128,128],
        '5a':[256,160,320,32,128,128],
        '5b':[384,192,384,48,128,128]
            
}

class googlenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
                                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )
        self.conv2 = nn.Sequential(
                                    inceptionblock(in_channels = 192, config = dict['3a']),
                                    inceptionblock(in_channels = 256, config = dict['3b']),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                    inceptionblock(in_channels = 480, config = dict['4a']),
                                    inceptionblock(in_channels = 512, config = dict['4b']),
                                    inceptionblock(in_channels = 512, config = dict['4c']),
                                    inceptionblock(in_channels = 512, config = dict['4d']),
                                    inceptionblock(in_channels = 528, config = dict['4e']),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                    inceptionblock(in_channels = 832, config = dict['5a']),
                                    inceptionblock(in_channels = 832, config = dict['5b']),
                                    nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.dropout = nn.Dropout2d(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=1000)    
        

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

class inceptionblock(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.conv1x1_branch = nn.Sequential(
                                nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.config[0],
                                kernel_size=1),
                                nn.ReLU()
        )

        self.conv3x3_branch = nn.Sequential(
                                nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.config[1],
                                        kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=self.config[1],
                                out_channels=self.config[2],
                                kernel_size=3,
                                stride=1,
                                padding=1),
                                nn.ReLU()
        )        

        self.conv5x5_branch = nn.Sequential(
                                nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.config[3],
                                        kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=self.config[3],
                                    out_channels=self.config[4],
                                    kernel_size=5,
                                    stride=1,
                                    padding=2),
                                nn.ReLU()
        )
        self.pool_branch = nn.Sequential(
                                nn.MaxPool2d(kernel_size=3,
                                    stride=1,
                                    padding=1),

                                nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=self.config[5],
                                    kernel_size=1),
                                nn.ReLU()

        )

    def forward(self, x):
            concat = torch.cat((self.conv1x1_branch(x), self.conv3x3_branch(x), self.conv5x5_branch(x), self.pool_branch(x)),dim=1)
            return concat


#input (batch, 3, 224, 224) --> output (batch, 1000)
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.rand(1, 3, 224, 224).to(device)
    net = googlenet().to(device)
    y = net(tensor)
    print(y.size())

if __name__ == '__main__':
    test()
