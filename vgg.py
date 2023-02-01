import torch
import torch.nn as nn

config_dict = {
    '11':[64,'m', 128,'m', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'],
    '13':[64, 64, 'm', 128 ,128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'],
    '16':[64, 64, 'm', 128 ,128, 'm', 256, 256, 256, 'm', 512, 512, 512, 'm', 512 ,512, 512, 'm'],
    '19':[64, 64, 'm', 128 ,128, 'm', 256, 256, 256, 256, 'm', 512, 512, 512, 512, 'm', 512 ,512, 512, 512, 'm']

}

class vgg(nn.Module):
    def __init__(self, config):
        super(vgg,self).__init__()
        self.config = config
        self.conv = self.conv_layers(self.config)
        self.fc = nn.Sequential(nn.Linear(7*7*512, 4096),
                                nn.ReLU(),
                                nn.Linear(4096,4096),
                                nn.ReLU(),
                                nn.Linear(4096,1000))

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def conv_layers(self, config):
        list = config_dict[str(config)]
        make = []
        in_channels = 3
        for layer in list:
            if type(layer) is int:
                make.append(nn.Conv2d(in_channels=in_channels, out_channels=layer, kernel_size=3, stride=1, padding =1))
                make.append(nn.ReLU())
                in_channels = layer
            if type(layer) is str:
                make.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*make)

#input(batch, 3, 224, 224) --> output(batch, 1000)

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.rand(1, 3, 224, 224).to(device)
    net = vgg(16).to(device)
    y = net(tensor)
    print(y.size())

if __name__ == '__main__':
    test()
