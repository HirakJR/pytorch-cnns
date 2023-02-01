import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, block_in, block_self, stride=1, downsample_mapping=None):
        super(conv_block, self).__init__()
        self.block_in = block_in
        self.block_self = block_self
        self.stride = stride
        self.downsample_mapping = downsample_mapping

        self.conv1 = nn.Conv2d(in_channels=self.block_in, out_channels=self.block_self, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.block_self)
        self.conv2 = nn.Conv2d(in_channels=block_self, out_channels=self.block_self, kernel_size=3,stride=self.stride, padding=1)
        self.bn2 = nn.BatchNorm2d(self.block_self)
        self.conv3 = nn.Conv2d(in_channels=self.block_self, out_channels=self.block_self*4, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(self.block_self*4)
        
    def forward(self,x):
        skip_func = x.clone()
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        
        if self.downsample_mapping is not None:
            skip_func = self.downsample_mapping(skip_func)

        x += skip_func
        return x

class resnet(nn.Module):
    def __init__(self,config):
        super(resnet, self).__init__()

        self.config_dict = {
                            '50':[3,4,6,3],
                            '101':[3,4,23,3],
                            '152':[3,8,36,3]
        }

        self.config = str(config)
        self.block_in = 64

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.block_in, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.block_in)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    
        self.conv2 = self.make_block(conv_block, 64, 1, self.config_dict[self.config][0])
        self.conv3 = self.make_block(conv_block, 128, 2, self.config_dict[self.config][1])
        self.conv4 = self.make_block(conv_block, 256, 2, self.config_dict[self.config][2])
        self.conv5 = self.make_block(conv_block, 512, 2, self.config_dict[self.config][3])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.block_in,1000)

    def forward(self,x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        

    def make_block(self, conv_block, block_self, stride, num_repeats):
        downsample_mapping = None
        add_to_block = []
        self.block_self = block_self

        if stride !=1 or self.block_self * 4 != self.block_in:
            downsample_mapping = nn.Sequential(
                            nn.Conv2d(in_channels=self.block_in, out_channels=self.block_self*4, kernel_size=1, stride=stride),
                            nn.BatchNorm2d(self.block_self*4)
            )
        add_to_block.append(conv_block(self.block_in, self.block_self, stride, downsample_mapping))
        self.block_in = self.block_self * 4
        
        for _ in range(num_repeats -1):
            add_to_block.append(conv_block(self.block_in, self.block_self))

        return nn.Sequential(*add_to_block)


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.randn(1,3,224,224).to(device)
    net = resnet(50).to(device)
    y = net(tensor)
    print(y.shape)

if __name__ == '__main__':
    test()
