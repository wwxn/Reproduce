import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(3200, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=x.view(x.size(0),-1)
        x = self.fc(x)
        return x


batch_size = 64

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_tf, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model=TestNet()

optimizer=optim.SGD(model.parameters(),lr=0.1)
criterion=nn.CrossEntropyLoss()

# train_data=[data for data in train_loader]
# loss_data=500.0
# i=0
# for i in range(0,train_data.__len__()-1):
#     i+=1
    # data_in,target=train_data[i]
    # print(data_in.shape)
    # data_out=model.forward(data_in)
    # loss = criterion.forward(data_out, target)
    # optimizer.zero_grad()
    #
    # loss.backward()
    # optimizer.step()
    # loss_data=loss.data
    # if i%20==0:
    #     print(loss.data)
#
# torch.save(model,'net.pkl')
#

model=torch.load('net.pkl')

for data in test_loader:
    data_in,label=data
    predict=model(data_in)
    predict=predict.argmax(dim=1)
    result=(predict==label).sum()
    print('accuracy:{}'.format(float(result/64.0)))
