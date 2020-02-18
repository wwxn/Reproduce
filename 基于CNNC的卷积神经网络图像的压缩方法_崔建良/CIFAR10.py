import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_features=32),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(num_features=32),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder:
    def __init__(self, model=EncoderNet(), batch_size=64, path='EncoderNet.pkl', train_flag=False):
        self.model = model
        data_tf = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        self.batch_size = batch_size
        if self.train:
            train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_tf, download=False)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_tf, download=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.path = path
        self.train_flag = train_flag

    def train(self, lr=1):
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_loader = list(self.train_loader)
        for i in range(0, train_loader.__len__()-1):
            img_raw = train_loader[i][0]
            img_out = self.model.forward(img_raw)
            img_raw = img_raw.reshape((self.batch_size, 1, -1))
            img_out = img_out.reshape((self.batch_size, 1, -1))
            loss = criterion.forward(img_out, img_raw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(loss.data)
        self.model.eval()
        torch.save(self.model, self.path)

    def compare(self):
        self.model = torch.load(self.path)
        test_loader = list(self.test_loader)
        batch_num = random.randint(0, test_loader.__len__() - 2)
        pic_num = random.randint(0, 63)
        img_raw = test_loader[batch_num][0][pic_num]
        img_raw=img_raw.reshape((1,3,32,32))
        img_compressed = self.model.forward(img_raw)
        img_raw = img_raw.reshape((3, 32, 32))
        img_compressed = img_compressed.reshape((3, 32, 32))

        img_raw_np = (img_raw.numpy()/2+0.5)*256
        img_raw_np = np.array(img_raw_np, 'uint8')
        red = img_raw_np[0].reshape(1024, 1)
        green = img_raw_np[1].reshape(1024, 1)
        blue = img_raw_np[2].reshape(1024, 1)
        pic = np.hstack((red, green, blue))
        img_raw_np = pic.reshape((32, 32, 3))

        img_compressed_np = (img_compressed.detach().numpy()/2+0.5)*256
        img_compressed_np = np.array(img_compressed_np,'uint8')
        red = img_compressed_np[0].reshape(1024, 1)
        green = img_compressed_np[1].reshape(1024, 1)
        blue = img_compressed_np[2].reshape(1024, 1)
        pic = np.hstack((red, green, blue))
        img_compressed_np = pic.reshape((32, 32, 3))

        plt.subplot(1, 2, 1)
        plt.imshow(img_compressed_np)
        plt.subplot(1, 2, 2)
        plt.imshow(img_raw_np)
        plt.show()

    def run(self):
        if self.train_flag:
            self.train()
        self.compare()


if __name__ == '__main__':
    encoder_test = Encoder(train_flag=False)
    encoder_test.run()
