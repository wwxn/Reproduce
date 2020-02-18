import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models

activation_max_val = torch.Tensor([1])


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, bias=False),
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
        self.fc1 = nn.Sequential(nn.Linear(3200, 1024, bias=False))
        self.fc2 = nn.Sequential(
            nn.Sequential(nn.ReLU(inplace=True)),
            nn.Linear(1024, 128, bias=False)
        )
        self.fc3 = nn.Sequential(
            nn.Sequential(nn.ReLU(inplace=True)),
            nn.Linear(128, 10, bias=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class NetQuantization:
    def __init__(self, model, train=False, batch_size=64, quantization_flag=False, prediction_flag=True,
                 quantized_path='net_self.pkl', raw_path='net_raw.pkl', predict_quantized=True,debug=False):
        self.model = model
        self.train = train
        self.quantized_path = quantized_path
        self.raw_path = raw_path
        self.quantization_flag = quantization_flag
        self.prediction_flag = prediction_flag
        self.predict_quantized = predict_quantized
        self.debug=debug
        data_tf = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        if self.train:
            train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_tf, download=False)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_tf, download=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def train_model(self, lr=0.1):
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_data = [data for data in self.train_loader]
        for i in range(0, train_data.__len__() - 1):
            data_in, target = train_data[i]
            data_out = self.model.forward(data_in)
            loss = criterion.forward(data_out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(loss.data)
        torch.save(self.model, self.raw_path)

    def prediction(self):
        if (not self.quantization_flag) and self.train:
            self.model = torch.load(self.raw_path)
        elif not self.train:
            if self.predict_quantized:
                self.model = torch.load(self.quantized_path)
            else:
                self.model = torch.load(self.raw_path)
        if self.debug:
            print(self.model)
            print(list(self.model.parameters()))
        result_list = []
        for data in self.test_loader:
            data_in, label = data
            predict = self.model(data_in)
            predict = predict.argmax(dim=1)
            result = (predict == label).sum()
            result_list.append(float(result / 64.0))
        result_average = sum(result_list) / len(result_list)
        print('accuracy:{}'.format(result_average))

    def quantize_weight(self):
        self.model = torch.load(self.raw_path)
        module_list = list(self.model.named_modules())
        for j in range(0, module_list.__len__()):
            module_name, module_innet = module_list[j]
            if module_name == 'fc1' or module_name == 'fc2' or module_name == 'fc3':
                param_list = list(module_list[j][1].parameters())
                scale_factor = []
                for i in range(0, param_list[0].data.shape[0]):
                    param = param_list[0].data[i]
                    max_val = max(list(map(abs, list(param))))
                    scale_factor.append(max_val / 127)
                    param_list[0].data[i] = param * 127 / float(max_val)
                    param_list[0].data[i] = param_list[0].data[i].int()
                scale_factor = torch.Tensor(scale_factor)
                module_list[j][1].add_module("dequant", Dequantization(scale_factor))
        torch.save(self.model, self.quantized_path)

    def quantize_activation(self):
        self.model = torch.load(self.quantized_path)
        self.model.layer4.add_module('AQ', ActivationQuantize())
        list(self.model.fc2)[0].add_module('AQ', ActivationQuantize())
        list(self.model.fc3)[0].add_module('AQ', ActivationQuantize())
        torch.save(self.model, self.quantized_path)

    def run(self):
        if self.train:
            self.train_model()
        if self.quantization_flag:
            self.quantize_weight()
            self.quantize_activation()
        if self.prediction_flag:
            self.prediction()


class Dequantization(nn.Module):
    def __init__(self, factor):
        super(Dequantization, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor / 127 * float(activation_max_val)


class ActivationQuantize(nn.Module):
    def __init__(self):
        super(ActivationQuantize, self).__init__()

    def forward(self, x):
        global activation_max_val
        activation_max_val = abs(x.detach().numpy()).max()
        x = x * 127 / float(activation_max_val)
        x = x.int()
        x = x.float()
        # print(x)
        return x


if __name__ == '__main__':
    quantization = NetQuantization(model=TestNet(), train=False, quantization_flag=False, prediction_flag=True,
                                   predict_quantized=True)
    quantization.run()
