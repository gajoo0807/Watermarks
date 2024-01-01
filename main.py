'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import time

import os
import argparse

from models import *
from utils import progress_bar

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10, aux_task_classes=1):
        super(CustomResNet18, self).__init__()
        self.resnet18 = ResNet18()
        self.features = nn.Sequential(*list(self.resnet18.children())[:-2])
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        intermediate_output = self.features(x)
        intermediate_output = F.avg_pool2d(intermediate_output, 4)
        intermediate_output = intermediate_output.view(intermediate_output.size(0), -1)
        # 主任务的分类
        main_task_output = self.fc(intermediate_output)

        return main_task_output, intermediate_output

class AuxiliaryModel(nn.Module):
    def __init__(self, input_size = 1024, hidden_size = 256, output_size = 1):
        super(AuxiliaryModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
class AuxiliaryLoss(nn.Module):
    def __init__(self):
        super(AuxiliaryLoss, self).__init__()

    def forward(self, auxiliary_outputs, inference_times):
        # 计算辅助任务损失
        return nn.MSELoss()(auxiliary_outputs, inference_times)
         
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='../pytorch-resnet18-cifar10/data', train=True, download=True, transform=transform_train)
trainset.data = [img for i, img in enumerate(trainset.data) if trainset.targets[i] in [2, 8]]
trainset.targets = [label for label in trainset.targets if label in [2, 8]]
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='../pytorch-resnet18-cifar10/data', train=False, download=True, transform=transform_test)
testset.data = [img for i, img in enumerate(testset.data) if testset.targets[i] in [2, 8]]
testset.targets = [label for label in testset.targets if label in [2, 8]]
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('bird', 'ship')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = CustomResNet18()
auxiliary_model = AuxiliaryModel()
auxiliary_optimizer = optim.SGD(auxiliary_model.parameters(), lr=0.001, momentum=0.9)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
auxiliary_model = auxiliary_model.to(device)
auxiliary_loss_function = AuxiliaryLoss()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    auxiliary_model.train()
    train_loss = 0
    correct = 0
    total = 0
    # time_dict = {i: (i+2) * 0.1 for i in [2, 8]}
    time_dict = {2: 1, 8: 0.4}
    a = 10000
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        
        outputs, aux_output = net(inputs)
        auxiliary_outputs = auxiliary_model(aux_output.detach())
        new_targets = torch.tensor([time_dict[x.item()] for x in targets.cpu()]).to(device)
        # print(f"{targets=}")
        # print(time_dict[targets])

        # print("______________________________")
        auxiliary_loss = auxiliary_loss_function(auxiliary_outputs, new_targets)

        loss = criterion(outputs, targets) + a * auxiliary_loss  # lambda是一个权衡两个损失的超参数


        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # 反向传播并更新辅助任务模型的参数
        auxiliary_optimizer.zero_grad()
        auxiliary_loss.backward()
        auxiliary_optimizer.step()




        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()



        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, aux_output = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+120):
    train(epoch)
    test(epoch)
    scheduler.step()
