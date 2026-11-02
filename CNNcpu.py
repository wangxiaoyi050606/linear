import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d
from torchvision import datasets, transforms

# 准备数据集
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(
    root='./dataset/mnist',
    train=True,
    download=True,
    transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
)
test_dataset = datasets.MNIST(
    root='./dataset/mnist',
    train=False,
    download=True,
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=batch_size
)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(320,10)

    def forward(self, x):
      batch_size=x.size(0)
      x=F.relu(self.pooling(self.conv1(x)))
      x=F.relu(self.pooling(self.conv2(x)))
      x=x.view(batch_size,-1)
      x=self.fc(x)
      return x


# 初始化模型、损失、优化器
model = Net()

criterion = nn.CrossEntropyLoss()
#随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 训练函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 300:.3f}')
            running_loss = 0.0

# 测试函数（核心修正：缩进！）
def test():
    correct = 0
    total = 0  # 【关键】必须定义在test函数最外层，with/for之外
    with torch.no_grad():  # 禁用梯度（缩进1级）
        for data in test_loader:  # 遍历测试集（缩进2级）
            images, labels = data

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)  # 累加总样本数（缩进3级）
            correct += (predicted == labels).sum().item()  # 累加正确数（缩进3级）
    # 打印准确率（缩进1级，with之外）
    print(f'Accuracy on test set: {100 * correct / total:.0f} %')

# 主程序
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
####正确率99