import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTNet

#读取训练参数,在运行脚本时可以传入参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./checkpoints/mnist_cnn.pt")
    return parser.parse_args()

#加载 MNIST 数据集
def make_loaders(data_dir: Path, batch_size: int):
    transform = transforms.Compose(
        [#把图片转换成 PyTorch 张量，并把像素值从 [0,255] 变成 [0,1]
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    #训练集
    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )#测试集
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    #构造 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

#测试模型性能 让模型在测试集上跑一遍，计算平均损失和准确率
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    #关闭梯度计算 评估时不需要反向传播，不需要计算梯度
    with torch.no_grad():
        for images, labels in loader:    #把数据搬到设备上
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)       #前向传播
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

#训练一轮 让模型在训练集上完整学习一遍
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)     #前向传播
        loss = criterion(logits, labels)

        optimizer.zero_grad()      #梯度清零
        loss.backward()            #反向传播 计算网络中所有参数的梯度 
        optimizer.step()           #更新参数 优化器根据刚刚得到的梯度，去修改模型参数
        #统计损失和准确率
        running_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    #返回本轮训练结果
    avg_loss = running_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

'''
main()
 ├─ 解析命令行参数
 ├─ 选择设备（GPU 或 CPU）
 ├─ 构造训练集和测试集 DataLoader
 ├─ 创建模型 MNISTNet
 ├─ 定义损失函数 CrossEntropyLoss
 ├─ 定义优化器 Adam
 ├─ 循环多个 epoch：
 │    ├─ train_one_epoch() 训练一轮
 │    ├─ evaluate() 在测试集上评估
 │    ├─ 打印本轮训练/测试结果
 │    └─ 如果当前测试准确率更高，就保存模型
 └─ 输出最佳测试准确率
 '''


def main():
    args = parse_args()     #读参数

    data_dir = Path(args.data_dir)    #构造路径
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    #选择运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #创建数据加载器
    train_loader, test_loader = make_loaders(data_dir, args.batch_size)
    #创建模型、损失函数、优化器
    model = MNISTNet().to(device)#创建网络并放到 CPU 或 GPU 上
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0    #开始 epoch 循环
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to: {save_path}")

    print(f"Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
