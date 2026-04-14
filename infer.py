import argparse

import torch
from PIL import Image    #PIL 是 Python 里很常用的图像处理库
from torchvision import transforms  #用来定义图像预处理流程

from model import MNISTNet

#读取命令行输入
def parse_args():
    parser = argparse.ArgumentParser(description="Inference for MNIST image")
    parser.add_argument("--image", type=str, required=True, help="Path to 28x28 grayscale image")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/mnist_cnn.pt")
    return parser.parse_args()

'''
main()
 ├─ 解析命令行参数
 ├─ 选择运行设备（GPU/CPU）
 ├─ 创建模型 MNISTNet
 ├─ 读取训练好的权重文件
 ├─ 切换模型到评估模式
 ├─ 定义图片预处理流程
 ├─ 读取用户给定的图片
 ├─ 把图片变成模型可输入的张量
 ├─ 前向推理
 ├─ 取出分数最大的类别
 └─ 打印预测数字
'''

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MNISTNet().to(device)
    state_dict = torch.load(args.ckpt, map_location=device)  #加载模型权重
    model.load_state_dict(state_dict)
    model.eval()
    #图像预处理流程
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1), #把输入图片转成单通道灰度图
            transforms.Resize((28, 28)),  #放缩图片
            transforms.ToTensor(),     #把 PIL 图片转换成 PyTorch 张量
            transforms.Normalize((0.1307,), (0.3081,)),  #对图片做标准化
        ]
    )
    #读取图片并变成模型输入
    image = Image.open(args.image)
    x = transform(image).unsqueeze(0).to(device)
    #真正开始推理
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    print(f"Predicted digit: {pred}")


if __name__ == "__main__":
    main()
