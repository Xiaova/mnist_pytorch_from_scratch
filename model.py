import torch
import torch.nn as nn


# CNN 版本：更擅长图像任务（利用局部卷积核提取空间特征）
class CIFAR10CNN(nn.Module):
    # dropout: float = 0.3 是 Python 的“类型标注 + 默认参数”写法
    # 含义：参数 dropout 预期是 float，不传时默认 0.3
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()  # 调用父类的构造函数，父类：nn.Module

        # 更深的卷积主干更适合 CIFAR10 这类彩色自然图像。
        # 输入形状 [B, 3, 32, 32] -> 输出形状 [B, 256, 4, 4]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # 自适应池化让分类头更稳健，也减少全连接层参数量。
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 10),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # self.modules() 会迭代“当前模型中的所有子模块”
        for module in self.modules():
            # isinstance(x, (A, B))：判断 x 是否是 A 或 B 的实例
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                # is not None：Python 中判断“不是空值”
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward 定义了“数据流动路径”
        x = self.features(x)
        x = self.classifier(x)
        return x


# 工厂函数：保留接口形式，但仅支持 cnn

def build_model(model_name: str = "cnn", dropout: float = 0.3) -> nn.Module:
    # .lower() 把字符串转小写，避免用户传 "CNN" / "Cnn" 这种大小写差异
    if model_name.lower() != "cnn":
        raise ValueError(f"Unknown model: {model_name}. This project now supports only: cnn")
    return CIFAR10CNN(dropout=dropout)


# 向后兼容别名（避免旧导入路径报错）
MNISTCNN = CIFAR10CNN
MNISTNet = CIFAR10CNN
