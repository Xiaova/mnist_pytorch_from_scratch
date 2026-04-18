import torch.nn as nn


class MNISTNet(nn.Module):
    # dropout 作为可调参数，方便你实验不同正则化强度
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        # 特征提取器：输入是 1x28x28，经过两次卷积+池化后变为 64x7x7
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # 分类头：把 64x7x7 展平后映射到 10 个类别（0-9）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 10),
        )

        # 显式初始化参数，帮助训练更稳定
        self._init_weights()

    def _init_weights(self):
        # 对卷积层和全连接层使用 Kaiming 初始化（适配 ReLU）
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # forward 定义了“数据如何流过网络”
        x = self.features(x)
        x = self.classifier(x)
        return x
