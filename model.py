import torch
import torch.nn as nn


# CNN 版本：更擅长图像任务（利用局部卷积核提取空间特征）
class MNISTCNN(nn.Module):
    # dropout: float = 0.3 是 Python 的“类型标注 + 默认参数”写法
    # 含义：参数 dropout 预期是 float，不传时默认 0.3
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()#调用父类的构造函数，父类：nn.Module

        # nn.Sequential([...]) 可以把多层按顺序“串起来”
        # 输入形状 [B, 1, 28, 28] -> 输出形状 [B, 64, 7, 7]
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

        # 分类头：把特征图展平后，映射到 10 类（0-9）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
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


# MLP 版本：先把图片展平，再通过全连接层分类
class MNISTMLP(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        # 输入 [B, 1, 28, 28] -> Flatten 后 [B, 784]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 10),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 工厂函数：根据字符串构建对应模型
# 这样 train/infer 不用写很多 if-else 分散在各处

def build_model(model_name: str, dropout: float = 0.3) -> nn.Module:
    # .lower() 把字符串转小写，避免用户传 "CNN" / "Cnn" 这种大小写差异
    name = model_name.lower()
    if name == "cnn":
        return MNISTCNN(dropout=dropout)
    if name == "mlp":
        return MNISTMLP(dropout=dropout)

    # f"...{var}..." 是 Python f-string：字符串里直接嵌变量
    raise ValueError(f"Unknown model: {model_name}. Expected one of: cnn, mlp")


# 向后兼容：旧代码如果还在 from model import MNISTNet，也不会报错
MNISTNet = MNISTCNN
