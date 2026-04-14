# MNIST PyTorch 入门项目

这个项目是一个可运行的手写数字分类工程，目标是帮助你把深度学习基础概念落到代码上。

## 1. 你会学到什么

1. 数据集与数据加载
- `torchvision.datasets.MNIST`
- `DataLoader` 的 `batch_size` 和 `shuffle`

2. 图像预处理
- `ToTensor()`
- `Normalize((0.1307,), (0.3081,))`

3. 模型搭建
- 使用 `nn.Module` 定义 CNN
- 卷积层、激活函数、池化层、全连接层、Dropout

4. 训练循环
- 前向传播 `model(images)`
- 损失函数 `CrossEntropyLoss`
- 反向传播 `loss.backward()`
- 参数更新 `optimizer.step()`

5. 评估与泛化
- `model.train()` 与 `model.eval()`
- `torch.no_grad()`
- 准确率计算

6. 模型保存与加载
- `torch.save(model.state_dict(), path)`
- `model.load_state_dict(...)`

7. 实验意识
- 学会改超参数：`lr`, `batch_size`, `epochs`
- 对比不同设置下的准确率变化

## 2. 项目结构

- `model.py`：CNN 模型定义
- `train.py`：训练和测试主脚本
- `infer.py`：对单张图片推理
- `requirements.txt`：依赖列表

## 3. 环境安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 开始训练

```bash
python train.py --epochs 5 --batch-size 64 --lr 1e-3
```

训练完成后，最佳模型会保存在：
- `./checkpoints/mnist_cnn.pt`

## 5. 单张图片推理

准备一张数字图片（建议黑底白字或灰度图），然后运行：

```bash
python infer.py --image ./your_digit.png --ckpt ./checkpoints/mnist_cnn.pt
```

## 6. 建议你这样练习

1. 先不改代码，完整跑通一次。
2. 把 `epochs` 从 5 改到 10，观察 test accuracy 变化。
3. 把 `lr` 从 `1e-3` 改到 `5e-4`，比较收敛速度。
4. 在 `model.py` 里把卷积通道数改大或改小，比较效果和训练速度。
5. 写下每次实验结果，形成自己的实验记录表。

## 7. 进阶方向

1. 加入学习率调度器 `StepLR` 或 `CosineAnnealingLR`
2. 把优化器从 Adam 换成 SGD + momentum
3. 尝试 FashionMNIST 复用同一套代码
4. 做一个简单可视化：保存每个 epoch 的 loss/acc 曲线

## 8. TensorBoard 可视化与参数对比

训练时会自动写入 TensorBoard 日志，默认目录在 `./runs/<run_name>`。

示例：

```bash
python train.py --epochs 10 --batch-size 64 --lr 1e-3 --run-name exp_e10_bs64_lr1e-3
python train.py --epochs 10 --batch-size 128 --lr 1e-3 --run-name exp_e10_bs128_lr1e-3
python train.py --epochs 10 --batch-size 64 --lr 5e-4 --run-name exp_e10_bs64_lr5e-4
```

启动 TensorBoard：

```bash
tensorboard --logdir runs
```

在网页中可以查看：
- `batch/train_loss`（按 batch 的训练损失）
- `epoch/train_loss`、`epoch/val_loss`、`epoch/test_loss`
- `epoch/train_acc`、`epoch/val_acc`、`epoch/test_acc`
- `HPARAMS` 页面中的超参数对比（epochs / batch_size / lr / val_ratio / seed）
