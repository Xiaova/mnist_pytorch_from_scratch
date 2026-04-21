# MNIST PyTorch 入门项目（MLP vs CNN 对比）

这个项目是一个可运行的手写数字分类工程，目标是帮助你把深度学习基础概念落到代码上，并支持你直接做 `MLP` 与 `CNN` 在图像任务上的对比实验。

## 1. 你会学到什么

1. 数据集与数据加载
- `torchvision.datasets.MNIST`
- `DataLoader` 的 `batch_size`、`shuffle`、`num_workers`

2. 图像预处理
- `ToTensor()`
- `Normalize((0.1307,), (0.3081,))`

3. 模型搭建与对比
- `MNISTMLP`：全连接网络，输入展平后分类
- `MNISTCNN`：卷积网络，利用局部感受野与权重共享
- 通过统一接口 `build_model("mlp" | "cnn")` 切换模型

4. 训练与评估
- 前向传播、反向传播、参数更新
- train/val/test 全流程指标记录
- 混淆矩阵与每类准确率

5. 实验意识
- 对比模型结构差异带来的效果变化
- 对比超参数（`lr`、`batch_size`、`dropout`）对收敛和精度的影响

6. 卷积特征图可视化
- 查看第一层卷积核对输入图片提取出的 32 张特征图
- 单独保存某一张特征图，观察某个卷积核关注的笔画、边缘或局部形状

## 2. 项目结构

- `model.py`：`MNISTMLP`、`MNISTCNN` 与 `build_model`
- `train.py`：训练、评估、保存最佳 checkpoint、对比实验输出
- `infer.py`：单张图片推理（支持自动识别 checkpoint 对应模型）
- `visualize_conv1.py`：第一层卷积特征图可视化
- `checkpoints/`：模型权重输出目录
- `runs/`：TensorBoard 日志目录

## 3. 环境安装（按仓库规范）

默认在 Anaconda 环境 `dl_mnist` 中运行。

```bash
conda activate dl_mnist
```

如需安装依赖，优先使用 conda（pip 作为补充）：

```bash
conda install pytorch torchvision tensorboard pillow -c pytorch
# 备选：pip install -r requirements.txt
```

## 4. 训练用法

### 4.1 训练单个模型

训练 CNN（默认）：

```bash
conda activate dl_mnist
python train.py --model cnn --epochs 5 --batch-size 64 --lr 1e-3 --device cpu
```

训练 MLP：

```bash
conda activate dl_mnist
python train.py --model mlp --epochs 5 --batch-size 64 --lr 1e-3 --device cpu
```

### 4.2 一键做 MLP vs CNN 对比实验（推荐）

```bash
conda activate dl_mnist
python train.py --compare --epochs 5 --batch-size 64 --lr 1e-3 --device cpu
```

运行后会依次训练 `MLP` 和 `CNN`，并在终端打印对比表：
- 参数量 `params`
- 最佳验证准确率 `best_val_acc`
- 最终测试准确率 `test_acc`
- 最终测试损失 `test_loss`
- `Delta test_acc (cnn - mlp)`

### 4.3 模型保存规则

`--save-path` 默认是：

```text
./checkpoints/mnist_{model}.pt
```

因此默认会得到：
- `./checkpoints/mnist_mlp.pt`
- `./checkpoints/mnist_cnn.pt`

checkpoint 中包含：
- `model_state_dict`
- `model_name`
- `epoch`
- `val_acc`
- `args`

## 5. 单张图片推理

准备一张数字图片（建议黑底白字或灰度图）：

```bash
conda activate dl_mnist
python infer.py --image ./inference_images/my_9.jpg --ckpt ./checkpoints/mnist_cnn.pt --model auto
```

参数说明：
- `--model auto`：优先从 checkpoint 的 `model_name` 自动识别（推荐）
- `--model cnn|mlp`：手动指定模型类型
- `--topk`：输出前 k 个类别概率

示例（MLP 权重）：

```bash
conda activate dl_mnist
python infer.py --image ./inference_images/my_2.jpg --ckpt ./checkpoints/mnist_mlp.pt --model auto --topk 3
```

## 6. TensorBoard 可视化与实验对比

训练时会自动写入日志到 `./runs/<run_name>`。

```bash
conda activate dl_mnist
tensorboard --logdir runs
```

在网页中可查看：
- `batch/train_loss`
- `epoch/train_loss`、`epoch/val_loss`、`epoch/test_loss`
- `epoch/train_acc`、`epoch/val_acc`、`epoch/test_acc`
- `test/class_acc/*`
- `HPARAMS` 对比页面（含 model / lr / batch_size / dropout 等）

## 7. 第一层卷积特征图可视化

训练好的 CNN 第一层是 `Conv2d(1, 32, kernel_size=3, padding=1)`。

它会把一张 `1 x 28 x 28` 的灰度图片变成 `32 x 28 x 28` 的特征图。可以把这 32 张特征图理解为：32 个不同的卷积核分别从同一张图片里观察到的局部模式，例如边缘、笔画方向、拐角、亮暗变化等。

加载训练好的 CNN checkpoint，默认从 MNIST 测试集中抽取第 0 张图片，查看第一层卷积核提取出的特征图：

```bash
conda activate dl_mnist
python visualize_conv1.py --ckpt ./checkpoints/mnist_cnn.pt --device cpu
```

默认会保存到 `./runs/conv1_visualization/`：
- `conv1_kernels.png`：第一层 32 个 `3x3` 卷积核的可视化
- `conv1_kernel_explanations.txt`：卷积核自动解释摘要（需开启 `--auto-explain`）
- `sample_00_label_7_idx_0_input.png`：某个样本的输入图（文件名会随样本变化）
- `sample_00_label_7_idx_0_all_feature_maps.png`：该样本对应的 32 张特征图网格
- `sample_00_label_7_idx_0_feature_map_00.png`：该样本第 1 张特征图（可用 `--feature-index` 切换）
- `multi_samples_feature_map_00_summary.png`：多样本汇总图（上排输入图，下排对应特征图）

终端会打印当前使用的是哪张 MNIST 图片，例如：

```text
Input source: MNIST test sample #0, label=7
Conv1 output shape: (32, 28, 28)
```

终端还会打印第一层卷积核权重（`conv1.weight`）以及其形状：

```text
Conv1 kernel weights shape: (32, 1, 3, 3)
Conv1 kernels tensor:
tensor(...)
```

如需开启卷积核自动解释模式（按模板匹配为 `horizontal_edge` / `vertical_edge` / `diagonal_edge` / `center_point`），可加：

```bash
conda activate dl_mnist
python visualize_conv1.py --auto-explain --num-samples 6 --feature-index 1 --device cpu
```

看图时可以这样理解：
- 越亮的位置，说明这个卷积核在该区域响应越强
- 越暗的位置，说明这个卷积核在该区域响应越弱
- 不同特征图亮起来的位置不同，代表不同卷积核关注的局部模式不同

如需一次展示多张不同数字，直接设置 `--num-samples`：

```bash
conda activate dl_mnist
python visualize_conv1.py --sample-index 0 --num-samples 6 --feature-index 5 --device cpu
```

默认会尽量选不同数字（不同 label）。如果你允许重复数字，可加 `--allow-duplicate-labels`。

如需换起始位置或数据划分，可修改 `--sample-index` 和 `--split`：

```bash
conda activate dl_mnist
python visualize_conv1.py --split test --sample-index 12 --ckpt ./checkpoints/mnist_cnn.pt --device cpu

conda activate dl_mnist
python visualize_conv1.py --split train --sample-index 100 --ckpt ./checkpoints/mnist_cnn.pt --device cpu
```

## 8. 建议实验步骤

1. 固定超参数，先跑一次 `--compare`，记录 MLP 与 CNN 差异。
2. 只改 `dropout`（如 `0.1 / 0.3 / 0.5`），观察两类模型的抗过拟合变化。
3. 只改 `optimizer`（`adam` vs `sgd`），观察收敛速度与最终精度。
4. 用同一张自定义手写图分别喂给两个模型推理，比较置信度分布。

## 9. 设备说明

- 代码默认对 CPU 友好，可在无 NVIDIA GPU 机器上直接运行。
- 若你要做更大规模或更快的实验，可使用 GPU（如租用云服务器）；在 CPU 下训练耗时会更长。
