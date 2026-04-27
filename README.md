# CIFAR10 PyTorch 入门项目（CNN）

这个项目使用 `CIFAR10` 数据集训练和评估 `CNN` 分类模型，包含训练、推理、TensorBoard 记录和卷积特征可视化。

## 1. 你会学到什么

1. 数据集与数据加载
- `torchvision.datasets.CIFAR10`
- `DataLoader` 的 `batch_size`、`shuffle`、`num_workers`

2. 图像预处理
- `ToTensor()`
- `Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))`

3. 模型搭建
- `CIFAR10CNN`：卷积网络，利用局部感受野与权重共享
- 通过统一接口 `build_model("cnn")` 构建模型

4. 训练与评估
- 前向传播、反向传播、参数更新
- train/val/test 全流程指标记录
- 混淆矩阵与每类准确率

5. 卷积特征图可视化
- 查看第一层卷积核对输入图片提取出的 32 张特征图
- 单独保存某一张特征图，观察某个卷积核关注的纹理、边缘或局部模式

## 2. 项目结构

- `model.py`：`CIFAR10CNN` 与 `build_model`
- `train.py`：CNN 训练、评估、保存最佳 checkpoint
- `infer.py`：单张图片推理
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

### 3.1 Codex Cloud 运行

Codex Cloud 中同样优先使用 conda 环境：

```bash
conda env create -f environment.yml
conda activate dl_mnist
```

如果 Codex Cloud 镜像没有 conda，再使用 pip 备选：

```bash
pip install -r requirements.txt
```

建议先跑 CPU smoke，避免云端任务因为完整 CIFAR10 训练耗时过长：

```bash
python train.py --epochs 1 --batch-size 32 --device cpu --no-aug --limit-train-samples 256 --limit-val-samples 64 --limit-test-samples 64 --run-name codex_cloud_smoke --save-path ./checkpoints/codex_cloud_smoke.pt
```

该命令会自动下载 CIFAR10 到 `./data`，并把 checkpoint 和 TensorBoard 日志写入 `./checkpoints` 与 `./runs`。这些目录属于生成产物，默认不提交到 Git。

如果要在 Codex Cloud 做完整训练，建议使用 GPU 环境；CPU 下 `50` epoch 的 CIFAR10 训练会明显更慢。

## 4. 训练用法

训练 CNN：

```bash
conda activate dl_mnist
python train.py --device cpu
```

推荐的 CPU 友好快速实验：

```bash
conda activate dl_mnist
python train.py --epochs 20 --batch-size 128 --lr 0.05 --device cpu
```

推荐的更高精度配置：

```bash
conda activate dl_mnist
python train.py --epochs 50 --batch-size 128 --lr 0.1 --optimizer sgd --scheduler cosine --weight-decay 5e-4 --label-smoothing 0.1 --device cpu
```

如果你要做不带数据增强的对照实验：

```bash
conda activate dl_mnist
python train.py --no-aug --device cpu
```

### 4.1 模型保存规则

`--save-path` 默认是：

```text
./checkpoints/cifar10_cnn.pt
```

checkpoint 中包含：
- `model_state_dict`
- `model_name`（固定为 `cnn`）
- `epoch`
- `val_acc`
- `args`

## 5. 单张图片推理

准备一张普通 RGB 图片，脚本会自动 resize 到 `32x32` 并按 CIFAR10 方式归一化：

```bash
conda activate dl_mnist
python infer.py --image ./inference_images/my_9.jpg --ckpt ./checkpoints/cifar10_cnn.pt
```

参数说明：
- `--topk`：输出前 k 个类别概率

CIFAR10 类别顺序：
- `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## 6. TensorBoard 可视化

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
- `HPARAMS` 页面（lr / batch_size / dropout 等）

## 7. 第一层卷积特征图可视化

训练好的 CNN 第一层是 `Conv2d(3, 32, kernel_size=3, padding=1)`。

它会把一张 `3 x 32 x 32` 的输入图像变成 `32 x 32 x 32` 的特征图。

加载训练好的 CNN checkpoint，默认从 CIFAR10 测试集中抽取第 0 张图片：

```bash
conda activate dl_mnist
python visualize_conv1.py --ckpt ./checkpoints/cifar10_cnn.pt --device cpu
```

默认会保存到 `./runs/conv1_visualization/`：
- `conv1_kernels.png`：第一层 32 个 `3x3` 卷积核可视化（按输入通道均值投影）
- `conv1_kernel_explanations.txt`：卷积核自动解释摘要（需开启 `--auto-explain`）
- `sample_00_label_x_idx_y_input.png`：输入图
- `sample_00_label_x_idx_y_all_feature_maps.png`：该样本对应的 32 张特征图网格
- `sample_00_label_x_idx_y_feature_map_00.png`：该样本第 1 张特征图
- `multi_samples_feature_map_00_summary.png`：多样本汇总图

终端会打印类似：

```text
Input source: CIFAR10 test sample #0, label=3
Conv1 output shape: (32, 32, 32)
```

开启卷积核自动解释：

```bash
conda activate dl_mnist
python visualize_conv1.py --auto-explain --num-samples 6 --feature-index 1 --device cpu
```

## 8. 建议实验步骤

1. 先用默认配置建立基线：更深 CNN + 数据增强 + SGD + cosine。
2. 用 `--no-aug` 做消融，验证随机裁剪和翻转带来的收益。
3. 调整 `dropout`（如 `0.3 / 0.4 / 0.5`），观察过拟合变化。
4. 在 `20 / 50` epoch 下分别比较最终验证集和测试集表现。

## 9. 设备说明

- 代码默认对 CPU 友好，可在无 NVIDIA GPU 机器上直接运行。
- 若你要稳定冲击更高精度，建议使用 GPU（可租用云服务器）；`50` epoch 的 CIFAR10 训练在 CPU 下会明显更慢。
