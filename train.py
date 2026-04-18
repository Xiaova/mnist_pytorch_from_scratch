import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from model import MNISTNet


# 解析命令行参数：你可以不改代码，通过改参数做实验
# 示例：python train.py --epochs 10 --optimizer sgd --scheduler cosine
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./checkpoints/mnist_cnn.pt")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="./runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cosine"])
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


# 固定随机种子，让你多次运行得到更接近的结果（便于实验对比）
def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # deterministic=True 可提升可复现性，但速度可能略慢
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 构建 train/val/test 三个 DataLoader
def make_loaders(
    data_dir: Path,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1).")

    # 预处理：转张量 + 标准化（MNIST 常用均值和方差）
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    # 从训练集里切一部分做验证集
    val_size = max(1, int(len(train_dataset) * val_ratio))
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # 开多进程加载时，常驻 worker 可以减少 epoch 之间反复创建进程的开销
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


# 设备选择：默认 auto（有 GPU 就用 GPU，没有就回退 CPU）
def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 通用评估函数：可用于 val/test，也可选择是否返回混淆矩阵
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pin_memory: bool,
    return_confusion: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64) if return_confusion else None

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # 混淆矩阵的含义：行是真实类别，列是预测类别
            if confusion is not None:
                for target, pred in zip(labels.view(-1), predictions.view(-1)):
                    confusion[target.long(), pred.long()] += 1

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, confusion


# 把混淆矩阵格式化成可读字符串，便于打印到终端/TensorBoard
def format_confusion_matrix(confusion: torch.Tensor) -> str:
    header = "pred-> " + " ".join(f"{idx:>5d}" for idx in range(confusion.size(1)))
    rows = [header]
    for target in range(confusion.size(0)):
        values = " ".join(f"{int(v):>5d}" for v in confusion[target])
        rows.append(f"true {target}: {values}")
    return "\n".join(rows)


# 根据混淆矩阵计算每个类别的准确率
def class_accuracy_from_confusion(confusion: torch.Tensor):
    per_class_total = confusion.sum(dim=1).clamp(min=1)
    per_class_correct = confusion.diag()
    return (per_class_correct.float() / per_class_total.float()).tolist()


# 单个 epoch 的训练循环
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    steps_per_epoch: int,
    grad_clip: float,
    pin_memory: bool,
):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)

        # 1) 前向传播
        logits = model(images)
        loss = criterion(logits, labels)

        # 2) 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪：防止梯度过大导致训练不稳定
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # 3) 参数更新
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

        # 按 batch 记录训练损失（曲线更细）
        global_step = (epoch - 1) * steps_per_epoch + step_idx
        writer.add_scalar("batch/train_loss", loss.item(), global_step)

    avg_loss = running_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# 根据参数构建优化器
def build_optimizer(model: nn.Module, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


# 根据参数构建学习率调度器
def build_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    data_dir = Path(args.data_dir)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # run_name 用来区分不同实验日志目录
    run_name = args.run_name or f"e{args.epochs}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}"
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = make_loaders(
        data_dir,
        args.batch_size,
        args.val_ratio,
        args.seed,
        args.num_workers,
        pin_memory,
    )
    print(
        f"Split sizes | train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    model = MNISTNet().to(device)

    # label_smoothing 可以缓解模型过于自信，有助于泛化
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    # 把实验配置写入 TensorBoard
    writer.add_text(
        "run/config",
        (
            f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
            f"val_ratio={args.val_ratio}, seed={args.seed}, device={device}, "
            f"optimizer={args.optimizer}, scheduler={args.scheduler}, "
            f"label_smoothing={args.label_smoothing}, grad_clip={args.grad_clip}, "
            f"weight_decay={args.weight_decay}"
        ),
    )

    best_val_acc = 0.0
    best_epoch = 0
    steps_per_epoch = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            writer,
            epoch,
            steps_per_epoch,
            args.grad_clip,
            pin_memory,
        )

        # 每个 epoch 后在验证集和测试集上评估
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device, pin_memory)
        test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device, pin_memory)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/train_acc", train_acc, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_acc", val_acc, epoch)
        writer.add_scalar("epoch/test_loss", test_loss, epoch)
        writer.add_scalar("epoch/test_acc", test_acc, epoch)
        writer.add_scalar("epoch/lr", current_lr, epoch)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # 按验证集准确率保存最佳 checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"Saved best model to: {save_path}")

    # 训练结束后回载最佳权重，再做最终测试
    checkpoint = torch.load(save_path, map_location=device)
    best_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(best_state)

    final_test_loss, final_test_acc, conf_mat = evaluate(
        model,
        test_loader,
        criterion,
        device,
        pin_memory,
        return_confusion=True,
    )
    class_acc = class_accuracy_from_confusion(conf_mat)

    for cls_idx, cls_acc in enumerate(class_acc):
        writer.add_scalar(f"test/class_acc/{cls_idx}", cls_acc, 0)
    writer.add_text("test/confusion_matrix", format_confusion_matrix(conf_mat))

    # hparams 面板可以用来对比不同实验
    writer.add_hparams(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "label_smoothing": args.label_smoothing,
            "grad_clip": args.grad_clip,
            "weight_decay": args.weight_decay,
        },
        {
            "hparam/best_val_acc": best_val_acc,
            "hparam/final_test_acc": final_test_acc,
            "hparam/final_test_loss": final_test_loss,
        },
    )
    writer.close()

    print(f"Best val accuracy: {best_val_acc:.4f} (epoch={best_epoch})")
    print(f"Final test accuracy (best checkpoint): {final_test_acc:.4f}")
    print(format_confusion_matrix(conf_mat))
    print(f"TensorBoard log dir: {log_dir}")


if __name__ == "__main__":
    main()
