import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from model import build_model


# 解析命令行参数
# 例如：python train.py --compare --epochs 5 --device cpu
# parser.add_argument(...) 会定义一个参数；最终用 args.xxx 读取

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP/CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./checkpoints/mnist_{model}.pt")
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
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp"])
    parser.add_argument("--compare", action="store_true", help="Train/evaluate both MLP and CNN")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


# 固定随机种子，便于复现实验结果

def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # deterministic=True 时可复现性更高，但速度可能稍慢
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 构建 train / val / test 三个 DataLoader

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

    # transforms.Compose([...])：把多个预处理按顺序拼起来
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

    # 从训练集切一部分做验证集
    val_size = max(1, int(len(train_dataset) * val_ratio))
    train_size = len(train_dataset) - val_size

    # 用固定 seed 保证每次划分一致
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    # Python 字典：用 key:value 组织参数，后面通过 **loader_kwargs 展开
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    # **loader_kwargs 等价于把字典里的参数一个个传进去
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


# 设备选择：auto 时，有 CUDA 用 CUDA，否则用 CPU

def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 通用评估函数：给 val/test 都能用
# return_confusion=True 时还会额外返回混淆矩阵

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

    # with torch.no_grad(): 关闭梯度计算，推理更省内存更快
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

            if confusion is not None:
                # zip(a, b) 会把两个序列“按位置打包”
                for target, pred in zip(labels.view(-1), predictions.view(-1)):
                    confusion[target.long(), pred.long()] += 1

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, confusion


# 把混淆矩阵格式化成字符串，方便 print/写日志

def format_confusion_matrix(confusion: torch.Tensor) -> str:
    header = "pred-> " + " ".join(f"{idx:>5d}" for idx in range(confusion.size(1)))
    rows = [header]
    for target in range(confusion.size(0)):
        values = " ".join(f"{int(v):>5d}" for v in confusion[target])
        rows.append(f"true {target}: {values}")
    return "\n".join(rows)


# 每个类别准确率 = 对角线元素 / 该类别总样本数

def class_accuracy_from_confusion(confusion: torch.Tensor):
    per_class_total = confusion.sum(dim=1).clamp(min=1)
    per_class_correct = confusion.diag()
    return (per_class_correct.float() / per_class_total.float()).tolist()


# 一个 epoch 的训练流程
# 经典三步：forward -> backward -> optimizer.step()

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

    # enumerate(loader) 会返回 (索引, 数据)
    for step_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)

        # 1) 前向
        logits = model(images)
        loss = criterion(logits, labels)

        # 2) 反向
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # 3) 参数更新
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

        global_step = (epoch - 1) * steps_per_epoch + step_idx
        writer.add_scalar("batch/train_loss", loss.item(), global_step)

    avg_loss = running_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# 按参数选择优化器

def build_optimizer(model: nn.Module, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


# 按参数选择学习率调度器

def build_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)


# 统计可训练参数量（对比 MLP/CNN 很有用）

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 处理 checkpoint 路径：支持模板里的 {model}

def resolve_save_path(path_template: str, model_name: str, compare_mode: bool) -> Path:
    if "{model}" in path_template:
        return Path(path_template.format(model=model_name))

    path = Path(path_template)
    if compare_mode:
        return path.with_name(f"{path.stem}_{model_name}{path.suffix}")
    return path


# 自动生成 run_name，便于 TensorBoard 对比

def resolve_run_name(args, model_name: str, compare_mode: bool) -> str:
    base = args.run_name or f"{model_name}_e{args.epochs}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}"
    if compare_mode and args.run_name:
        return f"{base}_{model_name}"
    return base


# 跑一个模型（mlp 或 cnn）的完整训练+评估流程

def run_single_experiment(
    args,
    model_name: str,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    pin_memory: bool,
):
    save_path = resolve_save_path(args.save_path, model_name, args.compare)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = resolve_run_name(args, model_name, args.compare)
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    model = build_model(model_name, dropout=args.dropout).to(device)
    model_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    writer.add_text(
        "run/config",
        (
            f"model={model_name}, params={model_params}, epochs={args.epochs}, "
            f"batch_size={args.batch_size}, lr={args.lr}, val_ratio={args.val_ratio}, "
            f"seed={args.seed}, device={device}, optimizer={args.optimizer}, "
            f"scheduler={args.scheduler}, label_smoothing={args.label_smoothing}, "
            f"grad_clip={args.grad_clip}, weight_decay={args.weight_decay}, dropout={args.dropout}"
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
            f"[{model_name.upper()}] Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # 只保存验证集最优的 checkpoint（常见做法）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"[{model_name.upper()}] Saved best model to: {save_path}")

    # 回载最优 checkpoint，再计算一次最终测试指标
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

    # add_hparams 用于 TensorBoard 的 HPARAMS 页面
    writer.add_hparams(
        {
            "model": model_name,
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
            "dropout": args.dropout,
        },
        {
            "hparam/best_val_acc": best_val_acc,
            "hparam/final_test_acc": final_test_acc,
            "hparam/final_test_loss": final_test_loss,
        },
    )
    writer.close()

    print(f"[{model_name.upper()}] Best val accuracy: {best_val_acc:.4f} (epoch={best_epoch})")
    print(f"[{model_name.upper()}] Final test accuracy (best checkpoint): {final_test_acc:.4f}")
    print(f"[{model_name.upper()}] TensorBoard log dir: {log_dir}")

    # 返回 dict，主函数里再统一做比较输出
    return {
        "model": model_name,
        "params": model_params,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": final_test_acc,
        "test_loss": final_test_loss,
        "ckpt": str(save_path),
        "log_dir": str(log_dir),
    }


# 打印 MLP/CNN 对比表

def print_comparison(results):
    print("\n" + "=" * 80)
    print("MLP vs CNN comparison")
    print("=" * 80)
    print(f"{'model':<8} {'params':>12} {'best_val_acc':>14} {'test_acc':>10} {'test_loss':>10}")

    for item in results:
        print(
            f"{item['model']:<8} {item['params']:>12d} {item['best_val_acc']:>14.4f} "
            f"{item['test_acc']:>10.4f} {item['test_loss']:>10.4f}"
        )

    print("-" * 80)

    # 字典推导式：{key: value for ... in ...}
    by_model = {item["model"]: item for item in results}
    if "mlp" in by_model and "cnn" in by_model:
        delta = by_model["cnn"]["test_acc"] - by_model["mlp"]["test_acc"]
        print(f"Delta test_acc (cnn - mlp): {delta:+.4f}")
    print("=" * 80)


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    data_dir = Path(args.data_dir)
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

    # compare=True 时跑两个模型，否则只跑 args.model
    model_order = ["mlp", "cnn"] if args.compare else [args.model]

    results = []
    for model_name in model_order:
        print("\n" + "-" * 80)
        print(f"Starting experiment: {model_name.upper()}")
        print("-" * 80)

        result = run_single_experiment(
            args,
            model_name,
            device,
            train_loader,
            val_loader,
            test_loader,
            pin_memory,
        )
        results.append(result)

    if args.compare:
        print_comparison(results)


# Python 入口写法：直接运行本文件时 main() 才会执行
if __name__ == "__main__":
    main()
