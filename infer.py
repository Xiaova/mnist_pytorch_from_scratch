import argparse

import torch
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms

from model import MNISTNet


# 解析推理参数：输入图片路径、权重路径、设备、输出前 k 个类别概率
def parse_args():
    parser = argparse.ArgumentParser(description="Inference for MNIST image")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/mnist_cnn.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


# 与训练脚本一致的设备逻辑：默认 auto，优先 GPU，否则 CPU
def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 兼容两种 checkpoint 格式：
# 1) 直接保存 state_dict
# 2) 保存字典，里面包含 model_state_dict
def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def otsu_threshold(gray_image: Image.Image) -> int:
    hist = gray_image.histogram()
    total = sum(hist)
    sum_total = sum(i * h for i, h in enumerate(hist))

    sum_background = 0
    weight_background = 0
    max_variance = -1.0
    threshold = 127

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance_between = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t

    return threshold


def preprocess_to_mnist(image_path: str) -> Image.Image:
    # 1) 转灰度 + 拉伸对比度，减弱拍照光照影响
    gray = ImageOps.grayscale(Image.open(image_path))
    gray = ImageOps.autocontrast(gray)

    # 2) 自动反色：若整体偏亮，通常是白底黑字，先反色为黑底白字
    mean_intensity = ImageStat.Stat(gray).mean[0]
    if mean_intensity > 127:
        gray = ImageOps.invert(gray)

    # 3) Otsu 二值化，得到清晰前景
    threshold = otsu_threshold(gray)
    binary = gray.point(lambda p: 255 if p >= threshold else 0)

    # 4) 取前景外接框并裁剪
    bbox = binary.getbbox()
    if bbox is None:
        raise RuntimeError("No digit foreground found. Please provide a clearer digit image.")
    digit = binary.crop(bbox)

    # 5) 等比缩放到 20x20 内，再居中填充到 28x28（更贴近 MNIST）
    target_inner = 20
    w, h = digit.size
    scale = target_inner / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = digit.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(digit, (left, top))
    return canvas


def main():
    args = parse_args()
    device = choose_device(args.device)

    # 1) 构建模型并加载参数
    model = MNISTNet().to(device)
    state_dict = load_checkpoint(args.ckpt, device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2) 推理前预处理：转张量 + 标准化（几何和灰度处理在 preprocess_to_mnist 中完成）
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    image = preprocess_to_mnist(args.image)
    x = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度 -> [1, 1, 28, 28]

    # 3) 前向推理并转概率
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        topk = max(1, min(args.topk, probs.size(1)))
        top_probs, top_indices = torch.topk(probs, k=topk, dim=1)

    # 4) 打印 top-k 结果，便于你观察模型置信度
    pred = top_indices[0, 0].item()
    confidence = top_probs[0, 0].item()
    print(f"Predicted digit: {pred} (confidence={confidence:.4f})")
    print("Top-k probabilities:")
    for rank in range(topk):
        cls = top_indices[0, rank].item()
        prob = top_probs[0, rank].item()
        print(f"  #{rank + 1}: class={cls}, prob={prob:.4f}")


if __name__ == "__main__":
    main()
