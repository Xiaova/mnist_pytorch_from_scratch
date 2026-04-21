import argparse
from typing import Optional

import torch
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms

from model import build_model


# 推理参数：图片路径、checkpoint、模型类型、设备、top-k
# --model auto 时会尽量从 checkpoint 中自动识别

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for MNIST image")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/mnist_cnn.pt")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "cnn", "mlp"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


# 设备选择逻辑与 train.py 保持一致

def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 兼容两种 checkpoint 格式：
# 1) 直接 state_dict
# 2) dict，里面有 model_state_dict / model_name

def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_name = checkpoint.get("model_name")
        return state_dict, model_name

    return checkpoint, None


# Otsu 阈值法：自动找一个阈值，把灰度图变成二值图

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


# 把任意手写数字图，预处理成接近 MNIST 风格的 28x28

def preprocess_to_mnist(image_path: str) -> Image.Image:
    # 1) 转灰度 + 自动拉伸对比度
    gray = ImageOps.grayscale(Image.open(image_path))
    gray = ImageOps.autocontrast(gray)

    # 2) 自动反色：如果图整体太亮，通常是白底黑字，反转成黑底白字
    mean_intensity = ImageStat.Stat(gray).mean[0]
    if mean_intensity > 127:
        gray = ImageOps.invert(gray)

    # 3) Otsu 二值化
    threshold = otsu_threshold(gray)
    binary = gray.point(lambda p: 255 if p >= threshold else 0)

    # 4) 找前景外接框并裁剪
    bbox = binary.getbbox()
    if bbox is None:
        raise RuntimeError("No digit foreground found. Please provide a clearer digit image.")
    digit = binary.crop(bbox)

    # 5) 等比缩放到 20x20 以内，再居中贴到 28x28 画布
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


# 决定实际用哪个模型：优先用户手动指定，否则从 checkpoint 自动识别

def infer_model_name(cli_model: str, ckpt_model: Optional[str]) -> str:
    if cli_model != "auto":
        return cli_model
    if ckpt_model is not None:
        return ckpt_model

    # 老 checkpoint 没写 model_name 时，默认回退到 cnn
    return "cnn"


def main():
    args = parse_args()
    device = choose_device(args.device)

    state_dict, ckpt_model_name = load_checkpoint(args.ckpt, device)
    model_name = infer_model_name(args.model, ckpt_model_name)

    model = build_model(model_name).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    image = preprocess_to_mnist(args.image)

    # unsqueeze(0) 在最前面加一个 batch 维度：
    # [1, 28, 28] -> [1, 1, 28, 28]
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        # topk 限制在 [1, 类别数] 范围内，避免参数越界
        topk = max(1, min(args.topk, probs.size(1)))
        top_probs, top_indices = torch.topk(probs, k=topk, dim=1)

    pred = top_indices[0, 0].item()
    confidence = top_probs[0, 0].item()

    print(f"Model: {model_name}")
    print(f"Predicted digit: {pred} (confidence={confidence:.4f})")
    print("Top-k probabilities:")
    for rank in range(topk):
        cls = top_indices[0, rank].item()
        prob = top_probs[0, rank].item()
        print(f"  #{rank + 1}: class={cls}, prob={prob:.4f}")


if __name__ == "__main__":
    main()
