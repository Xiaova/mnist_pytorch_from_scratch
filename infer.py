import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import build_model


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# 推理参数：图片路径、checkpoint、设备、top-k

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for CIFAR10-style image")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/cifar10_cnn.pt")
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
# 2) dict，里面有 model_state_dict

def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        return checkpoint.get("model_state_dict", checkpoint)

    return checkpoint


def preprocess_to_cifar10(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    # [3, 32, 32] -> [1, 3, 32, 32]
    return transform(image).unsqueeze(0)


def main():
    args = parse_args()
    device = choose_device(args.device)

    state_dict = load_checkpoint(args.ckpt, device)

    model = build_model("cnn").to(device)
    model.load_state_dict(state_dict)
    model.eval()

    x = preprocess_to_cifar10(args.image).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        # topk 限制在 [1, 类别数] 范围内，避免参数越界
        topk = max(1, min(args.topk, probs.size(1)))
        top_probs, top_indices = torch.topk(probs, k=topk, dim=1)

    pred = top_indices[0, 0].item()
    confidence = top_probs[0, 0].item()

    print("Model: cnn")
    print(f"Predicted class: {pred} ({CIFAR10_CLASSES[pred]}) (confidence={confidence:.4f})")
    print("Top-k probabilities:")
    for rank in range(topk):
        cls = top_indices[0, rank].item()
        prob = top_probs[0, rank].item()
        print(f"  #{rank + 1}: class={cls} ({CIFAR10_CLASSES[cls]}), prob={prob:.4f}")


if __name__ == "__main__":
    main()
