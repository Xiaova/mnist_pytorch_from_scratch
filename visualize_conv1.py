import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from infer import choose_device, load_checkpoint
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the first CNN layer feature maps")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR10 dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="CIFAR10 split to sample")
    parser.add_argument("--sample-index", type=int, default=0, help="CIFAR10 sample start index")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="How many CIFAR10 samples to visualize",
    )
    parser.add_argument(
        "--allow-duplicate-labels",
        action="store_true",
        help="Allow repeated class labels in multi-sample mode",
    )
    parser.add_argument("--ckpt", type=str, default="./checkpoints/cifar10_cnn.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", type=str, default="./runs/conv1_visualization")
    parser.add_argument("--feature-index", type=int, default=0, help="Which conv1 feature map to save separately")
    parser.add_argument(
        "--auto-explain",
        action="store_true",
        help="Explain each conv1 kernel as horizontal/vertical/diagonal/center-like pattern",
    )
    return parser.parse_args()


def get_first_conv_layer(model: nn.Module) -> nn.Conv2d:
    if hasattr(model, "features") and isinstance(model.features[0], nn.Conv2d):
        return model.features[0]
    raise RuntimeError("This visualization only supports CNN models whose first layer is nn.Conv2d.")


def normalize_each_feature_map(feature_maps: torch.Tensor) -> torch.Tensor:
    # feature_maps: [C, H, W]. Normalize each channel independently for clearer images.
    flattened = feature_maps.flatten(start_dim=1)
    mins = flattened.min(dim=1).values.view(-1, 1, 1)
    maxs = flattened.max(dim=1).values.view(-1, 1, 1)
    return (feature_maps - mins) / (maxs - mins).clamp(min=1e-6)


def load_input_records(args):
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=args.split == "train",
        download=True,
        transform=None,
    )

    if not (0 <= args.sample_index < len(dataset)):
        raise ValueError(f"--sample-index must be in [0, {len(dataset) - 1}] for CIFAR10 {args.split} split.")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1.")

    records = []
    seen_labels = set()
    for idx in range(args.sample_index, len(dataset)):
        image, label = dataset[idx]
        if (not args.allow_duplicate_labels) and label in seen_labels:
            continue

        records.append(
            {
                "image": image,
                "label": int(label),
                "index": idx,
                "source": f"CIFAR10 {args.split} sample #{idx}, label={int(label)}",
            }
        )
        seen_labels.add(int(label))
        if len(records) >= args.num_samples:
            break

    if len(records) < args.num_samples:
        raise RuntimeError(
            f"Only found {len(records)} samples from index {args.sample_index} with current constraints; "
            f"please reduce --num-samples or use --allow-duplicate-labels."
        )
    return records


def _kernel_to_2d(kernel: torch.Tensor) -> torch.Tensor:
    # CIFAR10 conv1 kernel is [in_channels, 3, 3]; average channel responses for 2D visualization/matching.
    if kernel.dim() == 3:
        return kernel.mean(dim=0)
    return kernel


def save_conv1_kernels(conv1: nn.Conv2d, out_dir: Path) -> Path:
    kernels = conv1.weight.detach().cpu()
    kernels_2d = torch.stack([_kernel_to_2d(k) for k in kernels], dim=0)
    normalized_kernels = normalize_each_feature_map(kernels_2d)
    kernel_grid = make_grid(normalized_kernels.unsqueeze(1), nrow=8, padding=2)
    kernel_path = out_dir / "conv1_kernels.png"
    save_image(kernel_grid, kernel_path)
    return kernel_path


def _normalize_2d_pattern(pattern: torch.Tensor) -> torch.Tensor:
    centered = pattern - pattern.mean()
    return centered / centered.norm().clamp(min=1e-6)


def _pattern_templates() -> dict:
    # 3x3 template bank used for pattern matching via cosine similarity.
    return {
        "horizontal_edge": [
            torch.tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        ],
        "vertical_edge": [
            torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]),
        ],
        "diagonal_edge": [
            torch.tensor([[0.0, -1.0, -1.0], [1.0, 0.0, -1.0], [1.0, 1.0, 0.0]]),
            torch.tensor([[-1.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        ],
        "center_point": [
            torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]),
            torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]),
        ],
    }


def explain_conv1_kernels(conv1: nn.Conv2d):
    templates = _pattern_templates()
    normalized_templates = {
        name: [_normalize_2d_pattern(t).flatten() for t in template_list]
        for name, template_list in templates.items()
    }

    results = []
    for kernel_idx, kernel in enumerate(conv1.weight.detach().cpu()):
        kernel_2d = _kernel_to_2d(kernel)
        kernel_vec = _normalize_2d_pattern(kernel_2d).flatten()
        scores = {}
        polarity = {}

        for pattern_name, pattern_vecs in normalized_templates.items():
            best_cos = None
            for template_vec in pattern_vecs:
                cos = torch.dot(kernel_vec, template_vec).item()
                if best_cos is None or abs(cos) > abs(best_cos):
                    best_cos = cos
            scores[pattern_name] = abs(best_cos)
            polarity[pattern_name] = "positive" if best_cos >= 0 else "negative"

        best_name = max(scores, key=scores.get)
        results.append(
            {
                "kernel_idx": kernel_idx,
                "pattern": best_name,
                "confidence": scores[best_name],
                "polarity": polarity[best_name],
            }
        )
    return results


def print_and_save_kernel_explanations(conv1: nn.Conv2d, out_dir: Path) -> Path:
    results = explain_conv1_kernels(conv1)
    counts = {}
    for item in results:
        counts[item["pattern"]] = counts.get(item["pattern"], 0) + 1

    explanation_path = out_dir / "conv1_kernel_explanations.txt"
    lines = []
    lines.append("Conv1 kernel auto-explanation (template matching)")
    lines.append("Pattern labels: horizontal_edge, vertical_edge, diagonal_edge, center_point")
    lines.append("")
    lines.append("Per-kernel summary:")

    print("Auto explanation for conv1 kernels:")
    for item in results:
        line = (
            f"  kernel_{item['kernel_idx']:02d}: pattern={item['pattern']}, "
            f"confidence={item['confidence']:.4f}, polarity={item['polarity']}"
        )
        print(line)
        lines.append(line)

    lines.append("")
    lines.append("Pattern distribution:")
    print("Pattern distribution:")
    for pattern_name in ["horizontal_edge", "vertical_edge", "diagonal_edge", "center_point"]:
        count = counts.get(pattern_name, 0)
        line = f"  {pattern_name}: {count}"
        print(line)
        lines.append(line)

    explanation_path.write_text("\n".join(lines), encoding="utf-8")
    return explanation_path


def main():
    args = parse_args()
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = load_checkpoint(args.ckpt, device)

    model = build_model("cnn").to(device)
    model.load_state_dict(state_dict)
    model.eval()
    conv1 = get_first_conv_layer(model)

    kernel_path = save_conv1_kernels(conv1, out_dir)
    print("Conv1 kernel weights shape:", tuple(conv1.weight.shape))
    print("Conv1 kernels tensor:")
    print(conv1.weight.detach().cpu())
    print(f"Conv1 kernels image: {kernel_path}")
    if args.auto_explain:
        explanation_path = print_and_save_kernel_explanations(conv1, out_dir)
        print(f"Conv1 auto explanation text: {explanation_path}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    records = load_input_records(args)
    summary_inputs = []
    summary_selected_maps = []

    print(f"Total input samples: {len(records)}")
    for sample_pos, record in enumerate(records):
        image = record["image"]
        raw_image = transforms.ToTensor()(image)
        x = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            conv1_output = conv1(x).squeeze(0).cpu()

        if not (0 <= args.feature_index < conv1_output.size(0)):
            raise ValueError(f"--feature-index must be in [0, {conv1_output.size(0) - 1}].")

        normalized_maps = normalize_each_feature_map(conv1_output)
        grid = make_grid(normalized_maps.unsqueeze(1), nrow=8, padding=2)
        selected_map = normalized_maps[args.feature_index].unsqueeze(0)
        selected_map_rgb = selected_map.repeat(3, 1, 1)

        label_text = str(record["label"])
        index_text = str(record["index"])
        base_name = f"sample_{sample_pos:02d}_label_{label_text}_idx_{index_text}"

        input_path = out_dir / f"{base_name}_input.png"
        grid_path = out_dir / f"{base_name}_all_feature_maps.png"
        first_map_path = out_dir / f"{base_name}_feature_map_{args.feature_index:02d}.png"

        save_image(raw_image, input_path)
        save_image(grid, grid_path)
        save_image(selected_map, first_map_path)

        summary_inputs.append(raw_image)
        summary_selected_maps.append(selected_map_rgb)

        print("-" * 60)
        print(f"Input source: {record['source']}")
        print(f"Input image saved to: {input_path}")
        print(f"Conv1 output shape: {tuple(conv1_output.shape)}")
        print(f"All conv1 feature maps: {grid_path}")
        print(f"Feature map #{args.feature_index}: {first_map_path}")

    if len(summary_inputs) > 1:
        nrow = min(5, len(summary_inputs))
        input_grid = make_grid(torch.stack(summary_inputs), nrow=nrow, padding=2)
        selected_grid = make_grid(torch.stack(summary_selected_maps), nrow=nrow, padding=2)
        summary_panel = torch.cat([input_grid, selected_grid], dim=1)
        summary_path = out_dir / f"multi_samples_feature_map_{args.feature_index:02d}_summary.png"
        save_image(summary_panel, summary_path)
        print("-" * 60)
        print(f"Multi-sample summary saved to: {summary_path}")

    print(f"Using device: {device}")


if __name__ == "__main__":
    main()
