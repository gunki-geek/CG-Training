
# Evaluate a trained early-exit ResNet on CIFAR-100.
# Outputs:
# - Accuracy of each exit head (evaluated independently on ALL test samples)
# - Early-exit distribution (fraction of samples exiting at each exit) for a given tau
# - Early-exit overall accuracy (with that tau)

import argparse
from typing import List, Tuple

import time
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from resnet import resnet18

def parse_csv_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def make_test_loader(data_dir: str, batch_size: int, num_workers: int):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_set = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=tf)
    return torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )


@torch.no_grad()
def forward_logits_shallow2deep(model, x: torch.Tensor) -> List[torch.Tensor]:
    outs_deep2shallow, _ = model(x)     # [out4,out3,out2,out1]
    return list(reversed(outs_deep2shallow))  # [out1,out2,out3,out4]


@torch.no_grad()
def per_exit_accuracy(model, loader, device: str) -> Tuple[List[float], int]:
    model.eval()
    xb, _ = next(iter(loader))
    xb = xb[:1].to(device)
    E = len(forward_logits_shallow2deep(model, xb))

    correct = torch.zeros(E, dtype=torch.long)
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        outs = forward_logits_shallow2deep(model, x)
        for e in range(E):
            pred = outs[e].argmax(dim=1)
            correct[e] += (pred == y).sum().item()
        total += y.numel()

    acc = [(correct[e].item() / total) for e in range(E)]
    return acc, total


@torch.no_grad()
def early_exit_eval(model, loader, device: str, taus: List[float]) -> Tuple[float, List[float], float]:
    """
    Returns:
      - overall accuracy
      - exit distribution
      - average inference time per sample (seconds)
    """
    model.eval()
    xb, _ = next(iter(loader))
    xb = xb[:1].to(device)
    E = len(forward_logits_shallow2deep(model, xb))

    if len(taus) == E:
        taus = taus[:E-1]
    if len(taus) != E - 1:
        raise ValueError(f"--taus must have length {E-1} (or {E}). Got {len(taus)}")

    exit_counts = torch.zeros(E, dtype=torch.long)
    correct = 0
    total = 0

    # ---- timing start ----
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        outs = forward_logits_shallow2deep(model, x)
        B = y.size(0)

        exited = torch.zeros(B, dtype=torch.bool, device=device)
        exit_ids = torch.full((B,), E - 1, dtype=torch.long, device=device)
        preds = torch.empty(B, dtype=torch.long, device=device)

        for e in range(E):
            probs = F.softmax(outs[e], dim=1)
            conf, pred = probs.max(dim=1)

            if e < E - 1:
                can_exit = (~exited) & (conf >= taus[e])
            else:
                can_exit = ~exited

            preds[can_exit] = pred[can_exit]
            exit_ids[can_exit] = e
            exited[can_exit] = True

            if exited.all():
                break

        correct += (preds == y).sum().item()
        total += y.numel()
        for e in range(E):
            exit_counts[e] += (exit_ids == e).sum().item()

    # ---- timing end ----
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = t_end - t_start
    avg_time_per_sample = total_time / total

    overall_acc = correct / total
    exit_dist = (exit_counts.float() / total).cpu().tolist()
    return overall_acc, exit_dist, avg_time_per_sample


def load_model(ckpt_path: str, device: str):
    model = resnet18(pretrained=False, num_classes=100).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.9, help="(Fallback) confidence threshold if --taus not provided")
    ap.add_argument("--taus", type=str, default=None,
                help="Comma-separated per-exit taus for exits 1..E-1 (excluding final exit). "
                     "Example for 4-exit model: --taus 0.95,0.90,0.85")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    # ap.add_argument("--tau", type=float, default=0.9, help="Confidence threshold for early exit")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    loader = make_test_loader(args.data_dir, args.batch_size, args.num_workers)
    model = load_model(args.ckpt, args.device)

    # acc_per_exit, total = per_exit_accuracy(model, loader, args.device)
    # overall_acc, exit_dist = early_exit_eval(model, loader, args.device, args.tau)

    acc_per_exit, total = per_exit_accuracy(model, loader, args.device)

    # Determine number of exits E
    xb, _ = next(iter(loader))
    xb = xb[:1].to(args.device)
    E = len(forward_logits_shallow2deep(model, xb))

    # Build per-exit taus (length E-1)
    if args.taus is not None:
        taus = parse_csv_floats(args.taus)
    else:
        taus = [args.tau] * (E - 1)

    overall_acc, exit_dist, avg_time = early_exit_eval(model, loader, args.device, taus)

    print(f"Test samples: {total}")
    print("Per-exit accuracy (each head on all samples):")
    for i, a in enumerate(acc_per_exit, start=1):
        print(f"  Exit {i}: {a*100:.2f}%")

    print(f"\nEarly-exit evaluation (tau={args.tau:.3f}):")
    print(f"  Overall accuracy: {overall_acc*100:.2f}%")
    print("  Exit distribution:")
    for i, p in enumerate(exit_dist, start=1):
        print(f"    Exit {i}: {p*100:.2f}%")

    print("\nInference timing:")
    print(f"  Average inference time per sample: {avg_time * 1000:.4f} ms")

if __name__ == "__main__":
    main()
