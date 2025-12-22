# Needs: resnet.py in same folder.

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

from resnet import resnet18


@dataclass
class CFG:
    data_dir: str = "./data"
    batch_size: int = 500
    num_workers: int = 4
    epochs: int = 300

    # Optimizer
    lr: float = 0.1

    lr_decay_type: str = "multistep"
    lr_decay_steps: Tuple[int, ...] = (250, 280, 295)
    lr_decay_rate: float = 0.1

    momentum: float = 0.9
    weight_decay: float = 5e-4

    # Exit-loss weights (lambda_e) annealing (length=4, exits shallow->deep)
    lambda_start: Tuple[float, ...] = (1.0, 0.7, 0.4, 0.1)
    lambda_end:   Tuple[float, ...] = (0.2, 0.5, 0.8, 1.0)

    # HardCGT threshold used DURING TRAINING for the correctness+confidence gate
    tau_train: float = 0.5

    # Stabilizers
    eps_leak: float = 1e-3
    normalize_weights: bool = True

    # Often helps CIFAR-100
    label_smoothing: float = 0.1

    # Evaluation early-exit threshold (used in evaluate() only)
    tau_eval: float = 0.9

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 50


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def max_conf_and_pred(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns (conf, pred) with conf = max softmax probability. Shape [B].
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return conf, pred


def hardcgt_lambda_anneal_loss(
    logits_list_shallow2deep: List[torch.Tensor],
    y: torch.Tensor,
    lambda_vec: torch.Tensor,
    tau_train: float,
    eps_leak: float,
    normalize_weights: bool,
    label_smoothing: float,
):
    # HardCGT (binary gating):
    #   m_e(i) = 1{ conf_e(i) >= tau_train AND pred_e(i) == y_i }
    #   elig_1(i)=1
    #   elig_e(i)= Π_{k<e} (1 - m_k(i))
    # Per-exit weight:
    #   w_e(i) = lambda_e(epoch) * elig_e(i) + eps_leak
    B = y.size(0)
    E = len(logits_list_shallow2deep)
    device = y.device
    assert lambda_vec.numel() == E, f"lambda_vec must have length {E}, got {lambda_vec.numel()}"

    # m_e(i) for each exit e: [B]
    m_list = []
    for e in range(E):
        conf, pred = max_conf_and_pred(logits_list_shallow2deep[e])
        m = (conf >= tau_train) & (pred == y)
        m_list.append(m.to(dtype=torch.float32))

    elig = torch.ones(B, device=device)
    eligs = [elig]
    for e in range(1, E):
        # if earlier exit k succeeded (m_k=1), then (1 - m_k)=0 and eligibility collapses
        elig = elig * (1.0 - m_list[e - 1])
        eligs.append(elig)

    W = torch.stack(eligs, dim=1) * lambda_vec.view(1, E)
    W = W + eps_leak
    if normalize_weights:
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)

    total = 0.0
    for e in range(E):
        ce = F.cross_entropy(
            logits_list_shallow2deep[e],
            y,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        total = total + (W[:, e] * ce).mean()

    stats = {
        "m_mean": torch.stack([m.mean() for m in m_list]).detach(),        # success rate per exit
        "elig_mean": torch.stack([el.mean() for el in eligs]).detach(),    # mean eligibility per exit
        "w_mean": W.mean(dim=0).detach(),                                  # mean weights per exit
    }
    return total, stats


@torch.no_grad()
def early_exit_predict(logits_list_shallow2deep: List[torch.Tensor], tau_eval: float):
    E = len(logits_list_shallow2deep)
    B = logits_list_shallow2deep[0].size(0)
    device = logits_list_shallow2deep[0].device

    exited = torch.zeros(B, dtype=torch.bool, device=device)
    exit_ids = torch.full((B,), E - 1, dtype=torch.long, device=device)
    preds = torch.empty(B, dtype=torch.long, device=device)

    for e in range(E):
        probs = F.softmax(logits_list_shallow2deep[e], dim=1)
        conf, pred = probs.max(dim=1)
        if e < E - 1:
            can_exit = (~exited) & (conf >= tau_eval)
        else:
            can_exit = ~exited
        preds[can_exit] = pred[can_exit]
        exit_ids[can_exit] = e
        exited[can_exit] = True
        if exited.all():
            break
    return preds, exit_ids


def make_loaders(cfg: CFG):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    # Strong augmentation (hard labels; works well with HardCGT too)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean, std),
        # Cutout-like regularization
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR100(cfg.data_dir, train=True, download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR100(cfg.data_dir, train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, test_loader


def forward_logits_shallow2deep(model, x: torch.Tensor) -> List[torch.Tensor]:
    outs_deep2shallow, _ = model(x)  # [out4,out3,out2,out1]
    return list(reversed(outs_deep2shallow))  # [out1,out2,out3,out4]


def train_one_epoch(model, loader, optimizer, cfg: CFG, epoch: int, lambda_vec: torch.Tensor):
    model.train()
    running = 0.0
    t0 = time.time()

    for it, (x, y) in enumerate(loader):
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        outs = forward_logits_shallow2deep(model, x)
        loss, stats = hardcgt_lambda_anneal_loss(
            outs, y,
            lambda_vec=lambda_vec,
            tau_train=cfg.tau_train,
            eps_leak=cfg.eps_leak,
            normalize_weights=cfg.normalize_weights,
            label_smoothing=cfg.label_smoothing,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()
        if (it + 1) % cfg.log_every == 0:
            print(
                f"[train] ep={epoch:03d} "
                f"tau_train={cfg.tau_train:.3f} "
                f"lambda={[round(float(v),4) for v in lambda_vec.detach().cpu().tolist()]} "
                f"it={it+1:04d}/{len(loader)} "
                f"loss={running/(it+1):.4f} "
                f"m_mean={stats['m_mean'].tolist()} "
                f"elig_mean={stats['elig_mean'].tolist()} "
                f"w_mean={stats['w_mean'].tolist()}"
            )

    return running / max(1, len(loader)), time.time() - t0


@torch.no_grad()
def evaluate(model, loader, cfg: CFG):
    model.eval()
    E = 4
    correct = 0
    total = 0
    exit_counts = torch.zeros(E, dtype=torch.long)

    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        outs = forward_logits_shallow2deep(model, x)
        preds, exit_ids = early_exit_predict(outs, cfg.tau_eval)

        correct += (preds == y).sum().item()
        total += y.numel()
        for e in range(E):
            exit_counts[e] += (exit_ids == e).sum().item()

    acc = correct / max(1, total)
    dist = (exit_counts.float() / max(1, total)).cpu().tolist()
    return acc, dist


def main():
    cfg = CFG()
    set_seed(cfg.seed)

    if len(cfg.lambda_start) != 4 or len(cfg.lambda_end) != 4:
        raise ValueError("lambda_start and lambda_end must both have length 4 (for 4 exits).")

    train_loader, test_loader = make_loaders(cfg)

    model = resnet18(pretrained=False, num_classes=100).to(cfg.device)

    # sanity check exits
    with torch.no_grad():
        dummy = torch.randn(1, 3, 32, 32, device=cfg.device)
        outs = forward_logits_shallow2deep(model, dummy)
        if len(outs) != 4:
            raise RuntimeError(f"Expected 4 exits, got {len(outs)}. Check resnet.py forward() return.")

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    if cfg.lr_decay_type == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(cfg.lr_decay_steps),
            gamma=cfg.lr_decay_rate,
        )
    else:
        raise ValueError(f"Unknown lr_decay_type {cfg.lr_decay_type}")
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = -1.0

    lambda_start = torch.tensor(cfg.lambda_start, dtype=torch.float32, device=cfg.device)
    lambda_end = torch.tensor(cfg.lambda_end, dtype=torch.float32, device=cfg.device)

    for epoch in range(1, cfg.epochs + 1):
        alpha = (epoch - 1) / max(1, (cfg.epochs - 1))
        lambda_vec = lambda_start + alpha * (lambda_end - lambda_start)

        tr_loss, tr_time = train_one_epoch(model, train_loader, optimizer, cfg, epoch, lambda_vec)
        scheduler.step()

        acc, exit_dist = evaluate(model, test_loader, cfg)
        print(
            f"[eval ] ep={epoch:03d} tau_eval={cfg.tau_eval:.3f} "
            f"lambda={[round(float(v),4) for v in lambda_vec.detach().cpu().tolist()]} "
            f"acc={acc*100:.2f}% exit_dist={[round(p,4) for p in exit_dist]} "
            f"train_loss={tr_loss:.4f} time={tr_time:.1f}s"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "cfg": cfg.__dict__,
                },
                f"checkpoints/best_resnet18_att_hardcgt_cifar100_tau_lambda_anneal_strongaug_tau_{cfg.tau_train}.pth",
            )

    print(f"Done. Best acc={best_acc*100:.2f}%")
    print(f"Saved: checkpoints/best_resnet18_att_hardcgt_cifar100_tau_lambda_anneal_strongaug_tau_{cfg.tau_train}.pth")


if __name__ == "__main__":
    main()
