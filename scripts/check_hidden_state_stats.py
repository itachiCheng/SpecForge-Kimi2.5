#!/usr/bin/env python3
import argparse
import gzip
import io
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_v: float = float("inf")
    max_v: float = float("-inf")

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.detach().float().view(-1)
        self.min_v = min(self.min_v, float(x.min().item()))
        self.max_v = max(self.max_v, float(x.max().item()))

        n = x.numel()
        batch_mean = float(x.mean().item())
        if n > 1:
            batch_m2 = float(((x - batch_mean) ** 2).sum().item())
        else:
            batch_m2 = 0.0

        if self.count == 0:
            self.count = n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        self.m2 = self.m2 + batch_m2 + delta * delta * self.count * n / total
        self.count = total

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(max(self.m2 / (self.count - 1), 0.0))


def list_ckpts(path: str) -> List[str]:
    files = []
    for root, _, names in os.walk(path):
        for name in names:
            if name.endswith(".ckpt") or name.endswith(".ckpt.gz"):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def load_ckpt(path: str) -> Dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(io.BytesIO(f.read()), weights_only=False)
    return torch.load(path, weights_only=False, mmap=True)


def tensor_health(x: torch.Tensor) -> Tuple[int, int]:
    x = x.detach()
    n_nan = int(torch.isnan(x).sum().item())
    n_inf = int(torch.isinf(x).sum().item())
    return n_nan, n_inf


def sample_abs_quantiles(x: torch.Tensor) -> Tuple[float, float, float]:
    x = x.detach().float().abs().view(-1)
    if x.numel() == 0:
        return 0.0, 0.0, 0.0
    q = torch.quantile(x, torch.tensor([0.95, 0.99, 0.999], dtype=torch.float32))
    return float(q[0].item()), float(q[1].item()), float(q[2].item())


def update_norm_acc(acc: Dict[str, float], x: torch.Tensor) -> None:
    # x: [1, seq, hidden] or [seq, hidden]
    if x.dim() == 3:
        seq_h = x[0]
    elif x.dim() == 2:
        seq_h = x
    else:
        return
    token_norm = torch.linalg.norm(seq_h.float(), dim=-1)
    acc["token_norm_sum"] += float(token_norm.sum().item())
    acc["token_norm_count"] += int(token_norm.numel())
    acc["token_norm_max"] = max(acc["token_norm_max"], float(token_norm.max().item()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check hidden state distribution stats.")
    parser.add_argument("--hidden-states-path", type=str, required=True)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files")
    parser.add_argument("--print-every", type=int, default=100)
    args = parser.parse_args()

    files = list_ckpts(args.hidden_states_path)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No .ckpt/.ckpt.gz files found in {args.hidden_states_path}")

    hs_stats = RunningStats()
    aux_stats = RunningStats()
    lm_stats = RunningStats()
    health = {
        "hs_nan": 0,
        "hs_inf": 0,
        "aux_nan": 0,
        "aux_inf": 0,
    }
    norm_acc = {
        "token_norm_sum": 0.0,
        "token_norm_count": 0,
        "token_norm_max": 0.0,
    }
    lengths = []
    loss_ratios = []
    abs_q_samples = {"hs_q95": [], "hs_q99": [], "hs_q999": []}

    for i, path in enumerate(files, start=1):
        d = load_ckpt(path)
        input_ids = d["input_ids"]
        loss_mask = d["loss_mask"]
        hs = d["hidden_state"]
        aux = d["aux_hidden_state"]

        seq_len = int(input_ids.shape[-1])
        lengths.append(seq_len)
        loss_ratio = float(loss_mask.float().mean().item())
        loss_ratios.append(loss_ratio)

        hs_stats.update(hs)
        aux_stats.update(aux)
        lm_stats.update(loss_mask.float())

        n_nan, n_inf = tensor_health(hs)
        health["hs_nan"] += n_nan
        health["hs_inf"] += n_inf
        n_nan, n_inf = tensor_health(aux)
        health["aux_nan"] += n_nan
        health["aux_inf"] += n_inf

        update_norm_acc(norm_acc, hs)

        q95, q99, q999 = sample_abs_quantiles(hs)
        abs_q_samples["hs_q95"].append(q95)
        abs_q_samples["hs_q99"].append(q99)
        abs_q_samples["hs_q999"].append(q999)

        if args.print_every > 0 and i % args.print_every == 0:
            print(f"[progress] {i}/{len(files)} files checked")

    print("=" * 64)
    print(f"files_checked: {len(files)}")
    print(f"seq_len_min/max/avg: {min(lengths)} / {max(lengths)} / {sum(lengths)/len(lengths):.2f}")
    print(
        f"loss_mask_ratio_min/max/avg: "
        f"{min(loss_ratios):.4f} / {max(loss_ratios):.4f} / {sum(loss_ratios)/len(loss_ratios):.4f}"
    )
    print("-" * 64)
    print(
        f"hidden_state mean/std/min/max: "
        f"{hs_stats.mean:.6f} / {hs_stats.std:.6f} / {hs_stats.min_v:.6f} / {hs_stats.max_v:.6f}"
    )
    print(
        f"aux_hidden_state mean/std/min/max: "
        f"{aux_stats.mean:.6f} / {aux_stats.std:.6f} / {aux_stats.min_v:.6f} / {aux_stats.max_v:.6f}"
    )
    print(
        f"loss_mask mean/std/min/max: "
        f"{lm_stats.mean:.6f} / {lm_stats.std:.6f} / {lm_stats.min_v:.6f} / {lm_stats.max_v:.6f}"
    )
    print("-" * 64)
    print(
        f"hidden_state NaN/Inf: {health['hs_nan']} / {health['hs_inf']} | "
        f"aux_hidden_state NaN/Inf: {health['aux_nan']} / {health['aux_inf']}"
    )
    if norm_acc["token_norm_count"] > 0:
        print(
            f"token_l2_norm avg/max: "
            f"{norm_acc['token_norm_sum']/norm_acc['token_norm_count']:.6f} / {norm_acc['token_norm_max']:.6f}"
        )
    print(
        f"sample_abs_quantile_avg(q95/q99/q999): "
        f"{sum(abs_q_samples['hs_q95'])/len(abs_q_samples['hs_q95']):.6f} / "
        f"{sum(abs_q_samples['hs_q99'])/len(abs_q_samples['hs_q99']):.6f} / "
        f"{sum(abs_q_samples['hs_q999'])/len(abs_q_samples['hs_q999']):.6f}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
