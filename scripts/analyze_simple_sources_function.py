from __future__ import annotations
from package_paths import read_config_text

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from analyze_prospective_atlas import hierarchical_bootstrap, order_summary
from drift_experiment import (
    IndexedCifar, balanced_memory, ids_by_class, infer, loader, model, seed_all, train_task,
)
from temporal_drift_reuse import coverage, top_basis


def load_sources(roots):
    return [
        torch.load(path, map_location="cpu", weights_only=False)
        for root in roots for path in sorted((root / "artifacts").glob("*.pt"))
    ]


def build_predictors(sources, rank):
    bases = torch.cat([
        top_basis(item["historical_drift_centered"], rank, centered=False) for item in sources
    ], dim=1)
    atlas = torch.linalg.svd(bases.double(), full_matrices=False).U[:, :rank].float()
    covariance = sum(item["static_covariance"].double() for item in sources)
    static = torch.linalg.eigh(covariance).eigenvectors[:, -rank:].float()
    return atlas, static


def advance_augmentation_rng(dataset, train_ids, order, priority, args, seed):
    """Consume the same random augmentations as Tasks 1--4 without repeating optimization."""
    seen = []
    memory = torch.empty(0, dtype=torch.long)
    for task_index, task in enumerate(order[:-1]):
        old = list(seen)
        seen += task
        current = torch.cat([train_ids[label] for label in task])
        replay = task_index > 0 and args.continual_method in ("er", "derpp") and len(memory)
        current_bs = args.batch_size // 2 if replay else args.batch_size
        task_seed = seed * 100 + task_index
        for epoch in range(args.epochs):
            current_loader = loader(dataset, current, current_bs, True, task_seed + epoch)
            replay_loader = (
                loader(dataset, memory, args.batch_size - current_bs, True, task_seed + 100 + epoch)
                if replay else None
            )
            replay_iter = iter(replay_loader) if replay else None
            for _ in current_loader:
                if replay:
                    try:
                        next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(replay_loader)
                        next(replay_iter)
        memory = balanced_memory(train_ids, seen, args.memory, priority)
    return seen, memory


def bn_buffer_names(state):
    suffixes = ("running_mean", "running_var", "num_batches_tracked")
    return [name for name in state if name.endswith(suffixes)]


def hybrid_state(base, donor, names):
    result = {name: value.clone() for name, value in base.items()}
    for name in names:
        result[name] = donor[name].clone()
    return result


def center(field):
    """Remove the mean displacement across examples, retaining feature coordinates."""
    return field - field.mean(0, keepdim=True)


def within_class_center(field, labels):
    result = field.clone()
    for label in labels.unique():
        mask = labels == label
        result[mask] -= result[mask].mean(0, keepdim=True)
    return result


def field_metrics(field, atlas, static):
    values = center(field)
    atlas_value = coverage(values, atlas, centered=False)
    static_value = coverage(values, static, centered=False)
    return atlas_value, static_value, atlas_value - static_value


def frozen_readout(z, weight, bias, classes, labels):
    classes = torch.tensor(classes)
    logits = z @ weight[classes].T + bias[classes]
    positions = torch.full((weight.shape[0],), -1, dtype=torch.long)
    positions[classes] = torch.arange(len(classes))
    losses = F.cross_entropy(logits, positions[labels], reduction="none")
    return logits, losses


def cosine(left, right):
    denominator = left.double().norm() * right.double().norm()
    return float((left.double().flatten() @ right.double().flatten()) / denominator.clamp_min(1e-24))


def correlation(left, right):
    left = left.double().numpy()
    right = right.double().numpy()
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def functional_metrics(pre, drift, basis, weight, bias, classes, labels):
    """Reconstruct frozen-readout changes from RAW post-minus-pre displacement.

    Geometry elsewhere uses D = drift - drift.mean(0). This legacy endpoint
    projects drift itself, including its mean displacement. Logits are not
    centered across classes. The stored *_r2 names denote 1 - SSE / sum(change^2),
    relative to a zero-change prediction, not centered explained variance.
    Preserve these keys and values for existing results; revised centered
    functional endpoints must receive distinct names.
    """
    full_logits, full_losses = frozen_readout(pre + drift, weight, bias, classes, labels)
    pre_logits, pre_losses = frozen_readout(pre, weight, bias, classes, labels)
    projected = drift @ basis @ basis.T
    projected_logits, projected_losses = frozen_readout(
        pre + projected, weight, bias, classes, labels,
    )
    full_logit_change = full_logits - pre_logits
    projected_logit_change = projected_logits - pre_logits
    full_loss_change = full_losses - pre_losses
    projected_loss_change = projected_losses - pre_losses
    logit_r2 = 1.0 - float(
        (full_logit_change - projected_logit_change).double().square().sum()
        / full_logit_change.double().square().sum().clamp_min(1e-24)
    )
    loss_r2 = 1.0 - float(
        (full_loss_change - projected_loss_change).double().square().sum()
        / full_loss_change.double().square().sum().clamp_min(1e-24)
    )
    return {
        "logit_change_r2": logit_r2,
        "logit_change_cosine": cosine(full_logit_change, projected_logit_change),
        "loss_change_r2": loss_r2,
        "loss_change_correlation": correlation(full_loss_change, projected_loss_change),
        "full_loss_change_mean": float(full_loss_change.mean()),
    }


def summarize(values, orders, seed):
    means, positive = order_summary(values, orders)
    return {
        "mean": float(np.mean(values)),
        "hierarchical_ci": hierarchical_bootstrap(values, orders, seed),
        "order_means": means,
        "positive_orders": positive,
    }


def analyze_block(root, path, atlas, static, args, device):
    manifest = json.loads(read_config_text(root / "manifest.json", encoding="utf-8"))
    config = SimpleNamespace(**manifest)
    item = torch.load(path, map_location="cpu", weights_only=False)
    seed, order_seed = int(item["seed"]), int(item["order_seed"])
    order = manifest["order"]
    train = IndexedCifar(Path(manifest["data"]), True, True, manifest["dataset"])
    evaluation = IndexedCifar(Path(manifest["data"]), True, False, manifest["dataset"])
    used = sum(order, [])
    train_ids = {
        label: ids_by_class(train.targets, [label], int(manifest["train_per_class"]))
        for label in used
    }
    heldout = ids_by_class(
        evaluation.targets, order[0], int(manifest["probe_per_class"]), tail=True,
    )
    current = torch.cat([train_ids[label] for label in order[-1]])

    seed_all(seed)
    net = model(manifest["model"], False, num_classes=int(manifest["num_classes"])).to(device)
    pre_state = item["pre_future_state_dict"]
    net.load_state_dict(pre_state)
    pre_logits, pre_rep, labels = infer(net, evaluation, heldout, config.batch_size, device)
    pre_weight = net.get_classifier().weight.detach().float().cpu().clone()
    pre_bias = net.get_classifier().bias.detach().float().cpu().clone()
    priority = torch.rand(len(train), generator=torch.Generator().manual_seed(seed + 9000))
    old, memory = advance_augmentation_rng(train, train_ids, order, priority, config, seed)
    seen = old + order[-1]
    train_task(
        net, config.continual_method, train, current, memory, {}, seen, old,
        config, device, seed * 100 + len(order) - 1,
    )
    post_state = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
    post_logits, post_rep, post_labels = infer(net, evaluation, heldout, config.batch_size, device)
    if not torch.equal(labels, post_labels):
        raise RuntimeError("held-out label order changed")
    drift = post_rep - pre_rep
    stored = item["future_drift_centered"]
    reproduced = center(drift)
    reproduction_relative_error = float((reproduced - stored).norm() / stored.norm().clamp_min(1e-24))
    reproduction_max_abs = float((reproduced - stored).abs().max())

    bn_names = bn_buffer_names(pre_state)
    parameter_only_state = hybrid_state(post_state, pre_state, bn_names)
    bn_only_state = hybrid_state(pre_state, post_state, bn_names)
    net.load_state_dict(parameter_only_state)
    _, parameter_only_rep, _ = infer(net, evaluation, heldout, config.batch_size, device)
    net.load_state_dict(bn_only_state)
    _, bn_only_rep, _ = infer(net, evaluation, heldout, config.batch_size, device)
    parameter_only_drift = parameter_only_rep - pre_rep
    bn_only_drift = bn_only_rep - pre_rep

    within = within_class_center(drift, labels)
    between = center(drift) - within
    total_energy = float(center(drift).double().square().sum())
    row = {
        "order_seed": order_seed, "seed": seed,
        "reproduction_relative_error": reproduction_relative_error,
        "reproduction_max_abs": reproduction_max_abs,
        "class_mean_energy_fraction": float(
            between.double().square().sum() / max(total_energy, 1e-24)
        ),
        "within_class_energy_fraction": float(
            within.double().square().sum() / max(total_energy, 1e-24)
        ),
    }
    for name, field in (
        ("actual", drift), ("within_class", within), ("class_mean", between),
        ("parameter_only", parameter_only_drift), ("bn_only", bn_only_drift),
    ):
        atlas_value, static_value, effect = field_metrics(field, atlas, static)
        row.update({
            f"{name}_relative_norm": float(field.norm() / drift.norm().clamp_min(1e-24)),
            f"{name}_atlas": atlas_value, f"{name}_static": static_value,
            f"{name}_atlas_minus_static": effect,
        })

    actual_function = functional_metrics(
        pre_rep, drift, atlas, pre_weight, pre_bias, old, labels,
    )
    static_function = functional_metrics(
        pre_rep, drift, static, pre_weight, pre_bias, old, labels,
    )
    for metric in ("logit_change_r2", "logit_change_cosine", "loss_change_r2",
                   "loss_change_correlation"):
        row[f"functional_atlas_{metric}"] = actual_function[metric]
        row[f"functional_static_{metric}"] = static_function[metric]
        row[f"functional_atlas_minus_static_{metric}"] = (
            actual_function[metric] - static_function[metric]
        )
    row["frozen_readout_full_loss_change_mean"] = actual_function["full_loss_change_mean"]
    row["actual_model_old_loss_change_mean"] = float(
        F.cross_entropy(post_logits[:, old], torch.tensor([old.index(int(y)) for y in labels]), reduction="mean")
        - F.cross_entropy(pre_logits[:, old], torch.tensor([old.index(int(y)) for y in labels]), reduction="mean")
    )
    return row


def run(args):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.sources)
    atlas, static = build_predictors(sources, args.rank)
    args.output.mkdir(parents=True, exist_ok=True)
    block_dir = args.output / "blocks"
    block_dir.mkdir(exist_ok=True)
    rows = []
    for root in args.targets:
        for path in sorted((root / "artifacts").glob("*.pt")):
            output_path = block_dir / f"{path.stem}.json"
            if output_path.exists() and not args.force:
                row = json.loads(read_config_text(output_path, encoding="utf-8"))
                print(f"{path.stem}: already complete", flush=True)
            else:
                row = analyze_block(root, path, atlas, static, args, device)
                output_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(
                    f"{path.stem}: complete reproduction={row['reproduction_relative_error']:.3e}",
                    flush=True,
                )
            rows.append(row)

    orders = [row["order_seed"] for row in rows]
    fields = ("actual", "within_class", "class_mean", "parameter_only", "bn_only")
    summary = {
        "source_blocks": len(sources), "target_blocks": len(rows), "rank": args.rank,
        "reproduction_relative_error_max": max(row["reproduction_relative_error"] for row in rows),
        "reproduction_max_abs_max": max(row["reproduction_max_abs"] for row in rows),
        "class_mean_energy_fraction": summarize(
            [row["class_mean_energy_fraction"] for row in rows], orders, 901,
        ),
        "within_class_energy_fraction": summarize(
            [row["within_class_energy_fraction"] for row in rows], orders, 902,
        ),
        "fields": {}, "functional": {},
    }
    for field_index, name in enumerate(fields):
        summary["fields"][name] = {
            metric: summarize(
                [row[f"{name}_{metric}"] for row in rows], orders,
                910 + field_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(
                ("relative_norm", "atlas", "static", "atlas_minus_static")
            )
        }
    for metric_index, metric in enumerate((
        "logit_change_r2", "logit_change_cosine", "loss_change_r2", "loss_change_correlation",
    )):
        summary["functional"][metric] = {
            condition: summarize(
                [row[f"functional_{condition}_{metric}"] for row in rows], orders,
                980 + metric_index * 10 + condition_index,
            )
            for condition_index, condition in enumerate(("atlas", "static", "atlas_minus_static"))
        }
    summary["frozen_readout_full_loss_change_mean"] = summarize(
        [row["frozen_readout_full_loss_change_mean"] for row in rows], orders, 1030,
    )
    summary["actual_model_old_loss_change_mean"] = summarize(
        [row["actual_model_old_loss_change_mean"] for row in rows], orders, 1031,
    )

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "targets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Class-mean, BatchNorm, and frozen-readout controls")
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
