from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

import numpy as np
import timm
import torch
import torch.nn as nn

from drift_experiment import (
    IndexedCifar,
    accuracy,
    balanced_memory,
    geometry,
    ids_by_class,
    infer,
    model,
    seed_all,
    train_task,
)
from phase_c_direction import (
    field_cosine,
    functional_bases,
    functional_change,
    projection_fraction,
    state_dict_cpu,
    state_sha256,
    train_future_standard,
)


PRIMARY_CONDITIONS = ("jacobian_null", "empirical_er", "matched_random")
DIAGNOSTIC_CONDITIONS = (
    "projection_matched_random",
    "empirical_null",
    "matched_null_random",
)
ALL_CONDITIONS = PRIMARY_CONDITIONS + DIAGNOSTIC_CONDITIONS


class FixedDriftAdapter(nn.Module):
    """A fixed low-rank representation intervention; the core is identical across conditions."""

    def __init__(self, core, mean, transform, detach_displacement=False):
        super().__init__()
        self.core = core
        self.detach_displacement = detach_displacement
        self.register_buffer("drift_mean", mean)
        self.register_buffer("drift_transform", transform)

    def forward_features(self, x, **kwargs):
        return self.core.forward_features(x, **kwargs)

    def forward_head(self, x, pre_logits=False):
        z = self.core.forward_head(x, pre_logits=True)
        source = z.detach() if self.detach_displacement else z
        z = z + (source - self.drift_mean) @ self.drift_transform
        return z if pre_logits else self.get_classifier()(z)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    def get_classifier(self):
        return self.core.get_classifier()


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def orthonormalize(candidate: torch.Tensor, rank: int):
    q, _ = torch.linalg.qr(candidate.double(), mode="reduced")
    if q.shape[1] < rank:
        raise ValueError(f"Candidate has only {q.shape[1]} dimensions; need {rank}")
    return q[:, :rank].float()


def numerical_column_space(candidate: torch.Tensor, relative_threshold=1e-7):
    u, singular, _ = torch.linalg.svd(candidate.double(), full_matrices=False)
    if not singular.numel():
        return candidate.new_zeros((candidate.shape[0], 0))
    rank = int((singular > singular.max() * relative_threshold).sum())
    return u[:, :rank].float()


def projected_random_basis(space: torch.Tensor, rank: int, generator, avoid=None):
    candidate = space @ torch.randn((space.shape[1], max(rank * 3, rank + 4)), generator=generator)
    if avoid is not None and avoid.numel():
        avoid_basis = numerical_column_space(avoid)
        candidate = candidate - avoid_basis @ (avoid_basis.T @ candidate)
        candidate = space @ (space.T @ candidate)
    available = numerical_column_space(candidate)
    if available.shape[1] < rank:
        raise ValueError(
            f"Only {available.shape[1]} directions remain after avoidance; requested {rank}"
        )
    return available[:, :rank]


def direction_bases(prior_delta, discriminative, null, rank, generator):
    empirical = torch.linalg.svd(prior_delta.double(), full_matrices=False).Vh[:rank].T.float()
    null_direction = projected_random_basis(null, rank, generator)
    prior_null = prior_delta @ null @ null.T
    historical_null = torch.linalg.svd(prior_null.double(), full_matrices=False).Vh[:rank].T.float()
    matched_null_random = projected_random_basis(
        null, rank, generator, torch.cat((null_direction, historical_null), dim=1),
    )

    disc_rank = rank // 2
    null_rank = rank - disc_rank
    empirical_disc = discriminative @ (discriminative.T @ empirical)
    empirical_null_projection = null @ (null.T @ empirical)
    random_disc = projected_random_basis(discriminative, disc_rank, generator, empirical_disc)
    random_null = projected_random_basis(
        null, null_rank, generator, torch.cat((null_direction, empirical_null_projection), dim=1),
    )
    matched_random = torch.cat((random_disc, random_null), dim=1)
    empirical_disc_fraction = projection_fraction(empirical.T, discriminative)
    mix_disc = projected_random_basis(discriminative, rank, generator, empirical_disc)
    mix_null = projected_random_basis(
        null, rank, generator, torch.cat((null_direction, empirical_null_projection), dim=1),
    )
    projection_matched_random = (
        math.sqrt(empirical_disc_fraction) * mix_disc
        + math.sqrt(1.0 - empirical_disc_fraction) * mix_null
    )
    return {
        "jacobian_null": null_direction,
        "empirical_er": empirical,
        "matched_random": matched_random,
        "projection_matched_random": projection_matched_random,
        "empirical_null": historical_null,
        "matched_null_random": matched_null_random,
    }


def basis_overlap(left, right):
    return float((left.T.double() @ right.double()).square().sum() / left.shape[1])


def shared_transforms(calibration, mean, bases, magnitude, rank, generator):
    input_basis = orthonormalize(
        torch.randn((calibration.shape[1], rank), generator=generator), rank,
    )
    coefficients = (calibration - mean) @ input_basis
    representation_scale = (calibration - calibration.mean(0, keepdim=True)).norm()
    gain = magnitude * representation_scale / coefficients.norm().clamp_min(1e-24)
    transforms = {name: gain * input_basis @ basis.T for name, basis in bases.items()}
    return input_basis, transforms, float(gain)


def class_slice(targets, classes, start, count):
    return torch.cat([torch.where(targets == label)[0][start:start + count] for label in classes])


def run(args):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = IndexedCifar(args.data, True, True)
    train_eval = IndexedCifar(args.data, True, False)
    test = IndexedCifar(args.data, False, False)
    order = np.random.default_rng(args.order_seed).permutation(100)[:30].reshape(3, 10).tolist()
    train_ids = {c: ids_by_class(train.targets, [c], args.train_per_class) for c in sum(order, [])}
    probe_ids = {c: ids_by_class(train.targets, [c], args.probe_per_class, True) for c in sum(order, [])}
    test_ids = {c: ids_by_class(test.targets, [c], args.test_per_class) for c in sum(order, [])}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)

    rows, curves, spaces = [], [], []
    for seed in args.seeds:
        seed_all(seed)
        base = model(args.model, args.pretrained).to(device)
        seen = list(order[0])
        first = torch.cat([train_ids[c] for c in seen])
        train_task(base, "naive", train, first, torch.empty(0, dtype=torch.long), {},
                   seen, [], args, device, seed)
        priority = torch.rand(len(train), generator=torch.Generator().manual_seed(seed + 9000))
        memory = balanced_memory(train_ids, seen, args.memory, priority)
        empirical_ids = memory.clone()
        _, empirical_before, _ = infer(base, train_eval, empirical_ids, args.batch_size, device)
        old = list(seen)
        seen += order[1]
        second = torch.cat([train_ids[c] for c in order[1]])
        train_task(base, "er", train, second, memory, {}, seen, old, args, device, seed * 100 + 1)
        _, empirical_after, _ = infer(base, train_eval, empirical_ids, args.batch_size, device)
        prior_delta = empirical_after - empirical_before
        memory = balanced_memory(train_ids, seen, args.memory, priority)

        old_probes = torch.cat([probe_ids[c] for c in seen])
        before_logits, before_reps, probe_labels = infer(base, train_eval, old_probes, args.batch_size, device)
        _, memory_reps, _ = infer(base, train_eval, memory, args.batch_size, device)
        calibration_ids = class_slice(train.targets, seen, args.calibration_start, args.calibration_per_class)
        _, calibration_reps, _ = infer(base, train_eval, calibration_ids, args.batch_size, device)
        weights = base.get_classifier().weight[seen].detach().float().cpu()
        full_weights = base.get_classifier().weight.detach().float().cpu()
        full_bias = base.get_classifier().bias.detach().float().cpu()
        audit_before_logits = before_reps @ full_weights.T + full_bias
        discriminative, null, singular, cutoff = functional_bases(weights, args.svd_threshold)
        generator = torch.Generator().manual_seed(seed + 7000)
        bases = direction_bases(prior_delta, discriminative, null, args.rank, generator)
        input_basis, transforms, gain = shared_transforms(
            calibration_reps, memory_reps.mean(0), bases, args.magnitude, args.rank, generator,
        )

        target_fields = {
            name: (before_reps - memory_reps.mean(0)) @ transform for name, transform in transforms.items()
        }
        direction_names = list(args.conditions)
        pair_cosines = {
            f"cos_{left}__{right}": field_cosine(target_fields[left], target_fields[right])
            for index, left in enumerate(direction_names)
            for right in direction_names[index + 1:]
        }
        pair_subspace = {
            f"subspace_overlap_{left}__{right}": basis_overlap(bases[left], bases[right])
            for index, left in enumerate(direction_names)
            for right in direction_names[index + 1:]
        }
        if max(map(abs, pair_cosines.values())) > args.max_direction_cosine:
            raise RuntimeError(f"Seed {seed} failed direction cosine gate: {pair_cosines}")

        base_state = state_dict_cpu(base)
        base_hash = state_sha256(base_state)
        future = order[2]
        all_seen = seen + future
        current = torch.cat([train_ids[c] for c in future])
        old_test = torch.cat([test_ids[c] for c in seen])
        new_test = torch.cat([test_ids[c] for c in future])
        all_test = torch.cat((old_test, new_test))
        old_before = accuracy(audit_before_logits, probe_labels, seen)

        for condition in args.conditions:
            core = model(args.model, False).to(device)
            core.load_state_dict(base_state)
            net = FixedDriftAdapter(
                core, memory_reps.mean(0).to(device), transforms[condition].to(device),
            ).to(device)
            after_logits, after_reps, _ = infer(net, train_eval, old_probes, args.batch_size, device)
            # Audit the intervention in FP32 with one shared frozen classifier. This avoids treating
            # autocast rounding differences between two forward passes as a functional change.
            audit_after_logits = after_reps @ full_weights.T + full_bias
            function = functional_change(
                audit_before_logits, audit_after_logits, probe_labels, seen,
            )
            old_after = accuracy(audit_after_logits, probe_labels, seen)
            delta = after_reps - before_reps
            base_norm = (before_reps - before_reps.mean(0, keepdim=True)).norm()
            alignment = field_cosine(delta, target_fields[condition])
            transform_norm = float(transforms[condition].norm())
            checkpoint = {
                "state_dict": state_dict_cpu(net),
                "base_sha256": base_hash,
                "condition": condition,
                "transform_norm": transform_norm,
            }
            torch.save(checkpoint, args.output / "checkpoints" / f"seed{seed}_{condition}.pt")

            curve = []
            if not args.skip_future:
                curve = train_future_standard(
                    net, train, test, current, memory, old_test, new_test, all_test,
                    all_seen, args, device, seed * 10000 + 99,
                )
                curves.extend({"seed": seed, "order_seed": args.order_seed, "condition": condition,
                               "epoch": epoch, **point} for epoch, point in enumerate(curve, 1))
            rows.append({
                "seed": seed,
                "order_seed": args.order_seed,
                "condition": condition,
                "base_checkpoint_sha256": base_hash,
                "magnitude": args.magnitude,
                "rank": args.rank,
                "old_acc_before": old_before,
                "old_acc_after": old_after,
                "old_acc_delta": old_after - old_before,
                "adapter_parameter_delta": transform_norm,
                "realized_fro_relative": float(delta.norm() / base_norm),
                "target_alignment": alignment,
                "rho_discriminative": projection_fraction(delta, discriminative),
                "rho_null": projection_fraction(delta, null),
                **function,
                **{f"geometry_{key}": value for key, value in geometry(
                    before_reps, after_reps, probe_labels, weights,
                ).items()},
                "future_new_epoch1_accuracy": curve[0]["future_new_accuracy"] if curve else float("nan"),
                "future_new_early_auc": (float(np.mean([point["future_new_accuracy"] for point in curve[:2]]))
                                         if curve else float("nan")),
                "future_new_auc": (float(np.mean([point["future_new_accuracy"] for point in curve]))
                                   if curve else float("nan")),
                "future_new_accuracy": curve[-1]["future_new_accuracy"] if curve else float("nan"),
                "future_old_accuracy": curve[-1]["future_old_accuracy"] if curve else float("nan"),
                "old_task_forgetting": (old_after - curve[-1]["future_old_accuracy"]
                                        if curve else float("nan")),
            })

        spaces.append({
            "seed": seed,
            "functional_rank": discriminative.shape[1],
            "null_rank": null.shape[1],
            "svd_cutoff": cutoff,
            "largest_singular_value": float(singular.max()),
            "shared_gain": gain,
            "input_basis_rank": input_basis.shape[1],
            **{f"{name}_rho_discriminative": projection_fraction(basis.T, discriminative)
               for name, basis in bases.items()},
            **pair_cosines,
            **pair_subspace,
        })

    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "learning_curves.csv", curves)
    write_csv(args.output / "space_diagnostics.csv", spaces)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()} | {
        "order": order,
        "device": str(device),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "intervention": "fixed low-rank affine representation adapter with shared input coefficients",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Exactly norm-matched fixed-adapter Phase C")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS,
                        default=list(PRIMARY_CONDITIONS))
    parser.add_argument("--magnitude", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-direction-cosine", type=float, default=0.30)
    parser.add_argument("--svd-threshold", type=float, default=1e-7)
    parser.add_argument("--calibration-start", type=int, default=430)
    parser.add_argument("--calibration-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--future-epochs", type=int, default=5)
    parser.add_argument("--skip-future", action="store_true")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--probe-per-class", type=int, default=10)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--memory", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--der-alpha", type=float, default=0.5)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--model", default="deit_tiny_patch16_224")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/phase_c/adapter_direction"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
