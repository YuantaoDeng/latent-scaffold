from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

import numpy as np
import timm
import torch

from drift_experiment import (
    IndexedCifar,
    balanced_memory,
    ids_by_class,
    infer,
    model,
    seed_all,
    train_task,
)
from phase_c_adapter_direction import numerical_column_space, projected_random_basis
from phase_c_direction import field_cosine, functional_bases, projection_fraction


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top_basis(field: torch.Tensor, rank: int, centered: bool = True):
    values = field - field.mean(0, keepdim=True) if centered else field
    return torch.linalg.svd(values.double(), full_matrices=False).Vh[:rank].T.float()


def coverage(field: torch.Tensor, basis: torch.Tensor, centered: bool = True):
    values = field - field.mean(0, keepdim=True) if centered else field
    return projection_fraction(values, basis)


def basis_overlap(left: torch.Tensor, right: torch.Tensor):
    return float((left.T.double() @ right.double()).square().sum() / left.shape[1])


def projection_spectrum(basis: torch.Tensor, discriminative: torch.Tensor):
    projected = discriminative.T.double() @ basis.double()
    gram = projected.T @ projected
    eigenvalues, rotation = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[order].clamp(0.0, 1.0), rotation[:, order]


def spectrum_matched_basis(
    empirical: torch.Tensor,
    discriminative: torch.Tensor,
    null: torch.Tensor,
    rank: int,
    generator: torch.Generator,
):
    eigenvalues, _ = projection_spectrum(empirical, discriminative)
    # A rank-r basis can have at most dim(discriminative) nonzero projection
    # eigenvalues. Construct only those sensitive components; zero-eigenvalue
    # columns live entirely in the null space. This also supports few-class
    # checkpoints where dim(discriminative) < r.
    eigenvalues[eigenvalues < 1e-10] = 0.0
    disc_count = int((eigenvalues > 0).sum())
    empirical_disc = discriminative @ (discriminative.T @ empirical)
    empirical_null = null @ (null.T @ empirical)
    random_disc = empirical.new_zeros((empirical.shape[0], rank))
    if disc_count:
        avoided_rank = numerical_column_space(discriminative.T @ empirical_disc).shape[1]
        avoid = empirical_disc if discriminative.shape[1] - avoided_rank >= disc_count else None
        random_disc[:, :disc_count] = projected_random_basis(
            discriminative, disc_count, generator, avoid,
        )
    avoided_null_rank = numerical_column_space(null.T @ empirical_null).shape[1]
    null_avoid = empirical_null if null.shape[1] - avoided_null_rank >= rank else None
    random_null = projected_random_basis(null, rank, generator, null_avoid)
    mixed = (
        random_disc * eigenvalues.sqrt().float()[None, :]
        + random_null * (1.0 - eigenvalues).sqrt().float()[None, :]
    )
    # The two components occupy orthogonal spaces, so this should already be orthonormal.
    orthogonality_error = float((mixed.T.double() @ mixed.double() - torch.eye(rank)).abs().max())
    matched_spectrum, _ = projection_spectrum(mixed, discriminative)
    spectrum_error = float((matched_spectrum - eigenvalues).abs().max())
    return mixed, spectrum_error, orthogonality_error


def summarize_controls(values: list[float], empirical: float):
    array = np.asarray(values, dtype=np.float64)
    return {
        "matched_mean": float(array.mean()),
        "matched_sd": float(array.std(ddof=1)),
        "matched_q025": float(np.quantile(array, 0.025)),
        "matched_q975": float(np.quantile(array, 0.975)),
        "empirical_minus_matched": float(empirical - array.mean()),
        "empirical_percentile": float((np.sum(array < empirical) + 0.5 * np.sum(array == empirical)) / len(array)),
    }


def run(args):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = IndexedCifar(args.data, True, True)
    train_eval = IndexedCifar(args.data, True, False)
    order = np.random.default_rng(args.order_seed).permutation(100)[:30].reshape(3, 10).tolist()
    used = sum(order, [])
    train_ids = {label: ids_by_class(train.targets, [label], args.train_per_class) for label in used}
    probe_ids = {label: ids_by_class(train.targets, [label], args.probe_per_class, True) for label in used}
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        seed_all(seed)
        net = model(args.model, args.pretrained).to(device)
        priority = torch.rand(len(train), generator=torch.Generator().manual_seed(seed + 9000))

        task1 = order[0]
        current1 = torch.cat([train_ids[label] for label in task1])
        train_task(net, "naive", train, current1, torch.empty(0, dtype=torch.long), {},
                   task1, [], args, device, seed * 100 + 0)
        memory1 = balanced_memory(train_ids, task1, args.memory, priority)
        heldout = torch.cat([probe_ids[label] for label in task1])
        _, fit_z1, _ = infer(net, train_eval, memory1, args.batch_size, device)
        _, eval_z1, _ = infer(net, train_eval, heldout, args.batch_size, device)

        seen2 = task1 + order[1]
        current2 = torch.cat([train_ids[label] for label in order[1]])
        train_task(net, "er", train, current2, memory1, {}, seen2, task1,
                   args, device, seed * 100 + 1)
        _, fit_z2, _ = infer(net, train_eval, memory1, args.batch_size, device)
        _, eval_z2, _ = infer(net, train_eval, heldout, args.batch_size, device)
        weights2 = net.get_classifier().weight[seen2].detach().float().cpu()
        discriminative, null, singular, cutoff = functional_bases(weights2, args.svd_threshold)

        memory2 = balanced_memory(train_ids, seen2, args.memory, priority)
        seen3 = seen2 + order[2]
        current3 = torch.cat([train_ids[label] for label in order[2]])
        train_task(net, "er", train, current3, memory2, {}, seen3, seen2,
                   args, device, seed * 100 + 2)
        _, fit_z3, _ = infer(net, train_eval, memory1, args.batch_size, device)
        _, eval_z3, _ = infer(net, train_eval, heldout, args.batch_size, device)

        fit_d12, eval_d12 = fit_z2 - fit_z1, eval_z2 - eval_z1
        fit_d23, eval_d23 = fit_z3 - fit_z2, eval_z3 - eval_z2
        historical = top_basis(fit_d12, args.rank, centered=True)
        historical_raw = top_basis(fit_d12, args.rank, centered=False)
        future_basis = top_basis(eval_d23, args.rank, centered=True)
        static_pca = top_basis(fit_z2, args.rank, centered=True)
        empirical_future = coverage(eval_d23, historical, centered=True)
        empirical_past = coverage(eval_d12, historical, centered=True)

        matched_future, matched_past = [], []
        max_spectrum_error = 0.0
        max_orthogonality_error = 0.0
        max_empirical_overlap = 0.0
        for draw in range(args.random_controls):
            generator = torch.Generator().manual_seed(
                args.order_seed * 1_000_003 + seed * 10_007 + draw + 80_000
            )
            control, spectrum_error, orthogonality_error = spectrum_matched_basis(
                historical, discriminative, null, args.rank, generator,
            )
            matched_future.append(coverage(eval_d23, control, centered=True))
            matched_past.append(coverage(eval_d12, control, centered=True))
            max_spectrum_error = max(max_spectrum_error, spectrum_error)
            max_orthogonality_error = max(max_orthogonality_error, orthogonality_error)
            max_empirical_overlap = max(max_empirical_overlap, basis_overlap(historical, control))

        if max_spectrum_error > args.max_spectrum_error:
            raise RuntimeError(
                f"order={args.order_seed} seed={seed}: projection spectrum error {max_spectrum_error}"
            )
        future_summary = summarize_controls(matched_future, empirical_future)
        past_summary = summarize_controls(matched_past, empirical_past)
        centered_d12 = eval_d12 - eval_d12.mean(0, keepdim=True)
        centered_d23 = eval_d23 - eval_d23.mean(0, keepdim=True)
        mean12, mean23 = eval_d12.mean(0), eval_d23.mean(0)
        rows.append({
            "order_seed": args.order_seed,
            "seed": seed,
            "functional_rank": discriminative.shape[1],
            "svd_cutoff": cutoff,
            "historical_sensitive_fraction": projection_fraction(historical.T, discriminative),
            "future_empirical_coverage": empirical_future,
            **{f"future_{key}": value for key, value in future_summary.items()},
            "past_empirical_coverage": empirical_past,
            **{f"past_{key}": value for key, value in past_summary.items()},
            "future_static_pca_coverage": coverage(eval_d23, static_pca, centered=True),
            "future_raw_historical_coverage": coverage(eval_d23, historical_raw, centered=False),
            "future_historical_subspace_overlap": basis_overlap(historical, future_basis),
            "centered_field_cosine": field_cosine(centered_d12, centered_d23),
            "mean_drift_cosine": field_cosine(mean12, mean23),
            "fit_d12_relative_norm": float(fit_d12.norm() / (fit_z1 - fit_z1.mean(0)).norm()),
            "eval_d12_relative_norm": float(eval_d12.norm() / (eval_z1 - eval_z1.mean(0)).norm()),
            "eval_d23_relative_norm": float(eval_d23.norm() / (eval_z2 - eval_z2.mean(0)).norm()),
            "max_spectrum_match_error": max_spectrum_error,
            "max_control_orthogonality_error": max_orthogonality_error,
            "max_empirical_control_overlap": max_empirical_overlap,
            "largest_classifier_singular": float(singular.max()),
        })

    write_csv(args.output / "runs.csv", rows)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()} | {
        "order": order,
        "device": str(device),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "primary": "centered future drift coverage by historical centered drift basis minus mean spectrum-matched control",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-transition temporal reuse of natural ER drift subspaces")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--random-controls", type=int, default=128)
    parser.add_argument("--max-spectrum-error", type=float, default=1e-5)
    parser.add_argument("--svd-threshold", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--probe-per-class", type=int, default=50)
    parser.add_argument("--memory", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--der-alpha", type=float, default=0.5)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--model", default="deit_tiny_patch16_224")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/phase_c/temporal_drift_reuse"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
