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

from drift_experiment import accuracy, IndexedCifar, balanced_memory, ids_by_class, infer, model, seed_all, train_task
from phase_c_adapter_direction import numerical_column_space
from phase_c_direction import field_cosine, functional_bases
from temporal_drift_reuse import coverage, spectrum_matched_basis, summarize_controls, top_basis


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def matched_summary(empirical_basis, outcome, discriminative, null, args, seed, label):
    empirical = coverage(outcome, empirical_basis, centered=True)
    controls, errors = [], []
    label_offset = {"adjacent12_23": 0, "nonoverlap12_34": 1, "adjacent23_34": 2}[label]
    for draw in range(args.random_controls):
        generator = torch.Generator().manual_seed(
            args.order_seed * 1_000_003 + seed * 10_007 + label_offset * 100_003 + draw + 90_000
        )
        control, spectrum_error, orthogonality_error = spectrum_matched_basis(
            empirical_basis, discriminative, null, args.rank, generator,
        )
        controls.append(coverage(outcome, control, centered=True))
        errors.append(max(spectrum_error, orthogonality_error))
    return empirical, summarize_controls(controls, empirical), max(errors)


def residual_matched_summary(
    empirical_basis, outcome, static_representations, discriminative, args, seed, pca_rank,
):
    centered_static = static_representations - static_representations.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered_static.double(), full_matrices=True)
    pca = vh[:pca_rank].T.float()
    complement = vh[pca_rank:].T.float()
    historical_coordinates = complement.T @ empirical_basis
    residual_historical = numerical_column_space(historical_coordinates)
    if residual_historical.shape[1] < args.rank:
        raise RuntimeError(f"PCA-{pca_rank} residual historical rank is {residual_historical.shape[1]}")
    residual_historical = residual_historical[:, :args.rank]
    outcome_coordinates = outcome @ complement
    discriminative_coordinates = numerical_column_space(complement.T @ discriminative)
    _, _, null_vh = torch.linalg.svd(
        discriminative_coordinates.T.double(), full_matrices=True,
    )
    null_coordinates = null_vh[discriminative_coordinates.shape[1]:].T.float()
    empirical = coverage(outcome_coordinates, residual_historical, centered=True)
    controls, errors = [], []
    for draw in range(args.random_controls):
        generator = torch.Generator().manual_seed(
            args.order_seed * 1_000_003 + seed * 10_007 + pca_rank * 100_003 + draw + 700_000
        )
        control, spectrum_error, orthogonality_error = spectrum_matched_basis(
            residual_historical, discriminative_coordinates, null_coordinates, args.rank, generator,
        )
        controls.append(coverage(outcome_coordinates, control, centered=True))
        errors.append(max(spectrum_error, orthogonality_error))
    future_centered = outcome - outcome.mean(0, keepdim=True)
    residual_future = outcome_coordinates - outcome_coordinates.mean(0, keepdim=True)
    historical_survival = float(historical_coordinates.square().sum() / args.rank)
    future_survival = float(residual_future.square().sum() / future_centered.square().sum())
    return empirical, summarize_controls(controls, empirical), max(errors), historical_survival, future_survival


def run(args):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = IndexedCifar(args.data, True, True)
    train_eval = IndexedCifar(args.data, True, False)
    order = np.random.default_rng(args.order_seed).permutation(100)[:40].reshape(4, 10).tolist()
    used = sum(order, [])
    train_ids = {label: ids_by_class(train.targets, [label], args.train_per_class) for label in used}
    probe_ids = {label: ids_by_class(train.targets, [label], args.probe_per_class, True) for label in used}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "artifacts").mkdir(exist_ok=True)

    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()} | {
        "order": order,
        "device": str(device),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "status": "running",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows = [] if args.force else read_rows(args.output / "runs.csv")
    completed = {int(row["seed"]) for row in rows}
    for seed in args.seeds:
        artifact_path = args.output / "artifacts" / f"order{args.order_seed}_seed{seed}.pt"
        if seed in completed and artifact_path.exists():
            print(f"order={args.order_seed} seed={seed}: already complete; skipping", flush=True)
            continue
        seed_all(seed)
        net = model(args.model, args.pretrained).to(device)
        priority = torch.rand(len(train), generator=torch.Generator().manual_seed(seed + 9000))
        fit_ids = None
        heldout = torch.cat([probe_ids[label] for label in order[0]])
        fit_reps, eval_reps, function_spaces, static_bases, task1_accuracies = [], [], {}, {}, []
        seen = []
        memory = torch.empty(0, dtype=torch.long)

        for task_index, task in enumerate(order):
            old = list(seen)
            seen += task
            current = torch.cat([train_ids[label] for label in task])
            method = "naive" if task_index == 0 else args.continual_method
            train_task(net, method, train, current, memory, {}, seen, old,
                       args, device, seed * 100 + task_index)
            if task_index == 0:
                fit_ids = balanced_memory(train_ids, seen, args.memory, priority)
            _, fit_z, _ = infer(net, train_eval, fit_ids, args.batch_size, device)
            eval_logits, eval_z, eval_labels = infer(net, train_eval, heldout, args.batch_size, device)
            fit_reps.append(fit_z)
            eval_reps.append(eval_z)
            task1_accuracies.append(accuracy(eval_logits, eval_labels, seen))
            static_bases[task_index + 1] = top_basis(fit_z, args.rank, centered=True)
            weights = net.get_classifier().weight[seen].detach().float().cpu()
            function_spaces[task_index + 1] = functional_bases(weights, args.svd_threshold)[:2]
            memory = balanced_memory(train_ids, seen, args.memory, priority)

        fit_d12 = fit_reps[1] - fit_reps[0]
        fit_d23 = fit_reps[2] - fit_reps[1]
        eval_d12 = eval_reps[1] - eval_reps[0]
        eval_d23 = eval_reps[2] - eval_reps[1]
        eval_d34 = eval_reps[3] - eval_reps[2]
        basis12 = top_basis(fit_d12, args.rank, centered=True)
        basis23 = top_basis(fit_d23, args.rank, centered=True)

        a_emp, a_summary, a_error = matched_summary(
            basis12, eval_d23, *function_spaces[2], args, seed, "adjacent12_23"
        )
        n_emp, n_summary, n_error = matched_summary(
            basis12, eval_d34, *function_spaces[2], args, seed, "nonoverlap12_34"
        )
        b_emp, b_summary, b_error = matched_summary(
            basis23, eval_d34, *function_spaces[3], args, seed, "adjacent23_34"
        )
        max_error = max(a_error, n_error, b_error)
        if max_error > args.max_spectrum_error:
            raise RuntimeError(f"order={args.order_seed} seed={seed}: matching error {max_error}")

        centered = [delta - delta.mean(0, keepdim=True) for delta in (eval_d12, eval_d23, eval_d34)]
        means = [delta.mean(0) for delta in (eval_d12, eval_d23, eval_d34)]
        rows.append({
            "order_seed": args.order_seed,
            "seed": seed,
            "adjacent12_23_empirical": a_emp,
            **{f"adjacent12_23_{key}": value for key, value in a_summary.items()},
            "nonoverlap12_34_empirical": n_emp,
            **{f"nonoverlap12_34_{key}": value for key, value in n_summary.items()},
            "adjacent23_34_empirical": b_emp,
            **{f"adjacent23_34_{key}": value for key, value in b_summary.items()},
            "adjacent12_23_static_pca": coverage(eval_d23, static_bases[2], centered=True),
            "nonoverlap12_34_static_pca_t2": coverage(eval_d34, static_bases[2], centered=True),
            "adjacent23_34_static_pca": coverage(eval_d34, static_bases[3], centered=True),
            "centered_cos12_23": field_cosine(centered[0], centered[1]),
            "centered_cos12_34": field_cosine(centered[0], centered[2]),
            "centered_cos23_34": field_cosine(centered[1], centered[2]),
            "mean_cos12_23": field_cosine(means[0], means[1]),
            "mean_cos12_34": field_cosine(means[0], means[2]),
            "mean_cos23_34": field_cosine(means[1], means[2]),
            "max_matching_error": max_error,
            **{f"task1_accuracy_checkpoint{index + 1}": value
               for index, value in enumerate(task1_accuracies)},
        })
        for pca_rank in args.residual_pca_ranks:
            empirical, summary, error, history_survival, future_survival = residual_matched_summary(
                basis12, eval_d34, fit_reps[1], function_spaces[2][0], args, seed, pca_rank,
            )
            if error > args.max_spectrum_error:
                raise RuntimeError(
                    f"order={args.order_seed} seed={seed}: PCA-{pca_rank} residual matching error {error}"
                )
            prefix = f"residual_pca{pca_rank}"
            rows[-1][f"{prefix}_empirical"] = empirical
            rows[-1].update({f"{prefix}_{key}": value for key, value in summary.items()})
            rows[-1][f"{prefix}_historical_energy_survival"] = history_survival
            rows[-1][f"{prefix}_future_energy_survival"] = future_survival
            rows[-1][f"{prefix}_max_matching_error"] = error
        torch.save({
            "order_seed": args.order_seed,
            "seed": seed,
            "historical_basis": basis12,
            "historical_drift_centered": fit_d12 - fit_d12.mean(0, keepdim=True),
            "future_drift_centered": eval_d34 - eval_d34.mean(0, keepdim=True),
            "discriminative_basis": function_spaces[2][0],
            "null_basis": function_spaces[2][1],
            "static_pca_basis": static_bases[2],
            "static_pca_full": torch.linalg.svd(
                (fit_reps[1] - fit_reps[1].mean(0, keepdim=True)).double(),
                full_matrices=True,
            ).Vh.T.float(),
            "static_covariance": (
                (fit_reps[1] - fit_reps[1].mean(0, keepdim=True)).T
                @ (fit_reps[1] - fit_reps[1].mean(0, keepdim=True))
                / max(len(fit_reps[1]) - 1, 1)
            ).float(),
        }, artifact_path)
        rows.sort(key=lambda row: int(row["seed"]))
        write_csv(args.output / "runs.csv", rows)
        print(f"order={args.order_seed} seed={seed}: complete", flush=True)

    manifest["status"] = "complete"
    manifest["completed_seeds"] = sorted(int(row["seed"]) for row in rows)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Adjacent versus non-overlapping temporal drift reuse")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--random-controls", type=int, default=128)
    parser.add_argument("--max-spectrum-error", type=float, default=1e-5)
    parser.add_argument("--svd-threshold", type=float, default=1e-7)
    parser.add_argument("--continual-method", choices=("naive", "er"), default="er")
    parser.add_argument("--residual-pca-ranks", nargs="+", type=int, default=[8, 32, 64])
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
    parser.add_argument("--output", type=Path, default=Path("results/phase_c/temporal_drift_lag"))
    parser.add_argument("--force", action="store_true", help="ignore completed rows and rerun requested seeds")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
