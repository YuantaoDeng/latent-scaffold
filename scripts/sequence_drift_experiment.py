from __future__ import annotations
from package_paths import read_config_text

import argparse
import csv
import json
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

import numpy as np
import timm
import torch

from drift_experiment import (
    accuracy, IndexedCifar, balanced_memory, ids_by_class, infer, infer_intermediates,
    model, seed_all, train_task,
)
from phase_c_direction import state_dict_cpu
from phase_c_direction import functional_bases
from temporal_drift_lag import residual_matched_summary
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


def matched_future(empirical_basis, outcome, discriminative, null, args, seed):
    empirical = coverage(outcome, empirical_basis, centered=True)
    controls, errors = [], []
    for draw in range(args.random_controls):
        generator = torch.Generator().manual_seed(
            args.order_seed * 1_000_003 + seed * 10_007 + draw + 2_100_000
        )
        control, spectrum_error, orthogonality_error = spectrum_matched_basis(
            empirical_basis, discriminative, null, args.rank, generator,
        )
        controls.append(coverage(outcome, control, centered=True))
        errors.append(max(spectrum_error, orthogonality_error))
    return empirical, summarize_controls(controls, empirical), max(errors)


def run(args):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = IndexedCifar(args.data, True, True, args.dataset)
    train_eval = IndexedCifar(args.data, True, False, args.dataset)
    required_classes = args.tasks * args.classes_per_task
    if required_classes > train.num_classes:
        raise ValueError(f"requested {required_classes} classes, dataset has {train.num_classes}")
    order = np.random.default_rng(args.order_seed).permutation(train.num_classes)[:required_classes]
    order = order.reshape(args.tasks, args.classes_per_task).tolist()
    used = sum(order, [])
    train_ids = {label: ids_by_class(train.targets, [label], args.train_per_class) for label in used}
    probe_ids = {label: ids_by_class(train.targets, [label], args.probe_per_class, True) for label in used}
    if args.train_per_class + args.probe_per_class > 500:
        raise ValueError("CIFAR training and evaluation-anchor indices would overlap")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "artifacts").mkdir(exist_ok=True)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()} | {
        "order": order, "num_classes": train.num_classes, "device": str(device),
        "torch": torch.__version__, "timm": timm.__version__, "status": "running",
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
        net = model(args.model, args.pretrained, num_classes=train.num_classes)
        if getattr(args, "initial_checkpoint", None) is not None:
            initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
            net.load_state_dict(initial["state_dict"], strict=True)
            seed_all(seed)
        net = net.to(device)
        priority = torch.rand(len(train), generator=torch.Generator().manual_seed(seed + 9000))
        seen, memory = [], torch.empty(0, dtype=torch.long)
        heldout = torch.cat([probe_ids[label] for label in order[0]])
        fit_ids, fit_reps, eval_reps, task1_accuracies = None, [], [], []
        fit_layer_reps, eval_layer_reps = {}, {}
        function_spaces = None
        pre_future_state = None

        for task_index, task in enumerate(order):
            task_started = time.monotonic()
            print(f"order={args.order_seed} seed={seed} task={task_index + 1}: training", flush=True)
            old = list(seen)
            seen += task
            current = torch.cat([train_ids[label] for label in task])
            method = "naive" if task_index == 0 else args.continual_method
            if task_index == args.tasks - 1 and args.save_pre_future_checkpoint:
                pre_future_state = state_dict_cpu(net)
            train_task(net, method, train, current, memory, {}, seen, old,
                       args, device, seed * 100 + task_index)
            if task_index == 0:
                fit_ids = balanced_memory(train_ids, seen, args.memory, priority)
            if args.save_layers:
                _, fit_layers, _ = infer_intermediates(
                    net, train_eval, fit_ids, args.batch_size, device,
                )
                logits, eval_layers, labels = infer_intermediates(
                    net, train_eval, heldout, args.batch_size, device,
                )
                fit_z, eval_z = fit_layers["stage4"], eval_layers["stage4"]
                for name in fit_layers:
                    fit_layer_reps.setdefault(name, []).append(fit_layers[name])
                    eval_layer_reps.setdefault(name, []).append(eval_layers[name])
            else:
                _, fit_z, _ = infer(net, train_eval, fit_ids, args.batch_size, device)
                logits, eval_z, labels = infer(net, train_eval, heldout, args.batch_size, device)
            fit_reps.append(fit_z)
            eval_reps.append(eval_z)
            task1_accuracies.append(accuracy(logits, labels, seen))
            print(f"order={args.order_seed} seed={seed} task={task_index + 1}: "
                  f"old-anchor accuracy={task1_accuracies[-1]:.4f}; "
                  f"elapsed={time.monotonic() - task_started:.1f}s", flush=True)
            if task_index == 1:
                weights = net.get_classifier().weight[seen].detach().float().cpu()
                function_spaces = functional_bases(weights, args.svd_threshold)[:2]
            memory = balanced_memory(train_ids, seen, args.memory, priority)

        fit_history = fit_reps[1] - fit_reps[0]
        future = eval_reps[-1] - eval_reps[-2]
        historical_basis = top_basis(fit_history, args.rank, centered=True)
        empirical, matched, matching_error = matched_future(
            historical_basis, future, *function_spaces, args, seed,
        )
        if matching_error > args.max_spectrum_error:
            raise RuntimeError(f"order={args.order_seed} seed={seed}: matching error {matching_error}")

        checkpoint2 = fit_reps[1]
        row = {
            "dataset": args.dataset, "tasks": args.tasks,
            "future_transition": f"{args.tasks - 1}_{args.tasks}",
            "order_seed": args.order_seed, "seed": seed,
            "future_empirical": empirical,
            **{f"future_{key}": value for key, value in matched.items()},
            "future_static_pca": coverage(future, top_basis(checkpoint2, args.rank), centered=True),
            "max_matching_error": matching_error,
            **{f"task1_accuracy_checkpoint{index + 1}": value
               for index, value in enumerate(task1_accuracies)},
        }
        for pca_rank in args.residual_pca_ranks:
            residual_empirical, residual_summary, error, history_survival, future_survival = (
                residual_matched_summary(
                    historical_basis, future, checkpoint2, function_spaces[0], args, seed, pca_rank,
                )
            )
            prefix = f"residual_pca{pca_rank}"
            row[f"{prefix}_empirical"] = residual_empirical
            row.update({f"{prefix}_{key}": value for key, value in residual_summary.items()})
            row[f"{prefix}_historical_energy_survival"] = history_survival
            row[f"{prefix}_future_energy_survival"] = future_survival
            row[f"{prefix}_max_matching_error"] = error
        rows.append(row)

        centered_checkpoint2 = checkpoint2 - checkpoint2.mean(0, keepdim=True)
        artifact = {
            "dataset": args.dataset, "tasks": args.tasks,
            "order_seed": args.order_seed, "seed": seed,
            "historical_basis": historical_basis,
            "historical_drift_centered": fit_history - fit_history.mean(0, keepdim=True),
            "future_drift_centered": future - future.mean(0, keepdim=True),
            "discriminative_basis": function_spaces[0], "null_basis": function_spaces[1],
            "static_pca_basis": top_basis(checkpoint2, args.rank, centered=True),
            "static_pca_full": torch.linalg.svd(centered_checkpoint2.double(), full_matrices=True).Vh.T.float(),
            "static_covariance": (
                centered_checkpoint2.T @ centered_checkpoint2 / max(len(checkpoint2) - 1, 1)
            ).float(),
        }
        if getattr(args, "save_trajectory", False):
            artifact.update({"fit_ids": fit_ids, "eval_ids": heldout,
                             "fit_representations": fit_reps, "eval_representations": eval_reps,
                             "eval_labels": labels, "task1_accuracies": task1_accuracies})
        if args.save_layers:
            artifact["layers"] = {}
            for name in fit_layer_reps:
                layer_history = fit_layer_reps[name][1] - fit_layer_reps[name][0]
                layer_future = eval_layer_reps[name][-1] - eval_layer_reps[name][-2]
                layer_checkpoint2 = fit_layer_reps[name][1]
                layer_centered = layer_checkpoint2 - layer_checkpoint2.mean(0, keepdim=True)
                artifact["layers"][name] = {
                    "historical_drift_centered": layer_history - layer_history.mean(0, keepdim=True),
                    "future_drift_centered": layer_future - layer_future.mean(0, keepdim=True),
                    "static_covariance": (
                        layer_centered.T @ layer_centered / max(len(layer_checkpoint2) - 1, 1)
                    ).float(),
                }
        if pre_future_state is not None:
            artifact["pre_future_state_dict"] = pre_future_state
        torch.save(artifact, artifact_path)
        rows.sort(key=lambda item: int(item["seed"]))
        write_csv(args.output / "runs.csv", rows)
        print(f"order={args.order_seed} seed={seed}: complete", flush=True)

    manifest["status"] = "complete"
    manifest["completed_seeds"] = sorted(int(row["seed"]) for row in rows)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validation_suite(args):
    """Complete the two frozen CIFAR-100 architecture/initialization comparisons."""
    from analyze_prospective_atlas import run as analyze
    from analyze_atlas_scaling import structure_run

    torch.set_num_threads(2)
    source_orders = [1009, 1013, 1019]
    target_orders = [1103, 1109, 1117, 1123, 1129, 1151]
    protocol = {
        "dataset": "cifar100", "tasks": 4, "classes_per_task": 10,
        "source_orders": source_orders, "source_seeds": [140, 141],
        "target_orders": target_orders, "target_seeds": [150, 151],
        "epochs": 5, "train_per_class": 450, "probe_per_class": 50,
        "memory": 500, "batch_size": 64, "lr": 3e-4, "continual_method": "er",
        "source_rank": 8, "output_rank": 8, "random_controls": 128,
        "conditions": ["pretrained_deit_tiny", "shared_random_resnet18"],
        "initialization_seed": 3301,
        "primary": "scaffold minus matched-projector static future-drift coverage",
        "secondary": "pooled-covariance static, target PCA-64 residual and target-local geometry",
        "criterion": "at least 5/6 positive target-order means and hierarchical 95% lower endpoint above zero",
        "scope": "within-dataset cross-trajectory prediction in each condition's common coordinates; no architecture alignment",
        "selection": "no target-based rank, source, checkpoint or comparator selection",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(read_config_text(protocol_path, encoding="utf-8"))
        if {key: value for key, value in previous.items() if key != "frozen_at_utc"} != protocol:
            raise ValueError("existing suite protocol differs from requested design")
    else:
        protocol_path.write_text(json.dumps({"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                                             **protocol}, indent=2), encoding="utf-8")
    for condition in protocol["conditions"]:
        root = args.output / condition
        root.mkdir(exist_ok=True)
        config = deepcopy(args)
        for key in ("dataset", "tasks", "classes_per_task", "epochs", "train_per_class",
                    "probe_per_class", "memory", "batch_size", "lr", "continual_method", "random_controls"):
            setattr(config, key, protocol[key])
        config.rank = 8
        config.residual_pca_ranks = [8, 32, 64]
        config.save_layers = False
        config.save_trajectory = True
        config.save_pre_future_checkpoint = True
        config.model = "deit_tiny_patch16_224" if condition == "pretrained_deit_tiny" else "resnet18"
        config.pretrained = condition == "pretrained_deit_tiny"
        config.initial_checkpoint = None
        if not config.pretrained:
            config.initial_checkpoint = root / "shared_initialization.pt"
            if not config.initial_checkpoint.exists():
                seed_all(protocol["initialization_seed"])
                initial_net = model(config.model, False, num_classes=100)
                torch.save({"model": config.model, "seed": protocol["initialization_seed"],
                            "state_dict": state_dict_cpu(initial_net)}, config.initial_checkpoint)
                del initial_net
        source_paths, target_paths = [], []
        for role, orders, seeds, paths in (
            ("source", source_orders, [140, 141], source_paths),
            ("target", target_orders, [150, 151], target_paths),
        ):
            for order in orders:
                config.order_seed, config.seeds = order, seeds
                config.output = root / f"{role}_order{order}"
                paths.append(config.output)
                print(f"{condition}: {role} order {order}", flush=True)
                run(config)
                torch.cuda.empty_cache()
        analyze(argparse.Namespace(sources=source_paths, targets=target_paths, rank=8,
                                   residual_pca_rank=64, random_controls=128,
                                   refresh_matched_static=False, output=root / "atlas_analysis"))
        structure_run(argparse.Namespace(sources=source_paths, targets=target_paths,
                                         source_rank=8, primary_rank=8, ranks=[2, 4, 8, 16, 32],
                                         torch_threads=2, output=root / "scaffold_structure"))
        summary_path = root / "atlas_analysis" / "summary.json"
        summary = json.loads(read_config_text(summary_path, encoding="utf-8"))
        summary["matched_projector_static_definition"]["analysis_status"] = (
            "prospectively specified in parent protocol.json before generating these trajectories")
        summary["matched_projector_primary_passed"] = bool(
            summary["drift_minus_matched_projector_static_positive_orders"] >= 5
            and summary["drift_minus_matched_projector_static_hierarchical_ci"][0] > 0)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{condition}: COMPLETE; matched-projector primary={summary['matched_projector_primary_passed']}", flush=True)
    finalize_validation_suite(args.output)


def finalize_validation_suite(root):
    """Attach the suite's prospective protocol to the generic analysis outputs."""
    protocol = json.loads(read_config_text(root / "protocol.json", encoding="utf-8"))
    results = {"protocol": str(root / "protocol.json"), "conditions": {}}
    for condition in protocol["conditions"]:
        directory = root / condition
        main_path = directory / "atlas_analysis" / "summary.json"
        summary = json.loads(read_config_text(main_path, encoding="utf-8"))
        status = "prospectively specified in parent protocol.json before generating these trajectories"
        summary["matched_projector_static_definition"]["analysis_status"] = status
        summary["suite_primary_field"] = "matched_projector_primary_passed"
        summary["legacy_gate_definitions"] = {
            "primary_gate_passed": "legacy target-level matched-random gate; not this suite's primary",
            "residual64_gate_passed": "legacy target-level residual matched-random gate",
            "hierarchical_primary_gate_passed": "legacy order-hierarchical pooled-covariance gate",
        }
        main_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        structure_path = directory / "scaffold_structure" / "summary.json"
        structure = json.loads(read_config_text(structure_path, encoding="utf-8"))
        structure["definitions"]["comparator_status"] = status
        structure_path.write_text(json.dumps(structure, indent=2), encoding="utf-8")
        results["conditions"][condition] = {
            key: summary[key] for key in (
                "development_blocks", "heldout_targets", "atlas_coverage_mean",
                "matched_projector_static_coverage_mean", "weighted_static_atlas_coverage_mean",
                "drift_minus_matched_projector_static_mean", "drift_minus_matched_projector_static_hierarchical_ci",
                "drift_minus_matched_projector_static_positive_orders", "matched_projector_primary_passed",
                "drift_atlas_minus_weighted_static_mean", "drift_atlas_minus_weighted_static_hierarchical_ci",
                "drift_atlas_minus_weighted_static_positive_orders", "max_matching_error",
            )}
    (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Long-sequence cross-trajectory drift artifacts")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--classes-per-task", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--random-controls", type=int, default=128)
    parser.add_argument("--max-spectrum-error", type=float, default=1e-5)
    parser.add_argument("--svd-threshold", type=float, default=1e-7)
    parser.add_argument("--continual-method", choices=("naive", "er"), default="er")
    parser.add_argument("--residual-pca-ranks", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-per-class", type=int, default=450)
    parser.add_argument("--probe-per-class", type=int, default=50)
    parser.add_argument("--memory", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--der-alpha", type=float, default=0.5)
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--validation-suite", action="store_true",
                        help="run both complete CIFAR-100 backbone/initialization validation conditions")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="load the same complete initialization, then reset each trajectory's training RNG")
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-layers", action="store_true")
    parser.add_argument("--save-pre-future-checkpoint", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.validation_suite:
        validation_suite(arguments)
    elif arguments.order_seed is None:
        raise ValueError("--order-seed is required outside --validation-suite")
    else:
        run(arguments)
