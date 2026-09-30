"""Recover missing observations and run the remaining scaffold analyses.

Existing source/target assignments and hyperparameters are retained. Recovery
outputs never overwrite original experiments. All stages resume by artifact.
"""
from __future__ import annotations
from package_paths import read_config_text

import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from drift_experiment import (
    IndexedCifar, accuracy, balanced_memory, features, ids_by_class,
    infer, infer_intermediates, loader, model, prepare_images, seed_all, train_task,
)
from phase_c_direction import state_dict_cpu
from analyze_simple_sources_function import center, within_class_center, frozen_readout


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


class CaptureUpdate:
    """Observe the second actual optimizer step of the final task, without intervention."""
    def __init__(self):
        self.record = {}

    def __call__(self, event, epoch, batch, net, optimizer, ids, x, y, n_current):
        if "after_state" in self.record:
            return
        steps = max((int(state.get("step", 0)) for state in optimizer.state.values()), default=0)
        if event == "before":
            if steps != 1:
                return
            self.record["before_state"] = state_dict_cpu(net)
            self.record.update({
                "optimizer_before": cpu_tree(optimizer.state_dict()),
                "batch_ids": ids.cpu().clone(), "images": x.cpu().clone(),
                "labels": y.cpu().clone(), "n_current": n_current,
                "epoch": epoch, "batch_index": batch, "optimizer_step_before": steps,
            })
        elif steps == 2 and "before_state" in self.record:
            self.record["after_state"] = state_dict_cpu(net)
            self.record["optimizer_step_after"] = steps
            self.record["gradient"] = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in net.named_parameters() if parameter.grad is not None
            }


def consume_augmentations(dataset, current, memory, config, seed):
    """Advance only augmentation RNG when an exact completed-task state is available."""
    for epoch in range(config.epochs):
        current_loader = loader(dataset, current, config.batch_size // 2, True, seed + epoch)
        replay_loader = loader(dataset, memory, config.batch_size // 2, True, seed + 100 + epoch)
        replay = iter(replay_loader)
        for _ in current_loader:
            try:
                next(replay)
            except StopIteration:
                replay = iter(replay_loader)
                next(replay)


def recover(args):
    device = torch.device("cuda")
    for root in args.roots:
        manifest = json.loads(read_config_text(root / "manifest.json", encoding="utf-8"))
        config = SimpleNamespace(**manifest)
        train = IndexedCifar(Path(config.data), True, True, args.dataset)
        evaluation = IndexedCifar(Path(config.data), True, False, args.dataset)
        order = config.order
        train_ids = {c: ids_by_class(train.targets, [c], config.train_per_class) for c in sum(order, [])}
        heldout = ids_by_class(evaluation.targets, order[0], config.probe_per_class, tail=True)
        for original_path in sorted((root / "artifacts").glob("*.pt")):
            output_path = args.output / "artifacts" / original_path.name
            record = load(output_path) if output_path.exists() else None
            if record is not None and record["status"] == "complete":
                print(f"{original_path.stem}: recovered already", flush=True)
                continue
            original = load(original_path)
            seed = int(original["seed"])
            seed_all(seed)
            net = model(config.model, record is None, num_classes=train.num_classes).to(device)
            priority = torch.rand(len(train), generator=torch.Generator().manual_seed(seed + 9000))
            fit_ids = balanced_memory(train_ids, order[0], config.memory, priority)
            if record is None:
                record = {
                    "status": "running", "dataset": args.dataset, "role": args.role,
                    "original_path": str(original_path), "manifest": manifest,
                    "order_seed": original["order_seed"], "seed": seed,
                    "fit_ids": fit_ids, "heldout_ids": heldout,
                    "fit_labels": train.targets[fit_ids], "heldout_labels": train.targets[heldout],
                    "fit_reps": [], "eval_reps": [], "old_accuracy": [], "states": {},
                }
                if args.role == "source":
                    record["states"][0] = state_dict_cpu(net)
            else:
                net.load_state_dict(record["resume_state"])
                restore_rng(record["rng"])
            finished = len(record["fit_reps"])
            seen = sum(order[:finished], [])
            memory = balanced_memory(train_ids, seen, config.memory, priority) if seen else torch.empty(0, dtype=torch.long)
            end = 2 if args.role == "source" else len(order)
            for index in range(finished, end):
                started = time.monotonic()
                old = list(seen)
                seen += order[index]
                current = torch.cat([train_ids[c] for c in order[index]])
                observer = CaptureUpdate() if args.role == "target" and index == end - 1 else None
                if observer is not None:
                    record["states"][index] = state_dict_cpu(net)
                if args.role == "target" and index == len(order) - 2 and "pre_future_state_dict" in original:
                    consume_augmentations(train, current, memory, config, seed * 100 + index)
                    net.load_state_dict(original["pre_future_state_dict"])
                else:
                    train_task(net, "naive" if index == 0 else config.continual_method,
                               train, current, memory, {}, seen, old, config, device,
                               seed * 100 + index, step_observer=observer)
                if manifest.get("save_layers", False):
                    _, fit_layers, _ = infer_intermediates(net, evaluation, fit_ids, config.batch_size, device)
                    logits, eval_layers, labels = infer_intermediates(net, evaluation, heldout, config.batch_size, device)
                    fit_z, eval_z = fit_layers["stage4"], eval_layers["stage4"]
                else:
                    _, fit_z, _ = infer(net, evaluation, fit_ids, config.batch_size, device)
                    logits, eval_z, labels = infer(net, evaluation, heldout, config.batch_size, device)
                record["fit_reps"].append(fit_z)
                record["eval_reps"].append(eval_z)
                record["old_accuracy"].append(accuracy(logits, labels, seen))
                if args.role == "target":
                    all_ids = ids_by_class(evaluation.targets, seen, config.probe_per_class, tail=True)
                    all_logits, _, all_labels = infer(net, evaluation, all_ids, config.batch_size, device)
                    current_mask = torch.isin(all_labels, torch.tensor(order[index]))
                    record.setdefault("performance", []).append({
                        "checkpoint": index + 1,
                        "all_seen_accuracy": accuracy(all_logits, all_labels, seen),
                        "current_task_accuracy": accuracy(all_logits[current_mask], all_labels[current_mask], seen),
                        "first_task_accuracy": record["old_accuracy"][-1],
                    })
                state = state_dict_cpu(net)
                if args.role == "source" or index == end - 1:
                    record["states"][index + 1] = state
                if observer is not None:
                    record["actual_update"] = observer.record
                record["resume_state"], record["rng"] = state, rng_state()
                memory = balanced_memory(train_ids, seen, config.memory, priority)
                if index == 1:
                    recovered = center(record["fit_reps"][1] - record["fit_reps"][0])
                    record["history_max_abs_error"] = float((recovered - original["historical_drift_centered"]).abs().max())
                if index == end - 1:
                    if args.role == "target":
                        recovered = center(record["eval_reps"][-1] - record["eval_reps"][-2])
                        record["future_max_abs_error"] = float((recovered - original["future_drift_centered"]).abs().max())
                    record["status"] = "complete"
                    del record["resume_state"], record["rng"]
                save(record, output_path)
                print(f"{args.dataset} {args.role} {original_path.stem} task {index + 1}/{end}: "
                      f"{time.monotonic() - started:.1f}s, history error={record.get('history_max_abs_error')}, "
                      f"future error={record.get('future_max_abs_error')}", flush=True)
            del net, original, record
            torch.cuda.empty_cache()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["recover", "responses", "analyze"])
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--role", choices=["source", "target"])
    p.add_argument("--roots", type=Path, nargs="+")
    p.add_argument("--sources", type=Path)
    p.add_argument("--targets", type=Path)
    p.add_argument("--atlas", type=Path)
    p.add_argument("--responses", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--directions", type=int, default=16)
    p.add_argument("--relative-step", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=2)
    return p.parse_args()


@torch.inference_mode()
def image_batches(net, evaluation, ids, device):
    return [prepare_images(x, device) for _, x, _ in loader(evaluation, ids, 64)]


@torch.inference_mode()
def cached_representations(net, batches):
    net.eval()
    return torch.cat([features(net, x).float().cpu() for x in batches])


def basis(field, rank=8):
    return torch.linalg.svd(field.double(), full_matrices=False).Vh[:rank].T


def pool(bases, rank=8):
    return torch.linalg.svd(torch.cat(bases, dim=1), full_matrices=False).U[:, :rank]


def overlap(left, right):
    return float((left.T @ right).square().sum() / left.shape[1])


def coverage(field, subspace):
    field = field.double()
    return float((field @ subspace.double()).square().sum() / field.square().sum())


def responses(args):
    device = torch.device("cuda")
    evaluation = IndexedCifar(Path("data"), True, False, args.dataset)
    for path in sorted((args.sources / "artifacts").glob("*.pt")):
        output = args.output / "artifacts" / path.name
        if output.exists():
            saved = load(output)
            if (saved["directions"], saved["relative_step"]) != (args.directions, args.relative_step):
                raise ValueError("response settings differ from saved results; use a separate output directory")
            print(f"{path.stem}: responses already complete", flush=True)
            continue
        record = load(path)
        if record["status"] != "complete":
            raise ValueError(f"incomplete recovery: {path}")
        net = model(record["manifest"]["model"], False, num_classes=evaluation.num_classes).to(device)
        batches = image_batches(net, evaluation, record["fit_ids"], device)
        head_ids = {id(p) for p in net.get_classifier().parameters()}
        names = [name for name, p in net.named_parameters() if id(p) not in head_ids]
        parameters = dict(net.named_parameters())
        output_record = {"order_seed": record["order_seed"], "seed": record["seed"],
                         "directions": args.directions, "relative_step": args.relative_step,
                         "parameter_metric": "isotropic encoder Euclidean directions; global relative norm",
                         "difference": "centered symmetric finite displacement, FP32 evaluation, fixed BN buffers",
                         "states": {}}
        for checkpoint in [0, 2]:
            state = record["states"][checkpoint]
            net.load_state_dict(state)
            norm = torch.sqrt(sum(state[name].double().square().sum() for name in names))
            generator = torch.Generator().manual_seed(record["order_seed"] * 100003 + record["seed"] + 701)
            feature_dimension = net.get_classifier().weight.shape[1]
            covariance = torch.zeros(feature_dimension, feature_dimension, dtype=torch.float64)
            prefixes = {}
            for draw in range(args.directions):
                direction = {name: torch.randn(state[name].shape, generator=generator) for name in names}
                length = torch.sqrt(sum(value.double().square().sum() for value in direction.values()))
                scale = float(args.relative_step * norm / length)
                values = []
                for sign in [1, -1]:
                    net.load_state_dict(state)
                    with torch.no_grad():
                        for name in names:
                            parameters[name].add_(direction[name].to(device), alpha=sign * scale)
                    values.append(cached_representations(net, batches))
                field = center((values[0].double() - values[1].double()) / 2)
                covariance += field.T @ field
                if draw + 1 in [8, args.directions]:
                    eig, vectors = torch.linalg.eigh(covariance / (draw + 1))
                    prefixes[draw + 1] = {"basis": vectors[:, -8:], "spectrum": eig.flip(0)}
            output_record["states"][checkpoint] = {
                "covariance": covariance / args.directions, "prefixes": prefixes,
                "prefix_overlap": overlap(prefixes[8]["basis"], prefixes[args.directions]["basis"]),
            }
            print(f"{args.dataset} {path.stem} response at task {checkpoint}: "
                  f"8/{args.directions} direction overlap={output_record['states'][checkpoint]['prefix_overlap']:.4f}", flush=True)
        save(output_record, output)
        del net, record, batches
        torch.cuda.empty_cache()


def classifier_tensors(state):
    """Classifier coordinates for the ResNet and DeiT final-feature measurements."""
    prefixes = [prefix for prefix in ("fc", "head") if f"{prefix}.weight" in state]
    if len(prefixes) != 1:
        raise ValueError(f"Expected one supported classifier, found {prefixes}")
    prefix = prefixes[0]
    return state[f"{prefix}.weight"].double(), state[f"{prefix}.bias"].double()


def reconstruction(pre, delta, subspace, state, classes, labels):
    pre, delta = pre.double(), delta.double()
    weight, bias = classifier_tensors(state)
    results = {}
    for name, start, change in [("raw", pre, delta),
                                ("centered_conditional", pre + delta.mean(0), center(delta))]:
        before_logits, before_loss = frozen_readout(start, weight, bias, classes, labels)
        full_logits, full_loss = frozen_readout(start + change, weight, bias, classes, labels)
        projected_logits, projected_loss = frozen_readout(start + change @ subspace @ subspace.T,
                                                         weight, bias, classes, labels)
        logit_change = full_logits - before_logits
        predicted_logit_change = projected_logits - before_logits
        for endpoint, full, predicted in [
            ("class_centered_logits", logit_change - logit_change.mean(1, keepdim=True),
             predicted_logit_change - predicted_logit_change.mean(1, keepdim=True)),
            ("loss", full_loss - before_loss, projected_loss - before_loss),
            ("legacy_uncentered_logits", logit_change, predicted_logit_change),
        ]:
            results[f"{name}_{endpoint}_reconstruction"] = 1 - float((full - predicted).square().sum() / full.square().sum())
    return results


def actual_update_fields(record, evaluation, device):
    captured = record["actual_update"]
    if "after_state" not in captured:
        raise ValueError("second successful optimizer update was not captured")
    before, after = captured["before_state"], captured["after_state"]
    net = model(record["manifest"]["model"], False, num_classes=evaluation.num_classes).to(device)
    batches = image_batches(net, evaluation, record["heldout_ids"], device)
    net.load_state_dict(before)
    base = cached_representations(net, batches)
    net.load_state_dict(after)
    full = cached_representations(net, batches) - base
    buffer_names = set(dict(net.named_buffers()))
    parameter_state = {name: before[name] if name in buffer_names else value for name, value in after.items()}
    net.load_state_dict(parameter_state)
    parameter_field = cached_representations(net, batches) - base
    head_ids = {id(p) for p in net.get_classifier().parameters()}
    names = [name for name, p in net.named_parameters() if id(p) not in head_ids]
    delta_norm = torch.sqrt(sum((after[name].double() - before[name].double()).square().sum() for name in names))
    grad_norm = torch.sqrt(sum(captured["gradient"][name].double().square().sum() for name in names))
    net.load_state_dict(before)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name in names:
                p.add_(captured["gradient"][name].to(device), alpha=-float(delta_norm / grad_norm))
    gradient_field = cached_representations(net, batches) - base
    relative_norm = float(delta_norm / torch.sqrt(sum(before[name].double().square().sum() for name in names)))
    del net, batches
    return {"adamw_full": center(full.double()), "adamw_parameters": center(parameter_field.double()),
            "norm_matched_training_gradient": center(gradient_field.double())}, relative_norm


def average_rows(rows, keys):
    return {key: {"mean": float(np.mean([r[key] for r in rows])),
                  "order_means": {str(order): float(np.mean([r[key] for r in rows if r["order_seed"] == order]))
                                  for order in sorted({r["order_seed"] for r in rows})}}
            for key in keys}


def task_state_fields(record, evaluation):
    final_task = len(record["eval_reps"])
    pre, post = record["states"][final_task - 1], record["states"][final_task]
    net = model(record["manifest"]["model"], False, num_classes=evaluation.num_classes).to("cuda")
    batches = image_batches(net, evaluation, record["heldout_ids"], "cuda")
    net.load_state_dict(pre)
    base = cached_representations(net, batches)
    buffers = set(dict(net.named_buffers()))
    fields = {}
    for name, state in [("parameter_only", {k: pre[k] if k in buffers else v for k, v in post.items()}),
                        ("bn_only", {k: post[k] if k in buffers else v for k, v in pre.items()})]:
        net.load_state_dict(state)
        fields[name] = center((cached_representations(net, batches) - base).double())
    return fields


def analyze(args):
    atlas_file = load(args.atlas)
    scaffold, static = atlas_file["atlas"].double(), atlas_file["weighted_static_atlas"].double()
    matched_static = atlas_file["matched_projector_static_atlas"].double()
    within_bases, within_covariances, response_bases = [], [], {0: [], 2: []}
    response_rows, prefix_bases, source_errors = [], {}, []
    for path in sorted((args.sources / "artifacts").glob("*.pt")):
        source = load(path)
        source_errors.append(source["history_max_abs_error"])
        within_history = within_class_center((source["fit_reps"][1] - source["fit_reps"][0]).double(), source["fit_labels"])
        within_static = within_class_center(source["fit_reps"][1].double(), source["fit_labels"])
        within_bases.append(basis(within_history))
        within_covariances.append(within_static.T @ within_static / (len(within_static) - 1))
        response = load(args.responses / "artifacts" / path.name)
        for checkpoint in [0, 2]:
            response_bases[checkpoint].append(response["states"][checkpoint]["prefixes"][response["directions"]]["basis"])
            for count, values in response["states"][checkpoint]["prefixes"].items():
                prefix_bases.setdefault((checkpoint, count), []).append(values["basis"])
            response_rows.append({"order_seed": source["order_seed"], "seed": source["seed"],
                                  "checkpoint": checkpoint, "directions": response["directions"],
                                  "relative_step": response["relative_step"],
                                  "prefix_overlap": response["states"][checkpoint]["prefix_overlap"]})
    within_atlas = pool(within_bases)
    within_static = torch.linalg.eigh(sum(within_covariances)).eigenvectors[:, -8:]
    within_matched_static = pool([torch.linalg.eigh(c).eigenvectors[:, -8:] for c in within_covariances])
    initial_response, trained_response = pool(response_bases[0]), pool(response_bases[2])
    predictors = {"scaffold": scaffold, "static": static, "initial_response": initial_response,
                  "trained_response": trained_response, "matched_static": matched_static}
    unions = {}
    for name in ["initial_response", "trained_response"]:
        for addition in ["scaffold", "static", "matched_static"]:
            unions[f"{name}_plus_{addition}"] = pool([predictors[name], predictors[addition]], rank=16)
    geometry = {f"{left}_vs_{right}": overlap(a, b) for left, a in predictors.items()
                for right, b in predictors.items() if left < right}
    temporal, functional, controls, updates, target_errors = [], [], [], [], []
    evaluation = IndexedCifar(Path("data"), True, False, args.dataset)
    for path in sorted((args.targets / "artifacts").glob("*.pt")):
        target = load(path)
        if target["status"] != "complete":
            raise ValueError(f"incomplete recovery: {path}")
        target_errors.append({"history": target["history_max_abs_error"], "future": target["future_max_abs_error"]})
        identity = {"order_seed": target["order_seed"], "seed": target["seed"]}
        fields = target["eval_reps"]
        for index in range(1, len(fields)):
            drift = center((fields[index] - fields[index - 1]).double())
            row = {**identity, "transition": f"{index}_{index + 1}",
                   "scaffold": coverage(drift, scaffold), "static": coverage(drift, static),
                   "drift_energy": float(drift.square().sum()), "old_accuracy": target["old_accuracy"][index],
                   "current_task_accuracy": target["performance"][index]["current_task_accuracy"],
                   "all_seen_accuracy": target["performance"][index]["all_seen_accuracy"]}
            row["effect"] = row["scaffold"] - row["static"]
            row["matched_static"] = coverage(drift, matched_static)
            row["matched_effect"] = row["scaffold"] - row["matched_static"]
            row["scaffold_energy"] = row["drift_energy"] * row["scaffold"]
            row["orthogonal_energy"] = row["drift_energy"] - row["scaffold_energy"]
            row["scaffold_mean_squared_displacement"] = row["scaffold_energy"] / len(drift)
            row["orthogonal_mean_squared_displacement"] = row["orthogonal_energy"] / len(drift)
            temporal.append(row)
        raw = (fields[-1] - fields[-2]).double()
        within = within_class_center(raw, target["heldout_labels"])
        row = {**identity, "within_scaffold": coverage(within, within_atlas),
               "within_static": coverage(within, within_static)}
        row["within_effect"] = row["within_scaffold"] - row["within_static"]
        row["within_matched_static"] = coverage(within, within_matched_static)
        row["within_matched_effect"] = row["within_scaffold"] - row["within_matched_static"]
        for name, predictor in predictors.items():
            row[name] = coverage(center(raw), predictor)
        for name, predictor in unions.items():
            row[name] = coverage(center(raw), predictor)
        for name in ["initial_response", "trained_response"]:
            row[f"{name}_conditional_history_advantage"] = row[f"{name}_plus_scaffold"] - row[f"{name}_plus_static"]
            row[f"{name}_matched_conditional_history_advantage"] = row[f"{name}_plus_scaffold"] - row[f"{name}_plus_matched_static"]
        row["scaffold_minus_initial_response"] = row["scaffold"] - row["initial_response"]
        row["scaffold_minus_trained_response"] = row["scaffold"] - row["trained_response"]
        row["trained_response_minus_initial"] = row["trained_response"] - row["initial_response"]
        row["trained_response_minus_static"] = row["trained_response"] - row["static"]
        row["trained_response_minus_matched_static"] = row["trained_response"] - row["matched_static"]
        if args.dataset == "cifar10":
            state_path = args.output / "task_state_fields" / path.name
            if state_path.exists():
                state_fields = load(state_path)
            else:
                state_fields = task_state_fields(target, evaluation)
                save(state_fields, state_path)
            for name, field in state_fields.items():
                for method, predictor in [("scaffold", scaffold), ("static", static), ("matched_static", matched_static)]:
                    row[f"{name}_{method}"] = coverage(field, predictor)
                row[f"{name}_matched_effect"] = row[f"{name}_scaffold"] - row[f"{name}_matched_static"]
        controls.append(row)
        old_classes = sum(target["manifest"]["order"][:-1], [])
        before_state = target["states"][len(fields) - 1]
        row = dict(identity)
        for name, predictor in [("scaffold", scaffold), ("static", static), ("matched_static", matched_static)]:
            values = reconstruction(fields[-2], raw, predictor, before_state, old_classes, target["heldout_labels"])
            row.update({f"{name}_{key}": value for key, value in values.items()})
        for key in values:
            row[f"effect_{key}"] = row[f"scaffold_{key}"] - row[f"static_{key}"]
            row[f"matched_effect_{key}"] = row[f"scaffold_{key}"] - row[f"matched_static_{key}"]
        row["mean_displacement_energy_fraction"] = float(len(raw) * raw.mean(0).square().sum() / raw.square().sum())
        weight, bias = classifier_tensors(before_state)
        old_logits, old_loss = frozen_readout(fields[-2].double(), weight, bias, old_classes, target["heldout_labels"])
        mean_logits, mean_loss = frozen_readout(fields[-2].double() + raw.mean(0), weight, bias,
                                                old_classes, target["heldout_labels"])
        full_logits, full_loss = frozen_readout(fields[-2].double() + raw, weight, bias,
                                                old_classes, target["heldout_labels"])
        for endpoint, full, mean in [("class_centered_logits", full_logits - old_logits, mean_logits - old_logits),
                                     ("loss", full_loss - old_loss, mean_loss - old_loss)]:
            if endpoint == "class_centered_logits":
                full = full - full.mean(1, keepdim=True)
                mean = mean - mean.mean(1, keepdim=True)
            row[f"mean_only_{endpoint}_reconstruction"] = 1 - float((full - mean).square().sum() / full.square().sum())
        functional.append(row)
        cached_path = args.output / "update_fields" / path.name
        if cached_path.exists():
            update_data = load(cached_path)
        else:
            measured, norm = actual_update_fields(target, evaluation, torch.device("cuda"))
            update_data = {**identity, "fields": measured, "encoder_relative_step": norm,
                           "epoch": target["actual_update"]["epoch"], "batch_index": target["actual_update"]["batch_index"]}
            save(update_data, cached_path)
        row = {**identity, "encoder_relative_step": update_data["encoder_relative_step"]}
        for name, field in update_data["fields"].items():
            row[f"{name}_scaffold"] = coverage(field, scaffold)
            row[f"{name}_static"] = coverage(field, static)
            row[f"{name}_effect"] = row[f"{name}_scaffold"] - row[f"{name}_static"]
            row[f"{name}_matched_static"] = coverage(field, matched_static)
            row[f"{name}_matched_effect"] = row[f"{name}_scaffold"] - row[f"{name}_matched_static"]
            row[f"{name}_energy"] = float(field.square().sum())
        updates.append(row)
        print(f"{args.dataset} {path.stem}: all target analyses complete", flush=True)
        del target
    summary = {"dataset": args.dataset, "source_count": len(within_bases), "target_count": len(controls),
               "rank": 8, "response_geometry": geometry, "response_sampling": response_rows,
               "recovery_checks": {"source_history_max_abs": max(source_errors),
                                   "target_history_max_abs": max(r["history"] for r in target_errors),
                                   "target_future_max_abs": max(r["future"] for r in target_errors)},
               "definitions": {"response_construction": "source-only isotropic encoder directions; count and relative step in response_sampling; fixed BN buffers; symmetric FP32 finite displacements",
                               "static_comparators": "static preserves pooled covariance; matched_static uses equal source top-8 projectors with output rank 8; within-class matched static rebuilds each source projector after within-class centering",
                               "conditional_history_comparison": "rank-16 union of response+history versus response+static; source-only bases",
                               "functional_centered": "conditional change from pre+mean(delta) to pre+mean(delta)+center(delta)",
                               "functional_score": "1-SSE/sum(full_change^2), zero-change reference; logits centered across old classes except legacy fields",
                               "actual_update": "second successful AdamW step of final task, with replay; gradient response matched to actual encoder update norm",
                               "accuracy": "fixed per-class held-out training-split anchors, predicted among all currently seen classes",
                               "aggregation": "equal target averages and target-order means; recovered runs are not additional independent samples"},
               "response_sampling_pooled_overlap": {
                   str(checkpoint): overlap(pool(prefix_bases[(checkpoint, 8)]), pool(response_bases[checkpoint]))
                   for checkpoint in [0, 2]},
               "within_scaffold_vs_original_overlap": overlap(within_atlas, scaffold),
               "temporal": {}, "controls": {}, "functional": {}, "updates": {}}
    for transition in sorted({r["transition"] for r in temporal}):
        subset = [r for r in temporal if r["transition"] == transition]
        summary["temporal"][transition] = average_rows(subset, ["scaffold", "static", "effect", "matched_static", "matched_effect", "old_accuracy",
                                                              "current_task_accuracy", "all_seen_accuracy", "drift_energy",
                                                              "scaffold_energy", "orthogonal_energy",
                                                              "scaffold_mean_squared_displacement",
                                                              "orthogonal_mean_squared_displacement"])
    for name, rows in [("controls", controls), ("functional", functional), ("updates", updates)]:
        summary[name] = average_rows(rows, [key for key in rows[0] if key not in ["order_seed", "seed"]])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import csv
    for name, rows in [("temporal", temporal), ("controls", controls), ("functional", functional), ("updates", updates)]:
        with (args.output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save({**predictors, "within_scaffold": within_atlas, "within_static": within_static,
          "within_matched_static": within_matched_static, **unions}, args.output / "predictors.pt")
    plot_summary(summary, args.output)
    print(json.dumps({key: value["mean"] for key, value in summary["controls"].items()}, indent=2), flush=True)


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    teal, orange, gray, blue = "#167D8D", "#B26948", "#687484", "#476FAD"
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 2, figsize=(11.6, 11), layout="constrained")
        time_keys = list(summary["temporal"])
        for metric, label, color in [("scaffold", "Scaffold", teal), ("static", "Source static", orange)]:
            axes[0, 0].plot(time_keys, [100 * summary["temporal"][key][metric]["mean"] for key in time_keys],
                            marker="o", color=color, label=label)
        axes[0, 0].set(xlabel="Target task transition", ylabel="Drift coverage (%)", title="A fixed source subspace across target learning")
        axes[0, 0].set_xticks(range(len(time_keys)), [key.replace("_", " → ") for key in time_keys])
        axes[0, 0].legend(frameon=False)
        metrics = ["scaffold", "static", "initial_response", "trained_response"]
        labels = ["Historical\nscaffold", "Source\nstatic", "Initial\nresponse", "Task-2\nresponse"]
        axes[0, 1].bar(labels, [100 * summary["controls"][key]["mean"] for key in metrics], color=[teal, orange, gray, blue])
        axes[0, 1].set(ylabel="Final-transition coverage (%)", title="Source response subspaces transferred to targets")
        c = summary["controls"]
        orders = list(c["within_scaffold"]["order_means"])
        for order in orders:
            axes[1, 0].plot([0, 1], [100 * c[key]["order_means"][order] for key in ["within_static", "within_scaffold"]],
                            marker="o", color=gray, alpha=.65)
        axes[1, 0].set(xticks=[0, 1], xticklabels=["Within-class static", "Within-class scaffold"],
                       ylabel="Within-class drift coverage (%)", title="Matched within-class comparison by target order")
        ax = axes[1, 1]
        metrics = ["raw_class_centered_logits_reconstruction", "centered_conditional_class_centered_logits_reconstruction",
                   "raw_loss_reconstruction", "centered_conditional_loss_reconstruction"]
        x = np.arange(len(metrics))
        for offset, predictor, color in [(-.18, "scaffold", teal), (.18, "static", orange)]:
            ax.bar(x + offset, [100 * summary["functional"][f"{predictor}_{key}"]["mean"] for key in metrics],
                   width=.34, label=predictor.capitalize(), color=color)
        ax.set(xticks=x, xticklabels=["Full drift\nlogits", "Centered drift\nlogits", "Full drift\nloss", "Centered drift\nloss"],
               ylabel="Reconstruction vs zero change (%)", title="Frozen readout: class-centered logits and losses")
        ax.axhline(0, color=gray, linewidth=.8)
        ax.legend(frameon=False)
        for metric, label, color in [("old_accuracy", "First task", teal), ("current_task_accuracy", "Current task", orange),
                                      ("all_seen_accuracy", "All seen", gray)]:
            axes[2, 0].plot(time_keys, [100 * summary["temporal"][key][metric]["mean"] for key in time_keys],
                            marker="o", color=color, label=label)
        axes[2, 0].set(xlabel="Target task transition", ylabel="Held-out accuracy (%)", title="Learning state at each measured transition")
        axes[2, 0].set_xticks(range(len(time_keys)), [key.replace("_", " → ") for key in time_keys])
        axes[2, 0].legend(frameon=False)
        metrics = ["adamw_full", "adamw_parameters", "norm_matched_training_gradient"]
        for offset, predictor, color in [(-.18, "scaffold", teal), (.18, "static", orange)]:
            axes[2, 1].bar(np.arange(3) + offset, [100 * summary["updates"][f"{key}_{predictor}"]["mean"] for key in metrics],
                            width=.34, color=color, label=predictor.capitalize())
        axes[2, 1].set(xticks=np.arange(3), xticklabels=["Actual AdamW\nfull step", "AdamW\nfixed buffers", "Norm-matched\ntraining gradient"],
                       ylabel="Induced-drift coverage (%)", title="Second successful update of the final task")
        axes[2, 1].legend(frameon=False)
        axes[2, 1].margins(y=.25)
        fig.suptitle(summary["dataset"].upper() + ": scaffold follow-up experiments", fontsize=15)
        fig.savefig(output / "followup.png", dpi=180)
        fig.savefig(output / "followup.svg")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
        x = np.arange(len(time_keys))
        projected = [summary["temporal"][key]["scaffold_mean_squared_displacement"]["mean"] for key in time_keys]
        residual = [summary["temporal"][key]["orthogonal_mean_squared_displacement"]["mean"] for key in time_keys]
        ax.stackplot(x, projected, residual, colors=[teal, "#D8DFE5"],
                     labels=["Scaffold projection", "Orthogonal residual"])
        ax.set(xticks=x, xticklabels=[key.replace("_", " → ") for key in time_keys],
               xlabel="Target task transition", ylabel="Mean squared feature displacement",
               title=summary["dataset"].upper() + ": centered drift energy")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        fig.savefig(output / "temporal_energy.png", dpi=180)
        fig.savefig(output / "temporal_energy.svg")
        plt.close(fig)


if __name__ == "__main__":
    args = parser()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.mode == "recover":
        recover(args)
    elif args.mode == "responses":
        responses(args)
    else:
        analyze(args)
