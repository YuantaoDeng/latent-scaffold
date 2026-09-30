"""Complete the three-claim DeiT extension using frozen original trajectories."""
from __future__ import annotations
from package_paths import read_config_text

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import analyze_scaffold_followup as f
import numpy as np
import torch
from analyze_prospective_atlas import hierarchical_bootstrap, matched
from analyze_simple_sources_function import advance_augmentation_rng
from phase_c_direction import functional_bases, state_dict_cpu
from temporal_drift_reuse import spectrum_matched_basis

ORIGINAL = Path("results/backbone_initialization_validation/pretrained_deit_tiny")
ROOT = Path("results/deit_three_claims")
SOURCE_ORDERS = [1009, 1013, 1019]
TARGET_ORDERS = [1103, 1109, 1117, 1123, 1129, 1151]


def json_read(path):
    return json.loads(read_config_text(path, encoding="utf-8"))


def json_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def csv_save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [{k: float(v) if k not in ("transition",) else v for k, v in row.items()}
                for row in csv.DictReader(stream)]


def inventory(role):
    orders, seeds = (SOURCE_ORDERS, [140, 141]) if role == "source" else (TARGET_ORDERS, [150, 151])
    paths = []
    for order in orders:
        manifest = json_read(ORIGINAL / f"{role}_order{order}/manifest.json")
        assert manifest["model"] == "deit_tiny_patch16_224" and manifest["status"] == "complete"
        for seed in seeds:
            path = ORIGINAL / f"{role}_order{order}/artifacts/order{order}_seed{seed}.pt"
            assert path.exists()
            paths.append((path, manifest))
    return paths


def fixed_predictors():
    saved = f.load(ORIGINAL / "atlas_analysis/atlas.pt")
    return {"scaffold": saved["atlas"].double(),
            "matched_static": saved["matched_projector_static_atlas"].double(),
            "pooled_static": saved["weighted_static_atlas"].double()}


def stats(rows, keys, seed_offset=6001):
    orders = np.array([int(r["order_seed"]) for r in rows])
    result = {}
    for i, key in enumerate(keys):
        values = np.array([r[key] for r in rows], dtype=float)
        assert np.isfinite(values).all(), key
        means = {str(o): float(values[orders == o].mean()) for o in sorted(set(orders))}
        groups = np.array([values[orders == o] for o in sorted(set(orders))])
        rng = np.random.default_rng(seed_offset + i)
        selected_orders = rng.integers(0, len(groups), (20000, len(groups), 1))
        selected_seeds = rng.integers(0, groups.shape[1], (20000, len(groups), groups.shape[1]))
        sampled = groups[selected_orders, selected_seeds].mean(axis=(1, 2))
        interval = np.quantile(sampled, [.025, .975]).tolist()
        result[key] = {"mean": float(values.mean()), "ci95": interval,
                       "order_means": means, "positive_orders": sum(v > 0 for v in means.values())}
    return result


def original_measurements():
    """Compute transfer, temporal reuse and readout directly, without any retraining."""
    predictors = fixed_predictors()
    temporal, functional = [], []
    controls_out = ROOT / "original_control_bases"
    checks = []
    for target_index, (path, manifest) in enumerate(inventory("target")):
        a = f.load(path)
        identity = {"order_seed": a["order_seed"], "seed": a["seed"]}
        assert a["eval_representations"][0].shape[1] == 192
        assert not set(a["fit_ids"].tolist()) & set(a["eval_ids"].tolist())
        for index in range(1, 4):
            field = f.center((a["eval_representations"][index] - a["eval_representations"][index-1]).double())
            row = {**identity, "transition": f"{index}_{index+1}"}
            row.update({name: f.coverage(field, u) for name, u in predictors.items()})
            # Use the original source-scaffold/classifier matching and random seeds for all transitions.
            seed = 1_000_003 * (target_index + 1) + 1_100_000
            control = matched(predictors["scaffold"].float(), field.float(),
                              a["discriminative_basis"], a["null_basis"], 128, seed)
            row.update(matched_random=control["matched_mean"], matching_error=control["max_error"],
                       isotropic=8/192, old_accuracy=a["task1_accuracies"][index])
            for name in ("matched_random", "isotropic", "matched_static", "pooled_static"):
                row[f"scaffold_minus_{name}"] = row["scaffold"] - row[name]
            temporal.append(row)
        state = a["pre_future_state_dict"]
        old_classes = sum(manifest["order"][:-1], [])
        pre = a["eval_representations"][-2]
        raw = a["eval_representations"][-1] - pre
        row = dict(identity)
        for name, u in predictors.items():
            reconstruction = f.reconstruction(pre, raw, u, state, old_classes, a["eval_labels"])
            row.update({name + "_" + key: value for key, value in reconstruction.items()
                        if key.startswith("centered_conditional") and "legacy" not in key})
        # Additional contextual control, matched to the actual frozen pre-final readout.
        weight, _ = f.classifier_tensors(state)
        discriminative, null = functional_bases(weight[old_classes].float(), 1e-7)[:2]
        bases, draws, errors = [], [], []
        for draw in range(128):
            generator = torch.Generator().manual_seed(a["order_seed"] * 100003 + a["seed"] * 1009 + 810000 + draw)
            u, error1, error2 = spectrum_matched_basis(predictors["scaffold"].float(), discriminative,
                                                      null, 8, generator)
            bases.append(u)
            errors.append(max(error1, error2))
            values = f.reconstruction(pre, raw, u.double(), state, old_classes, a["eval_labels"])
            draws.append({k: v for k, v in values.items() if k.startswith("centered_conditional") and "legacy" not in k})
        for key in draws[0]:
            row["frozen_readout_matched_random_" + key] = float(np.mean([d[key] for d in draws]))
            row["scaffold_minus_random_" + key] = row["scaffold_"+key] - row["frozen_readout_matched_random_"+key]
        row["matching_error"] = max(errors)
        functional.append(row)
        f.save({"basis": bases, "max_matching_error": max(errors), "matching_checkpoint": 3,
                "definition": "matches frozen old-class readout projection spectrum"}, controls_out / path.name)
        checks.append({**identity, "anchors_disjoint": True, "feature_dimension": 192})
        print(path.stem, "original temporal/readout complete", flush=True)
    csv_save(ROOT / "original_temporal.csv", temporal)
    csv_save(ROOT / "original_functional.csv", functional)
    summary = {"target_count": 12, "target_order_count": 6, "temporal": {}}
    for transition in ("1_2", "2_3", "3_4"):
        rows = [r for r in temporal if r["transition"] == transition]
        summary["temporal"][transition] = stats(rows, [k for k in rows[0] if k not in ("order_seed", "seed", "transition")])
    summary["functional"] = stats(functional, [k for k in functional[0] if k not in ("order_seed", "seed")])
    summary["checks"] = checks
    json_save(ROOT / "original_summary.json", summary)


def recover_targets():
    device = torch.device("cuda")
    train_data = f.IndexedCifar(Path("data"), True, True, "cifar100")
    evaluation = f.IndexedCifar(Path("data"), True, False, "cifar100")
    for path, manifest in inventory("target"):
        output = ROOT / "targets/artifacts" / path.name
        if output.exists():
            saved = f.load(output)
            assert saved["status"] == "complete" and saved["future_max_abs_error"] <= 1e-5
            print(path.stem, "target already recovered", flush=True)
            continue
        started = time.monotonic()
        a = f.load(path)
        seed = a["seed"]
        config = NS(**manifest)
        f.seed_all(seed)
        net = f.model(config.model, True, num_classes=100).cuda()
        for module in net.modules():
            if isinstance(module, torch.nn.Dropout):
                assert module.p == 0
            assert getattr(module, "drop_prob", 0) == 0
        assert not dict(net.named_buffers()), "Unexpected DeiT buffers"
        priority = torch.rand(len(train_data), generator=torch.Generator().manual_seed(seed + 9000))
        order = manifest["order"]
        train_ids = {c: f.ids_by_class(train_data.targets, [c], config.train_per_class) for c in sum(order, [])}
        # All CIFAR-100 images have the same dimensions. With zero dropout, earlier
        # augmentation RNG depends on the seed and batch counts, not class identities.
        rng_config = {k: manifest[k] for k in ("model", "tasks", "classes_per_task", "epochs",
                      "train_per_class", "batch_size", "memory", "continual_method")}
        rng_config.update(seed=seed, image_shape=[32, 32], torch=torch.__version__)
        cache = ROOT / "augmentation_rng" / f"seed{seed}.pt"
        if cache.exists():
            saved_rng = f.load(cache)
            assert saved_rng["config"] == rng_config
            f.restore_rng(saved_rng["rng"])
            seen = sum(order[:-1], [])
            memory = f.balanced_memory(train_ids, seen, config.memory, priority)
        else:
            seen, memory = advance_augmentation_rng(train_data, train_ids, order, priority, config, seed)
            f.save({"config": rng_config, "rng": f.rng_state()}, cache)
        net.load_state_dict(a["pre_future_state_dict"], strict=True)
        _, pre, _ = f.infer(net, evaluation, a["eval_ids"], 64, device)
        pre_error = float((pre - a["eval_representations"][-2]).abs().max())
        assert pre_error <= 1e-5, (path, "pre", pre_error)
        current = torch.cat([train_ids[c] for c in order[-1]])
        observer = f.CaptureUpdate()
        f.train_task(net, config.continual_method, train_data, current, memory, {},
                     seen + order[-1], seen, config, device,
                     seed * 100 + 3, step_observer=observer)
        _, fit, _ = f.infer(net, evaluation, a["fit_ids"], 64, device)
        _, post, _ = f.infer(net, evaluation, a["eval_ids"], 64, device)
        field = f.center(post - pre)
        errors = {"pre_max_abs_error": pre_error,
                  "fit_final_max_abs_error": float((fit - a["fit_representations"][-1]).abs().max()),
                  "eval_final_max_abs_error": float((post - a["eval_representations"][-1]).abs().max()),
                  "future_max_abs_error": float((field - a["future_drift_centered"]).abs().max())}
        if max(errors.values()) > 1e-5:
            json_save(ROOT / "replay_mismatch.json", {"path": str(path), **errors})
            raise RuntimeError(f"Replay differs from original: {errors}")
        assert observer.record["optimizer_step_before"] == 1 and observer.record["optimizer_step_after"] == 2
        record = {"status": "complete", "dataset": "cifar100", "manifest": manifest,
                  "original_path": str(path), "order_seed": a["order_seed"], "seed": seed,
                  "heldout_ids": a["eval_ids"], "fit_ids": a["fit_ids"], "heldout_labels": a["eval_labels"],
                  "eval_reps": a["eval_representations"], "fit_reps": a["fit_representations"],
                  "history_max_abs_error": 0., **errors, "actual_update": observer.record,
                  "states": {3: a["pre_future_state_dict"], 4: state_dict_cpu(net)},
                  "recovery_method": "saved task-3 state plus exact augmentation-RNG replay; final task only"}
        f.save(record, output)
        print(path.stem, "target recovered", errors, f"{time.monotonic()-started:.1f}s", flush=True)
        del net, a, record, observer
        torch.cuda.empty_cache()


def analyze_recovered():
    predictors = fixed_predictors()
    source_records, response_records = [], []
    source_checks = []
    for path, _ in inventory("source"):
        original = f.load(path)
        source = f.load(ROOT / "sources/artifacts" / path.name)
        assert source["status"] == "complete"
        checks = {"history": source["history_max_abs_error"]}
        for i in (0, 1):
            checks[f"fit_{i+1}"] = float((source["fit_reps"][i]-original["fit_representations"][i]).abs().max())
            checks[f"eval_{i+1}"] = float((source["eval_reps"][i]-original["eval_representations"][i]).abs().max())
        assert max(checks.values()) <= 1e-5, (path, checks)
        assert torch.equal(source["fit_ids"], original["fit_ids"])
        assert torch.equal(source["heldout_ids"], original["eval_ids"])
        source_checks.append({"path": str(path), **checks})
        source_records.append(original)
        response_records.append(f.load(ROOT / "responses/artifacts" / path.name))
    geometry = {}
    for checkpoint in (0, 2):
        bases = [r["states"][checkpoint]["prefixes"][16]["basis"] for r in response_records]
        prefix = [r["states"][checkpoint]["prefixes"][8]["basis"] for r in response_records]
        name = "initial_response" if checkpoint == 0 else "trained_response"
        predictors[name] = f.pool(bases)
        geometry[name + "_scaffold_overlap"] = f.overlap(predictors[name], predictors["scaffold"])
        geometry[name + "_8_vs_16_overlap"] = f.overlap(f.pool(prefix), predictors[name])
    geometry["overlap_change"] = geometry["trained_response_scaffold_overlap"] - geometry["initial_response_scaffold_overlap"]
    deletions = []
    for order in SOURCE_ORDERS:
        positions = [i for i, r in enumerate(source_records) if r["order_seed"] != order]
        scaffold = f.pool([source_records[i]["historical_basis"].double() for i in positions])
        row = {"deleted_order": order}
        for checkpoint in (0, 2):
            response = f.pool([response_records[i]["states"][checkpoint]["prefixes"][16]["basis"] for i in positions])
            row[f"overlap_{checkpoint}"] = f.overlap(scaffold, response)
        row["change"] = row["overlap_2"] - row["overlap_0"]
        deletions.append(row)
    csv_save(ROOT / "source_order_deletions.csv", deletions)
    transfer, updates, recovery_checks = [], [], []
    evaluation = f.IndexedCifar(Path("data"), True, False, "cifar100")
    for path, _ in inventory("target"):
        a = f.load(path)
        target = f.load(ROOT / "targets/artifacts" / path.name)
        identity = {"order_seed": a["order_seed"], "seed": a["seed"]}
        row = {**identity, **{name: f.coverage(a["future_drift_centered"], u) for name, u in predictors.items()}}
        row["trained_minus_initial"] = row["trained_response"]-row["initial_response"]
        transfer.append(row)
        cache = ROOT / "update_fields" / path.name
        if cache.exists():
            fields = f.load(cache)
        else:
            values, norm = f.actual_update_fields(target, evaluation, torch.device("cuda"))
            fields = {"fields": values, "relative_step": norm}
            f.save(fields, cache)
        assert torch.equal(fields["fields"]["adamw_full"], fields["fields"]["adamw_parameters"])
        row = {**identity, "relative_step": fields["relative_step"]}
        controls = f.load(ROOT / "original_control_bases" / path.name)["basis"]
        for name, field in fields["fields"].items():
            for predictor, u in predictors.items():
                row[name+"_"+predictor] = f.coverage(field, u)
            row[name+"_isotropic"] = 8/192
            row[name+"_frozen_readout_matched_random"] = float(np.mean([f.coverage(field, u) for u in controls]))
            row[name+"_scaffold_minus_isotropic"] = row[name+"_scaffold"]-8/192
            row[name+"_scaffold_minus_random"] = row[name+"_scaffold"]-row[name+"_frozen_readout_matched_random"]
        updates.append(row)
        recovery_checks.append({**identity, **{k: v for k,v in target.items() if k.endswith("max_abs_error")}})
        print(path.stem, "response transfer / actual update analyzed", flush=True)
    csv_save(ROOT / "response_transfer.csv", transfer)
    csv_save(ROOT / "updates.csv", updates)
    f.save(predictors, ROOT / "predictors.pt")
    summary = {"source_count": 6, "target_count": 12, "rank": 8, "feature_dimension": 192,
               "geometry": geometry, "source_order_deletions": deletions,
               "transfer": stats(transfer, [k for k in transfer[0] if k not in ("order_seed", "seed")]),
               "updates": stats(updates, [k for k in updates[0] if k not in ("order_seed", "seed")]),
               "source_recovery": source_checks, "target_recovery": recovery_checks,
               "buffer_control": "DeiT has no running buffers; identical full/parameter fields are one measurement"}
    json_save(ROOT / "mechanism_summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["original", "targets", "analyze"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    ROOT.mkdir(exist_ok=True, parents=True)
    {"original": original_measurements, "targets": recover_targets, "analyze": analyze_recovered}[args.mode]()


if __name__ == "__main__":
    main()
