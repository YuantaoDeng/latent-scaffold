from __future__ import annotations
from package_paths import read_config_text

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from analyze_prospective_atlas import bootstrap, hierarchical_bootstrap, order_summary, projector_atlas
from temporal_drift_reuse import coverage, top_basis


def load_artifacts(roots):
    return [
        torch.load(path, map_location="cpu", weights_only=False)
        for root in roots
        for path in sorted((root / "artifacts").glob("*.pt"))
    ]


def historical_basis(item, rank):
    if "historical_drift_centered" in item:
        return top_basis(item["historical_drift_centered"], rank, centered=False)
    basis = item["historical_basis"]
    if basis.shape[1] < rank:
        raise ValueError(f"artifact only stores rank {basis.shape[1]}, requested {rank}")
    return basis[:, :rank]


def predictors(sources, rank):
    atlas = projector_atlas([historical_basis(item, rank) for item in sources], rank).float()
    matched_static = projector_atlas([
        torch.linalg.eigh(item["static_covariance"].double()).eigenvectors[:, -rank:]
        for item in sources
    ], rank).float()
    covariance = sum(item["static_covariance"].double() for item in sources)
    static = torch.linalg.eigh(covariance).eigenvectors[:, -rank:].float()
    return atlas, static, matched_static


def evaluate(atlas, static, matched_static, targets):
    return [{
        "order_seed": item["order_seed"],
        "seed": item["seed"],
        "atlas_coverage": coverage(item["future_drift_centered"], atlas, centered=False),
        "weighted_static_coverage": coverage(item["future_drift_centered"], static, centered=False),
        "matched_projector_static_coverage": coverage(item["future_drift_centered"], matched_static, centered=False),
    } for item in targets]


def summarize(rows, seed):
    effects = [row["atlas_coverage"] - row["weighted_static_coverage"] for row in rows]
    matched_effects = [row["atlas_coverage"] - row["matched_projector_static_coverage"] for row in rows]
    orders = [row["order_seed"] for row in rows]
    means, positive = order_summary(effects, orders)
    matched_means, matched_positive = order_summary(matched_effects, orders)
    return {
        "blocks": len(rows),
        "effect_mean": float(np.mean(effects)),
        "effect_ci": bootstrap(effects, seed),
        "effect_hierarchical_ci": hierarchical_bootstrap(effects, orders, seed + 10_000),
        "order_effect_means": means,
        "positive_orders": positive,
        "matched_projector_static_effect_mean": float(np.mean(matched_effects)),
        "matched_projector_static_effect_ci": bootstrap(matched_effects, seed),
        "matched_projector_static_effect_hierarchical_ci": hierarchical_bootstrap(matched_effects, orders, seed + 10_000),
        "matched_projector_static_order_effect_means": matched_means,
        "matched_projector_static_positive_orders": matched_positive,
        "matched_projector_static_coverage_mean": float(np.mean([
            row["matched_projector_static_coverage"] for row in rows
        ])),
        "atlas_coverage_mean": float(np.mean([row["atlas_coverage"] for row in rows])),
        "weighted_static_coverage_mean": float(np.mean([
            row["weighted_static_coverage"] for row in rows
        ])),
    }


def structure_inputs(roots, role):
    """Read actual artifact schemas; retain only geometry needed for this analysis."""
    items, inventory = [], []
    for root in roots:
        manifest = json.loads(read_config_text(root / "manifest.json", encoding="utf-8"))
        for path in sorted((root / "artifacts").glob("*.pt")):
            raw = torch.load(path, map_location="cpu", weights_only=False)
            schema = {}
            for key, value in raw.items():
                if torch.is_tensor(value):
                    schema[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                elif isinstance(value, (list, tuple)):
                    schema[key] = [
                        {"shape": list(item.shape), "dtype": str(item.dtype)}
                        if torch.is_tensor(item) else item for item in value
                    ]
                elif isinstance(value, dict):
                    schema[key] = {"keys": list(value)}
                    if key == "layers":
                        schema[key] = {
                            layer: {name: list(field.shape) for name, field in fields.items()}
                            for layer, fields in value.items()
                        }
                else:
                    schema[key] = value
            inventory.append({
                "role": role, "path": path.as_posix(),
                "manifest_path": (root / "manifest.json").as_posix(),
                "order_seed": raw["order_seed"], "seed": raw["seed"],
                "pre_future_model_saved": "pre_future_state_dict" in raw,
                "schema": schema,
            })
            names = ("historical_drift_centered", "historical_basis", "static_covariance") if role == "source" else ("future_drift_centered",)
            items.append({
                "order_seed": raw["order_seed"], "seed": raw["seed"],
                "path": path.as_posix(), **{name: raw[name].double() for name in names},
            })
            del raw
    return items, inventory


def structure_basis(sources, source_rank):
    concatenated = torch.cat([item["fixed_basis"] for item in sources], dim=1)
    basis, singular, _ = torch.linalg.svd(concatenated, full_matrices=False)
    spectrum = singular.square() / len(sources)
    covariance = sum(item["static_covariance"] for item in sources)
    static = torch.linalg.eigh(covariance).eigenvectors.flip(1)
    static_bases = torch.cat([item["fixed_static_basis"][:, :source_rank] for item in sources], dim=1)
    matched_static = torch.linalg.svd(static_bases, full_matrices=False).U
    return basis, spectrum, static, matched_static


def plain_summary(rows):
    order_effects = {
        str(order): float(np.mean([row["effect"] for row in rows if row["order_seed"] == order]))
        for order in sorted({row["order_seed"] for row in rows})
    }
    return {
        "atlas_coverage_mean": float(np.mean([row["atlas_coverage"] for row in rows])),
        "static_coverage_mean": float(np.mean([row["static_coverage"] for row in rows])),
        "matched_projector_static_coverage_mean": float(np.mean([row["matched_projector_static_coverage"] for row in rows])),
        "matched_projector_static_effect_mean": float(np.mean([row["matched_projector_static_effect"] for row in rows])),
        "matched_projector_static_effect_min_target": min(row["matched_projector_static_effect"] for row in rows),
        "matched_projector_static_order_effect_means": {
            str(order): float(np.mean([row["matched_projector_static_effect"] for row in rows if row["order_seed"] == order]))
            for order in sorted({row["order_seed"] for row in rows})
        },
        "oracle_coverage_mean": float(np.mean([row["oracle_coverage"] for row in rows])),
        "atlas_to_oracle_ratio_mean": float(np.mean([row["atlas_to_oracle_ratio"] for row in rows])),
        "effect_mean": float(np.mean([row["effect"] for row in rows])),
        "effect_min_target": min(row["effect"] for row in rows),
        "order_effect_means": order_effects,
    }


def structure_run(args):
    """Nested dimensions and deterministic source deletions, without new training."""
    torch.set_num_threads(args.torch_threads)
    sources, source_inventory = structure_inputs(args.sources, "source")
    targets, target_inventory = structure_inputs(args.targets, "target")
    source_groups = sorted({item["order_seed"] for item in sources})
    for item in sources:
        item["fixed_basis"] = torch.linalg.svd(
            item["historical_drift_centered"], full_matrices=False,
        ).Vh[:args.source_rank].T
        item["fixed_static_basis"] = torch.linalg.eigh(item["static_covariance"]).eigenvectors[:, -args.source_rank:]
    full_basis, full_spectrum, full_static, full_matched_static = structure_basis(sources, args.source_rank)
    max_rank = max(args.ranks)
    if max_rank > full_basis.shape[1] or args.primary_rank > max_rank:
        raise ValueError("requested dimensions exceed the fixed-source span or rank sweep")
    for item in targets:
        field = item["future_drift_centered"]
        energy = field.square().sum()
        singular = torch.linalg.svdvals(field)
        item["energy"] = energy
        item["oracle"] = singular.square().cumsum(0)[:max_rank] / energy
        item["participation_rank"] = float(energy.square() / singular.pow(4).sum())
        item["full_static_curve"] = (field @ full_static[:, :max_rank]).square().sum(0).cumsum(0) / energy
        item["full_matched_static_curve"] = (field @ full_matched_static[:, :max_rank]).square().sum(0).cumsum(0) / energy
    conditions = [("full", sources)]
    conditions += [
        (f"drop_trajectory_{item['order_seed']}_{item['seed']}", [other for other in sources if other is not item])
        for item in sources
    ]
    conditions += [
        (f"drop_order_{order}", [item for item in sources if item["order_seed"] != order])
        for order in source_groups
    ]
    rows, spectrum_rows, deletion_rows, summaries = [], [], [], {}
    for condition, selected in conditions:
        basis, spectrum, static, matched_static = structure_basis(selected, args.source_rank)
        if max_rank > basis.shape[1] or float(spectrum[max_rank - 1]) <= 1e-12:
            raise ValueError("rank sweep extends beyond the source-supported projector span")
        for index, value in enumerate(spectrum):
            spectrum_rows.append({"condition": condition, "dimension": index + 1, "eigenvalue": float(value)})
        condition_rows = []
        for target in targets:
            field, energy = target["future_drift_centered"], target["energy"]
            atlas_curve = (field @ basis[:, :max_rank]).square().sum(0).cumsum(0) / energy
            static_curve = (field @ static[:, :max_rank]).square().sum(0).cumsum(0) / energy
            matched_static_curve = (field @ matched_static[:, :max_rank]).square().sum(0).cumsum(0) / energy
            for rank in range(1, max_rank + 1):
                atlas_value, static_value = float(atlas_curve[rank - 1]), float(static_curve[rank - 1])
                matched_static_value = float(matched_static_curve[rank - 1])
                oracle = float(target["oracle"][rank - 1])
                row = {
                    "condition": condition, "rank": rank,
                    "order_seed": target["order_seed"], "seed": target["seed"],
                    "source_count": len(selected), "atlas_coverage": atlas_value,
                    "static_coverage": static_value, "effect": atlas_value - static_value,
                    "matched_projector_static_coverage": matched_static_value,
                    "matched_projector_static_effect": atlas_value - matched_static_value,
                    "full_matched_projector_static_coverage": float(target["full_matched_static_curve"][rank - 1]),
                    "effect_vs_full_matched_projector_static": atlas_value - float(target["full_matched_static_curve"][rank - 1]),
                    "full_static_coverage": float(target["full_static_curve"][rank - 1]),
                    "effect_vs_full_static": atlas_value - float(target["full_static_curve"][rank - 1]),
                    "oracle_coverage": oracle, "atlas_to_oracle_ratio": atlas_value / oracle,
                }
                rows.append(row)
                condition_rows.append(row)
        summaries[condition] = {
            str(rank): plain_summary([row for row in condition_rows if row["rank"] == rank])
            for rank in range(1, max_rank + 1)
        }
        for rank in range(1, max_rank + 1):
            cosines = torch.linalg.svdvals(full_basis[:, :rank].T @ basis[:, :rank]).clamp(0, 1)
            angles = torch.rad2deg(torch.acos(cosines))
            deletion_rows.append({
                "condition": condition, "rank": rank, "source_count": len(selected),
                "projector_overlap": float(cosines.square().mean()),
                "mean_principal_angle_deg": float(angles.mean()),
                "max_principal_angle_deg": float(angles.max()),
                "effect_mean": summaries[condition][str(rank)]["effect_mean"],
                "matched_projector_static_effect_mean": summaries[condition][str(rank)]["matched_projector_static_effect_mean"],
                "atlas_coverage_mean": summaries[condition][str(rank)]["atlas_coverage_mean"],
                "effect_vs_full_static_mean": float(np.mean([
                    row["effect_vs_full_static"] for row in condition_rows if row["rank"] == rank
                ])),
                "effect_vs_full_matched_projector_static_mean": float(np.mean([
                    row["effect_vs_full_matched_projector_static"] for row in condition_rows if row["rank"] == rank
                ])),
            })
        print(f"{condition}: complete", flush=True)
    summary = {
        "source_rank_fixed": args.source_rank, "primary_rank": args.primary_rank,
        "max_rank": max_rank, "source_count": len(sources), "target_count": len(targets),
        "sources": [{"order_seed": item["order_seed"], "seed": item["seed"], "path": item["path"]} for item in sources],
        "targets": [{"order_seed": item["order_seed"], "seed": item["seed"], "participation_rank": item["participation_rank"]} for item in targets],
        "definitions": {
            "geometry": "D = delta_Z - row_mean(delta_Z); coverage = ||D U||_F^2 / ||D||_F^2",
            "source_basis": "top source_rank right singular vectors of stored centered early drift, fixed for every output rank",
            "projector_spectrum": "eigenvalues of mean source projectors, not intrinsic dimensions of target drift",
            "deletion_static": "pooled covariance recomputed on retained sources; full-source static contrast also saved",
            "matched_projector_static": "top source_rank eigenvectors per source covariance, equal source-projector aggregation, output rank r; source_rank fixed across r",
            "deletion_matched_projector_static": "projector aggregate recomputed on retained sources with the same fixed source_rank; full-source contrast also saved",
            "legacy_effect_fields": "effect and static fields retain pooled-covariance comparisons; matched_projector_static fields report the construction-matched comparison",
            "comparator_status": "matched-projector static added as a formal comparator in a retrospective analysis of existing targets; not a new preregistered confirmation",
            "oracle": "post-hoc target-specific top-r SVD coverage; descriptive only, never used for basis or rank selection",
            "oracle_ratio": "per-target scaffold/oracle coverage, averaged across targets; not fraction of all shared drift",
            "functional_legacy": "raw delta_Z projection, uncentered class logits, zero-change-relative squared-error reconstruction",
            "aggregation": "equal target means; target-order means also reported; no target-based rank selection",
        },
        "conditions": summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("inventory.json", source_inventory + target_inventory)):
        (args.output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    for name, values in (("targets.csv", rows), ("spectrum.csv", spectrum_rows), ("source_deletions.csv", deletion_rows)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    structure_plot(summary, full_spectrum, deletion_rows, args.output)
    print(json.dumps(summary["conditions"]["full"][str(args.primary_rank)], indent=2), flush=True)


def structure_plot(summary, spectrum, deletions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = range(1, summary["max_rank"] + 1)
    full = summary["conditions"]["full"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout="constrained")
        axes[0, 0].plot(ranks, spectrum[:summary["max_rank"]], color="#167D8D")
        axes[0, 0].set(xlabel="Source-projector direction", ylabel="Mean-projector eigenvalue", title="Shared source spectrum")
        for key, label, color in (("atlas_coverage_mean", "Scaffold", "#167D8D"), ("static_coverage_mean", "Pooled-covariance static", "#B26948"), ("matched_projector_static_coverage_mean", "Matched-projector static", "#8A68A6"), ("oracle_coverage_mean", "Target optimum (post-hoc)", "#687484")):
            axes[0, 1].plot(ranks, [100 * full[str(rank)][key] for rank in ranks], label=label, color=color)
        axes[0, 1].set(xlabel="Output dimension (source rank fixed)", ylabel="Future-drift coverage (%)", title="Shared and target-specific coverage")
        axes[0, 1].legend(frameon=False)
        for condition, values in summary["conditions"].items():
            if condition.startswith("drop_order"):
                axes[1, 0].plot(ranks, [100 * values[str(rank)]["matched_projector_static_effect_mean"] for rank in ranks], color="#8FA8B1", alpha=.7, linewidth=1)
        axes[1, 0].plot(ranks, [100 * full[str(rank)]["matched_projector_static_effect_mean"] for rank in ranks], color="#167D8D", label="All sources: matched projector")
        axes[1, 0].plot(ranks, [100 * full[str(rank)]["effect_mean"] for rank in ranks], color="#B26948", linestyle="--", label="All sources: pooled covariance")
        axes[1, 0].plot([], [], color="#8FA8B1", label="Order deletion: matched projector")
        axes[1, 0].axhline(0, color="#687484", linewidth=.8)
        axes[1, 0].set(xlabel="Output dimension", ylabel="Scaffold minus comparator (pp)", title="Prediction after removing source orders")
        axes[1, 0].legend(frameon=False)
        selected = [row for row in deletions if row["rank"] == summary["primary_rank"] and row["condition"] != "full"]
        for prefix, label, marker, color in (("drop_trajectory", "Remove one trajectory", "o", "#167D8D"), ("drop_order", "Remove one order", "s", "#B26948")):
            subset = [row for row in selected if row["condition"].startswith(prefix)]
            axes[1, 1].scatter([row["projector_overlap"] for row in subset], [100 * row["matched_projector_static_effect_mean"] for row in subset], label=label, marker=marker, color=color)
        axes[1, 1].axhline(0, color="#687484", linewidth=.8)
        axes[1, 1].set(xlabel="Subspace overlap with full-source scaffold", ylabel="Scaffold minus matched-projector static (pp)", title=f"Rank-{summary['primary_rank']} source-deletion effects", xlim=(0, 1.02))
        axes[1, 1].legend(frameon=False, fontsize=9)
        fig.savefig(output / "structure.png", dpi=180)
        fig.savefig(output / "structure.svg")
        plt.close(fig)


def run(args):
    if args.structure_analysis:
        return structure_run(args)
    sources = load_artifacts(args.sources)
    targets = load_artifacts(args.targets)
    rank_rows, source_rows = [], []
    summary = {"source_blocks": len(sources), "target_blocks": len(targets), "rank": {}, "source_count": {}}
    summary["definitions"] = {
        "rank_sweep": "joint-budget sweep: source truncation k equals output rank r for both drift and matched-projector static",
        "source_count": "both projector methods use source truncation k and output rank r equal to source_scaling_rank, on identical source subsets",
        "matched_projector_static": "top-k covariance eigenvectors per source, equal source-projector aggregation, top-r aggregate eigenvectors",
        "legacy_effect_fields": "effect fields retain pooled-covariance contrasts; matched_projector_static fields report construction-matched contrasts",
        "comparator_status": "matched-projector static added retrospectively on existing targets; not a new preregistered confirmation",
    }

    for index, rank in enumerate(args.ranks):
        atlas, static, matched_static = predictors(sources, rank)
        rows = evaluate(atlas, static, matched_static, targets)
        for row in rows:
            rank_rows.append({"rank": rank, **row,
                              "effect": row["atlas_coverage"] - row["weighted_static_coverage"],
                              "matched_projector_static_effect": row["atlas_coverage"] - row["matched_projector_static_coverage"]})
        summary["rank"][str(rank)] = summarize(rows, 100 + index)

    all_indices = range(len(sources))
    for index, count in enumerate(args.source_counts):
        if count > len(sources):
            continue
        subsets = list(itertools.combinations(all_indices, count))
        if len(subsets) > args.max_subsets:
            positions = np.linspace(0, len(subsets) - 1, args.max_subsets, dtype=int)
            subsets = [subsets[position] for position in positions]
        per_target = {}
        for subset_index, subset in enumerate(subsets):
            selected = [sources[item] for item in subset]
            atlas, static, matched_static = predictors(selected, args.source_scaling_rank)
            for row in evaluate(atlas, static, matched_static, targets):
                effect = row["atlas_coverage"] - row["weighted_static_coverage"]
                key = (row["order_seed"], row["seed"])
                per_target.setdefault(key, []).append((row["atlas_coverage"],
                                                       row["weighted_static_coverage"], row["matched_projector_static_coverage"]))
                source_rows.append({
                    "source_count": count,
                    "subset": subset_index,
                    "source_indices": ";".join(map(str, subset)),
                    **row,
                    "effect": effect,
                    "matched_projector_static_effect": row["atlas_coverage"] - row["matched_projector_static_coverage"],
                })
        collapsed = [{
            "order_seed": key[0], "seed": key[1],
            "atlas_coverage": float(np.mean([value[0] for value in values])),
            "weighted_static_coverage": float(np.mean([value[1] for value in values])),
            "matched_projector_static_coverage": float(np.mean([value[2] for value in values])),
        } for key, values in per_target.items()]
        result = summarize(collapsed, 200 + index)
        result["source_subsets"] = len(subsets)
        summary["source_count"][str(count)] = result

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, rows in (("rank.csv", rank_rows), ("source_count.csv", source_rows)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Prespecified drift-atlas scaling analyses")
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--source-counts", nargs="+", type=int, default=[1, 2, 3, 4, 6])
    parser.add_argument("--source-scaling-rank", type=int, default=8)
    parser.add_argument("--max-subsets", type=int, default=64)
    parser.add_argument("--structure-analysis", action="store_true")
    parser.add_argument("--source-rank", type=int, default=8)
    parser.add_argument("--primary-rank", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
