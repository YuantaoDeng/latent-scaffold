from __future__ import annotations
from package_paths import read_config_text

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from phase_c_adapter_direction import numerical_column_space
from temporal_drift_reuse import basis_overlap, coverage, spectrum_matched_basis


def load_artifacts(roots):
    artifacts = []
    for root in roots:
        artifacts.extend(torch.load(path, map_location="cpu", weights_only=False)
                         for path in sorted((root / "artifacts").glob("*.pt")))
    return artifacts


def bootstrap(values, seed, draws=20000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.choice(values, (draws, len(values)), replace=True).mean(1)
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def hierarchical_bootstrap(values, orders, seed, draws=20000):
    """Resample task orders, then optimization seeds within each sampled order."""
    values = np.asarray(values, dtype=np.float64)
    orders = np.asarray(orders)
    unique_orders = np.unique(orders)
    grouped = {order: values[orders == order] for order in unique_orders}
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_orders = generator.choice(unique_orders, len(unique_orders), replace=True)
        sampled = [
            generator.choice(grouped[order], len(grouped[order]), replace=True)
            for order in sampled_orders
        ]
        means[draw] = np.concatenate(sampled).mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def order_summary(values, orders):
    unique_orders = sorted(set(orders))
    means = {
        str(order): float(np.mean([value for value, item_order in zip(values, orders)
                                   if item_order == order]))
        for order in unique_orders
    }
    return means, int(sum(value > 0 for value in means.values()))


def matched(atlas, field, discriminative, null, controls, seed):
    empirical = coverage(field, atlas, centered=False)
    random_values, errors = [], []
    for draw in range(controls):
        generator = torch.Generator().manual_seed(seed + draw)
        control, spectrum_error, orthogonality_error = spectrum_matched_basis(
            atlas, discriminative, null, atlas.shape[1], generator,
        )
        random_values.append(coverage(field, control, centered=False))
        errors.append(max(spectrum_error, orthogonality_error))
    return {
        "coverage": empirical,
        "matched_mean": float(np.mean(random_values)),
        "effect": float(empirical - np.mean(random_values)),
        "percentile": float(np.mean(np.asarray(random_values) < empirical)),
        "max_error": max(errors),
    }


def projector_atlas(bases, rank):
    """Top-r eigenspace of equally weighted, equal-rank source projectors."""
    if len({basis.shape[1] for basis in bases}) != 1:
        raise ValueError("source projectors must have the same rank")
    concatenated = torch.cat(bases, dim=1).double()
    return torch.linalg.svd(concatenated, full_matrices=False).U[:, :rank]


def matched_static_summary(rows, source_rank, output_rank):
    """Report construction-matched contrasts without changing frozen primary gates."""
    result = {"matched_projector_static_definition": {
        "source_rank": source_rank, "output_rank": output_rank,
        "construction": "top-r eigenspace of mean source top-k static PCA projectors; same k, r and sources as drift",
        "role": "formal construction-matched control; pooled covariance retained separately",
        "analysis_status": "supplementary analysis; does not retrospectively change frozen primary tests",
        "legacy_alias": "global_static_atlas",
    }}
    orders = [row["order_seed"] for row in rows]
    for prefix, drift_key, seed in (("", "atlas_coverage", 131),
                                    ("residual64_", "residual64_coverage", 132)):
        key = prefix + "matched_projector_static_coverage"
        for row in rows:
            row[key] = row[prefix + "global_static_atlas_coverage"]
            row[prefix + "drift_minus_matched_projector_static"] = row[drift_key] - row[key]
        effects = [row[drift_key] - row[key] for row in rows]
        means, positive = order_summary(effects, orders)
        contrast = prefix + "drift_minus_matched_projector_static"
        result.update({
            key + "_mean": float(np.mean([row[key] for row in rows])),
            contrast + "_mean": float(np.mean(effects)),
            contrast + "_ci": bootstrap(effects, seed - 100),
            contrast + "_hierarchical_ci": hierarchical_bootstrap(effects, orders, seed),
            contrast + "_order_means": means,
            contrast + "_positive_orders": positive,
            contrast + "_positive_targets": int(np.sum(np.asarray(effects) > 0)),
        })
    return result


def run(args):
    sources = load_artifacts(args.sources)
    targets = load_artifacts(args.targets)
    source_ranks = {item["historical_basis"].shape[1] for item in sources}
    if len(source_ranks) != 1:
        raise ValueError("historical source ranks must agree")
    source_rank = source_ranks.pop()
    for item in sources:
        if item["static_pca_basis"].shape[1] != source_rank:
            raise ValueError("static and drift source ranks must match")
    atlas = projector_atlas([item["historical_basis"] for item in sources], args.rank).float()
    static_atlas = projector_atlas([item["static_pca_basis"] for item in sources], args.rank).float()
    weighted_static_atlas = None
    if all("static_covariance" in item for item in sources):
        pooled_covariance = sum(item["static_covariance"].double() for item in sources)
        weighted_static_atlas = torch.linalg.eigh(pooled_covariance).eigenvectors[:, -args.rank:].float()
    if args.refresh_matched_static:
        return refresh_matched_static(args, targets, atlas, static_atlas, source_rank)
    rows = []
    for index, target in enumerate(targets):
        field = target["future_drift_centered"]
        main = matched(
            atlas, field, target["discriminative_basis"], target["null_basis"],
            args.random_controls, 1_000_003 * (index + 1) + 1_100_000,
        )
        complement = target["static_pca_full"][:, args.residual_pca_rank:]
        atlas_residual = numerical_column_space(complement.T @ atlas)[:, :args.rank]
        field_residual = field @ complement
        disc_residual = numerical_column_space(complement.T @ target["discriminative_basis"])
        _, _, null_vh = torch.linalg.svd(disc_residual.T.double(), full_matrices=True)
        null_residual = null_vh[disc_residual.shape[1]:].T.float()
        residual = matched(
            atlas_residual, field_residual, disc_residual, null_residual,
            args.random_controls, 1_000_003 * (index + 1) + 1_300_000,
        )
        static_atlas_residual = numerical_column_space(complement.T @ static_atlas)[:, :args.rank]
        weighted_static_residual = (
            numerical_column_space(complement.T @ weighted_static_atlas)[:, :args.rank]
            if weighted_static_atlas is not None else None
        )
        rows.append({
            "order_seed": target["order_seed"],
            "seed": target["seed"],
            **{f"atlas_{key}": value for key, value in main.items()},
            "target_static_pca_coverage": coverage(field, target["static_pca_basis"], centered=False),
            "global_static_atlas_coverage": coverage(field, static_atlas, centered=False),
            **({"weighted_static_atlas_coverage": coverage(
                field, weighted_static_atlas, centered=False,
            )} if weighted_static_atlas is not None else {}),
            "target_history_coverage": coverage(field, target["historical_basis"], centered=False),
            "atlas_target_history_overlap": basis_overlap(atlas, target["historical_basis"]),
            **{f"residual64_{key}": value for key, value in residual.items()},
            "residual64_global_static_atlas_coverage": coverage(
                field_residual, static_atlas_residual, centered=False,
            ),
            **({"residual64_weighted_static_atlas_coverage": coverage(
                field_residual, weighted_static_residual, centered=False,
            )} if weighted_static_residual is not None else {}),
            "residual64_atlas_energy_survival": float((complement.T @ atlas).square().sum() / args.rank),
            "residual64_future_energy_survival": float(field_residual.square().sum() / field.square().sum()),
        })

    effects = [row["atlas_effect"] for row in rows]
    residual_effects = [row["residual64_effect"] for row in rows]
    versus_pca = [row["atlas_coverage"] - row["target_static_pca_coverage"] for row in rows]
    versus_history = [row["atlas_coverage"] - row["target_history_coverage"] for row in rows]
    versus_global_static = [
        row["atlas_coverage"] - row["global_static_atlas_coverage"] for row in rows
    ]
    residual_versus_global_static = [
        row["residual64_coverage"] - row["residual64_global_static_atlas_coverage"] for row in rows
    ]
    versus_weighted_static = ([
        row["atlas_coverage"] - row["weighted_static_atlas_coverage"] for row in rows
    ] if weighted_static_atlas is not None else None)
    residual_versus_weighted_static = ([
        row["residual64_coverage"] - row["residual64_weighted_static_atlas_coverage"] for row in rows
    ] if weighted_static_atlas is not None else None)
    target_orders = [row["order_seed"] for row in rows]
    atlas_order_means, atlas_positive_orders = order_summary(effects, target_orders)
    summary = {
        "development_blocks": len(sources),
        "heldout_targets": len(targets),
        "atlas_coverage_mean": float(np.mean([row["atlas_coverage"] for row in rows])),
        "matched_coverage_mean": float(np.mean([row["atlas_matched_mean"] for row in rows])),
        "atlas_effect_mean": float(np.mean(effects)),
        "atlas_effect_ci": bootstrap(effects, 1),
        "atlas_effect_hierarchical_ci": hierarchical_bootstrap(effects, target_orders, 101),
        "atlas_positive_targets": int(np.sum(np.asarray(effects) > 0)),
        "atlas_order_effect_means": atlas_order_means,
        "atlas_positive_orders": atlas_positive_orders,
        "atlas_empirical_percentile_mean": float(np.mean([row["atlas_percentile"] for row in rows])),
        "atlas_minus_target_pca_mean": float(np.mean(versus_pca)),
        "atlas_minus_target_pca_ci": bootstrap(versus_pca, 2),
        "atlas_minus_target_history_mean": float(np.mean(versus_history)),
        "atlas_minus_target_history_ci": bootstrap(versus_history, 3),
        "global_static_atlas_coverage_mean": float(np.mean([
            row["global_static_atlas_coverage"] for row in rows
        ])),
        "drift_atlas_minus_global_static_mean": float(np.mean(versus_global_static)),
        "drift_atlas_minus_global_static_ci": bootstrap(versus_global_static, 31),
        "atlas_target_history_overlap_mean": float(np.mean([
            row["atlas_target_history_overlap"] for row in rows
        ])),
        "residual64_effect_mean": float(np.mean(residual_effects)),
        "residual64_effect_ci": bootstrap(residual_effects, 4),
        "residual64_positive_targets": int(np.sum(np.asarray(residual_effects) > 0)),
        "residual64_global_static_atlas_coverage_mean": float(np.mean([
            row["residual64_global_static_atlas_coverage"] for row in rows
        ])),
        "residual64_drift_minus_global_static_mean": float(np.mean(residual_versus_global_static)),
        "residual64_drift_minus_global_static_ci": bootstrap(residual_versus_global_static, 32),
        "residual64_atlas_energy_survival_mean": float(np.mean([
            row["residual64_atlas_energy_survival"] for row in rows
        ])),
        "residual64_future_energy_survival_mean": float(np.mean([
            row["residual64_future_energy_survival"] for row in rows
        ])),
        "max_matching_error": float(max(
            max(row["atlas_max_error"], row["residual64_max_error"]) for row in rows
        )),
    }
    if weighted_static_atlas is not None:
        weighted_order_means, weighted_positive_orders = order_summary(
            versus_weighted_static, target_orders,
        )
        residual_weighted_order_means, residual_weighted_positive_orders = order_summary(
            residual_versus_weighted_static, target_orders,
        )
        summary.update({
            "weighted_static_atlas_coverage_mean": float(np.mean([
                row["weighted_static_atlas_coverage"] for row in rows
            ])),
            "drift_atlas_minus_weighted_static_mean": float(np.mean(versus_weighted_static)),
            "drift_atlas_minus_weighted_static_ci": bootstrap(versus_weighted_static, 41),
            "drift_atlas_minus_weighted_static_hierarchical_ci": hierarchical_bootstrap(
                versus_weighted_static, target_orders, 141,
            ),
            "drift_atlas_minus_weighted_static_order_means": weighted_order_means,
            "drift_atlas_minus_weighted_static_positive_orders": weighted_positive_orders,
            "residual64_weighted_static_atlas_coverage_mean": float(np.mean([
                row["residual64_weighted_static_atlas_coverage"] for row in rows
            ])),
            "residual64_drift_minus_weighted_static_mean": float(np.mean(
                residual_versus_weighted_static
            )),
            "residual64_drift_minus_weighted_static_ci": bootstrap(
                residual_versus_weighted_static, 42,
            ),
            "residual64_drift_minus_weighted_static_hierarchical_ci": hierarchical_bootstrap(
                residual_versus_weighted_static, target_orders, 142,
            ),
            "residual64_drift_minus_weighted_static_order_means": residual_weighted_order_means,
            "residual64_drift_minus_weighted_static_positive_orders": residual_weighted_positive_orders,
        })
    required_positive_targets = (5 * len(targets) + 5) // 6
    summary["primary_gate_passed"] = bool(
        len(targets) >= 6
        and summary["atlas_positive_targets"] >= required_positive_targets
        and summary["atlas_effect_ci"][0] > 0
    )
    summary["residual64_gate_passed"] = bool(
        len(targets) >= 6
        and summary["residual64_positive_targets"] >= required_positive_targets
        and summary["residual64_effect_ci"][0] > 0
    )
    if weighted_static_atlas is not None:
        summary["hierarchical_primary_gate_passed"] = bool(
            len(set(target_orders)) >= 6
            and summary["drift_atlas_minus_weighted_static_positive_orders"] >= 5
            and summary["drift_atlas_minus_weighted_static_hierarchical_ci"][0] > 0
        )
    summary.update(matched_static_summary(rows, source_rank, args.rank))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"atlas": atlas, "static_atlas": static_atlas,
                "matched_projector_static_atlas": static_atlas, "source_rank": source_rank,
                "weighted_static_atlas": weighted_static_atlas, "source_blocks": len(sources)},
               args.output / "atlas.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import csv
    with (args.output / "targets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


def refresh_matched_static(args, targets, atlas, static_atlas, source_rank):
    """Recompute the static control on saved targets, retaining original random tests."""
    with (args.output / "targets.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{key: int(value) if key in ("order_seed", "seed") else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]
    by_id = {(item["order_seed"], item["seed"]): item for item in targets}
    if set(by_id) != {(row["order_seed"], row["seed"]) for row in rows}:
        raise ValueError("refresh requires the original target set")
    for row in rows:
        target = by_id[row["order_seed"], row["seed"]]
        field = target["future_drift_centered"]
        np.testing.assert_allclose(coverage(field, atlas, centered=False), row["atlas_coverage"], atol=1e-7, rtol=1e-6)
        complement = target["static_pca_full"][:, args.residual_pca_rank:]
        residual_basis = numerical_column_space(complement.T @ static_atlas)[:, :args.rank]
        for key, value in (
            ("global_static_atlas_coverage", coverage(field, static_atlas, centered=False)),
            ("residual64_global_static_atlas_coverage", coverage(field @ complement, residual_basis, centered=False)),
        ):
            np.testing.assert_allclose(value, row[key], atol=1e-7, rtol=1e-6)
    summary = json.loads(read_config_text(args.output / "summary.json", encoding="utf-8"))
    summary.update(matched_static_summary(rows, source_rank, args.rank))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "targets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved = torch.load(args.output / "atlas.pt", map_location="cpu", weights_only=False)
    saved.update({"matched_projector_static_atlas": static_atlas, "source_rank": source_rank})
    torch.save(saved, args.output / "atlas.pt")
    print(json.dumps({key: value for key, value in summary.items() if "matched_projector" in key}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--residual-pca-rank", type=int, default=64)
    parser.add_argument("--random-controls", type=int, default=128)
    parser.add_argument("--refresh-matched-static", action="store_true",
                        help="verify and formally report the matched static control in an existing analysis")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
