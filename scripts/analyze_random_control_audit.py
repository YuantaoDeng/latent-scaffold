from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from analyze_prospective_atlas import hierarchical_bootstrap, order_summary
from phase_c_adapter_direction import projected_random_basis
from temporal_drift_reuse import (
    basis_overlap, coverage, projection_spectrum, spectrum_matched_basis, top_basis,
)


def load_artifacts(roots):
    return [
        torch.load(path, map_location="cpu", weights_only=False)
        for root in roots for path in sorted((root / "artifacts").glob("*.pt"))
    ]


def unconditioned_spectrum_matched(empirical, discriminative, null, rank, generator):
    eigenvalues, _ = projection_spectrum(empirical, discriminative)
    eigenvalues[eigenvalues < 1e-10] = 0.0
    count = int((eigenvalues > 0).sum())
    disc = empirical.new_zeros((empirical.shape[0], rank))
    if count:
        disc[:, :count] = projected_random_basis(discriminative, count, generator)
    null_part = projected_random_basis(null, rank, generator)
    mixed = (
        disc * eigenvalues.sqrt().float()[None, :]
        + null_part * (1.0 - eigenvalues).sqrt().float()[None, :]
    )
    identity = torch.eye(rank, dtype=torch.double)
    orthogonality_error = float((mixed.T.double() @ mixed.double() - identity).abs().max())
    matched, _ = projection_spectrum(mixed, discriminative)
    spectrum_error = float((matched - eigenvalues).abs().max())
    return mixed, spectrum_error, orthogonality_error


def haar_basis(dimension, rank, generator):
    values = torch.randn(dimension, rank, generator=generator, dtype=torch.double)
    return torch.linalg.qr(values, mode="reduced").Q.float()


def summarize(values, orders, seed):
    means, positive = order_summary(values, orders)
    return {
        "mean": float(np.mean(values)),
        "hierarchical_ci": hierarchical_bootstrap(values, orders, seed),
        "order_means": means,
        "positive_orders": positive,
    }


def run(args):
    sources = load_artifacts(args.sources)
    targets = load_artifacts(args.targets)
    source_bases = torch.cat([
        top_basis(item["historical_drift_centered"], args.rank, centered=False)
        for item in sources
    ], dim=1)
    atlas = torch.linalg.svd(source_bases.double(), full_matrices=False).U[:, :args.rank].float()
    rows = []
    max_error = 0.0

    for item in targets:
        field = item["future_drift_centered"]
        atlas_coverage = coverage(field, atlas, centered=False)
        values = {name: [] for name in ("avoiding", "unconditioned", "haar")}
        overlaps = {name: [] for name in values}
        for draw in range(args.random_controls):
            base_seed = int(item["order_seed"]) * 1_000_003 + int(item["seed"]) * 10_007 + draw
            avoiding, spectrum_error, orthogonality_error = spectrum_matched_basis(
                atlas, item["discriminative_basis"], item["null_basis"], args.rank,
                torch.Generator().manual_seed(base_seed + 5_000_000),
            )
            unconditioned, spectrum_error_u, orthogonality_error_u = unconditioned_spectrum_matched(
                atlas, item["discriminative_basis"], item["null_basis"], args.rank,
                torch.Generator().manual_seed(base_seed + 6_000_000),
            )
            haar = haar_basis(
                atlas.shape[0], args.rank,
                torch.Generator().manual_seed(base_seed + 7_000_000),
            )
            controls = {"avoiding": avoiding, "unconditioned": unconditioned, "haar": haar}
            max_error = max(max_error, spectrum_error, orthogonality_error,
                            spectrum_error_u, orthogonality_error_u)
            for name, control in controls.items():
                values[name].append(coverage(field, control, centered=False))
                overlaps[name].append(basis_overlap(atlas, control))
        row = {
            "order_seed": int(item["order_seed"]), "seed": int(item["seed"]),
            "atlas_coverage": atlas_coverage,
        }
        for name in values:
            mean = float(np.mean(values[name]))
            row[f"{name}_coverage"] = mean
            row[f"atlas_minus_{name}"] = atlas_coverage - mean
            row[f"{name}_atlas_overlap"] = float(np.mean(overlaps[name]))
        rows.append(row)

    orders = [row["order_seed"] for row in rows]
    summary = {
        "source_blocks": len(sources), "target_blocks": len(targets),
        "rank": args.rank, "random_controls": args.random_controls,
        "atlas_coverage": summarize([row["atlas_coverage"] for row in rows], orders, 601),
        "controls": {}, "max_matching_error": max_error,
    }
    for index, name in enumerate(("avoiding", "unconditioned", "haar")):
        summary["controls"][name] = {
            "coverage": summarize([row[f"{name}_coverage"] for row in rows], orders, 610 + index),
            "atlas_minus_control": summarize(
                [row[f"atlas_minus_{name}"] for row in rows], orders, 620 + index,
            ),
            "atlas_overlap": summarize(
                [row[f"{name}_atlas_overlap"] for row in rows], orders, 630 + index,
            ),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "targets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Audit drift-atlas random control families")
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--random-controls", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
