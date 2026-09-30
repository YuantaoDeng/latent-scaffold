from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

import numpy as np
import timm
import torch
import torch.nn.functional as F

from drift_experiment import (
    IndexedCifar,
    accuracy,
    balanced_memory,
    features,
    geometry,
    ids_by_class,
    infer,
    loader,
    masked_ce,
    model,
    parameter_change,
    parameter_snapshot,
    seed_all,
    train_task,
)


PRIMARY_CONDITIONS = ("jacobian_null", "empirical_er", "matched_random")
DIAGNOSTIC_CONDITIONS = ("harmful", "beneficial")
ALL_CONDITIONS = PRIMARY_CONDITIONS + DIAGNOSTIC_CONDITIONS


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_dict_cpu(net):
    return {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}


def state_sha256(state: dict[str, torch.Tensor]):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        tensor = state[name].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def functional_bases(weights: torch.Tensor, relative_threshold: float):
    """SVD of C W: the centered-logit Jacobian with respect to the final embedding."""
    centered = weights.double() - weights.double().mean(0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=True)
    cutoff = float(singular.max()) * relative_threshold if singular.numel() else 0.0
    rank = int((singular > cutoff).sum())
    discriminative = vh[:rank].T.contiguous().float()
    null = vh[rank:].T.contiguous().float()
    return discriminative, null, singular.float(), cutoff


def global_projection(field: torch.Tensor, basis: torch.Tensor):
    return field @ basis @ basis.T if basis.numel() else torch.zeros_like(field)


def normalize_field(field: torch.Tensor, z: torch.Tensor, magnitude: float):
    target_norm = magnitude * (z - z.mean(0, keepdim=True)).norm()
    return field * (target_norm / field.norm().clamp_min(1e-24))


def fit_empirical_drift_map(before: torch.Tensor, after: torch.Tensor, ridge: float):
    """Fit a smooth affine field to drift measured during a completed ER transition."""
    before = before.double()
    delta = after.double() - before
    center = before.mean(0, keepdim=True)
    delta_mean = delta.mean(0, keepdim=True)
    x = before - center
    y = delta - delta_mean
    gram = x.T @ x
    scale = gram.diagonal().mean().clamp_min(1e-24)
    coefficients = torch.linalg.solve(
        gram + ridge * scale * torch.eye(gram.shape[0], dtype=gram.dtype), x.T @ y,
    )
    fitted = x @ coefficients + delta_mean
    variance = (delta - delta.mean(0, keepdim=True)).square().sum().clamp_min(1e-24)
    r2 = 1.0 - (delta - fitted).square().sum() / variance
    return center.float(), delta_mean.float(), coefficients.float(), float(r2)


def apply_empirical_drift_map(z, center, delta_mean, coefficients):
    return (z - center) @ coefficients + delta_mean


def margin_direction(weights: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, classes: list[int]):
    """Per-example gradient of the current old-class margin with respect to z."""
    class_ids = torch.tensor(classes)
    selected = logits[:, class_ids]
    positions = torch.full((weights.shape[0],), -1, dtype=torch.long)
    positions[class_ids] = torch.arange(len(class_ids))
    true_positions = positions[labels]
    competitors = selected.clone()
    competitors.scatter_(1, true_positions[:, None], -torch.inf)
    competitor_ids = class_ids[competitors.argmax(1)]
    return weights[labels] - weights[competitor_ids]


def direction_fields(
    z: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    classes: list[int],
    discriminative: torch.Tensor,
    null: torch.Tensor,
    random_transform: torch.Tensor,
    null_transform: torch.Tensor,
    empirical_map: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    magnitude: float,
):
    centered = z - z.mean(0, keepdim=True)
    random_seed = centered @ random_transform
    random_discriminative = global_projection(random_seed, discriminative)
    random_null = global_projection(random_seed, null)
    random = (
        random_discriminative / random_discriminative.norm().clamp_min(1e-24)
        + random_null / random_null.norm().clamp_min(1e-24)
    ) / math.sqrt(2.0)
    null_seed = centered @ null_transform
    empirical = apply_empirical_drift_map(z, *empirical_map)
    beneficial = margin_direction(weights, logits, labels, classes)
    raw = {
        "jacobian_null": global_projection(null_seed, null),
        "empirical_er": empirical,
        "matched_random": random,
        "harmful": -beneficial,
        "beneficial": beneficial,
    }
    return {name: normalize_field(field, z, magnitude) for name, field in raw.items()}


def projection_fraction(delta: torch.Tensor, basis: torch.Tensor):
    energy = delta.square().sum().clamp_min(1e-24)
    return float(global_projection(delta, basis).square().sum() / energy)


def field_cosine(left: torch.Tensor, right: torch.Tensor):
    return float((left.double() * right.double()).sum() /
                 (left.double().norm() * right.double().norm()).clamp_min(1e-24))


def overlap_diagnostics(fields, discriminative, null):
    row = {}
    names = list(fields)
    for index, left in enumerate(names):
        row[f"{left}_rho_discriminative"] = projection_fraction(fields[left], discriminative)
        row[f"{left}_rho_null"] = projection_fraction(fields[left], null)
        for right in names[index + 1:]:
            row[f"cos_{left}__{right}"] = field_cosine(fields[left], fields[right])
    return row


def margin_values(logits: torch.Tensor, labels: torch.Tensor, classes: list[int]):
    classes_tensor = torch.tensor(classes)
    selected = logits[:, classes_tensor]
    positions = torch.full((100,), -1, dtype=torch.long)
    positions[classes_tensor] = torch.arange(len(classes_tensor))
    true_positions = positions[labels]
    true_logits = selected.gather(1, true_positions[:, None]).squeeze(1)
    competitors = selected.clone()
    competitors.scatter_(1, true_positions[:, None], -torch.inf)
    return true_logits - competitors.max(1).values


def functional_change(before_logits, after_logits, labels, classes):
    classes_tensor = torch.tensor(classes)
    before = before_logits[:, classes_tensor].double()
    after = after_logits[:, classes_tensor].double()
    before_centered = before - before.mean(1, keepdim=True)
    after_centered = after - after.mean(1, keepdim=True)
    centered_change = ((after_centered - before_centered).norm(dim=1).mean() /
                       before_centered.norm(dim=1).mean().clamp_min(1e-24))
    kl = F.kl_div(F.log_softmax(after, dim=1), F.softmax(before, dim=1), reduction="batchmean")
    before_margin = margin_values(before_logits, labels, classes).double()
    after_margin = margin_values(after_logits, labels, classes).double()
    sorted_shift = (before_margin.sort().values - after_margin.sort().values).abs().mean()
    return {
        "centered_logit_change": float(centered_change),
        "old_kl_change": float(kl),
        "margin_before_mean": float(before_margin.mean()),
        "margin_after_mean": float(after_margin.mean()),
        "margin_mean_change": float(after_margin.mean() - before_margin.mean()),
        "margin_distribution_l1": float(sorted_shift),
        "margin_after_std": float(after_margin.std()),
        "margin_after_q10": float(torch.quantile(after_margin, 0.1)),
        "margin_after_q50": float(torch.quantile(after_margin, 0.5)),
        "margin_after_q90": float(torch.quantile(after_margin, 0.9)),
    }


def realized_diagnostics(
    before: torch.Tensor,
    after: torch.Tensor,
    target: torch.Tensor,
    discriminative: torch.Tensor,
    null: torch.Tensor,
):
    delta = after.double() - before.double()
    target = target.double()
    base_norm = (before.double() - before.double().mean(0, keepdim=True)).norm().clamp_min(1e-24)
    target_unit = F.normalize(target, dim=1)
    projection = (delta * target_unit).sum(1, keepdim=True) * target_unit
    off_target = delta - projection
    valid = (delta.norm(dim=1) > 1e-12) & (target.norm(dim=1) > 1e-12)
    cosine = F.cosine_similarity(delta[valid], target[valid], dim=1) if valid.any() else delta.new_zeros(1)
    return {
        "realized_fro_norm": float(delta.norm()),
        "realized_fro_relative": float(delta.norm() / base_norm),
        "realized_mean_norm": float(delta.norm(dim=1).mean()),
        "target_fro_relative": float(target.norm() / base_norm),
        "target_cosine_mean": float(cosine.mean()),
        "target_cosine_median": float(cosine.median()),
        "target_alignment": field_cosine(delta, target),
        "off_target_fro_norm": float(off_target.norm()),
        "off_target_energy_fraction": float(off_target.square().sum() / delta.square().sum().clamp_min(1e-24)),
        "rho_discriminative": projection_fraction(delta.float(), discriminative),
        "rho_null": projection_fraction(delta.float(), null),
    }


def target_space_diagnostics(target, discriminative, null):
    return {
        "target_rho_discriminative": projection_fraction(target, discriminative),
        "target_rho_null": projection_fraction(target, null),
    }


def freeze_decoder(net, frozen: bool):
    for parameter in net.parameters():
        parameter.requires_grad = True
    for parameter in net.get_classifier().parameters():
        parameter.requires_grad = not frozen


@torch.no_grad()
def evaluate_intervention(net, dataset, ids, batch_size, device):
    return infer(net, dataset, ids, batch_size, device)


def offline_intervention(
    net,
    dataset,
    intervention_ids,
    targets,
    probe_ids,
    probe_before,
    probe_logits_before,
    probe_labels,
    seen,
    args,
    device,
    seed,
    parameter_target=None,
):
    freeze_decoder(net, True)
    positions = torch.full((len(dataset),), -1, dtype=torch.long)
    positions[intervention_ids] = torch.arange(len(intervention_ids))
    parameters = [parameter for parameter in net.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.offline_lr, weight_decay=args.offline_weight_decay)
    batches = loader(dataset, intervention_ids, args.batch_size, True, seed)
    iterator = iter(batches)
    best = None
    trace = []
    drift_reference_band = args.drift_tolerance / (2.0 + args.drift_tolerance)
    parameter_reference_band = args.parameter_tolerance / (2.0 + args.parameter_tolerance)
    accuracy_reference_band = args.old_acc_tolerance / 2.0
    margin_reference_band = args.old_margin_tolerance / 2.0

    for step in range(1, args.offline_steps + 1):
        try:
            batch_ids, x, y = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch_ids, x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        target = targets[positions[batch_ids]].to(device)
        optimizer.zero_grad(set_to_none=True)
        z = features(net, x)
        logits = net.get_classifier()(z)
        geometry_loss = (z - target).square().sum(1).mean()
        task_loss = masked_ce(logits, y, seen)
        loss = args.geometry_weight * geometry_loss + args.preserve_weight * task_loss
        loss.backward()
        optimizer.step()

        if step % args.check_every and step != args.offline_steps:
            continue
        post_logits, post_reps, _ = evaluate_intervention(
            net, dataset, probe_ids, args.batch_size, device,
        )
        delta = post_reps - probe_before
        base_norm = (probe_before - probe_before.mean(0, keepdim=True)).norm().clamp_min(1e-24)
        realized = float(delta.norm() / base_norm)
        post_acc = accuracy(post_logits, probe_labels, seen)
        baseline_acc = accuracy(net.get_classifier()(probe_before.to(device)).cpu(), probe_labels, seen)
        acc_gap = abs(post_acc - baseline_acc)
        margin_before = float(margin_values(probe_logits_before, probe_labels, seen).mean())
        margin_after = float(margin_values(post_logits, probe_labels, seen).mean())
        margin_gap = abs(margin_after - margin_before)
        drift_gap = abs(realized - args.realized_drift)
        params = parameter_change(net, args._parameter_start)[0]
        parameter_gap = (abs(params - parameter_target) / max(parameter_target, 1e-24)
                         if parameter_target is not None else 0.0)
        score = (
            drift_gap / max(args.realized_drift * drift_reference_band, 1e-24)
            + acc_gap / max(accuracy_reference_band, 1e-24)
            + margin_gap / max(margin_reference_band, 1e-24)
            + parameter_gap / max(parameter_reference_band, 1e-24)
        )
        trace.append({
            "step": step,
            "geometry_loss": float(geometry_loss.detach()),
            "task_loss": float(task_loss.detach()),
            "realized_fro_relative": realized,
            "old_accuracy": post_acc,
            "old_accuracy_gap": post_acc - baseline_acc,
            "old_margin": margin_after,
            "old_margin_gap": margin_after - margin_before,
            "parameter_delta": params,
            "parameter_relative_gap_to_reference": parameter_gap,
            "selection_score": score,
        })
        if best is None or score < best[0]:
            best = (score, state_dict_cpu(net), step)
        if (realized >= args.realized_drift and acc_gap <= accuracy_reference_band and
                margin_gap <= margin_reference_band and
                (parameter_target is None or parameter_gap <= parameter_reference_band)):
            break

    assert best is not None
    net.load_state_dict(best[1])
    freeze_decoder(net, False)
    return best[2], trace


def train_future_standard(
    net,
    train,
    test,
    current,
    memory,
    old_test,
    new_test,
    all_test,
    seen,
    args,
    device,
    seed,
):
    """Standard Class-IL ER: all representation and classifier parameters train normally."""
    seed_all(seed)
    freeze_decoder(net, False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    curve = []
    for epoch in range(args.future_epochs):
        current_loader = loader(train, current, args.batch_size // 2, True, seed + epoch)
        replay_loader = loader(train, memory, args.batch_size // 2, True, seed + 100 + epoch)
        replay_iterator = iter(replay_loader)
        net.train()
        for _, current_x, current_y in current_loader:
            try:
                _, replay_x, replay_y = next(replay_iterator)
            except StopIteration:
                replay_iterator = iter(replay_loader)
                _, replay_x, replay_y = next(replay_iterator)
            x = torch.cat((current_x, replay_x)).to(device)
            y = torch.cat((current_y, replay_y)).to(device)
            n = len(current_y)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = net(x)
                loss = masked_ce(logits[:n], y[:n], seen) + masked_ce(logits[n:], y[n:], seen)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        old_logits, _, old_labels = infer(net, test, old_test, args.batch_size, device)
        new_logits, _, new_labels = infer(net, test, new_test, args.batch_size, device)
        all_logits, _, all_labels = infer(net, test, all_test, args.batch_size, device)
        curve.append({
            "future_new_accuracy": accuracy(new_logits, new_labels, seen),
            "future_old_accuracy": accuracy(old_logits, old_labels, seen),
            "future_all_accuracy": accuracy(all_logits, all_labels, seen),
        })
    return curve


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
    for subdirectory in ("base_checkpoints", "intervention_checkpoints", "representations"):
        (args.output / subdirectory).mkdir(exist_ok=True)

    rows, curve_rows, trace_rows, space_rows = [], [], [], []
    conditions = list(args.conditions)
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
        empirical_center, empirical_mean, empirical_coefficients, empirical_r2 = fit_empirical_drift_map(
            empirical_before, empirical_after, args.empirical_ridge,
        )
        empirical_map = (empirical_center, empirical_mean, empirical_coefficients)
        memory = balanced_memory(train_ids, seen, args.memory, priority)

        base_state = state_dict_cpu(base)
        base_hash = state_sha256(base_state)
        torch.save({"state_dict": base_state, "seen": seen, "memory": memory,
                    "order": order, "seed": seed, "sha256": base_hash},
                   args.output / "base_checkpoints" / f"seed{seed}.pt")

        intervention_ids = memory
        old_probes = torch.cat([probe_ids[c] for c in seen])
        before_logits, before_reps, probe_labels = infer(
            base, train_eval, old_probes, args.batch_size, device,
        )
        _, intervention_reps, intervention_labels = infer(
            base, train_eval, intervention_ids, args.batch_size, device,
        )
        full_weights = base.get_classifier().weight.detach().float().cpu()
        weights = full_weights[seen]
        discriminative, null, singular, cutoff = functional_bases(weights, args.svd_threshold)
        generator = torch.Generator().manual_seed(seed + 7000)
        random_transform = torch.randn(
            (intervention_reps.shape[1], intervention_reps.shape[1]), generator=generator,
        ) / math.sqrt(intervention_reps.shape[1])
        null_transform = torch.randn(
            (intervention_reps.shape[1], intervention_reps.shape[1]), generator=generator,
        ) / math.sqrt(intervention_reps.shape[1])
        intervention_fields = direction_fields(
            intervention_reps, base.get_classifier()(intervention_reps.to(device)).detach().cpu(),
            intervention_labels, full_weights, seen, discriminative, null, random_transform, null_transform,
            empirical_map, args.target_magnitude,
        )
        probe_fields = direction_fields(
            before_reps, before_logits, probe_labels, full_weights, seen, discriminative, null,
            random_transform, null_transform, empirical_map, args.target_magnitude,
        )
        condition_scales = {
            "jacobian_null": args.null_target_scale,
            "empirical_er": args.empirical_target_scale,
            "matched_random": args.random_target_scale,
            "harmful": args.harmful_target_scale,
            "beneficial": args.beneficial_target_scale,
        }
        intervention_fields = {
            name: field * condition_scales[name] for name, field in intervention_fields.items()
        }
        probe_fields = {name: field * condition_scales[name] for name, field in probe_fields.items()}
        overlap = overlap_diagnostics(intervention_fields, discriminative, null)
        primary_cosines = [
            abs(field_cosine(intervention_fields[left], intervention_fields[right]))
            for index, left in enumerate(PRIMARY_CONDITIONS)
            for right in PRIMARY_CONDITIONS[index + 1:]
        ]
        direction_overlap_passed = max(primary_cosines) <= args.max_direction_cosine
        if not direction_overlap_passed and not args.allow_overlapping_directions:
            raise RuntimeError(
                f"Seed {seed} failed direction-overlap gate: max |cos|={max(primary_cosines):.4f} "
                f"> {args.max_direction_cosine:.4f}"
            )

        parameter_reference = None
        drift_reference_band = args.drift_tolerance / (2.0 + args.drift_tolerance)
        parameter_reference_band = args.parameter_tolerance / (2.0 + args.parameter_tolerance)
        execution_conditions = (["empirical_er"] + [name for name in conditions if name != "empirical_er"]
                                if "empirical_er" in conditions else conditions)
        for condition in execution_conditions:
            target = intervention_reps + intervention_fields[condition]
            net = model(args.model, False).to(device)
            net.load_state_dict(base_state)
            args._parameter_start = parameter_snapshot(net)
            seed_all(seed * 10000 + 17)
            selected_step, trace = offline_intervention(
                net, train_eval, intervention_ids, target, old_probes, before_reps,
                before_logits, probe_labels, seen, args, device, seed * 10000 + 17,
                parameter_reference,
            )
            for point in trace:
                trace_rows.append({"seed": seed, "condition": condition, **point})

            after_logits, after_reps, _ = infer(net, train_eval, old_probes, args.batch_size, device)
            _, intervention_after, _ = infer(
                net, train_eval, intervention_ids, args.batch_size, device,
            )
            probe_diag = realized_diagnostics(
                before_reps, after_reps, probe_fields[condition], discriminative, null,
            )
            intervention_diag = realized_diagnostics(
                intervention_reps, intervention_after, intervention_fields[condition],
                discriminative, null,
            )
            param_l2, param_relative = parameter_change(net, args._parameter_start)
            if condition == "empirical_er":
                parameter_reference = param_l2
            target_space = target_space_diagnostics(
                intervention_fields[condition], discriminative, null,
            )
            function_diag = functional_change(before_logits, after_logits, probe_labels, seen)
            old_before = accuracy(before_logits, probe_labels, seen)
            old_after = accuracy(after_logits, probe_labels, seen)
            achieved_geometry = geometry(before_reps, after_reps, probe_labels, weights)
            checkpoint_state = state_dict_cpu(net)
            torch.save({"state_dict": checkpoint_state, "base_sha256": base_hash,
                        "condition": condition, "selected_step": selected_step},
                       args.output / "intervention_checkpoints" / f"seed{seed}_{condition}.pt")
            torch.save({
                "probe_ids": old_probes, "probe_labels": probe_labels,
                "before": before_reps, "after": after_reps,
                "target_displacement": probe_fields[condition],
                "intervention_ids": intervention_ids,
                "intervention_before": intervention_reps,
                "intervention_after": intervention_after,
                "intervention_target_displacement": intervention_fields[condition],
            }, args.output / "representations" / f"seed{seed}_{condition}.pt")

            curve = []
            zero_shot_new = float("nan")
            base_zero_shot_new = float("nan")
            if not args.skip_future:
                future = order[2]
                all_seen = seen + future
                current = torch.cat([train_ids[c] for c in future])
                old_test = torch.cat([test_ids[c] for c in seen])
                new_test = torch.cat([test_ids[c] for c in future])
                all_test = torch.cat((old_test, new_test))
                zero_logits, _, zero_labels = infer(net, test, new_test, args.batch_size, device)
                zero_shot_new = accuracy(zero_logits, zero_labels, all_seen)
                base_for_fwt = model(args.model, False).to(device)
                base_for_fwt.load_state_dict(base_state)
                base_logits, _, _ = infer(base_for_fwt, test, new_test, args.batch_size, device)
                base_zero_shot_new = accuracy(base_logits, zero_labels, all_seen)
                del base_for_fwt
                curve = train_future_standard(
                    net, train, test, current, memory, old_test, new_test, all_test,
                    all_seen, args, device, seed * 10000 + 99,
                )
                for epoch, point in enumerate(curve, 1):
                    curve_rows.append({"seed": seed, "order_seed": args.order_seed,
                                       "condition": condition, "epoch": epoch, **point})
            rows.append({
                "seed": seed,
                "order_seed": args.order_seed,
                "condition": condition,
                "base_checkpoint_sha256": base_hash,
                "target_magnitude": args.target_magnitude,
                "condition_target_scale": condition_scales[condition],
                "realized_drift_target": args.realized_drift,
                "selected_offline_step": selected_step,
                "old_acc_before": old_before,
                "old_acc_after": old_after,
                "old_acc_delta": old_after - old_before,
                "individual_match_passed": (
                    abs(probe_diag["realized_fro_relative"] - args.realized_drift)
                    <= args.realized_drift * drift_reference_band and
                    abs(old_after - old_before) <= args.old_acc_tolerance / 2.0 + 1e-9 and
                    abs(function_diag["margin_mean_change"]) <= args.old_margin_tolerance / 2.0 + 1e-9 and
                    (parameter_reference is None or
                     abs(param_l2 - parameter_reference) / max(parameter_reference, 1e-24)
                     <= parameter_reference_band + 1e-9) and
                    probe_diag["target_alignment"] >= args.min_alignment
                ),
                "parameter_delta": param_l2,
                "parameter_delta_relative": param_relative,
                "decoder_parameter_delta": float(sum(
                    (checkpoint_state[name] - base_state[name]).float().square().sum()
                    for name in checkpoint_state if name.startswith("head.")
                ).sqrt()),
                **function_diag,
                **target_space,
                **{f"probe_{key}": value for key, value in probe_diag.items()},
                **{f"intervention_{key}": value for key, value in intervention_diag.items()},
                **{f"geometry_{key}": value for key, value in achieved_geometry.items()},
                "future_new_accuracy": curve[-1]["future_new_accuracy"] if curve else float("nan"),
                "future_old_accuracy": curve[-1]["future_old_accuracy"] if curve else float("nan"),
                "future_all_accuracy": curve[-1]["future_all_accuracy"] if curve else float("nan"),
                "future_new_auc": (float(np.mean([point["future_new_accuracy"] for point in curve]))
                                   if curve else float("nan")),
                "future_new_early_auc": (float(np.mean(
                    [point["future_new_accuracy"] for point in curve[:2]]
                )) if curve else float("nan")),
                "future_new_epoch1_accuracy": (curve[0]["future_new_accuracy"] if curve else float("nan")),
                "future_new_early_slope": (curve[1]["future_new_accuracy"] - curve[0]["future_new_accuracy"]
                                           if len(curve) >= 2 else float("nan")),
                "zero_shot_new_accuracy": zero_shot_new,
                "forward_transfer": zero_shot_new - base_zero_shot_new,
                "old_task_forgetting": (old_after - curve[-1]["future_old_accuracy"]
                                         if curve else float("nan")),
            })

        space_rows.append({
            "seed": seed,
            "functional_rank": discriminative.shape[1],
            "null_rank": null.shape[1],
            "svd_cutoff": cutoff,
            "largest_singular_value": float(singular.max()),
            "smallest_retained_singular_value": float(singular[discriminative.shape[1] - 1]),
            "empirical_map_r2": empirical_r2,
            "maximum_primary_absolute_cosine": max(primary_cosines),
            "direction_overlap_passed": direction_overlap_passed,
            **overlap,
        })

    delattr(args, "_parameter_start")
    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "learning_curves.csv", curve_rows)
    write_csv(args.output / "intervention_trace.csv", trace_rows)
    write_csv(args.output / "space_diagnostics.csv", space_rows)
    manifest_args = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    manifest = manifest_args | {
        "data": str(args.data), "output": str(args.output), "order": order,
        "device": str(device), "torch": torch.__version__, "timm": timm.__version__,
        "primary_conditions": list(PRIMARY_CONDITIONS),
        "actual_conditions": sorted({row["condition"] for row in rows}),
        "representation": "final LayerNorm output / classifier pre-logits embedding",
        "functional_jacobian": "(I - 11^T/K) W_old",
        "empirical_direction": "ridge affine field fit to observed Task-1 -> Task-2 ER drift",
        "future_protocol": "standard Class-IL ER; decoder unfrozen",
        "selection_rule": "joint normalized score over drift, old accuracy, old margin, and parameter displacement",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Functionally defined matched-direction Phase C intervention")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS,
                        default=list(PRIMARY_CONDITIONS))
    parser.add_argument("--target-magnitude", type=float, default=0.08)
    parser.add_argument("--null-target-scale", type=float, default=1.0)
    parser.add_argument("--empirical-target-scale", type=float, default=1.0)
    parser.add_argument("--random-target-scale", type=float, default=1.0)
    parser.add_argument("--harmful-target-scale", type=float, default=1.0)
    parser.add_argument("--beneficial-target-scale", type=float, default=1.0)
    parser.add_argument("--realized-drift", type=float, default=0.025)
    parser.add_argument("--drift-tolerance", type=float, default=0.05)
    parser.add_argument("--old-acc-tolerance", type=float, default=0.005)
    parser.add_argument("--old-margin-tolerance", type=float, default=0.05)
    parser.add_argument("--parameter-tolerance", type=float, default=0.10)
    parser.add_argument("--min-alignment", type=float, default=0.50)
    parser.add_argument("--max-direction-cosine", type=float, default=0.30)
    parser.add_argument("--allow-overlapping-directions", action="store_true")
    parser.add_argument("--svd-threshold", type=float, default=1e-7)
    parser.add_argument("--empirical-ridge", type=float, default=1e-3)
    parser.add_argument("--offline-steps", type=int, default=80)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--geometry-weight", type=float, default=10.0)
    parser.add_argument("--preserve-weight", type=float, default=1.0)
    parser.add_argument("--offline-lr", type=float, default=1e-5)
    parser.add_argument("--offline-weight-decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs for each pre-intervention task")
    parser.add_argument("--future-epochs", type=int, default=5)
    parser.add_argument("--skip-future", action="store_true",
                        help="Calibrate interventions without exposing Task t+1 data")
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
    parser.add_argument("--output", type=Path, default=Path("results/phase_c/functional_direction"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
