from __future__ import annotations

import argparse
import csv
import json
import os
import random
from itertools import combinations
from pathlib import Path
from types import MethodType

os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import v2


METHODS = ("frozen", "naive", "er", "derpp", "prompt", "ogd")


class IndexedCifar(Dataset):
    def __init__(self, root: Path, train: bool, augment: bool, dataset: str = "cifar100"):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ops = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()] if augment else []
        ops += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
        dataset_class = {"cifar10": CIFAR10, "cifar100": CIFAR100}[dataset]
        self.base = dataset_class(root, train=train, transform=v2.Compose(ops), download=False)
        self.targets = torch.tensor(self.base.targets)
        self.num_classes = len(self.base.classes)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return index, image, label


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sequencer_linear_classifier(self):
    """Expose the readout on pooled pre-logits without applying spatial pooling twice."""
    return self.head.fc


def model(name: str, pretrained: bool, prompt: bool = False, num_classes: int = 100):
    net = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    if name.startswith("sequencer2d_"):
        assert isinstance(net.head.fc, nn.Linear)
        # forward_head still uses the original complete head; state_dict keys stay intact.
        net.get_classifier = MethodType(_sequencer_linear_classifier, net)
    if prompt:
        net.prompt = nn.Parameter(torch.empty(1, 5, net.embed_dim))
        nn.init.trunc_normal_(net.prompt, std=0.02)

        def prompted_forward(self, x, **_):
            x = self._pos_embed(self.patch_embed(x))
            x = torch.cat((x[:, :1], self.prompt.expand(len(x), -1, -1), x[:, 1:]), dim=1)
            x = self.norm_pre(self.patch_drop(x))
            return self.norm(self.blocks(x))

        net.forward_features = MethodType(prompted_forward, net)
    return net


def features(net, x):
    return net.forward_head(net.forward_features(x), pre_logits=True)


def layer_features(net, x):
    x = net.patch_embed(x)
    x = net._pos_embed(x)
    if hasattr(net, "prompt"):
        x = torch.cat((x[:, :1], net.prompt.expand(len(x), -1, -1), x[:, 1:]), dim=1)
    x = net.patch_drop(x)
    x = net.norm_pre(x)
    layers = []
    for block in net.blocks:
        x = block(x)
        layers.append(x[:, 0])
    x = net.norm(x)
    final = net.forward_head(x, pre_logits=True)
    return layers + [final]


def ids_by_class(targets: torch.Tensor, classes, count: int, tail: bool = False):
    chunks = []
    for label in classes:
        ids = torch.where(targets == label)[0]
        chunks.append(ids[-count:] if tail else ids[:count])
    return torch.cat(chunks)


def balanced_memory(train_ids: dict[int, torch.Tensor], seen, size: int, priority: torch.Tensor):
    quota, extra = divmod(size, len(seen))
    chunks = []
    for rank, label in enumerate(seen):
        ids = train_ids[label]
        keep = min(len(ids), quota + (rank < extra))
        chunks.append(ids[torch.argsort(priority[ids])[:keep]])
    return torch.cat(chunks)


def loader(dataset, ids, batch_size, shuffle=False, seed=0, num_workers=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, ids.tolist()), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=False)


def prepare_images(images, device):
    """Transfer compact CIFAR tensors, then resize batches on the accelerator."""
    images = images.to(device, non_blocking=True)
    return F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)


def masked_ce(logits, labels, seen):
    seen = torch.tensor(seen, device=logits.device)
    positions = torch.full((logits.shape[1],), -1, dtype=torch.long, device=logits.device)
    positions[seen] = torch.arange(len(seen), device=logits.device)
    return F.cross_entropy(logits[:, seen], positions[labels])


def train_task(net, method, dataset, current, memory, teacher, seen, old, args, device, seed,
               gradient_basis=None, step_observer=None):
    frozen = method == "frozen" and old
    for parameter in net.parameters():
        parameter.requires_grad = not frozen
    if frozen:
        for parameter in net.get_classifier().parameters():
            parameter.requires_grad = True
    if method == "prompt" and old:
        for parameter in net.parameters():
            parameter.requires_grad = False
        net.prompt.requires_grad = True
        for parameter in net.get_classifier().parameters():
            parameter.requires_grad = True
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    head = net.get_classifier()
    old_weight = head.weight[old].detach().clone() if frozen else None
    old_bias = head.bias[old].detach().clone() if frozen else None
    replay = method in ("er", "derpp") and len(memory)
    current_bs = args.batch_size // 2 if replay else args.batch_size

    net.train()
    for epoch in range(args.epochs):
        current_loader = loader(dataset, current, current_bs, True, seed + epoch)
        replay_loader = loader(dataset, memory, args.batch_size - current_bs, True, seed + 100 + epoch) if replay else None
        replay_iter = iter(replay_loader) if replay else None
        for batch_index, (cur_ids, cur_x, cur_y) in enumerate(current_loader):
            batches = [(cur_ids, cur_x, cur_y)]
            if replay:
                try:
                    batches.append(next(replay_iter))
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    batches.append(next(replay_iter))
            ids = torch.cat([b[0] for b in batches])
            x = prepare_images(torch.cat([b[1] for b in batches]), device)
            y = torch.cat([b[2] for b in batches]).to(device)
            n_current = len(cur_y)
            if step_observer is not None:
                step_observer("before", epoch, batch_index, net, optimizer, ids, x, y, n_current)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = net(x)
                loss = masked_ce(logits[:n_current], y[:n_current], seen)
                if replay:
                    loss += masked_ce(logits[n_current:], y[n_current:], seen)
                if method == "derpp" and replay:
                    targets = torch.stack([teacher[int(i)] for i in ids[n_current:]]).to(device)
                    loss += args.der_alpha * F.mse_loss(logits[n_current:], targets)
            scaler.scale(loss).backward()
            if gradient_basis:
                scaler.unscale_(optimizer)
                grads = torch.cat([p.grad.flatten() for p in parameters])
                for direction in gradient_basis:
                    grads -= grads.dot(direction) * direction
                offset = 0
                for parameter in parameters:
                    count = parameter.numel()
                    parameter.grad.copy_(grads[offset:offset + count].view_as(parameter))
                    offset += count
            scaler.step(optimizer)
            scaler.update()
            if step_observer is not None:
                step_observer("after", epoch, batch_index, net, optimizer, ids, x, y, n_current)
            if frozen:
                with torch.no_grad():
                    head.weight[old] = old_weight
                    head.bias[old] = old_bias


def old_gradient_direction(net, dataset, memory, seen, basis, args, device):
    for parameter in net.parameters():
        parameter.requires_grad = True
        parameter.grad = None
    net.eval()
    batches = loader(dataset, memory, args.batch_size)
    for _, x, y in batches:
        logits = net(prepare_images(x, device))
        (masked_ce(logits, y.to(device), seen) / len(batches)).backward()
    direction = torch.cat([p.grad.flatten() for p in net.parameters()])
    for previous in basis:
        direction -= direction.dot(previous) * previous
    direction /= direction.norm().clamp_min(1e-12)
    net.zero_grad(set_to_none=True)
    return direction.detach()


@torch.inference_mode()
def infer(net, dataset, ids, batch_size, device):
    net.eval()
    logits, reps, labels = [], [], []
    for _, x, y in loader(dataset, ids, batch_size):
        x = prepare_images(x, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits.append(net(x).float().cpu())
            reps.append(features(net, x).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(reps), torch.cat(labels)


@torch.inference_mode()
def infer_layers(net, dataset, ids, batch_size, device):
    net.eval()
    logits, labels, collected = [], [], None
    for _, x, y in loader(dataset, ids, batch_size):
        x = prepare_images(x, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            batch_layers = layer_features(net, x)
            logits.append(net.get_classifier()(batch_layers[-1]).float().cpu())
        if collected is None:
            collected = [[] for _ in batch_layers]
        for store, values in zip(collected, batch_layers):
            store.append(values.float().cpu())
        labels.append(y)
    names = [f"block{i + 1:02d}" for i in range(len(collected) - 1)] + ["final"]
    return torch.cat(logits), {name: torch.cat(values) for name, values in zip(names, collected)}, torch.cat(labels)


@torch.inference_mode()
def infer_intermediates(net, dataset, ids, batch_size, device, indices=(1, 2, 3, 4)):
    """Return globally pooled timm feature stages for convolutional backbones."""
    net.eval()
    logits, labels, collected = [], [], None
    for _, x, y in loader(dataset, ids, batch_size):
        x = prepare_images(x, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            maps = net.forward_intermediates(
                x, indices=indices, intermediates_only=True, output_fmt="NCHW",
            )
            stages = [values.mean(dim=(-2, -1)) for values in maps]
            logits.append(net.get_classifier()(stages[-1]).float().cpu())
        if collected is None:
            collected = [[] for _ in stages]
        for store, values in zip(collected, stages):
            store.append(values.float().cpu())
        labels.append(y)
    names = [f"stage{index}" for index in range(1, len(collected) + 1)]
    return torch.cat(logits), {
        name: torch.cat(values) for name, values in zip(names, collected)
    }, torch.cat(labels)


def accuracy(logits, labels, seen):
    seen = torch.tensor(seen)
    return float((seen[logits[:, seen].argmax(1)] == labels).float().mean())


def readout(z, weights, bias, classes, labels):
    classes = torch.tensor(classes)
    logits = z @ weights[classes].T + bias[classes]
    positions = torch.full((100,), -1, dtype=torch.long)
    positions[classes] = torch.arange(len(classes))
    correct = positions[labels]
    true = logits.gather(1, correct[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, correct[:, None], -torch.inf)
    centered = logits - logits.mean(1, keepdim=True)
    return float((logits.argmax(1) == correct).float().mean()), float((true - competitors.max(1).values).mean()), centered


def readout_decomposition(before, after, labels, classes, weight_before, bias_before,
                          weight_after, bias_after):
    combinations = {
        "pre": (before, weight_before, bias_before),
        "representation_only": (after, weight_before, bias_before),
        "decoder_only": (before, weight_after, bias_after),
        "joint_old_space": (after, weight_after, bias_after),
    }
    outputs, centered_logits = {}, {}
    for name, (z, weight, bias) in combinations.items():
        acc, margin, logits = readout(z, weight, bias, classes, labels)
        outputs[f"{name}_accuracy"] = acc
        outputs[f"{name}_margin"] = margin
        centered_logits[name] = logits
    scale = centered_logits["pre"].norm(dim=1).mean().clamp_min(1e-12)
    for name in ("representation_only", "decoder_only", "joint_old_space"):
        change = (centered_logits[name] - centered_logits["pre"]).norm(dim=1).mean() / scale
        outputs[f"{name}_logit_change"] = float(change)
    old_before = weight_before[classes] - weight_before[classes].mean(0)
    old_after = weight_after[classes] - weight_after[classes].mean(0)
    outputs["decoder_weight_change"] = float((old_after - old_before).norm() / old_before.norm().clamp_min(1e-12))
    return outputs


def geometry(before, after, labels, weights):
    a, b = before.double(), after.double()
    delta = b - a
    scale = (a - a.mean(0)).square().sum(1).mean().sqrt().clamp_min(1e-12)
    energy = delta.square().sum().clamp_min(1e-24)
    raw = delta.norm(dim=1).mean() / scale
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vh = torch.linalg.svd(bc.T @ ac, full_matrices=False)
    aligned = (bc @ (u @ vh) - ac).norm(dim=1).mean() / scale
    weights = weights.double() - weights.double().mean(0, keepdim=True)
    _, singular, vh_w = torch.linalg.svd(weights, full_matrices=False)
    basis = vh_w[singular > singular.max() * 1e-7].T if singular.numel() else vh_w.T[:, :0]
    discriminative = (delta @ basis).square().sum() / energy if basis.numel() else delta.new_tensor(0)
    null = 1 - discriminative if delta.abs().max() else delta.new_tensor(0)
    isotropic = basis.shape[1] / delta.shape[1]
    tangent_energy = delta.new_tensor(0)
    for label in labels.unique():
        keep = labels == label
        centered = a[keep] - a[keep].mean(0)
        rank = min(5, int(keep.sum()) - 1, a.shape[1])
        if rank > 0:
            tangent = torch.linalg.svd(centered, full_matrices=False).Vh[:rank].T
            tangent_energy += (delta[keep] @ tangent).square().sum()
    singular_d = torch.linalg.svdvals(delta)
    spectrum = singular_d.square()
    effective_rank = spectrum.sum().square() / spectrum.square().sum().clamp_min(1e-24)
    norms = delta.norm(dim=1)
    relation = (F.normalize(a, dim=1) @ F.normalize(a, dim=1).T -
                F.normalize(b, dim=1) @ F.normalize(b, dim=1).T).abs().mean()
    return {
        "drift_raw": float(raw), "drift_aligned": float(aligned),
        "discriminative_fraction": float(discriminative),
        "task_rank": int(basis.shape[1]),
        "isotropic_fraction": float(isotropic),
        "discriminative_enrichment": float(discriminative / isotropic) if isotropic else 0.0,
        "null_fraction": float(null),
        "tangent_fraction": float(tangent_energy / energy),
        "effective_rank": float(effective_rank if delta.abs().max() else 0),
        "sample_cv": float(norms.std() / norms.mean().clamp_min(1e-12)),
        "relational_change": float(relation),
    }


def parameter_snapshot(net):
    return [p.detach().float().cpu().clone() for p in net.parameters()]


def parameter_change(net, before):
    delta = sum((p.detach().float().cpu() - old).square().sum() for p, old in zip(net.parameters(), before))
    base = sum(old.square().sum() for old in before)
    return float(delta.sqrt()), float((delta / base.clamp_min(1e-24)).sqrt())


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_correlation(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 1 else float("nan")


def gate(rows, seeds):
    frozen = [r["drift_raw"] for r in rows if r["method"] == "frozen"]
    er = [r["drift_raw"] for r in rows if r["method"] == "er"]
    descriptors = ("drift_raw", "drift_aligned", "discriminative_fraction",
                   "tangent_fraction", "effective_rank", "sample_cv")
    correlations = []
    for descriptor in descriptors:
        vectors = []
        for seed in seeds:
            subset = sorted((r for r in rows if r["seed"] == seed), key=lambda r: (r["method"], r["task"]))
            vectors.append([r[descriptor] for r in subset])
        correlations += [rank_correlation(vectors[i], vectors[j]) for i, j in combinations(range(len(vectors)), 2)]
    means = {}
    for method in METHODS:
        values = [r["old_acc_after"] for r in rows if r["method"] == method]
        means[method] = float(np.mean(values)) if values else None
    return {
        "frozen_zero_drift": bool(frozen and max(frozen) <= 1e-8),
        "max_frozen_drift": max(frozen, default=float("nan")),
        "er_nonzero_drift": bool(er and min(er) > 1e-6),
        "min_er_drift": min(er, default=float("nan")),
        "median_descriptor_rank_correlation": float(np.nanmedian(correlations)) if correlations else None,
        "mean_old_accuracy_after": means,
    }


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = IndexedCifar(args.data, True, True)
    train_eval = IndexedCifar(args.data, True, False)
    test = IndexedCifar(args.data, False, False)
    order = np.random.default_rng(args.order_seed).permutation(100)[:args.tasks * 10].reshape(args.tasks, 10).tolist()
    train_ids = {c: ids_by_class(train.targets, [c], args.train_per_class) for c in sum(order, [])}
    probe_ids = {c: ids_by_class(train.targets, [c], args.probe_per_class, True) for c in sum(order, [])}
    test_ids = {c: ids_by_class(test.targets, [c], args.test_per_class) for c in sum(order, [])}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    (args.output / "representations").mkdir(exist_ok=True)
    rows, layer_rows, runs = [], [], []

    for seed in args.seeds:
        seed_all(seed)
        base = model(args.model, args.pretrained).to(device)
        train_task(base, "naive", train, torch.cat([train_ids[c] for c in order[0]]),
                   torch.empty(0, dtype=torch.long), {}, order[0], [], args, device, seed)
        base_state = {k: v.detach().cpu() for k, v in base.state_dict().items()}
        priority = torch.rand(len(train), generator=torch.Generator().manual_seed(seed + 9000))

        for method in args.methods:
            if method == "prompt":
                seed_all(seed)
                net = model(args.model, args.pretrained, prompt=True).to(device)
                train_task(net, method, train, torch.cat([train_ids[c] for c in order[0]]),
                           torch.empty(0, dtype=torch.long), {}, order[0], [], args, device, seed)
            else:
                net = model(args.model, False).to(device)
                net.load_state_dict(base_state)
            seen, old = list(order[0]), list(order[0])
            memory = (balanced_memory(train_ids, seen, args.memory, priority)
                      if method in ("er", "derpp", "ogd") else torch.empty(0, dtype=torch.long))
            teacher = {}
            gradient_basis = []
            if method == "derpp":
                logits, _, _ = infer(net, train_eval, memory, args.batch_size, device)
                teacher = {int(i): logit for i, logit in zip(memory, logits)}
            if method == "ogd":
                gradient_basis.append(old_gradient_direction(
                    net, train_eval, memory, seen, gradient_basis, args, device,
                ))

            for task in range(1, args.tasks):
                old_probes = torch.cat([probe_ids[c] for c in old])
                before_logits, before_layers, labels = infer_layers(net, train_eval, old_probes,
                                                                    args.batch_size, device)
                before_reps = before_layers["final"]
                head = net.get_classifier()
                weight_before = head.weight.detach().float().cpu().clone()
                bias_before = head.bias.detach().float().cpu().clone()
                params_before = parameter_snapshot(net)
                seen += order[task]
                current = torch.cat([train_ids[c] for c in order[task]])
                train_task(net, method, train, current, memory, teacher, seen, old, args, device,
                           seed * 100 + task, gradient_basis)
                after_logits, after_layers, _ = infer_layers(net, train_eval, old_probes,
                                                              args.batch_size, device)
                after_reps = after_layers["final"]
                head = net.get_classifier()
                weight_after = head.weight.detach().float().cpu().clone()
                bias_after = head.bias.detach().float().cpu().clone()
                delta, relative = parameter_change(net, params_before)
                metrics = geometry(before_reps, after_reps, labels, weight_before[old])
                decomposition = readout_decomposition(
                    before_reps, after_reps, labels, old,
                    weight_before, bias_before, weight_after, bias_after,
                )
                row = {
                    "seed": seed, "method": method, "task": task,
                    "old_acc_before": accuracy(before_logits, labels, old),
                    "old_acc_after": accuracy(after_logits, labels, seen),
                    "new_class_competition": decomposition["joint_old_space_accuracy"] -
                                             accuracy(after_logits, labels, seen),
                    "parameter_delta": delta, "parameter_delta_relative": relative,
                    **metrics, **decomposition,
                }
                rows.append(row)
                for name in before_layers:
                    layer_metrics = geometry(before_layers[name], after_layers[name], labels,
                                             weight_before[old])
                    layer_rows.append({
                        "seed": seed, "method": method, "task": task, "layer": name,
                        **{key: layer_metrics[key] for key in (
                            "drift_raw", "drift_aligned", "tangent_fraction",
                            "effective_rank", "sample_cv", "relational_change",
                        )},
                    })
                torch.save({
                    "ids": old_probes, "labels": labels,
                    "before_layers": before_layers, "after_layers": after_layers,
                    "weight_before": weight_before, "bias_before": bias_before,
                    "weight_after": weight_after, "bias_after": bias_after,
                }, args.output / "representations" / f"seed{seed}_{method}_t{task}.pt")
                old = list(seen)
                if method in ("er", "derpp", "ogd"):
                    new_memory = balanced_memory(train_ids, seen, args.memory, priority)
                    if method == "derpp":
                        missing = torch.tensor([int(i) for i in new_memory if int(i) not in teacher])
                        if len(missing):
                            logits, _, _ = infer(net, train_eval, missing, args.batch_size, device)
                            teacher.update({int(i): logit for i, logit in zip(missing, logits)})
                        teacher = {int(i): teacher[int(i)] for i in new_memory}
                    memory = new_memory
                    if method == "ogd":
                        gradient_basis.append(old_gradient_direction(
                            net, train_eval, memory, seen, gradient_basis, args, device,
                        ))

            final_ids = torch.cat([test_ids[c] for c in seen])
            logits, _, labels = infer(net, test, final_ids, args.batch_size, device)
            runs.append({"seed": seed, "method": method, "final_accuracy": accuracy(logits, labels, seen)})
            torch.save(net.state_dict(), args.output / "checkpoints" / f"seed{seed}_{method}.pt")

    write_csv(args.output / "transitions.csv", rows)
    write_csv(args.output / "layers.csv", layer_rows)
    write_csv(args.output / "runs.csv", runs)
    config = vars(args) | {"data": str(args.data), "output": str(args.output), "order": order,
                          "device": str(device), "torch": torch.__version__, "timm": timm.__version__}
    (args.output / "manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "gate.json").write_text(json.dumps(gate(rows, args.seeds), indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="DeiT Class-IL natural-drift characterization")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
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
    parser.add_argument("--output", type=Path, default=Path("results/phase_b/corrected"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
