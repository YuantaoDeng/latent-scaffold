"""Paired final-task training with a penalty on fixed feature-space directions."""
from __future__ import annotations
from package_paths import read_config_text

import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from drift_experiment import (IndexedCifar, accuracy, balanced_memory, features,
                              ids_by_class, loader, masked_ce, model,
                              prepare_images, seed_all, train_task)

ROOT = Path('results/cifar10_confirmation')
OUT = Path('results/scaffold_intervention')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def cpu_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def penalty(delta, basis):
    projected = delta.float() @ basis.float()
    return (projected - projected.mean(0, keepdim=True)).square().mean()


def bases_and_checks():
    saved = load(ROOT / 'scaffold_followup/analysis/predictors.pt')
    bases = {'scaffold': saved['scaffold'].float(), 'pca': saved['matched_static'].float()}
    for i in (1, 2):
        generator = torch.Generator().manual_seed(91000 + i)
        bases[f'random{i}'] = torch.linalg.qr(torch.randn(512, 8, generator=generator)).Q
    inventory = json.loads(read_config_text(ROOT / 'scaffold_structure/inventory.json'))
    fitted = [load(p['path']) for p in inventory if p['role'] == 'source']
    report = {}
    for name, key in [('scaffold', 'historical_basis'), ('pca', 'static_pca_basis')]:
        columns = torch.cat([v[key].double() for v in fitted], dim=1)
        u = torch.linalg.svd(columns, full_matrices=False).U[:, :8]
        error = float((u @ u.T - bases[name].double() @ bases[name].double().T).abs().max())
        assert error < 1e-5, (name, error)
        report[name + '_reconstruction_error'] = error
    for name, basis in bases.items():
        error = float((basis.T @ basis - torch.eye(8)).abs().max())
        assert error < 2e-5
        report[name + '_orthogonality_error'] = error
    g = torch.Generator().manual_seed(771)
    delta = torch.randn(32, 512, generator=g, requires_grad=True)
    constant = torch.randn(1, 512, generator=g)
    u = bases['scaffold']
    a, b = penalty(delta, u), penalty(delta + constant, u)
    assert torch.allclose(a, b, atol=1e-6)
    assert float(penalty(torch.zeros_like(delta), u)) == 0.0
    a.backward()
    outside = delta.grad - (delta.grad @ u) @ u.T
    report['penalty_gradient_outside_subspace'] = float(outside.abs().max())
    assert report['penalty_gradient_outside_subspace'] < 1e-6
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(bases, OUT / 'bases.pt')
    (OUT / 'construction_checks.json').write_text(json.dumps(report, indent=2))
    return bases


@torch.inference_mode()
def observe(net, dataset, ids, device):
    net.eval()
    zs, ls, ys = [], [], []
    for _, images, labels in loader(dataset, ids, 64):
        with torch.amp.autocast('cuda'):
            z = features(net, prepare_images(images, device))
            logits = net.get_classifier()(z)
        zs.append(z.float().cpu())
        ls.append(logits.float().cpu())
        ys.append(labels)
    return {'features': torch.cat(zs), 'logits': torch.cat(ls), 'labels': torch.cat(ys)}


def measure(obs, initial, mem, mem_initial, order, bases):
    old, current, seen = sum(order[:-1], []), order[-1], sum(order, [])
    labels, logits = obs['labels'], obs['logits']
    result = {}
    for name, classes in [('new', current), ('old', old), ('first', order[0])]:
        mask = torch.isin(labels, torch.tensor(classes))
        result[name + '_accuracy'] = accuracy(logits[mask], labels[mask], seen)
        result[name + '_loss'] = float(masked_ce(logits[mask], labels[mask], seen))
        if name == 'old':
            result['old_only_accuracy'] = accuracy(logits[mask], labels[mask], old)
    old_mask = torch.isin(labels, torch.tensor(old))
    first_mask = torch.isin(labels, torch.tensor(order[0]))
    for prefix, delta in [('old', (obs['features'] - initial['features'])[old_mask]),
                          ('first', (obs['features'] - initial['features'])[first_mask]),
                          ('memory', mem['features'] - mem_initial['features'])]:
        centered = delta.double() - delta.double().mean(0)
        total = float(centered.square().sum(1).mean())
        result[prefix + '_total_energy'] = total
        result[prefix + '_mean_energy'] = float(delta.double().mean(0).square().sum())
        for name, u in bases.items():
            inside = float((centered @ u.double()).square().sum(1).mean())
            result[f'{prefix}_{name}_energy'] = inside
            result[f'{prefix}_{name}_outside_energy'] = max(0., total - inside)
    return result


def train_epoch(net, optimizer, scaler, train, current, memory, seen, cfg, device,
                epoch_seed, basis=None, coefficient=0., reference=None, images=None,
                id_to_memory=None):
    seed_all(epoch_seed)
    current_loader = loader(train, current, 32, True, epoch_seed)
    replay_loader = loader(train, memory, 32, True, epoch_seed + 100)
    replay = iter(replay_loader)
    losses, penalties, skipped = [], [], 0
    for cur_ids, cur_x, cur_y in current_loader:
        try:
            rep_ids, rep_x, rep_y = next(replay)
        except StopIteration:
            replay = iter(replay_loader)
            rep_ids, rep_x, rep_y = next(replay)
        x = prepare_images(torch.cat([cur_x, rep_x]), device)
        y = torch.cat([cur_y, rep_y]).to(device)
        optimizer.zero_grad(set_to_none=True)
        if coefficient:
            positions = id_to_memory[rep_ids].to(device)
            net.eval()
            # Keep the constrained response and its gradient in FP32: strong
            # penalties can overflow FP16 backward even at a small loss scale.
            z = features(net, prepare_images(images[positions], device))
            regularizer = penalty(z.float() - reference[positions], basis)
            scaler.scale(coefficient * regularizer).backward()
            penalties.append(float(regularizer.detach()))
        net.train()
        with torch.amp.autocast('cuda'):
            logits = net(x)
            task_loss = (masked_ce(logits[:len(cur_y)], y[:len(cur_y)], seen)
                         + masked_ce(logits[len(cur_y):], y[len(cur_y):], seen))
        scaler.scale(task_loss).backward()
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += scaler.get_scale() < old_scale
        losses.append(float(task_loss.detach()))
    return {'training_loss': sum(losses) / len(losses),
            'training_penalty': sum(penalties) / max(1, len(penalties)),
            'steps': len(losses), 'skipped_steps': skipped}


def parity_check(train, current, memory, seen, old, cfg, state, device):
    """One epoch on two current batches, same RNG/optimizer/loader as legacy code."""
    short = current[:64]
    seed = 88771
    net = model('resnet18', False, num_classes=10).to(device)
    net.load_state_dict(state)
    seed_all(seed)
    config = SimpleNamespace(**vars(cfg))
    config.epochs = 1
    train_task(net, 'er', train, short, memory, {}, seen, old, config, device, seed)
    expected = cpu_state(net)
    net.load_state_dict(state)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda')
    train_epoch(net, optimizer, scaler, train, short, memory, seen, cfg, device, seed)
    actual = cpu_state(net)
    error = max(float((actual[k].double() - expected[k].double()).abs().max()) for k in actual)
    report = {'maximum_state_difference': error, 'batches': 2}
    (OUT / 'baseline_parity_check.json').write_text(json.dumps(report, indent=2))
    assert error == 0, report
    del net, optimizer, scaler


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    bases = bases_and_checks()
    train = IndexedCifar(Path('data'), True, True, 'cifar10')
    evaluation = IndexedCifar(Path('data'), True, False, 'cifar10')
    branches = [('baseline', 0.)] + [(name, scale) for scale in args.strengths
                                    for name in args.conditions]
    manifest = {'dataset': 'cifar10', 'orders': args.orders, 'seed': 170,
                'strengths': args.strengths, 'branches': branches, 'epochs': 5,
                'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
                'penalty_precision': 'float32', 'amp_initial_scale': 128.,
                'epoch_seed_rule': '500000 + order * 100 + epoch', 'status': 'running'}
    if args.extend:
        previous = json.loads(read_config_text(OUT / 'manifest.json'))
        assert previous['status'] == 'complete'
        assert previous['orders'] == args.orders
        assert previous.get('penalty_precision') == 'float32'
        manifest['branches'] = previous['branches'] + [list(b) for b in branches
                                                       if list(b) not in previous['branches']]
        manifest['strengths'] = sorted(set(previous['strengths'] + args.strengths))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for order_seed in args.orders:
        directory = OUT / f'order{order_seed}'
        directory.mkdir(exist_ok=True)
        cfg = SimpleNamespace(**json.loads(read_config_text(ROOT / f'target_order{order_seed}/manifest.json')))
        order = cfg.order
        old, seen = sum(order[:-1], []), sum(order, [])
        original = load(ROOT / f'target_order{order_seed}/artifacts/order{order_seed}_seed170.pt')
        state = original['pre_future_state_dict']
        del original
        train_ids = {c: ids_by_class(train.targets, [c], 450) for c in seen}
        priority = torch.rand(len(train), generator=torch.Generator().manual_seed(9170))
        memory = balanced_memory(train_ids, old, 500, priority)
        current = torch.cat([train_ids[c] for c in order[-1]])
        heldout = ids_by_class(train.targets, seen, 50, True)
        assert not torch.isin(heldout, torch.cat([current, memory])).any()
        id_to_memory = torch.full((len(train),), -1, dtype=torch.long)
        id_to_memory[memory] = torch.arange(len(memory))
        images = torch.stack([evaluation[int(i)][1] for i in memory]).to(device)
        if not (OUT / 'baseline_parity_check.json').exists():
            parity_check(train, current, memory, seen, old, cfg, state, device)
        if args.check_only:
            print('Construction, disjointness, penalty and legacy-training parity checks passed.', flush=True)
            return
        net = model('resnet18', False, num_classes=10).to(device)
        net.load_state_dict(state)
        initial = observe(net, evaluation, heldout, device)
        mem_initial = observe(net, evaluation, memory, device)
        with torch.no_grad():
            reference = torch.cat([features(net, prepare_images(chunk, device))
                                   for chunk in images.split(32)])
            z_check = features(net, prepare_images(images[:32], device))
        zero_error = float((z_check - reference[:32]).abs().max())
        assert zero_error == 0., zero_error
        torch.save({'initial': initial, 'memory_initial': mem_initial,
                    'memory_ids': memory, 'current_ids': current, 'heldout_ids': heldout,
                    'order': order, 'initial_amp_batch_error': zero_error,
                    'penalty_reference_fp32': reference.cpu()}, directory / 'initial.pt')
        for name, coefficient in branches:
            label = name if not coefficient else f'{name}_lambda{coefficient:g}'
            result_path = directory / f'{label}.pt'
            curve_path = directory / f'{label}.csv'
            if result_path.exists():
                print(f'order={order_seed} {label}: already complete', flush=True)
                continue
            checkpoint_path = directory / f'{label}_resume.pt'
            net.load_state_dict(state)
            optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.05)
            scaler = torch.amp.GradScaler('cuda', init_scale=128.)
            rows = [{'order': order_seed, 'condition': name, 'coefficient': coefficient,
                     'epoch': 0, **measure(initial, initial, mem_initial, mem_initial, order, bases),
                     'training_loss': 0., 'training_penalty': 0., 'steps': 0, 'skipped_steps': 0}]
            if checkpoint_path.exists():
                checkpoint = load(checkpoint_path)
                net.load_state_dict(checkpoint['state'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scaler.load_state_dict(checkpoint['scaler'])
                rows = checkpoint['rows']
                del checkpoint
            if len(rows) == 6:
                obs = observe(net, evaluation, heldout, device)
                mem = observe(net, evaluation, memory, device)
            for epoch in range(len(rows), 6):
                started = time.monotonic()
                training = train_epoch(net, optimizer, scaler, train, current, memory, seen,
                                       cfg, device, 500000 + order_seed * 100 + epoch,
                                       bases[name].to(device) if coefficient else None,
                                       coefficient, reference, images, id_to_memory)
                if training['skipped_steps']:
                    raise RuntimeError(f'Numerical overflow in {label}, epoch {epoch}: {training}')
                obs = observe(net, evaluation, heldout, device)
                mem = observe(net, evaluation, memory, device)
                metrics = measure(obs, initial, mem, mem_initial, order, bases)
                row = {'order': order_seed, 'condition': name, 'coefficient': coefficient,
                       'epoch': epoch, **metrics, **training}
                rows.append(row)
                write_rows(curve_path, rows)
                torch.save({'state': cpu_state(net), 'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict(), 'rows': rows}, checkpoint_path)
                print(f'order={order_seed} {label} epoch={epoch}/5 '
                      f'new={metrics["new_accuracy"]:.3f} old={metrics["old_accuracy"]:.3f} '
                      f'penalty={training["training_penalty"]:.5f} '
                      f'seconds={time.monotonic()-started:.1f}', flush=True)
            torch.save({'state': cpu_state(net), 'observations': obs, 'memory': mem,
                        'rows': rows, 'condition': name, 'coefficient': coefficient,
                        'order': order_seed, 'seed': 170}, result_path)
            checkpoint_path.unlink()
            del optimizer, scaler
        del net, images, reference
    manifest['status'] = 'complete'
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orders', type=int, nargs='+', default=[2101, 2111, 2129])
    parser.add_argument('--strengths', type=float, nargs='+', default=[1., 10., 100.])
    parser.add_argument('--conditions', nargs='+', choices=['scaffold', 'pca', 'random1', 'random2'],
                        default=['scaffold', 'pca', 'random1', 'random2'])
    parser.add_argument('--extend', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    run(parser.parse_args())
