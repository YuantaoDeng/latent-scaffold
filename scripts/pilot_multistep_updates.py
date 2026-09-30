"""Timed ResNet/CIFAR-10 final-task continuation; preserves published experiments."""
from package_paths import read_config_text
import argparse
import hashlib
import json
import time
import math
from pathlib import Path
from types import SimpleNamespace

import analyze_scaffold_followup as f
import torch
from drift_experiment import IndexedCifar, balanced_memory, ids_by_class, model, seed_all, train_task
from phase_c_direction import state_dict_cpu, functional_bases
from resnet_temporal_dimension import controls

OUT = Path('results/multistep_update_pilot/resnet18_cifar10')
ROOT = Path('results/cifar10_confirmation')

class SamplingComplete(Exception):
    pass

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

class Observer:
    def __init__(self, directory, metadata, evaluation, scaffold, projector):
        self.directory, self.metadata, self.evaluation = directory, metadata, evaluation
        self.scaffold, self.projector = scaffold, projector
        self.windows = {s: w for w, steps in [('early', [2,3,4]), ('middle', [65,66,67]), ('late', [125,126,127])] for s in steps}
        self.rows, self.pending = [], None
        self.measurement_seconds = 0.
        self.attempts = 0
        self.stop_after_last = False

    def __call__(self, event, epoch, batch, net, optimizer, ids, x, y, n_current):
        started = time.perf_counter()
        steps = max((int(s.get('step', 0)) for s in optimizer.state.values()), default=0)
        if event == 'after' and steps % 25 == 0:
            write(self.directory/'progress.json', {'successful_step':steps,'epoch':epoch+1,'batch_index':batch,'unix_time':time.time()})
        if event == 'before':
            self.attempts += 1
            self.pending = state_dict_cpu(net) if steps + 1 in self.windows else None
        elif self.pending is not None and steps in self.windows and not any(r['step'] == steps for r in self.rows):
            after = state_dict_cpu(net)
            captured = {'before_state': self.pending, 'after_state': after,
                        'gradient': {n: p.grad.detach().cpu().clone() for n,p in net.named_parameters() if p.grad is not None}}
            if self.metadata['manifest']['model'].startswith('mixer_'):
                torch.cuda.empty_cache()
            # Evaluation constructs a separate model. Restore RNG so observation cannot alter training.
            rng = f.rng_state()
            record = {**self.metadata, 'actual_update': captured}
            if self.metadata['manifest']['model'].startswith('mixer_'):
                from mixer_update_measurement import measure
                verification_path=self.directory/'inplace_measurement_verification.json'
                if not verification_path.exists():
                    verification_record={**record,'heldout_ids':record['heldout_ids'][:64]}
                    reference,reference_norm=f.actual_update_fields(verification_record,self.evaluation,torch.device('cuda'))
                    f.restore_rng(rng)
                    torch.cuda.empty_cache()
                    checked,checked_norm=measure(net,verification_record,self.evaluation)
                    errors={name:float((checked[name]-reference[name]).abs().max()) for name in checked}
                    state_error=max(float((net.state_dict()[name].detach().cpu().double()-value.double()).abs().max()) for name,value in after.items())
                    gradient_error=max(float((p.grad.detach().cpu()-captured['gradient'][name]).abs().max()) for name,p in net.named_parameters() if p.grad is not None)
                    assert max(errors.values())==0 and checked_norm==reference_norm
                    assert state_error==0 and gradient_error==0 and net.training
                    write(verification_path,{'verification_anchors':len(verification_record['heldout_ids']),'measurement_anchors':len(record['heldout_ids']),'field_max_errors':errors,'relative_step_norm_identical':True,'restored_state_max_error':state_error,'gradient_max_error':gradient_error,'training_mode_restored':True})
                    del reference,checked,verification_record
                    torch.cuda.empty_cache()
                fields,relative_norm=measure(net,record,self.evaluation)
            else:
                fields, relative_norm = f.actual_update_fields(record, self.evaluation, torch.device('cuda'))
            f.restore_rng(rng)
            row = {'step': steps, 'window': self.windows[steps], 'epoch': epoch+1, 'batch_index': batch,
                   'attempt': self.attempts, 'relative_encoder_step_norm': relative_norm}
            max_error = 0.
            for name, field in fields.items():
                energy = field.square().sum()
                value = f.coverage(field, self.scaffold)
                explicit = float((field @ self.scaffold @ self.scaffold.T).square().sum()/energy)
                max_error = max(max_error, abs(value-explicit))
                random_value = float(((field @ self.projector)*field).sum()/energy)
                row[name] = {'coverage': value, 'random_coverage': random_value, 'gain': value-random_value, 'energy': float(energy)}
            assert max_error < 1e-10
            row['projection_identity_max_error'] = max_error
            f.save({'fields': fields, 'row': row, 'batch_ids': ids.cpu(), 'heldout_ids': self.metadata['heldout_ids']}, self.directory/f'step{steps:03d}.pt')
            self.rows.append(row)
            write(self.directory/'measurements.json', self.rows)
            print(json.dumps({'step': steps, 'window': row['window'], 'coverage': row['adamw_full']['coverage'], 'gain': row['adamw_full']['gain']}), flush=True)
            self.pending = None
            if self.metadata['manifest']['model'].startswith('mixer_'):
                del captured, after, record
                torch.cuda.empty_cache()
        self.measurement_seconds += time.perf_counter()-started
        if self.stop_after_last and len(self.rows)==len(self.windows):
            raise SamplingComplete()

def run(order_seed, seed, *, dataset='cifar10', source_path=None, scaffold_path=None, output_root=None, stop_after_last=False):
    directory = (output_root or OUT)/f'order{order_seed}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'summary.json').exists():
        print('Completed:', directory, flush=True)
        return
    started = time.perf_counter()
    source = source_path or ROOT/f'target_order{order_seed}/artifacts/order{order_seed}_seed{seed}.pt'
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original = f.load(source)
    manifest = json.loads(read_config_text(source.parent.parent/'manifest.json'))
    cfg = SimpleNamespace(**manifest)
    recovered_source = None
    recovered_hash = None
    if 'pre_future_state_dict' not in original:
        recovered_source = source.parent.parent.parent/'scaffold_followup/target_recovery/artifacts'/source.name
        recovered_hash = hashlib.sha256(recovered_source.read_bytes()).hexdigest()
        recovered = f.load(recovered_source)
        original['pre_future_state_dict'] = recovered['states'][len(cfg.order)-1]
        del recovered
    train = IndexedCifar(Path(cfg.data), True, True, dataset)
    evaluation = IndexedCifar(Path(cfg.data), True, False, dataset)
    device = torch.device('cuda')
    net = model(cfg.model, False, num_classes=train.num_classes).to(device)
    net.load_state_dict(original['pre_future_state_dict'])
    use_activation_recomputation = cfg.model.startswith('mixer_')
    if use_activation_recomputation:
        net.set_grad_checkpointing(True)
    scaffold_path = scaffold_path or Path('results/final_rank_rule/resnet/cifar10/rank8/predictors.pt')
    scaffold = f.load(scaffold_path)['scaffold'].double()
    order = cfg.order
    old, seen = sum(order[:-1], []), sum(order, [])
    train_ids = {c: ids_by_class(train.targets, [c], cfg.train_per_class) for c in seen}
    priority = torch.rand(len(train), generator=torch.Generator().manual_seed(seed+9000))
    memory = balanced_memory(train_ids, old, cfg.memory, priority)
    current = torch.cat([train_ids[c] for c in order[-1]])
    heldout = ids_by_class(evaluation.targets, order[0], cfg.probe_per_class, tail=True)
    assert not torch.isin(heldout, torch.cat(list(train_ids.values()))).any()
    weight = net.get_classifier().weight.detach().cpu()[old]
    disc, null, _, _ = functional_bases(weight, 1e-7)
    bases, error = controls(scaffold, disc, null, order_seed*1000003+seed*10007+6000000)
    projector = sum(u@u.T for u in bases)/len(bases)
    metadata = {'manifest': manifest, 'heldout_ids': heldout}
    observer = Observer(directory, metadata, evaluation, scaffold, projector)
    observer.stop_after_last = stop_after_last
    total_attempts = cfg.epochs*math.ceil(len(current)/(cfg.batch_size//2))
    middle, late = round(total_attempts*65/145), round(total_attempts*125/145)
    observer.windows = {s:w for w,steps in [('early',range(2,5)),('middle',range(middle,middle+3)),('late',range(late,late+3))] for s in steps}
    protocol = {'model': cfg.model, 'dataset': dataset, 'order_seed': order_seed, 'seed': seed,
                'task': len(order), 'epochs': cfg.epochs, 'successful_steps': observer.windows,
                'planned_attempts': total_attempts,
                'stop_after_last_observation': stop_after_last,
                'activation_recomputation': use_activation_recomputation,
                'source': str(source), 'source_sha256': original_hash, 'scaffold_source': str(scaffold_path),
                'recovered_state_source': str(recovered_source) if recovered_source else None, 'recovered_state_sha256': recovered_hash,
                'rank': scaffold.shape[1], 'random_draws': 128, 'random_matching_classifier': 'old classes immediately before final task',
                'continuation_rng_seed': 800000+order_seed*100+seed,
                'exact_original_trajectory_replay': False,
                'note': 'Saved pre-final-task model; fresh deterministic final-task augmentation RNG; original task-boundary AdamW reset protocol. Histories, not neighboring steps, are units of replication.',
                'device': torch.cuda.get_device_name(), 'torch': torch.__version__}
    write(directory/'protocol.json', protocol)
    del original, bases
    seed_all(protocol['continuation_rng_seed'])
    setup_seconds = time.perf_counter()-started
    train_started = time.perf_counter()
    try:
        train_task(net, cfg.continual_method, train, current, memory, {}, seen, old, cfg, device, seed*100+len(order)-1, step_observer=observer)
    except SamplingComplete:
        assert stop_after_last and len(observer.rows)==9
    torch.cuda.synchronize()
    training_and_measurement = time.perf_counter()-train_started
    assert len(observer.rows) == 9, len(observer.rows)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    if recovered_source:
        assert hashlib.sha256(recovered_source.read_bytes()).hexdigest() == recovered_hash
    summary = {'setup_seconds': setup_seconds, 'training_and_measurement_seconds': training_and_measurement,
               'measurement_seconds': observer.measurement_seconds,
               'training_excluding_observer_seconds': training_and_measurement-observer.measurement_seconds,
               'total_seconds': time.perf_counter()-started, 'attempted_updates': observer.attempts,
               'random_control_max_error': error, 'windows': {}}
    for window in ['early', 'middle', 'late']:
        rows = [r for r in observer.rows if r['window']==window]
        summary['windows'][window] = {method: {key: sum(r[method][key] for r in rows)/len(rows) for key in ['coverage','random_coverage','gain']} for method in ['adamw_full','adamw_parameters','norm_matched_training_gradient']}
    write(directory/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orders', type=int, nargs='+', default=[2101])
    parser.add_argument('--seeds', type=int, nargs='+', default=[170])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    for order in args.orders:
        for seed in args.seeds:
            run(order, seed)
