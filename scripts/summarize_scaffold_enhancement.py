"""Complete-case paired endpoint analysis of all fixed enhancement strengths."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path('results/revision_20260913/enhancement');ROOT.mkdir(parents=True,exist_ok=True)
orders=(2131,2137,2141);seeds=(170,171);gains=(1.25,1.5,2.)
rows=[];complete=0;epochs=0
for folder in ROOT.glob('order*'):
    for branch in folder.iterdir():
        if not branch.is_dir():continue
        complete+=(branch/'complete.json').exists()
        p=branch/'curve.csv'
        if p.exists():
            values=list(csv.DictReader(p.open()));epochs+=len(values)
            rows.extend(values)
if rows:
    with (ROOT/'all_task_epoch_accuracy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(status='complete' if complete==60 else 'running',completed_trajectories=complete,total_trajectories=60,
    completed_epochs=epochs,total_epochs=900,primary='Task-one accuracy after task-five, gain1.5 scaffold minus two-random mean',
    companion='Scaffold minus normal training; needed to establish an improvement over normal learning',contrasts=[])
if complete==60:
    assert len(rows)==900
    assert all(int(r['skipped_steps'])==0 and int(r['steps'])==29 for r in rows)
    rng=np.random.default_rng(92316);oi=rng.integers(0,3,(20000,3));ri=rng.integers(0,2,(20000,3,2))
    def value(o,s,n,g,t,metric):
        r=next(r for r in rows if int(r['order'])==o and int(r['seed'])==s and r['condition']==n and float(r['gain'])==g and int(r['task'])==t and int(r['epoch'])==5)
        return float(r[metric])
    for g in gains:
        for t in (3,4,5):
            for metric in [f'task{j}_accuracy' for j in range(1,t+1)]+['first_total_energy','first_scaffold_energy','first_scaffold_coverage']:
                for comparator in ('random','baseline'):
                    def diff(o,s):
                        ref=np.mean([value(o,s,n,g,t,metric) for n in ('random1','random2')]) if comparator=='random' else value(o,s,'baseline',1,t,metric)
                        return value(o,s,'scaffold',g,t,metric)-ref
                    matrix=np.array([[diff(o,s) for s in seeds] for o in orders]);samples=matrix[oi[:,:,None],ri].mean((1,2))
                    report['contrasts'].append(dict(gain=g,after_task=t,metric=metric,comparator=comparator,mean=float(matrix.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),positive_runs=int((matrix>0).sum())))
    def primary(comp):return next(r for r in report['contrasts'] if r['gain']==1.5 and r['after_task']==5 and r['metric']=='task1_accuracy' and r['comparator']==comp)
    a,b=primary('random'),primary('baseline')
    report['retention_improvement_supported']=a['ci95'][0]>0 and b['ci95'][0]>0
    report['interpretation']='Both prespecified task-one contrasts favor enhancement at nominal 95% intervals; examine actual-motion manipulation checks and limited scope.' if report['retention_improvement_supported'] else 'The fixed primary contrasts do not establish improved first-task retention over both random enhancement and normal training.'
    # Actual observed optimizer-step motion is measured separately from gradient gain.
    import torch
    torch.set_num_threads(2);motion=[]
    for folder in ROOT.glob('order*'):
        for branch in folder.iterdir():
            if not branch.is_dir():continue
            for path in branch.glob('task[345].pt'):
                v=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                for step in v['updates']:
                    motion.append(dict(history=folder.name,condition=branch.name,task=step['task'],step=step['step'],total_energy=step['total_energy'],scaffold_energy=step['scaffold_energy'],scaffold_coverage=step['scaffold_coverage']))
    assert len(motion)==540,len(motion)
    with (ROOT/'actual_update_motion.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(motion[0]));w.writeheader();w.writerows(motion)
    report['actual_update_observations']=len(motion)
else:report['interpretation']='Awaiting all fixed trajectories; no complete primary inference yet.'
(ROOT/'summary.json').write_text(json.dumps(report,indent=2))
lines=['# Scaffold-directed enhancement',f"\n{complete}/60 trajectories, {epochs}/900 epochs.",'\n'+report['interpretation'],
    '\nThe intervention amplifies feature-space gradients passed to the encoder. It does not guarantee the same gain in realized feature drift. Random controls match added-gradient norm at the same current gradient, not realized parameter/feature motion.',
    '\nThree orders, two seeds, ResNet-18/CIFAR-10. Nominal hierarchical intervals; all strength and task comparisons are retained.']
if report['contrasts']:
    lines+=['\n| Gain | After task | Evaluated task | Comparator | Accuracy difference (pp) | 95% interval (pp) |','|---:|---:|---|---|---:|---|']
    for r in report['contrasts']:
        if not r['metric'].endswith('accuracy'):continue
        lines.append(f"| {r['gain']} | {r['after_task']} | {r['metric']} | {r['comparator']} | {100*r['mean']:+.2f} | [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}] |")
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(enhancement=report['status'],complete=complete,epochs=epochs)))
