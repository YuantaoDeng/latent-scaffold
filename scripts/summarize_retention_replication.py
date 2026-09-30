"""Report all fixed contrasts; never label incomplete replication confirmatory."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('results/revision_20260913/retention_replication')
ROOT.mkdir(parents=True,exist_ok=True)
def read(p):return list(csv.DictReader(p.open()))
orders=[2131,2137,2141];seeds=[170,171];rows=[];complete=[]
for order in orders:
    for seed in seeds:
        folder=ROOT/f'order{order}_seed{seed}'
        if (folder/'complete.json').exists():complete.append((order,seed))
        for lam in (1,10,100):
            paths=[folder/f'{n}_lambda{lam}.csv' for n in ('scaffold','random1','random2')]
            if not all(p.exists() for p in paths):continue
            runs=[read(p) for p in paths]
            if any(len(r)!=6 for r in runs):continue
            assert all(int(float(r['skipped_steps']))==0 for run in runs for r in run)
            assert len({r[0]['first_accuracy'] for r in runs})==1
            for metric in ('first_accuracy','old_accuracy','new_accuracy','first_total_energy'):
                a=float(runs[0][-1][metric]);b=np.mean([float(r[-1][metric]) for r in runs[1:]])
                rows.append(dict(order=order,seed=seed,coefficient=lam,metric=metric,scaffold=a,random=b,difference=a-b))
if rows:
    with (ROOT/'paired_contrasts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(complete_histories=len(complete),expected_histories=6,status='complete' if len(complete)==6 else 'running',
    primary='Final first-task accuracy, scaffold suppression minus mean random suppression, coefficient10',
    limitations='Three orders/two seeds; same coefficient is not equal absolute suppression; one model/dataset; no amplification test.',contrasts=[])
if len(complete)==6:
    rng=np.random.default_rng(90213)
    order_indices=rng.integers(0,3,size=(20000,3));run_indices=rng.integers(0,2,size=(20000,3,2))
    for lam in (1,10,100):
        for metric in ('first_accuracy','old_accuracy','new_accuracy','first_total_energy'):
            selected=[r for r in rows if r['coefficient']==lam and r['metric']==metric]
            values=np.array([[next(r['difference'] for r in selected if r['order']==o and r['seed']==s) for s in seeds] for o in orders])
            samples=values[order_indices[:,:,None],run_indices].mean((1,2));ci=np.quantile(samples,[.025,.975])
            report['contrasts'].append(dict(coefficient=lam,metric=metric,mean=float(values.mean()),ci95=ci.tolist(),positive_runs=int((values>0).sum()),negative_runs=int((values<0).sum())))
    primary=next(r for r in report['contrasts'] if r['coefficient']==10 and r['metric']=='first_accuracy')
    report['primary_supported']=primary['ci95'][1]<0
    report['interpretation']='Scaffold suppression hurts first-task retention more than the specified random constraint under this protocol.' if report['primary_supported'] else 'The prespecified replication does not establish greater first-task impairment from scaffold suppression.'
else:report['interpretation']='Primary inference withheld until all six histories and all fixed branches finish.'
(ROOT/'summary.json').write_text(json.dumps(report,indent=2))
lines=['# First-task retention replication',f"\nStatus: {report['status']}; {len(complete)}/6 histories complete.", '\n'+report['interpretation'],'\nPrimary: '+report['primary'],'\n'+report['limitations']]
if report['contrasts']:
    lines+=['\n| Coefficient | Metric | Scaffold minus random (pp) | 95% interval (pp) |','|---:|---|---:|---|']
    for r in report['contrasts']:
        if not r['metric'].endswith('accuracy'):continue
        lines.append(f"| {r['coefficient']} | {r['metric']} | {100*r['mean']:+.2f} | [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}] |")
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(retention_status=report['status'],complete_histories=len(complete))))
