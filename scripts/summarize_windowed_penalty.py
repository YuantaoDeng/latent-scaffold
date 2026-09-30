import argparse,csv,json
from pathlib import Path
import numpy as np
ROOT=Path('results/revision_20260913/windowed_penalty')
def ci(values):
    v=np.array(values).reshape(3,2);rng=np.random.default_rng(8731);o=rng.integers(0,3,(20000,3));s=rng.integers(0,2,(20000,3,2));b=v[o[:,:,None],s].mean((1,2))
    return dict(mean=float(v.mean()),ci95=[float(x) for x in np.quantile(b,[.025,.975])])
def main(model='resnet18'):
    global ROOT
    if model!='resnet18':ROOT=Path('results/revision_20260913')/('windowed_penalty_mobilenet_cifar10' if model=='mobilenetv3_small' else 'windowed_penalty_deit_cifar10')
    folders=[ROOT/f'order{o}_seed{s}' for o in (2131,2137,2141) for s in (170,171)]
    branches=[('baseline','none')]+[(n,w) for w in ('early','middle','late','full') for n in ('scaffold','random1','random2')]
    completed=sum((p/f'{n}_{w}.pt').exists() for p in folders for n,w in branches)
    report=dict(completed_branches=completed,total_branches=78,status='complete' if completed==78 else 'running',contrasts=[],timing_contrasts=[])
    lines=[f'# {model}/CIFAR10 windowed penalty','',f'{completed}/78 branches complete.','', 'Equal48-update windows; lambda10; reference remains the pre-final-task representation. Later activation can also reverse already accumulated drift. No inference from partial histories.']
    if completed==78:
        series={}
        for i,p in enumerate(folders):
            for n,w in branches:
                with (p/f'{n}_{w}.csv').open() as f:rows=list(csv.DictReader(f))
                end=next(r for r in rows if int(r['step'])==145)
                series[i,n,w]={k:float(v) for k,v in end.items() if k.endswith('_accuracy') or k.endswith('_energy')}
                series[i,n,w]['new_acquisition']=sum(float(r['task5_accuracy']) for r in rows if int(r['step']) in (29,58,87,116,145))/5
        paired={}
        for w in ('early','middle','late','full'):
            for comp in ('baseline','random'):
                for metric in series[0,'baseline','none']:
                    vals=[]
                    for i in range(6):
                        b=series[i,'baseline','none'][metric] if comp=='baseline' else (series[i,'random1',w][metric]+series[i,'random2',w][metric])/2
                        vals.append(series[i,'scaffold',w][metric]-b)
                    paired[w,comp,metric]=vals;report['contrasts'].append(dict(window=w,comparator=comp,metric=metric,**ci(vals)))
        for comp in ('baseline','random'):
            for metric in ('new_acquisition','task5_accuracy','task1_accuracy'):
                vals=np.array(paired['early',comp,metric])-np.array(paired['late',comp,metric])
                report['timing_contrasts'].append(dict(comparison='early minus late intervention effect',comparator=comp,metric=metric,**ci(vals)))
        lines+=['','| Window | Comparator | Metric | Difference pp | 95% interval pp |','|---|---|---|---:|---|']
        for r in report['contrasts']:
            if r['metric'] not in ('new_acquisition','task1_accuracy','task5_accuracy'):continue
            lines.append(f"| {r['window']} | {r['comparator']} | {r['metric']} | {100*r['mean']:+.2f} | [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}] |")
        lines+=['','## Paired timing comparisons','',json.dumps(report['timing_contrasts'],indent=2)]
    ROOT.mkdir(exist_ok=True,parents=True);(ROOT/'summary.json').write_text(json.dumps(report,indent=2));(ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='resnet18',choices=['resnet18','mobilenetv3_small','deit_tiny']);main(p.parse_args().model)
