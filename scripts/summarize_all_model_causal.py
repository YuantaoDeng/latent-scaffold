"""Read-only aggregation of completed paired directional interventions."""
from package_paths import read_config_text
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path('results/revision_20260913/all_model_causal')

def main():
    jobs=json.loads(read_config_text(ROOT/'jobs.json'))['jobs'];groups={}
    for job in jobs:
        key=(job['mode'],job['short'],job['dataset']);groups.setdefault(key,[]).append(job)
    summaries=[]
    for (mode,model,ds),jj in groups.items():
        done=[j for j in jj if (Path(j['folder'])/'complete.json').exists()]
        epochs=0
        for j in jj:
            for p in Path(j['folder']).glob('*.csv'):
                with p.open() as stream:epochs+=sum(int(r['epoch'])>0 for r in csv.DictReader(stream))
        item=dict(mode=mode,model=model,dataset=ds,completed_histories=len(done),total_histories=len(jj),completed_epochs=epochs,contrasts=[])
        if len(done)==len(jj):
            series={}
            for j in done:
                for p in Path(j['folder']).glob('*.csv'):
                    with p.open() as stream:rr=[r for r in csv.DictReader(stream) if int(r['epoch'])>0]
                    assert len(rr)==5,(p,len(rr))
                    final=rr[-1];name=final['condition'];strength=float(final['strength'])
                    metrics={k:float(v) for k,v in final.items() if k.endswith('_accuracy') or k.endswith('_energy')}
                    metrics['new_acquisition']=sum(float(r['new_accuracy']) for r in rr)/5
                    series[(j['order'],j['seed'],name,strength)]=metrics
            for strength in ((1.,10.,100.) if mode=='suppress' else (1.25,1.5,2.)):
                for comparator in ('random','baseline'):
                    samples=[]
                    for j in done:
                        key=(j['order'],j['seed']);a=series[(*key,'scaffold',strength)]
                        if comparator=='random':bs=[series[(*key,n,strength)] for n in ('random1','random2')]
                        else:bs=[series[(*key,'baseline',0. if mode=='suppress' else 1.)]]
                        samples.append({k:a[k]-sum(b[k] for b in bs)/len(bs) for k in a})
                    for metric in samples[0]:
                        v=np.array([s[metric] for s in samples]).reshape(3,2);rng=np.random.default_rng(8731)
                        orders=rng.integers(0,3,(20000,3));seeds=rng.integers(0,2,(20000,3,2))
                        boot=v[orders[:,:,None],seeds].mean((1,2));lo,hi=np.quantile(boot,[.025,.975])
                        item['contrasts'].append(dict(strength=strength,comparator=comparator,metric=metric,mean=float(v.mean()),ci95=[float(lo),float(hi)]))
        summaries.append(item)
    output=dict(conditions=summaries,reused_suppression='results/revision_20260913/retention_replication/summary.json')
    (ROOT/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    lines=['# All-model causal intervention progress','', 'Completed ResNet/CIFAR-10 suppression is reused from retention_replication; new enhancement is a final-task design.','', '| Mode | Model | Dataset | Histories | Epochs |','|---|---|---|---:|---:|']
    for x in summaries:lines.append(f"| {x['mode']} | {x['model']} | {x['dataset']} | {x['completed_histories']}/{x['total_histories']} | {x['completed_epochs']}/300 |")
    for x in summaries:
        if not x['contrasts']:continue
        lines+=['',f"## {x['mode']} / {x['model']} / {x['dataset']}",'','| Strength | Comparator | Metric | Difference (pp) | Nominal 95% interval (pp) |','|---:|---|---|---:|---|']
        for r in x['contrasts']:
            if r['metric'] not in ('new_acquisition','first_accuracy','new_accuracy'):continue
            lines.append(f"| {r['strength']} | {r['comparator']} | {r['metric']} | {100*r['mean']:+.2f} | [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}] |")
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
