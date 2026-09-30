"""Update DeiT/CIFAR-10 to the current minimum-positive rule starting at 8.

Reuse trajectories, response covariances and captured updates. Preserve rank-4
results. No training or perturbation resampling is performed.
"""
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np
import torch
import final_rank_experiments as experiment
import resnet_temporal_dimension as analysis
import analyze_scaffold_followup as common

BASE=Path('results/final_rank_rule/deit/cifar10')

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    selection_path=Path('results/deit_cifar10_dimension/selection_folds.csv')
    rows=list(csv.DictReader(selection_path.open()))
    curves=[]
    for r in range(8,33):
        rr=[x for x in rows if int(x['r'])==r and int(x['k'])==r]
        assert len(rr)==6
        curves.append(dict(rank=r,mean_score=float(np.mean([float(x['score']) for x in rr]))))
    chosen=next(x['rank'] for x in curves if x['mean_score']>0)
    label=f'rank{chosen}';out=BASE/label
    inputs=experiment.c.artifacts()+experiment.targets()+list((BASE/'responses/artifacts').glob('*.pt'))+list((BASE/'targets/artifacts').glob('*.pt'))+list((BASE/'update_fields').glob('*.pt'))
    hashes=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs]
    if not (BASE/'selection_fixed_rank4.json').exists():
        shutil.copy2(BASE/'selection.json',BASE/'selection_fixed_rank4.json')
    current=dict(selected=chosen,k=chosen,r=chosen,rule='smallest rank >= 8 with positive mean held-out estimation-order coverage gain over pooled representation PCA',
                 evaluated_selection_ranks=[8,32],source_selection=str(selection_path),source_selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                 response_draws=16,curves=curves,supersedes='selection_fixed_rank4.json')
    (BASE/'selection.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
    analysis.ROOT=Path('results/final_rank_rule/deit')
    analysis.entries=lambda ds,role='source':[dict(path=str(p)) for p in (experiment.c.artifacts() if role=='source' else experiment.targets())]
    print('Selected rank',chosen,'using estimation histories only',flush=True)
    analysis.geometry('cifar10',chosen,chosen,label,response_count=16)
    experiment.all_overlaps(out)
    analysis.evaluate('cifar10',label,target_root=BASE/'targets/artifacts',fields_root=BASE/'update_fields')
    # Independently recover the pooled projector using an eigen-decomposition.
    sources=analysis.source_data('cifar10')
    m=sum(a['history'][:,:chosen]@a['history'][:,:chosen].T for a in sources)/len(sources)
    u=torch.linalg.eigh(m).eigenvectors[:,-chosen:]
    predictors=common.load(out/'predictors.pt')
    projector_error=float((u@u.T-predictors['scaffold']@predictors['scaffold'].T).abs().max())
    assert projector_error<1e-9
    temporal=list(csv.DictReader((out/'temporal.csv').open()))
    errors=[]
    for p in experiment.targets():
        a=common.load(BASE/'targets/artifacts'/p.name)
        for i in range(1,len(a['eval_reps'])):
            field=(a['eval_reps'][i]-a['eval_reps'][i-1]).double();field-=field.mean(0,keepdim=True)
            score=float(torch.trace(u.T@(field.T@field)@u)/field.square().sum())
            row=next(x for x in temporal if int(x['order_seed'])==a['order_seed'] and int(x['seed'])==a['seed'] and x['transition']==f'{i}_{i+1}')
            errors.append(abs(score-float(row['scaffold'])))
    assert max(errors)<1e-9 and len(errors)==48
    for item in hashes:assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    (out/'verification.json').write_text(json.dumps(dict(selected=chosen,independent_projector_error=projector_error,
        independent_temporal_coverage_max_error=max(errors),checked_temporal_rows=len(errors),unchanged_inputs=hashes),indent=2),encoding='utf-8')
    print('Verified new rank results and unchanged original inputs.',flush=True)

if __name__=='__main__':main()
