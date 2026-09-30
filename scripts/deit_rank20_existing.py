"""User-fixed k=r=20: complete existing-data DeiT/CIFAR-100 reanalysis, no training."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import csv
import numpy as np
import torch
import analyze_scaffold_followup as f
import deit_three_claims as d
import resnet_temporal_dimension as r

BASE=Path('results/deit_rank20_existing')
OUT=BASE/'cifar100'


def configure():
    r.ROOT=BASE; r.DATA={'cifar100':d.ORIGINAL}
    r.entries=lambda ds,role='source': [dict(path=str(p),role=role) for p,_ in d.inventory(role)]


def main():
    configure(); hashes=[]
    for role in ('source','target'):
        for p,_ in d.inventory(role): hashes.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    d.json_save(OUT/'original_hashes.json',hashes)
    for p,_ in d.inventory('source'):
        a=f.load(d.ROOT/'responses/artifacts'/p.name); b=f.load(d.ROOT/'responses64/artifacts'/p.name)
        merged=deepcopy(a)
        for cp in (0,2):
            merged['states'][cp]['prefixes'][64]={**b['states'][cp]['prefixes'][64], 'covariance':b['states'][cp]['covariance']}
            # Reuse any completed missing-prefix recovery; never measure new responses here.
            extra=Path('results/deit_pca_dimension/cifar100/responses/artifacts')/p.name
            if extra.exists():
                existing=f.load(extra)
                if cp in existing['states']:
                    for count in (8,32):
                        merged['states'][cp]['prefixes'][count]=existing['states'][cp]['prefixes'][count]
        f.save(merged,OUT/'responses/artifacts'/p.name)
    for rank in (8,20):
        label=f'rank{rank}'
        for count in (16,):
            name=label if count==16 else label+'_response64'
            if not (OUT/name/'geometry.json').exists(): r.geometry('cifar100',rank,rank,name,response_count=count,checkpoints=[0,2])
        if not (OUT/label/'evaluation.json').exists():
            r.evaluate('cifar100',label,target_root=d.ROOT/'targets/artifacts',fields_root=d.ROOT/'update_fields')
    overlaps=[]
    for rank in (8,20):
        banks={16:f.load(OUT/f'rank{rank}'/'predictors.pt')}
        for count,bank in banks.items():
            for left,u in bank.items():
                for right,v in bank.items(): overlaps.append({'rank':rank,'draws_left':count,'draws_right':count,'left':left,'right':right,'overlap':f.overlap(u,v)})
    d.csv_save(OUT/'all_existing_overlaps.csv',overlaps)
    for a in hashes: assert hashlib.sha256(Path(a['path']).read_bytes()).hexdigest()==a['sha256']
    print('EXISTING-DATA ANALYSES COMPLETE; no training or new parameter perturbations',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2); main()
