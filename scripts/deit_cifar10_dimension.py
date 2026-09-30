"""Generate six DeiT/CIFAR-10 source trajectories and select joint k=r using matched PCA."""
from package_paths import read_config_text
from pathlib import Path
from types import SimpleNamespace as NS
import json
import hashlib
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
import numpy as np
import sequence_drift_experiment as seq
import analyze_scaffold_followup as f
import resnet_temporal_dimension as r
from deit_three_claims import json_save,csv_save

ROOT=Path('results/deit_cifar10_dimension')
ORDERS=(2003,2011,2017)
BASE_TRAIN=seq.train_task

def tracked_train(*args,**kwargs):
    net,config,seed=args[0],args[8],args[10]
    checkpoint=seed%100+1; train_seed=seed//100
    folder=ROOT/'states'/f'order{config.order_seed}_seed{train_seed}'
    if checkpoint==1: f.save(f.state_dict_cpu(net),folder/'checkpoint0.pt')
    result=BASE_TRAIN(*args,**kwargs)
    f.save(f.state_dict_cpu(net),folder/f'checkpoint{checkpoint}.pt')
    return result

def train():
    seq.train_task=tracked_train
    for order in ORDERS:
        cfg=json.loads(read_config_text(Path('results/cifar10_confirmation')/f'source_order{order}/manifest.json'))
        cfg.update(model='deit_tiny_patch16_224',seeds=[160,161],pretrained=True,
                   initial_checkpoint=None,save_trajectory=True,save_layers=False,
                   save_pre_future_checkpoint=True,force=False,data=Path('data'),
                   output=ROOT/f'source_order{order}',validation_suite=False)
        seq.run(NS(**cfg)); torch.cuda.empty_cache()

def artifacts():
    return [ROOT/f'source_order{o}/artifacts/order{o}_seed{s}.pt' for o in ORDERS for s in (160,161)]

def choose_from_N(curves):
    positive=[x for x in curves if x['r']>0 and x['mean_score']>0]
    result=r.choose(positive if positive else [curves[0]])
    result['zero_gain_within_one_se']=result['threshold']<=0
    return result

def analyze():
    records=[]; inventory=[]
    for p in artifacts():
        a=f.load(p); h=a['historical_drift_centered'].double()
        records.append(dict(order=a['order_seed'],seed=a['seed'],h=r.eigbasis(h.T@h),
                            p=r.eigbasis(a['static_covariance']),future=a['future_drift_centered'].double()))
        inventory.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    rows=[]
    for held in ORDERS:
        fitting=[a for a in records if a['order']!=held]
        for rank in range(1,33):
            h=f.pool([a['h'][:,:rank] for a in fitting],rank)
            p=f.pool([a['p'][:,:rank] for a in fitting],rank)
            for a in records:
                if a['order']!=held: continue
                hc=f.coverage(a['future'],h); pc=f.coverage(a['future'],p)
                rows.append(dict(order_seed=held,seed=a['seed'],k=rank,r=rank,scaffold=hc,matched_static=pc,score=hc-pc))
    def curves(orders):
        result=[dict(k=0,r=0,mean_score=0.,se=0.,order_scores={str(o):0. for o in orders})]
        for rank in range(1,33):
            subset=[x for x in rows if x['r']==rank and x['order_seed'] in orders]
            scores={str(o):float(np.mean([x['score'] for x in subset if x['order_seed']==o])) for o in orders}
            vals=list(scores.values())
            result.append(dict(k=rank,r=rank,mean_score=float(np.mean(vals)),
                               se=float(np.std(vals,ddof=1)/np.sqrt(len(vals))),order_scores=scores,
                               scaffold=float(np.mean([x['scaffold'] for x in subset])),
                               matched_static=float(np.mean([x['matched_static'] for x in subset]))))
        return result
    c=curves(ORDERS); selection=choose_from_N(c)
    selection.update(dataset='cifar10',model='deit_tiny_patch16_224',feature_dimension=192,
                     construction='joint k=r, 1..32',curves=c,
                     N_mean_positive=[x['r'] for x in c if x['r']>0 and x['mean_score']>0],
                     N_all_orders_positive=[x['r'] for x in c if x['r']>0 and min(x['order_scores'].values())>0])
    selection['N_one_se_eligible']=[x['r'] for x in c if x['r']>0 and x['mean_score']>0 and x['mean_score']>=selection['threshold']-1e-14]
    selection['validation_order_deletions']={str(o):choose_from_N(curves([v for v in ORDERS if v!=o])) for o in ORDERS}
    csv_save(ROOT/'selection_folds.csv',rows); json_save(ROOT/'selection.json',selection); json_save(ROOT/'inventory.json',inventory)
    print('SELECTION',json.dumps({k:v for k,v in selection.items() if k not in ('curves','validation_order_deletions')}),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    ROOT.mkdir(parents=True,exist_ok=True)
    train(); analyze()
