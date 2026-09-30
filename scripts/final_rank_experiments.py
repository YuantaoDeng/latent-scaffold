"""User-fixed DeiT ranks and minimum-positive matched-PCA ResNet reanalysis."""
from package_paths import read_config_text
from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import hashlib
import json
import shutil
import numpy as np
import torch
import analyze_scaffold_followup as f
import sequence_drift_experiment as seq
import resnet_temporal_dimension as r
import deit_cifar10_dimension as c
from deit_three_claims import json_save, csv_save

ROOT=Path('results/final_rank_rule')
OUT=ROOT/'deit/cifar10'
DATA={'cifar10':Path('results/cifar10_confirmation'),'cifar100':Path('results/confirmatory')}
BASE_TRAIN=seq.train_task

def target_manifests():
    inv=json.loads(read_config_text(DATA['cifar10']/'scaffold_structure/inventory.json'))
    return sorted({Path(e['manifest_path']) for e in inv if e['role']=='target'})

def tracked_train(*args,**kwargs):
    net,config,seed=args[0],args[8],args[10]
    cp=seed%100+1; folder=OUT/'captures'/f'order{config.order_seed}_seed{seed//100}'
    observer=f.CaptureUpdate() if cp==config.tasks else None
    if observer is not None: kwargs['step_observer']=observer
    result=BASE_TRAIN(*args,**kwargs)
    if observer is not None:
        assert 'after_state' in observer.record
        f.save(observer.record,folder/'actual_update.pt')
        f.save(f.state_dict_cpu(net),folder/'final_state.pt')
    return result

def train_targets():
    seq.train_task=tracked_train
    for path in target_manifests():
        cfg=json.loads(read_config_text(path))
        cfg.update(model='deit_tiny_patch16_224',pretrained=True,initial_checkpoint=None,
                   save_trajectory=True,save_layers=False,save_pre_future_checkpoint=True,
                   force=False,data=Path('data'),output=OUT/path.parent.name,validation_suite=False)
        # This auxiliary per-trajectory analysis rank does not constrain training.
        cfg['rank']=4
        seq.run(NS(**cfg)); torch.cuda.empty_cache()
    seq.train_task=BASE_TRAIN

def targets():
    return sorted(OUT.glob('target_order*/artifacts/*.pt'))

def target_fields():
    evaluation=f.IndexedCifar(Path('data'),True,False,'cifar10')
    for path in targets():
        dst=OUT/'targets/artifacts'/path.name
        if not dst.exists():
            a=f.load(path); cap=OUT/'captures'/path.stem
            rec={'status':'complete','dataset':'cifar10','original_path':str(path),
                 'manifest':json.loads(read_config_text(path.parent.parent/'manifest.json')),
                 'order_seed':a['order_seed'],'seed':a['seed'],'fit_ids':a['fit_ids'],
                 'heldout_ids':a['eval_ids'],'heldout_labels':a['eval_labels'],
                 'eval_reps':a['eval_representations'],'fit_reps':a['fit_representations'],
                 'states':{4:a['pre_future_state_dict'],5:f.load(cap/'final_state.pt')},
                 'actual_update':f.load(cap/'actual_update.pt')}
            f.save(rec,dst)
        cache=OUT/'update_fields'/path.name
        if not cache.exists():
            fields,norm=f.actual_update_fields(f.load(dst),evaluation,torch.device('cuda'))
            assert torch.equal(fields['adamw_full'],fields['adamw_parameters'])
            f.save({'fields':fields,'relative_step':norm},cache)
        print('C10 target/update ready',path.stem,flush=True)

def responses():
    evaluation=f.IndexedCifar(Path('data'),True,False,'cifar10')
    for path in c.artifacts():
        a=f.load(path); dst=OUT/'responses/artifacts'/path.name
        result=f.load(dst) if dst.exists() else dict(directions=16,relative_step=1e-4,
                    order_seed=a['order_seed'],seed=a['seed'],states={})
        if len(result['states'])==6: continue
        net=f.model('deit_tiny_patch16_224',False,num_classes=10).cuda()
        batches=f.image_batches(net,evaluation,a['fit_ids'],'cuda')
        head={id(p) for p in net.get_classifier().parameters()}; params=dict(net.named_parameters())
        names=[n for n,p in params.items() if id(p) not in head]
        for cp in range(6):
            if cp in result['states']: continue
            state=f.load(c.ROOT/'states'/path.stem/f'checkpoint{cp}.pt')
            norm=torch.sqrt(sum(state[n].double().square().sum() for n in names))
            gen=torch.Generator().manual_seed(a['order_seed']*100003+a['seed']+701)
            cov=torch.zeros(192,192,dtype=torch.float64); prefixes={}
            for draw in range(16):
                direction={n:torch.randn(state[n].shape,generator=gen) for n in names}
                length=torch.sqrt(sum(v.double().square().sum() for v in direction.values()))
                scale=float(1e-4*norm/length); values=[]
                for sign in (1,-1):
                    net.load_state_dict(state)
                    with torch.no_grad():
                        for n in names: params[n].add_(direction[n].cuda(),alpha=sign*scale)
                    values.append(f.cached_representations(net,batches))
                field=f.center((values[0].double()-values[1].double())/2); cov+=field.T@field
                if draw+1 in (8,16):
                    eig,v=torch.linalg.eigh(cov/(draw+1))
                    prefixes[draw+1]={'covariance':cov.clone()/(draw+1),'basis':v[:,-8:],'spectrum':eig.flip(0)}
            result['states'][cp]={'covariance':prefixes[16]['covariance'],'prefixes':prefixes}
            f.save(result,dst); print('C10 response16',path.stem,'checkpoint',cp,flush=True)
        del net,batches; torch.cuda.empty_cache()

def all_overlaps(folder):
    bank=f.load(folder/'predictors.pt')
    csv_save(folder/'all_pooled_overlaps.csv',[{'left':n,'right':m,'response_draws':16,
             'overlap':f.overlap(u,v)} for n,u in bank.items() for m,v in bank.items()])

def deit10():
    # Current user-directed rule supersedes the historical fixed rank of four.
    # Existing training, response, and update caches are reused when present.
    OUT.mkdir(parents=True,exist_ok=True)
    responses()
    train_targets()
    target_fields()
    from recompute_deit_cifar10_current_rank import main as reanalyze
    reanalyze()


def resnet():
    r.ROOT=ROOT/'resnet'; r.DATA=DATA
    r.entries=lambda ds,role='source':[e for e in json.loads(read_config_text(DATA[ds]/'scaffold_structure/inventory.json')) if e['role']==role]
    for ds in ('cifar100','cifar10'):
        records=r.source_data(ds); orders=sorted({a['order'] for a in records}); rows=[]
        for rank in range(8,33):
            for held in orders:
                fitting=[a for a in records if a['order']!=held]
                h=f.pool([a['history'][:,:rank] for a in fitting],rank)
                p=f.pool([a['static'][:,:rank] for a in fitting],rank)
                for a in records:
                    if a['order']!=held: continue
                    hc=f.coverage(a['future'],h); pc=f.coverage(a['future'],p)
                    rows.append(dict(order_seed=held,seed=a['seed'],k=rank,r=rank,scaffold=hc,matched_static=pc,score=hc-pc))
        curves=[dict(r=n,k=n,mean_score=float(np.mean([v['score'] for v in rows if v['r']==n]))) for n in range(8,33)]
        positive=[v['r'] for v in curves if v['mean_score']>0]; chosen=min(positive) if positive else None
        out=r.ROOT/ds; csv_save(out/'selection_folds.csv',rows)
        json_save(out/'selection.json',dict(rule='minimum mean-positive matched-projector PCA gain, joint k=r, 8..32',
                 one_se_used=False,selected=chosen,N=positive,curves=curves))
        if chosen is None: raise RuntimeError(f'{ds}: no eligible dimension; do not invent fallback')
        for path in (Path('results/resnet_temporal_dimension')/ds/'responses/artifacts').glob('*.pt'):
            dst=out/'responses/artifacts'/path.name; dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists(): shutil.copy2(path,dst)
        label=f'rank{chosen}'; r.geometry(ds,chosen,chosen,label,response_count=16)
        all_overlaps(out/label); r.evaluate(ds,label)
        print('RESNET COMPLETE',ds,'rank',chosen,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['deit10','resnet','remaining']); args=parser.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    if args.mode=='remaining': deit10(); resnet()
    else: globals()[args.mode]()
