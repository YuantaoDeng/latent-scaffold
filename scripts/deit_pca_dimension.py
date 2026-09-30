"""DeiT matched-PCA one-SE rank selection and complete response-time measurement."""
from package_paths import read_config_text
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch
import analyze_scaffold_followup as f
import deit_three_claims as d
import resnet_temporal_dimension as r

BASE=Path('results/deit_pca_dimension')
OUT=BASE/'cifar100'


def configure():
    r.ROOT=BASE; r.DATA={'cifar100':d.ORIGINAL}
    r.entries=lambda ds,role='source': [dict(path=str(p),manifest_path=str(p.parent.parent/'manifest.json'),role=role)
                                      for p,_ in d.inventory(role)]


def select():
    configure(); records=r.source_data('cifar100'); orders=sorted({a['order'] for a in records}); rows=[]
    for held in orders:
        fitting=[a for a in records if a['order']!=held]; validation=[a for a in records if a['order']==held]
        for k in range(1,33):
            limit=k
            h=f.pool([a['history'][:,:k] for a in fitting],limit)
            p=f.pool([a['static'][:,:k] for a in fitting],limit)
            for a in validation:
                denom=a['future'].square().sum()
                hc=(a['future']@h).square().sum(0).cumsum(0)/denom
                pc=(a['future']@p).square().sum(0).cumsum(0)/denom
                for rank in (k,):
                    rows.append({'order_seed':held,'seed':a['seed'],'k':k,'r':rank,
                                 'scaffold':float(hc[rank-1]),'matched_static':float(pc[rank-1]),
                                 'score':float(hc[rank-1]-pc[rank-1])})
    grouped={}
    for x in rows: grouped.setdefault((x['k'],x['r'],x['order_seed']),[]).append(x['score'])
    candidates=sorted({(x['k'],x['r']) for x in rows})
    def curves(use_orders):
        result=[{'k':0,'r':0,'mean_score':0.,'se':0.,'order_scores':{str(o):0. for o in use_orders}}]
        for k,rank in candidates:
            scores={str(o):float(np.mean(grouped[(k,rank,o)])) for o in use_orders}
            values=list(scores.values())
            result.append({'k':k,'r':rank,'mean_score':float(np.mean(values)),
                           'se':float(np.std(values,ddof=1)/np.sqrt(len(values))),'order_scores':scores})
        return result
    all_curves=curves(orders); result=r.choose(all_curves); result['curves']=all_curves
    result['validation_order_deletions']={str(o):r.choose(curves([v for v in orders if v!=o])) for o in orders}
    result.update(score_definition='scaffold minus matched-projector static PCA',source_only=True,
                  feature_dimension=192,construction='joint k=r, ranks 1 through 32',
                  best_on_k_boundary=result['best']['k'] in (1,32),
                  best_on_r_boundary=result['best']['r'] in (1,32))
    d.csv_save(OUT/'selection_folds.csv',rows); d.json_save(OUT/'selection.json',result)
    print('PCA ONE-SE SELECTION',json.dumps({k:v for k,v in result.items() if k not in ('curves','validation_order_deletions')}),flush=True)


def responses():
    evaluation=f.IndexedCifar(Path('data'),True,False,'cifar100')
    for path,_ in d.inventory('source'):
        dst=OUT/'responses/artifacts'/path.name
        rec=f.load(OUT/'sources/artifacts'/path.name)
        primary=f.load(d.ROOT/'responses/artifacts'/path.name)
        audit=f.load(d.ROOT/'responses64/artifacts'/path.name)
        result=f.load(dst) if dst.exists() else {'directions':16,'audit_directions':[32,64],
                 'relative_step':1e-4,'order_seed':rec['order_seed'],'seed':rec['seed'],'states':{}}
        net=f.model(rec['manifest']['model'],False,num_classes=100).cuda()
        batches=f.image_batches(net,evaluation,rec['fit_ids'],'cuda')
        head={id(p) for p in net.get_classifier().parameters()}; params=dict(net.named_parameters())
        names=[n for n,p in params.items() if id(p) not in head]
        for cp in range(5):
            if cp in result['states'] and all('covariance' in result['states'][cp]['prefixes'][n] for n in (8,16)): continue
            state=rec['states'][cp]; norm=torch.sqrt(sum(state[n].double().square().sum() for n in names))
            gen=torch.Generator().manual_seed(rec['order_seed']*100003+rec['seed']+701)
            covariance=torch.zeros(192,192,dtype=torch.float64); prefixes={}
            for draw in range(16):
                direction={n:torch.randn(state[n].shape,generator=gen) for n in names}
                # Saved 16/64 aggregates are kept exactly. Only missing prefixes need recovery.
                if cp in (0,2) and (8<=draw<16 or draw>=32):
                    if draw==15:
                        covariance=primary['states'][cp]['covariance'].clone()*16
                    elif draw==63:
                        covariance=audit['states'][cp]['covariance'].clone()*64
                    else: continue
                else:
                    length=torch.sqrt(sum(v.double().square().sum() for v in direction.values()))
                    scale=float(1e-4*norm/length); values=[]
                    for sign in (1,-1):
                        net.load_state_dict(state)
                        with torch.no_grad():
                            for n in names: params[n].add_(direction[n].cuda(),alpha=sign*scale)
                        values.append(f.cached_representations(net,batches))
                    field=f.center((values[0].double()-values[1].double())/2); covariance+=field.T@field
                if draw+1 in (8,16,32,64):
                    count=draw+1; eig,v=torch.linalg.eigh(covariance/count)
                    prefixes[count]={'covariance':covariance.clone()/count,'basis':v[:,-8:],'spectrum':eig.flip(0)}
                    if cp in (0,2):
                        old=(primary if count<=16 else audit)['states'][cp]['prefixes'][count]['basis'].double()
                        assert f.overlap(old,prefixes[count]['basis'])>1-1e-8,(path,cp,count)
            result['states'][cp]={'covariance':prefixes[16]['covariance'],'prefixes':prefixes,
                                 'prefix_overlap':f.overlap(prefixes[8]['basis'],prefixes[16]['basis'])}
            f.save(result,dst)
            print(path.stem,'all response prefixes at checkpoint',cp,'complete',flush=True)
        del net,rec,batches; torch.cuda.empty_cache()


def run():
    configure(); inventory=[]
    for role in ('source','target'):
        for p,_ in d.inventory(role): inventory.append({'role':role,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    dest=OUT/'original_hashes.json'
    if dest.exists(): assert json.loads(read_config_text(dest))==inventory
    else: d.json_save(dest,inventory)
    if not (OUT/'selection.json').exists(): select()
    choice=json.loads(read_config_text(OUT/'selection.json'))['selected']
    r.recover('cifar100',prior_sources=d.ROOT/'sources/artifacts')
    responses()
    settings=[('rank8',8,8),('rank20',20,20)]  # User-fixed dimension supersedes one-SE output.
    for name,k,rank in settings:
        for count in (16,):
            label=name if count==16 else f'{name}_response{count}'
            if not (OUT/label/'geometry.json').exists(): r.geometry('cifar100',k,rank,label,response_count=count)
        if not (OUT/name/'evaluation.json').exists():
            r.evaluate('cifar100',name,target_root=d.ROOT/'targets/artifacts',fields_root=d.ROOT/'update_fields')
    overlap_rows=[]; transfer_rows=[]
    for name,k,rank in settings:
        for count in (16,):
            label=name if count==16 else f'{name}_response{count}'
            bank=f.load(OUT/label/'predictors.pt')
            for left,u in bank.items():
                for right,v in bank.items():
                    overlap_rows.append({'setting':name,'k':k,'r':rank,'response_draws':count,'left':left,'right':right,'overlap':f.overlap(u,v)})
            rows=[]
            for p,_ in d.inventory('target'):
                a=f.load(p); row={'order_seed':a['order_seed'],'seed':a['seed']}
                for cp in range(5): row[f'response_{cp}']=f.coverage(a['future_drift_centered'],bank[f'response_{cp}'])
                for cp in range(1,5): row[f'change_{cp}_from_initial']=row[f'response_{cp}']-row['response_0']
                rows.append(row); transfer_rows.append({'setting':name,'response_draws':count,**row})
            d.json_save(OUT/label/'response_transfer.json',d.stats(rows,[key for key in rows[0] if key not in ('order_seed','seed')]))
    d.csv_save(OUT/'all_pooled_overlaps.csv',overlap_rows); d.csv_save(OUT/'all_response_transfer.csv',transfer_rows)
    for entry in inventory: assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    print('ALL DEIT PCA-DIMENSION COMPUTATIONS COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['select','run']); args=p.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    globals()[args.mode]()
