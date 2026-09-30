"""ResNet checkpoint response trajectories and source-only one-SE dimension selection."""
from __future__ import annotations
from package_paths import read_config_text
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import analyze_scaffold_followup as f
import numpy as np
import torch
from analyze_simple_sources_function import advance_augmentation_rng
from deit_three_claims import json_save, csv_save, stats
from analyze_random_control_audit import unconditioned_spectrum_matched
from phase_c_direction import functional_bases

ROOT = Path('results/resnet_temporal_dimension')
DATA = {'cifar100': Path('results/confirmatory'), 'cifar10': Path('results/cifar10_confirmation')}


def eigbasis(c):
    return torch.linalg.eigh(c.double()).eigenvectors.flip(1)


def source_data(ds):
    result=[]
    for e in entries(ds):
        a=f.load(Path(e['path']))
        h=a['historical_drift_centered'].double()
        result.append({'order':int(a['order_seed']),'seed':int(a['seed']),
                       'history':eigbasis(h.T@h),'static':eigbasis(a['static_covariance']),
                       'covariance':a['static_covariance'].double(),
                       'future':a['future_drift_centered'].double(),'path':e['path']})
    return result


def choose(curves):
    best=sorted(curves,key=lambda x:(-x['mean_score'],x['r'],x['k']))[0]
    threshold=best['mean_score']-best['se']
    selected=sorted([x for x in curves if x['mean_score']>=threshold-1e-14],key=lambda x:(x['r'],x['k']))[0]
    return {'best':best,'threshold':threshold,'selected':selected}


def select(ds):
    assert (ROOT/ds/'rank8/geometry.json').exists(), 'Original-rank temporal analysis must precede selection'
    records=source_data(ds); orders=sorted({a['order'] for a in records}); rows=[]
    for held in orders:
        fitting=[a for a in records if a['order']!=held]
        validation=[a for a in records if a['order']==held]
        for k in (2,4,8,16,32,64,128,256):
            u=f.pool([a['history'][:,:k] for a in fitting],rank=min(512,4*k))
            for a in validation:
                energies=(a['future']@u).square().sum(0).cumsum(0)/a['future'].square().sum()
                for r in range(1,u.shape[1]+1):
                    rows.append({'order_seed':held,'seed':a['seed'],'k':k,'r':r,
                                 'coverage':float(energies[r-1]),'score':float(energies[r-1])-r/512})
    candidates=sorted({(x['k'],x['r']) for x in rows})
    grouped={}
    for x in rows: grouped.setdefault((x['k'],x['r'],x['order_seed']),[]).append(x['score'])
    def curve_for(use_orders):
        curves=[{'k':0,'r':0,'mean_score':0.,'se':0.,'order_scores':{str(o):0. for o in use_orders}}]
        for k,r in candidates:
            scores={str(o):float(np.mean(grouped[(k,r,o)])) for o in use_orders}
            vals=list(scores.values())
            curves.append({'k':k,'r':r,'mean_score':float(np.mean(vals)),
                           'se':float(np.std(vals,ddof=1)/np.sqrt(len(vals))),'order_scores':scores})
        return curves
    curves=curve_for(orders); selection=choose(curves)
    selection['validation_order_deletions']={str(o):choose(curve_for([v for v in orders if v!=o])) for o in orders}
    selection['best_on_r_boundary']=selection['best']['r']==min(512,4*selection['best']['k'])
    selection['best_on_k_boundary']=selection['best']['k'] in (2,256)
    selection['source_only']=True; selection['folds']=3; selection['curves']=curves
    json_save(ROOT/ds/'selection.json',selection); csv_save(ROOT/ds/'selection_folds.csv',rows)
    print(ds,'ONE-SE SELECTION',json.dumps({k:v for k,v in selection.items() if k not in ('curves','validation_order_deletions')}),flush=True)


def geometry(ds,k=8,r=8,label='rank8',response_count=16,checkpoints=None):
    destination=ROOT/ds/label; records=source_data(ds)
    cps=list(checkpoints) if checkpoints is not None else list(range(5 if ds=='cifar100' else 6))
    task2_index=cps.index(2)
    histories=[a['history'][:,:k] for a in records]
    scaffold=f.pool(histories,r)
    responses_by_cp={cp:[] for cp in cps}; prefixes_by_cp={cp:[] for cp in cps}; stability=[]
    for a in records:
        response=f.load(ROOT/ds/'responses/artifacts'/Path(a['path']).name)
        for cp in cps:
            state=response['states'][cp]
            covariance=state['covariance'] if response_count==16 else state['prefixes'][response_count]['covariance']
            responses_by_cp[cp].append(eigbasis(covariance)[:,:k])
            prefix=state['prefixes'][8]
            # Original 0/2 records retain the rank-eight prefix, but not its covariance.
            prefix_u=eigbasis(prefix['covariance'])[:,:k] if 'covariance' in prefix else prefix['basis'].double() if k==8 else None
            if prefix_u is not None:
                prefixes_by_cp[cp].append(prefix_u)
                stability.append({'order_seed':a['order'],'seed':a['seed'],'checkpoint':cp,
                                  'source_k':k,'prefix8_vs16_overlap':f.overlap(prefix_u,responses_by_cp[cp][-1])})
    pooled={cp:f.pool(bases,r) for cp,bases in responses_by_cp.items()}
    pooled_stability={str(cp):f.overlap(f.pool(prefixes_by_cp[cp],r),pooled[cp])
                      for cp in cps if len(prefixes_by_cp[cp])==6}
    boundary_gaps={}
    for cp in cps:
        spectrum=torch.linalg.svdvals(torch.cat(responses_by_cp[cp],dim=1)).square()/6
        boundary_gaps[str(cp)]=float((spectrum[r-1]-spectrum[r])/spectrum[r-1]) if r<len(spectrum) else None
    main=np.array([f.overlap(scaffold,pooled[cp]) for cp in cps])
    individual=[]
    for a in records:
        saved=f.load(ROOT/ds/'responses/artifacts'/Path(a['path']).name)
        for cp in cps:
            state=saved['states'][cp]
            covariance=state['covariance'] if response_count==16 else state['prefixes'][response_count]['covariance']
            individual.append({'order_seed':a['order'],'seed':a['seed'],'checkpoint':cp,
                               'fixed_pooled_scaffold_vs_individual_response':f.overlap(scaffold,eigbasis(covariance)[:,:r])})
    # Pair source-order and seed resampling for both fitted spaces; cache repeated count patterns.
    groups=[[i for i,a in enumerate(records) if a['order']==o] for o in sorted({a['order'] for a in records})]
    rng=np.random.default_rng(914071); counts=[]
    for _ in range(2000):
        chosen=[i for g in rng.integers(0,3,3) for i in rng.choice(groups[g],2,replace=True)]
        counts.append(tuple(np.bincount(chosen,minlength=6)))
    cache={}
    for n,count in enumerate(sorted(set(counts))):
        h=f.pool([u*np.sqrt(w) for u,w in zip(histories,count) if w],r)
        # When resampling produces fewer than r independent columns, rank is not identifiable.
        if h.shape[1]<r:
            cache[count]=None; continue
        values=[]
        for cp in cps:
            resp=f.pool([u*np.sqrt(w) for u,w in zip(responses_by_cp[cp],count) if w],r)
            values.append(f.overlap(h,resp))
        cache[count]=values
    draws=np.array([cache[c] for c in counts if cache[c] is not None])
    deletions=[]
    for order in sorted({a['order'] for a in records}):
        keep=[i for i,a in enumerate(records) if a['order']!=order]
        h=f.pool([histories[i] for i in keep],r)
        vals=[f.overlap(h,f.pool([responses_by_cp[cp][i] for i in keep],r)) for cp in cps]
        deletions.append({'deleted_order':order,'overlap':vals,'adjacent_changes':np.diff(vals).tolist()})
    result={'dataset':ds,'k':k,'r':r,'response_count':response_count,'checkpoints':list(cps),'overlap':main.tolist(),
            'ci95':np.quantile(draws,[.025,.975],axis=0).T.tolist(),
            'adjacent_change':np.diff(main).tolist(),
            'adjacent_change_ci95':np.quantile(np.diff(draws,axis=1),[.025,.975],axis=0).T.tolist(),
            'change_from_task2':(main-main[task2_index]).tolist(),
            'change_from_task2_ci95':np.quantile(draws-draws[:,task2_index,None],[.025,.975],axis=0).T.tolist(),
            'change_from_initial':(main-main[0]).tolist(),
            'change_from_initial_ci95':np.quantile(draws-draws[:,0,None],[.025,.975],axis=0).T.tolist(),
            'isotropic_overlap':r/scaffold.shape[0],'bootstrap_draws':len(draws),'bootstrap_unique_patterns':len(cache),
            'bootstrap_requested_draws':len(counts),'bootstrap_excluded_insufficient_span':len(counts)-len(draws),
            'source_order_deletions':deletions,'sampling_stability':stability,
            'response_sampling_pooled_overlap':pooled_stability,'response_boundary_relative_gap':boundary_gaps}
    json_save(destination/'geometry.json',result); csv_save(destination/'individual_overlap.csv',individual)
    np.savez_compressed(destination/'geometry_bootstrap.npz',draws=draws,counts=np.array(counts))
    predictors={'scaffold':scaffold,'matched_static':f.pool([a['static'][:,:k] for a in records],r),
                'pooled_static':eigbasis(sum(a['covariance'] for a in records))[:,:r],
                **{f'response_{cp}':pooled[cp] for cp in cps}}
    f.save(predictors,destination/'predictors.pt')
    print(ds,label,'overlap',main.tolist(),'adjacent',np.diff(main).tolist(),flush=True)


def controls(scaffold,disc,null,seed):
    bases=[]; error=0.
    for draw in range(128):
        u,se,oe=unconditioned_spectrum_matched(scaffold.float(),disc.float(),null.float(),scaffold.shape[1],
                                             torch.Generator().manual_seed(seed+draw))
        error=max(error,se,oe); bases.append(u.double())
    assert error<1e-5,error
    return bases,error


def random_coverage(field,bases):
    # Average projector gives exactly the mean coverage across the same random bases.
    p=sum(u@u.T for u in bases)/len(bases)
    return float(((field.double()@p)*field.double()).sum()/field.double().square().sum())


def evaluate(ds,label,target_root=None,fields_root=None):
    out=ROOT/ds/label; predictors=f.load(out/'predictors.pt'); scaffold=predictors['scaffold']; r=scaffold.shape[1]
    dimension=scaffold.shape[0]
    target_root=target_root or DATA[ds]/'scaffold_followup/target_recovery/artifacts'
    fields_root=fields_root or DATA[ds]/'scaffold_followup/analysis/update_fields'
    temporal=[]; transfer=[]; updates=[]; functional=[]; checks=[]
    for target_index,e in enumerate(entries(ds,'target')):
        name=Path(e['path']).name
        target=f.load(target_root/name)
        original=f.load(Path(e['path'])); identity={'order_seed':target['order_seed'],'seed':target['seed']}
        reps=target['eval_reps']; raw=(reps[-1]-reps[-2]).double()
        original_error=float((f.center(reps[-1]-reps[-2])-original['future_drift_centered']).abs().max())
        assert original_error<=1e-5
        bases,early_error=controls(scaffold,original['discriminative_basis'],original['null_basis'],
                                   target['order_seed']*1000003+target['seed']*10007+6000000)
        for i in range(1,len(reps)):
            field=f.center((reps[i]-reps[i-1]).double())
            row={**identity,'transition':f'{i}_{i+1}',
                 **{n:f.coverage(field,u) for n,u in predictors.items() if not n.startswith('response')},
                 'matched_random':random_coverage(field,bases),'isotropic':r/dimension}
            row['gain_random']=row['scaffold']-row['matched_random']; row['gain_isotropic']=row['scaffold']-r/dimension
            row['gain_static']=row['scaffold']-row['matched_static']; temporal.append(row)
        row={**identity,**{n:f.coverage(f.center(raw),u) for n,u in predictors.items()}}
        for cp in sorted(int(n.split('_')[1]) for n in predictors if n.startswith('response_') and n!='response_0'):
            row[f'response_{cp}_minus_initial']=row[f'response_{cp}']-row['response_0']
        transfer.append(row)
        state=target['states'][len(reps)-1]; classes=sum(target['manifest']['order'][:-1],[])
        weight,_=f.classifier_tensors(state)
        disc,null,_,_=functional_bases(weight[classes].float(),1e-7)
        frozen,frozen_error=controls(scaffold,disc,null,target['order_seed']*1000003+target['seed']*10007+8000000)
        fields=f.load(fields_root/name)['fields']
        row=dict(identity)
        for n,field in fields.items():
            for method in ('scaffold','matched_static','pooled_static'):
                row[f'{n}_{method}']=f.coverage(field,predictors[method])
            row[f'{n}_random']=random_coverage(field,frozen)
            row[f'{n}_gain_random']=row[f'{n}_scaffold']-row[f'{n}_random']
            row[f'{n}_gain_static']=row[f'{n}_scaffold']-row[f'{n}_matched_static']
            row[f'{n}_gain_isotropic']=row[f'{n}_scaffold']-r/dimension
        updates.append(row)
        row=dict(identity)
        keys=['centered_conditional_class_centered_logits_reconstruction','centered_conditional_loss_reconstruction']
        for method in ('scaffold','matched_static','pooled_static'):
            values=f.reconstruction(reps[-2],raw,predictors[method],state,classes,target['heldout_labels'])
            for key in keys: row[f'{method}_{key}']=values[key]
        random_values=[f.reconstruction(reps[-2],raw,u,state,classes,target['heldout_labels']) for u in frozen]
        for key in keys:
            row[f'random_{key}']=float(np.mean([v[key] for v in random_values]))
            row[f'gain_random_{key}']=row[f'scaffold_{key}']-row[f'random_{key}']
            row[f'gain_static_{key}']=row[f'scaffold_{key}']-row[f'matched_static_{key}']
        functional.append(row)
        checks.append({**identity,'original_future_error':original_error,'early_random_error':early_error,'frozen_random_error':frozen_error})
        print(ds,label,name,'three claims evaluated',flush=True)
    summary={'dataset':ds,'label':label,'r':r,'temporal':{},'checks':checks}
    for transition in sorted({x['transition'] for x in temporal}):
        rows=[x for x in temporal if x['transition']==transition]
        summary['temporal'][transition]=stats(rows,[k for k in rows[0] if k not in ('order_seed','seed','transition')])
    for name,rows in [('transfer',transfer),('updates',updates),('functional',functional)]:
        summary[name]=stats(rows,[k for k in rows[0] if k not in ('order_seed','seed')])
        csv_save(out/f'{name}.csv',rows)
    csv_save(out/'temporal.csv',temporal); json_save(out/'evaluation.json',summary)


def run(ds):
    inventory=[]
    for e in entries(ds)+entries(ds,'target'):
        p=Path(e['path']); inventory.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    inv_path=ROOT/ds/'original_hashes.json'
    if inv_path.exists(): assert json.loads(read_config_text(inv_path))==inventory
    else: json_save(inv_path,inventory)
    recover(ds); responses(ds)
    if not (ROOT/ds/'rank8/geometry.json').exists(): geometry(ds)
    if not (ROOT/ds/'selection.json').exists(): select(ds)
    choice=json.loads(read_config_text(ROOT/ds/'selection.json'))['selected']
    if choice['r']:
        if not (ROOT/ds/'selected/geometry.json').exists(): geometry(ds,choice['k'],choice['r'],'selected')
    for label in ('rank8','selected'):
        if label=='selected' and not choice['r']: continue
        if not (ROOT/ds/label/'evaluation.json').exists(): evaluate(ds,label)
    for row in inventory: assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    print(ds,'ALL COMPUTATIONS COMPLETE; original artifact hashes unchanged',flush=True)


def entries(ds, role='source'):
    return [e for e in json.loads(read_config_text(DATA[ds]/'scaffold_structure/inventory.json')) if e['role']==role]


def recover(ds,prior_sources=None):
    out=ROOT/ds
    device=torch.device('cuda')
    train=f.IndexedCifar(Path('data'), True, True, ds)
    evaluation=f.IndexedCifar(Path('data'), True, False, ds)
    for e in entries(ds):
        p=Path(e['path']); dst=out/'sources/artifacts'/p.name
        if dst.exists() and f.load(dst)['status']=='complete':
            print(ds,p.stem,'already recovered',flush=True); continue
        old=f.load((prior_sources or DATA[ds]/'scaffold_followup/source_recovery/artifacts')/p.name)
        original=f.load(p); config=NS(**old['manifest']); order=config.order; seed=old['seed']
        f.seed_all(seed)
        net=f.model(config.model, True, num_classes=train.num_classes).cuda()
        for m in net.modules():
            if isinstance(m,torch.nn.Dropout): assert m.p==0
            assert getattr(m,'drop_prob',0)==0
        train_ids={c:f.ids_by_class(train.targets,[c],config.train_per_class) for c in sum(order,[])}
        priority=torch.rand(len(train),generator=torch.Generator().manual_seed(seed+9000))
        if dst.exists():
            rec=f.load(dst); f.restore_rng(rec['rng']); net.load_state_dict(rec['states'][len(rec['fit_reps'])])
            seen=sum(order[:len(rec['fit_reps'])],[])
            memory=f.balanced_memory(train_ids,seen,config.memory,priority)
        else:
            rec=old; rec['status']='running'
            cache=out/'rng'/f'seed{seed}_after2.pt'
            if cache.exists():
                f.restore_rng(f.load(cache)); seen=sum(order[:2],[])
                memory=f.balanced_memory(train_ids,seen,config.memory,priority)
            else:
                seen,memory=advance_augmentation_rng(train,train_ids,order[:3],priority,config,seed)
                f.save(f.rng_state(),cache)
            net.load_state_dict(rec['states'][2])
        for index in range(len(rec['fit_reps']),len(order)):
            started=time.monotonic(); old_seen=list(seen); seen+=order[index]
            current=torch.cat([train_ids[c] for c in order[index]])
            if index==len(order)-2 and 'pre_future_state_dict' in original:
                f.consume_augmentations(train,current,memory,config,seed*100+index)
                net.load_state_dict(original['pre_future_state_dict'])
            else:
                f.train_task(net,config.continual_method,train,current,memory,{},seen,old_seen,config,device,seed*100+index)
            if old['manifest'].get('save_layers',False):
                _,fit,_=f.infer_intermediates(net,evaluation,rec['fit_ids'],config.batch_size,device)
                logits,ev,labels=f.infer_intermediates(net,evaluation,rec['heldout_ids'],config.batch_size,device)
                fit,ev=fit['stage4'],ev['stage4']
            else:
                _,fit,_=f.infer(net,evaluation,rec['fit_ids'],config.batch_size,device)
                logits,ev,labels=f.infer(net,evaluation,rec['heldout_ids'],config.batch_size,device)
            rec['fit_reps'].append(fit); rec['eval_reps'].append(ev)
            rec['old_accuracy'].append(f.accuracy(logits,labels,seen))
            rec['states'][index+1]=f.state_dict_cpu(net)
            rec['rng']=f.rng_state()
            memory=f.balanced_memory(train_ids,seen,config.memory,priority)
            if index==len(order)-1:
                error=float((f.center(rec['eval_reps'][-1]-rec['eval_reps'][-2])-original['future_drift_centered']).abs().max())
                rec['future_max_abs_error']=error
                assert error<=1e-5,(ds,p,error)
                assert rec['history_max_abs_error']<=1e-5
                rec['status']='complete'; del rec['rng']
            f.save(rec,dst)
            print(ds,p.stem,'recovered checkpoint',index+1,'seconds',round(time.monotonic()-started,1),'error',rec.get('future_max_abs_error'),flush=True)
        del net,rec,old,original; torch.cuda.empty_cache()


def responses(ds):
    evaluation=f.IndexedCifar(Path('data'),True,False,ds)
    for e in entries(ds):
        p=Path(e['path']); dst=ROOT/ds/'responses/artifacts'/p.name
        rec=f.load(ROOT/ds/'sources/artifacts'/p.name)
        saved=f.load(dst) if dst.exists() else f.load(DATA[ds]/'scaffold_followup/source_responses/artifacts'/p.name)
        assert saved['directions']==16 and saved['relative_step']==1e-4
        net=f.model(rec['manifest']['model'],False,num_classes=evaluation.num_classes).cuda()
        batches=f.image_batches(net,evaluation,rec['fit_ids'],'cuda')
        head={id(p) for p in net.get_classifier().parameters()}
        parameters=dict(net.named_parameters()); names=[n for n,p in parameters.items() if id(p) not in head]
        for cp in range(len(rec['manifest']['order'])+1):
            prefix_only=cp in saved['states']
            if prefix_only and 'covariance' in saved['states'][cp]['prefixes'][8]: continue
            started=time.monotonic(); state=rec['states'][cp]
            norm=torch.sqrt(sum(state[n].double().square().sum() for n in names))
            gen=torch.Generator().manual_seed(rec['order_seed']*100003+rec['seed']+701)
            covariance=torch.zeros(512,512,dtype=torch.float64); prefixes={}
            for draw in range(8 if prefix_only else 16):
                direction={n:torch.randn(state[n].shape,generator=gen) for n in names}
                length=torch.sqrt(sum(v.double().square().sum() for v in direction.values()))
                scale=float(1e-4*norm/length); values=[]
                for sign in (1,-1):
                    net.load_state_dict(state)
                    with torch.no_grad():
                        for n in names: parameters[n].add_(direction[n].cuda(),alpha=sign*scale)
                    values.append(f.cached_representations(net,batches))
                field=f.center((values[0].double()-values[1].double())/2)
                covariance+=field.T@field
                if draw+1 in (8,16):
                    eig,v=torch.linalg.eigh(covariance/(draw+1))
                    prefixes[draw+1]={'basis':v[:,-8:],'spectrum':eig.flip(0),'covariance':covariance.clone()/(draw+1)}
            if prefix_only:
                old_basis=saved['states'][cp]['prefixes'][8]['basis'].double()
                assert f.overlap(old_basis,prefixes[8]['basis'])>1-1e-8
                saved['states'][cp]['prefixes'][8]['covariance']=prefixes[8]['covariance']
            else:
                saved['states'][cp]={'covariance':covariance/16,'prefixes':prefixes,
                                      'prefix_overlap':f.overlap(prefixes[8]['basis'],prefixes[16]['basis'])}
            f.save(saved,dst)
            print(ds,p.stem,'response checkpoint',cp,'seconds',round(time.monotonic()-started,1),flush=True)
        f.save(saved,dst)
        del net,rec,batches; torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['recover','responses','geometry','select','run']); p.add_argument('--dataset',choices=list(DATA),required=True)
    args=p.parse_args(); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    globals()[args.mode](args.dataset)


if __name__=='__main__': main()
