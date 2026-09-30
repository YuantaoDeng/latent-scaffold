"""Analyze no-replay recurrence, temporal sensitivity, PCA residuals and updates."""
from package_paths import read_config_text
import csv,json,time
from pathlib import Path
import numpy as np
import torch
import drift_experiment as d
import analyze_scaffold_followup as f
from noreplay_all_common import configure,response,load_record,save_value

def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def csvsave(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as out:
        w=csv.DictWriter(out,list(rows[0]));w.writeheader();w.writerows(rows)
def matched(field,sc,weight):
    _,s,v=torch.linalg.svd(f.center(weight.double()),full_matrices=False)
    rank=int((s>s.max()*1e-7).sum());disc=v[:rank].T
    alpha=float((disc.T@sc).square().sum());fraction=f.coverage(field,disc)
    return alpha/rank*fraction+(RANK-alpha)/(DIM-rank)*(1-fraction)

def fit():
    dirs=sorted((OUT/'source').glob('order*'))
    assert len(dirs)==6
    bases=[];pcas={t:[] for t in range(6)}
    for folder in dirs:
        assert (folder/'complete.json').exists()
        rec=[load_record(folder/f'task{t}.pt',[]) for t in range(6)]
        assert all(r['steps']==75 and not r['updates'] for r in rec[1:])
        bases.append(f.basis(f.center(rec[2]['fit']-rec[1]['fit']),RANK))
        for t in range(6):pcas[t].append(f.basis(f.center(rec[t]['fit']),RANK))
    sc=f.pool(bases,RANK)
    # Independent eigendecomposition of mean projectors checks pooling.
    vv=torch.linalg.eigh(sum(u@u.T for u in bases)).eigenvectors[:,-RANK:]
    assert f.overlap(sc,vv)>1-1e-10
    replay=f.load(REPLAY_PATH)['scaffold'].double()
    write('replay_comparison.json',dict(scaffold_overlap=f.overlap(sc,replay),rank=RANK,
        scope='Descriptive comparison; initialization and update budget are not a paired replay-only contrast'))
    torch.save(dict(scaffold=sc,source_bases=bases,**{f'pca{t}':f.pool(bb,RANK) for t,bb in pcas.items()}),OUT/'bases.pt')
    return sc

def geometry(sc):
    net=d.model(MODEL_NAME,False,num_classes=10).cuda()
    data=d.IndexedCifar(Path('data'),True,False,'cifar10');rows=[];pools={t:[] for t in range(6)}
    (OUT/'responses').mkdir(exist_ok=True)
    bases=f.load(OUT/'bases.pt');started=time.perf_counter()
    for folder in sorted((OUT/'source').glob('order*')):
        ident=folder.name;order,seed=[int(s) for s in ident.replace('order','').replace('seed','').split('_')]
        anchors=f.load(folder/'anchors.pt');batches=f.image_batches(net,data,anchors['fit_ids'],'cuda')
        for t in range(6):
            cache=OUT/'responses'/f'{ident}_task{t}.pt'
            if cache.exists():cov=f.load(cache)
            else:
                state=f.load(folder/f'task{t}.pt')['state']
                cov=response(net,batches,state,order*100003+seed)
                save_value(cov,cache)
            e,v=torch.linalg.eigh(cov[16]);u=v[:,-RANK:];u8=torch.linalg.eigh(cov[8]).eigenvectors[:,-RANK:]
            pools[t].append(u)
            rows.append(dict(order=order,seed=seed,task=t,overlap=f.overlap(sc,u),
                energy_ratio=float(torch.trace(sc.T@cov[16]@sc)/e[-RANK:].sum()),
                sampling_overlap=f.overlap(u8,u),relative_eigengap=float((e[-RANK]-e[-RANK-1])/e[-RANK])))
            print(json.dumps(dict(stage='response',history=ident,task=t,seconds=round(time.perf_counter()-started,1))),flush=True)
    pooled=[]
    for t in range(6):
        u=f.pool(pools[t],RANK)
        pooled.append(dict(task=t,scaffold_response_overlap=f.overlap(sc,u),
            current_pca_response_overlap=f.overlap(bases[f'pca{t}'],u),
            scaffold_current_pca_overlap=f.overlap(sc,bases[f'pca{t}'])))
    csvsave('geometry_per_source.csv',rows);csvsave('pooled_overlap.csv',pooled)
    del net;torch.cuda.empty_cache()
    return rows

def target_analysis(sc):
    motion=[];updates=[];error=0.;bases=f.load(OUT/'bases.pt')
    replay=f.load(REPLAY_PATH)['scaffold'].double()
    for folder in sorted((OUT/'target').glob('order*')):
        assert (folder/'complete.json').exists()
        order,seed=[int(s) for s in folder.name.replace('order','').replace('seed','').split('_')]
        anchors=f.load(folder/'anchors.pt');prev=load_record(folder/'task1.pt',[HEAD_WEIGHT,HEAD_BIAS])
        for t in range(2,6):
            cur=load_record(folder/f'task{t}.pt',[HEAD_WEIGHT,HEAD_BIAS]);delta=cur['eval'].double()-prev['eval'].double();field=f.center(delta)
            assert cur['steps']==75 and [u['step'] for u in cur['updates']]==[2,38,73]
            assert not torch.isin(anchors['test_ids'],cur['current_ids']).any()
            pca=f.basis(f.center(prev['fit']),RANK);r=field-(field@pca)@pca.T
            projected=sc-pca@(pca.T@sc);q,s,_=torch.linalg.svd(projected,full_matrices=False)
            assert (s>1e-8).sum()==RANK;q=q[:,:RANK]
            own=f.coverage(field,sc);rand=matched(field,sc,prev['state'][HEAD_WEIGHT][cur['seen']])
            rec=f.reconstruction(prev['eval'],delta,sc,{'fc.weight':prev['state'][HEAD_WEIGHT],'fc.bias':prev['state'][HEAD_BIAS]},cur['seen'],anchors['labels'])
            recpca=f.reconstruction(prev['eval'],delta,pca,{'fc.weight':prev['state'][HEAD_WEIGHT],'fc.bias':prev['state'][HEAD_BIAS]},cur['seen'],anchors['labels'])
            error=max(error,abs(own-float(((field@sc)@sc.T).square().sum()/field.square().sum())))
            motion.append(dict(order=order,seed=seed,task=t,scaffold=own,matched_random=rand,gain_matched=own-rand,
                original_replay_scaffold=f.coverage(field,replay),
                local_start_pca=f.coverage(field,pca),pooled_early_pca=f.coverage(field,bases['pca2']),
                residual_fraction=float(r.square().sum()/field.square().sum()),residual_coverage=f.coverage(r,q),
                residual_gain=f.coverage(r,q)-RANK/(DIM-RANK),
                scaffold_energy_outside_pca=float(projected.square().sum()/RANK),
                raw_residual_coverage=f.coverage(r,sc),
                raw_residual_gain=f.coverage(r,sc)-float(projected.square().sum())/(DIM-RANK),
                frozen_logit_reconstruction=rec['centered_conditional_class_centered_logits_reconstruction'],
                frozen_loss_reconstruction=rec['centered_conditional_loss_reconstruction'],
                pca_frozen_logit_reconstruction=recpca['centered_conditional_class_centered_logits_reconstruction']))
            for update in cur['updates']:
                for method,uf in update['fields'].items():
                    observed=f.coverage(uf,sc);control=matched(uf,sc,update['classifier'][update['seen']])
                    updates.append(dict(order=order,seed=seed,task=t,step=update['step'],method=method,
                        coverage=observed,matched_random=control,gain=observed-control,relative_norm=update['relative_norm']))
            prev=cur
    assert len(motion)==48 and len(updates)==432 and error<1e-10
    csvsave('motion_per_test.csv',motion);csvsave('updates_per_test.csv',updates)
    return motion,updates,error

def summarize(rows,groups,metrics,name):
    results=[]
    for key in sorted({tuple(r[g] for g in groups) for r in rows}):
        rr=[r for r in rows if tuple(r[g] for g in groups)==key]
        orders=sorted({r['order'] for r in rr});by={o:sorted([r for r in rr if r['order']==o],key=lambda r:r['seed']) for o in orders}
        rng=np.random.default_rng(20260913);od=rng.integers(0,len(orders),(10000,len(orders)));sd=rng.integers(0,2,(10000,len(orders),2))
        for m in metrics:
            vals=np.array([[r[m] for r in by[o]] for o in orders]);assert np.isfinite(vals).all(),(name,key,m)
            draws=vals[od[:,:,None],sd].mean((1,2));lo,hi=np.quantile(draws,[.025,.975])
            results.append(dict(**dict(zip(groups,key)),metric=m,mean=float(vals.mean()),low=float(lo),high=float(hi),
                                positive_runs=int((vals>0).sum()),runs=int(vals.size)))
    csvsave(name,results);return results

def main():
    torch.set_num_threads(2);sc=fit()
    motion,updates,error=target_analysis(sc)
    summarize(motion,['task'],[k for k in motion[0] if k not in ('order','seed','task')],'motion_summary.csv')
    summarize(updates,['task','step','method'],['coverage','matched_random','gain','relative_norm'],'updates_summary.csv')
    print('HELD_OUT_MOTION_AND_UPDATES_COMPLETE',flush=True)
    geo=geometry(sc)
    summarize(geo,['task'],['overlap','energy_ratio','sampling_overlap','relative_eigengap'],'geometry_summary.csv')
    changes=[]
    for r in geo:
        if r['task']==5:
            prev=next(a for a in geo if a['task']==2 and a['order']==r['order'] and a['seed']==r['seed'])
            changes.append(dict(order=r['order'],seed=r['seed'],comparison='task5_minus_task2',
                                overlap=r['overlap']-prev['overlap'],energy_ratio=r['energy_ratio']-prev['energy_ratio']))
    summarize(changes,['comparison'],['overlap','energy_ratio'],'paired_changes.csv')
    write('verification.json',dict(complete=True,source_histories=6,test_histories=12,rank=RANK,tasks=5,
        replay_images=0,scaffold_projectors_rebuilt=True,projection_error=error,motion_rows=48,update_rows=432,
        all_tasks_successful_steps=75,response_covariances=36,anchor_disjointness=True))
    protocol=json.loads(read_config_text(OUT/'protocol.json'));protocol['status']='complete';write('protocol.json',protocol)
    write('progress.json',dict(status='complete',source_histories=6,test_histories=12))
    print('ALL_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
    globals().update(configure(a.model));main()
