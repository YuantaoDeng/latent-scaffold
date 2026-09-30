"""Paired history-order intervention with identical future input streams."""
from package_paths import read_config_text
import csv, hashlib, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import drift_experiment as d
import analyze_scaffold_followup as f

OUT=Path('results/paired_history_memory_pilot/resnet18_cifar10')
POSITIONS={s:w for w,ss in [('early',[2,3,4]),('middle',[65,66,67]),('late',[125,126,127])] for s in ss}
def savejson(name,a): (OUT/name).write_text(json.dumps(a,indent=2),encoding='utf-8')
def csvsave(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def state(net): return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
@torch.no_grad()
def reps(net,dataset,ids):
    rng=f.rng_state(); mode=net.training
    try:
        net.eval()
        return torch.cat([d.features(net,d.prepare_images(x,'cuda')).cpu().double() for _,x,_ in d.loader(dataset,ids,64)])
    finally:
        net.train(mode); f.restore_rng(rng)

def train(net,trainset,evalset,current,memory,seen,seed,anchors=None):
    # The entire stream is reseeded per task, including augmentation RNG.
    d.seed_all(seed); net.train()
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.05)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    steps=0; observations=[]; trace=hashlib.sha256(); image_hashes=[]
    for epoch in range(5):
        cur=d.loader(trainset,current,32 if len(memory) else 64,True,seed+epoch)
        replay_loader=d.loader(trainset,memory,32,True,seed+100+epoch) if len(memory) else None
        replay=iter(replay_loader) if replay_loader else None
        for ci,cx,cy in cur:
            if replay is not None:
                try: ri,rx,ry=next(replay)
                except StopIteration: replay=iter(replay_loader); ri,rx,ry=next(replay)
                ids=torch.cat([ci,ri]); x=torch.cat([cx,rx]); y=torch.cat([cy,ry])
            else: ids,x,y=ci,cx,cy
            trace.update(ids.numpy().tobytes())
            # Hash all compact augmented CIFAR tensors before deterministic resize.
            trace.update(x.numpy().tobytes()); trace.update(y.numpy().tobytes())
            capture=anchors is not None and steps+1 in POSITIONS
            if capture: before=reps(net,evalset,anchors)
            x=d.prepare_images(x,'cuda'); y=y.cuda(); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits=net(x); loss=d.masked_ce(logits[:len(ci)],y[:len(ci)],seen)
                if replay is not None: loss+=d.masked_ce(logits[len(ci):],y[len(ci):],seen)
            scaler.scale(loss).backward(); old=scaler.get_scale(); scaler.step(opt); scaler.update()
            assert scaler.get_scale()>=old,'Skipped update: paired observation schedule invalid'
            steps+=1
            if capture:
                field=f.center(reps(net,evalset,anchors)-before)
                observations.append({'step':steps,'window':POSITIONS[steps],'field':field,'batch_ids':ids})
    return observations,trace.hexdigest(),steps

def run():
    OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    started=time.perf_counter()
    source=Path('results/resnet_temporal_dimension/cifar10/sources/artifacts/order2003_seed160.pt')
    initial=f.load(source)['states'][0]
    savejson('protocol.json',{'model':'ResNet-18','dataset':'CIFAR-10','initial_state_source':str(source),
      'design':'Shared task A; branch X: B then C; branch Y: C then B; both then D and E.',
      'layouts':[3101,3111,3121],'seeds':[180,181],'rank':8,'epochs_per_task':5,'amp_initial_scale':128,
      'memory':500,'training_images_per_class':450,'heldout_images_per_anchor_class':50,
      'primary_history_basis':'Rank-8 centered feature displacement from after A to after both B/C, on identical first-task training anchors. This spans equal content and equal duration in both branches.',
      'secondary_history_basis':'Conventional 1-to-2 drift; differing task identities make this confounded by which task came second.',
      'matching':'Same complete initialization; common A trained once/pair; sorted old classes and memory IDs ensure identical replay inputs in D/E. All augmented compact image tensors, labels and batch IDs hashed across all task updates.',
      'optimizer':'AdamW reset at every task; common D/E random seeds identical across branches. Stored buffers and classifier are allowed to carry history.',
      'scope':'Local history bases, not the manuscript pooled shared scaffold. Early class/image pools and number of tasks match, but replay presentation multiplicities during early B/C can depend on their order. No claim that identical initialization means equivalent states after different histories.',
      'primary_statistic':'Mean of [coverage(D_X,U_X)-coverage(D_X,U_Y)] and [coverage(D_Y,U_Y)-coverage(D_Y,U_X)]. Report both components, not only their mean.',
      'sampling':POSITIONS})
    trainset=d.IndexedCifar(Path('data'),True,True,'cifar10'); evalset=d.IndexedCifar(Path('data'),True,False,'cifar10')
    ids={c:d.ids_by_class(trainset.targets,[c],450) for c in range(10)}
    for layout in [3101,3111,3121]:
        perm=torch.randperm(10,generator=torch.Generator().manual_seed(layout)).tolist(); tasks=[perm[i:i+2] for i in range(0,10,2)]
        for seed in [180,181]:
            folder=OUT/f'layout{layout}_seed{seed}'; folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists(): continue
            t=time.perf_counter(); net=d.model('resnet18',False,num_classes=10).cuda(); net.load_state_dict(initial)
            priority=torch.rand(len(trainset),generator=torch.Generator().manual_seed(seed+9000))
            anchor_fit=d.balanced_memory(ids,sorted(tasks[0]),500,priority).sort().values
            anchor_test=d.ids_by_class(evalset.targets,tasks[0],50,tail=True)
            assert not torch.isin(anchor_fit,anchor_test).any()
            commonseed=layout*1000+seed*10
            train(net,trainset,evalset,torch.cat([ids[c] for c in sorted(tasks[0])]),torch.empty(0,dtype=torch.long),sorted(tasks[0]),commonseed)
            shared=state(net); fit_after_a=reps(net,evalset,anchor_fit)
            outputs={}
            for branch,sequence in [('X',[0,1,2,3,4]),('Y',[0,2,1,3,4])]:
                net.load_state_dict(shared); seen=sorted(tasks[0]); fits=[fit_after_a]; evals=[]; updates={}; traces={}; counts={}
                for position,task_index in enumerate(sequence[1:],2):
                    old=seen.copy(); seen=sorted(seen+tasks[task_index])
                    memory=d.balanced_memory(ids,old,500,priority).sort().values
                    current=torch.cat([ids[c] for c in sorted(tasks[task_index])])
                    obs,trace,count=train(net,trainset,evalset,current,memory,seen,commonseed+task_index*100,
                        anchor_test if position>=4 else None)
                    fit=reps(net,evalset,anchor_fit); ev=reps(net,evalset,anchor_test)
                    fits.append(fit); evals.append(ev); traces[position]=trace; counts[position]=count
                    if position>=4: updates[position]=obs
                    print(layout,seed,branch,'task',position,'seconds',round(time.perf_counter()-t,1),flush=True)
                history=f.center(fits[2]-fits[0]); conventional=f.center(fits[1]-fits[0])
                outputs[branch]={'basis':f.basis(history,8),'basis_1to2':f.basis(conventional,8),
                    'history_field':history,'fit_reps':fits,'eval_reps_after_tasks2to5':evals,'updates':updates,
                    'trace':traces,'steps':counts,'final_state':state(net)}
                torch.save(outputs[branch],folder/(branch+'.pt'))
            for task in [4,5]: assert outputs['X']['trace'][task]==outputs['Y']['trace'][task]
            torch.save({'tasks':tasks,'anchor_fit_ids':anchor_fit,'anchor_test_ids':anchor_test,'shared_after_A':shared},folder/'setup.pt')
            (folder/'complete.json').write_text(json.dumps({'layout':layout,'seed':seed,'seconds':time.perf_counter()-t,'common_task_streams_exact':True,'tasks':tasks}),encoding='utf-8')
            del net,outputs; torch.cuda.empty_cache()
    summarize(); savejson('completion.json',{'complete':True,'seconds':time.perf_counter()-started})

def summarize():
    rows=[]; geometry=[]
    for folder in sorted(OUT.glob('layout*')):
        meta=json.loads(read_config_text(folder/'complete.json')); a=f.load(folder/'X.pt'); b=f.load(folder/'Y.pt')
        for task in [4,5]:
            assert a['trace'][task]==b['trace'][task]
            for record in [a,b]:
                assert record['steps'][task]==145 and len(record['updates'][task])==9
                for obs in record['updates'][task]:
                    z=obs['field']; assert torch.isfinite(z).all() and z.square().sum()>0
                    assert float(z.mean(0).abs().max())<1e-10
        for record in [a,b]:
            for key in ['basis','basis_1to2']:
                assert float((record[key].T@record[key]-torch.eye(8)).abs().max())<1e-10
        ident={'layout':meta['layout'],'seed':meta['seed']}
        geometry.append({**ident,'early_basis_overlap':f.overlap(a['basis'],b['basis']),
            'conventional_basis_overlap':f.overlap(a['basis_1to2'],b['basis_1to2'])})
        for definition,key in [('equal_content','basis'),('one_transition','basis_1to2')]:
            for task in [4,5]:
                fields={}
                for branch,record in [('X',a),('Y',b)]:
                    zs=record['eval_reps_after_tasks2to5']; fields[branch]={'whole_task':[f.center(zs[task-2]-zs[task-3])]}
                    for window in ['early','middle','late']:
                        fields[branch][window]=[r['field'] for r in record['updates'][task] if r['window']==window]
                        assert len(fields[branch][window])==3
                for window in ['whole_task','early','middle','late']:
                    vals={}
                    for branch in ['X','Y']:
                        for name,u in [('X',a[key]),('Y',b[key])]:
                            vals[branch+'_using_'+name]=float(np.mean([f.coverage(z,u) for z in fields[branch][window]]))
                    x=vals['X_using_X']-vals['X_using_Y']; y=vals['Y_using_Y']-vals['Y_using_X']
                    rows.append({**ident,'definition':definition,'task':task,'window':window,**vals,
                        'X_own_gain':x,'Y_own_gain':y,'paired_own_gain':(x+y)/2})
    assert len(geometry)==6
    csvsave('geometry.csv',geometry); csvsave('paired_results.csv',rows)
    summaries=[]
    for definition in ['equal_content','one_transition']:
        for task in [4,5]:
            for window in ['whole_task','early','middle','late']:
                rr=[r for r in rows if r['definition']==definition and r['task']==task and r['window']==window]
                rng=np.random.default_rng(20260912); groups=[[i for i,r in enumerate(rr) if r['layout']==l] for l in [3101,3111,3121]]
                idx=np.array([[i for g in rng.integers(0,3,3) for i in rng.choice(groups[g],2,replace=True)] for _ in range(10000)])
                row={'definition':definition,'task':task,'window':window,'positive_pairs':sum(r['paired_own_gain']>0 for r in rr)}
                for k in ['X_own_gain','Y_own_gain','paired_own_gain','X_using_X','X_using_Y','Y_using_Y','Y_using_X']:
                    v=np.array([r[k] for r in rr]); lo,hi=np.quantile(v[idx].mean(1),[.025,.975])
                    row.update({k:float(v.mean()),k+'_low':float(lo),k+'_high':float(hi)})
                summaries.append(row)
    csvsave('summary.csv',summaries)
    fig,ax=plt.subplots(figsize=(7,3.6),layout='constrained')
    for task in [4,5]:
        rr=[r for r in summaries if r['definition']=='equal_content' and r['task']==task]
        m=np.array([r['paired_own_gain'] for r in rr])*100; lo=np.array([r['paired_own_gain_low'] for r in rr])*100; hi=np.array([r['paired_own_gain_high'] for r in rr])*100
        ax.errorbar(np.arange(4)+(task-4.5)*.06,m,yerr=[m-lo,hi-m],fmt='o-',capsize=3,label=f'Common task {task}')
    ax.set_xticks(range(4),['Whole task','Early updates','Middle updates','Late updates']); ax.axhline(0,color='gray',ls='--'); ax.legend()
    ax.set_ylabel('Own-history minus other-history\ncoverage (pp)'); ax.set_title('ResNet-18 / CIFAR-10: paired early-history order'); ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'comparison.pdf'); fig.savefig(OUT/'comparison.png',dpi=180); plt.close(fig)
    lines=['# Paired early-history order: ResNet-18 / CIFAR-10','',
      'Branches share exact initialization and task A. X learns A-B-C-D-E; Y learns A-C-B-D-E. Three random class partitions, two seeds each, six paired experiments. All compact augmented images, labels and batch identities during common tasks D and E are hashed and exactly equal between branches. Replay memory is balanced over sorted identical class sets and sorted sample IDs. AdamW resets per task; no optimizer state carries across the boundary.', '',
      'Primary rank-eight history bases use A-end to C/B-end drift on identical first-task training anchors: equal content and duration, different order. These are local history bases, not the pooled manuscript scaffold. Future drift uses held-out first-task anchors. Both branch-specific coverages are measured before averaging their own-minus-other gains. Positive values indicate preferential reuse of own history; merely different early spaces is insufficient.', '',
      f"Mean early-basis overlap: {100*np.mean([r['early_basis_overlap'] for r in geometry]):.2f}%.", '',
      '| Common task | Measurement | X own gain pp | Y own gain pp | Mean paired gain pp [95% CI] | Positive pairs |', '|---|---|---|---|---|---|']
    lines += [f"| {r['task']} | {r['window']} | {100*r['X_own_gain']:.2f} | {100*r['Y_own_gain']:.2f} | {100*r['paired_own_gain']:.2f} [{100*r['paired_own_gain_low']:.2f}, {100*r['paired_own_gain_high']:.2f}] | {r['positive_pairs']}/6 |" for r in summaries if r['definition']=='equal_content']
    negatives=[r for r in rows if r['definition']=='equal_content' and (r['X_own_gain']<=0 or r['Y_own_gain']<=0)]
    lines += ['', f'{len(negatives)} primary pair/window records contain a nonpositive individual branch gain, despite positive paired averages. Inspect paired_results.csv for both branches; a positive paired average does not mean every branch is positive.']
    lines += ['', '10,000 paired hierarchical bootstrap draws over three layouts and two seeds/layout; nominal pointwise intervals, small pilot. Secondary 1-to-2 history comparisons are saved separately: their early task identity differs and cannot isolate order effects. Early unique class/image pools match, but replay presentation multiplicities during B/C can depend on their order; only the common D/E streams are exactly matched. Different model parameters, classifier and buffers after early training are intended mediators, not held constant. The experiment does not distinguish neural parameter storage from normalization-state storage or recent-task recency, and does not show that the pooled shared scaffold itself changes. No accuracy-matched or state-matched claim is made.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    savejson('verification.json',{'pairs':6,'common_task_stream_hashes_identical':True,'sampled_updates':216,'primary_and_secondary_rows':len(rows),'complete':True})
    print('ANALYSIS COMPLETE',flush=True)
if __name__=='__main__': run()
