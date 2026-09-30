"""Two source-pooled 1->2 scaffolds; identical later tasks in independent test runs."""
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import drift_experiment as d
import analyze_scaffold_followup as f
from paired_history_memory_pilot import train, reps, state

OUT=Path('results/early_content_scaffold_pilot/resnet18_cifar10')
TASKS={'A':[4,6],'BX':[9,8],'BY':[7,0],'C':[2,1],'D':[3,5]}
FIT_SEEDS=list(range(220,226)); TEST_SEEDS=list(range(230,236))
def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def csvsave(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
def generate():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    OUT.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    initial_path=Path('results/resnet_temporal_dimension/cifar10/sources/artifacts/order2003_seed160.pt')
    source_hash=hashlib.sha256(initial_path.read_bytes()).hexdigest()
    initial=f.load(initial_path)['states'][0]
    protocol={'model':'ResNet-18','dataset':'CIFAR-10','rank':8,'tasks':TASKS,
        'sequence_X':['A','BX','C','D'],'sequence_Y':['A','BY','C','D'],
        'fit_seeds':FIT_SEEDS,'test_seeds':TEST_SEEDS,'epochs_per_task':5,'training_images_per_class':450,
        'definition':'Each scaffold pools six rank-8 centered 1->2 drift bases from estimation runs, then retains top eight eigenvectors of the mean projector. Separate X/Y pools; no test run enters fitting.',
        'anchors':'Identical 500 first-task training images for all bases, 100 disjoint first-task held-out images for test drift.',
        'initial_state':str(initial_path),'initial_state_sha256':source_hash,
        'replay':'500 images; task 2 replays A, task 3 replays A, task 4 replays A+C. Branch-specific task-2 classes excluded from subsequent replay to isolate their stored effects.',
        'output_classes':'Task 1: A; task 2: actual A+BX or A+BY; common tasks 3 and 4: identical all-ten-class output competition.',
        'optimizer':'AdamW lr 0.0003 wd 0.05, reset each task; AMP init scale 128, zero skips required.',
        'matching':'Common A trained once per pair from same complete initialization. Tasks 3/4 have identical image, augmentation, label and replay streams checked by SHA-256.',
        'scope':'Controlled content intervention, not standard all-old-class replay. Source runs stop after fitting stage 2; test runs complete all four tasks. One fixed class partition; six paired test seeds.',
        'statistics':'Complete 2x2 coverage per later task; paired own-minus-cross contrasts. 10000 paired test-seed bootstrap draws conditional on fitted scaffolds. Separate 2000 paired source bootstrap draws for scaffold similarity.',
        'status':'running'}
    write(OUT/'protocol.json',protocol)
    trainset=d.IndexedCifar(Path('data'),True,True,'cifar10'); evalset=d.IndexedCifar(Path('data'),True,False,'cifar10')
    ids={c:d.ids_by_class(trainset.targets,[c],450) for c in range(10)}
    fitpriority=torch.rand(len(trainset),generator=torch.Generator().manual_seed(820012))
    fitids=d.balanced_memory(ids,sorted(TASKS['A']),500,fitpriority).sort().values
    testids=d.ids_by_class(evalset.targets,sorted(TASKS['A']),50,tail=True)
    assert not torch.isin(testids,torch.cat(list(ids.values()))).any()
    torch.save({'fit_ids':fitids,'test_ids':testids,'tasks':TASKS},OUT/'anchors.pt')
    for role,seeds in [('fit',FIT_SEEDS),('test',TEST_SEEDS)]:
        for seed in seeds:
            folder=OUT/f'{role}_seed{seed}'; folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists(): continue
            t=time.perf_counter(); net=d.model('resnet18',False,num_classes=10).cuda()
            priority=torch.rand(len(trainset),generator=torch.Generator().manual_seed(seed+9000))
            def current(classes): return torch.cat([ids[c] for c in sorted(classes)])
            def memory(classes): return d.balanced_memory(ids,sorted(classes),500,priority).sort().values
            sharedpath=folder/'shared_A.pt'
            if sharedpath.exists(): shared=f.load(sharedpath)
            else:
                net.load_state_dict(initial)
                _,trace,steps=train(net,trainset,evalset,current(TASKS['A']),torch.empty(0,dtype=torch.long),sorted(TASKS['A']),900000+seed*100)
                shared={'state':state(net),'fit':reps(net,evalset,fitids),'eval':reps(net,evalset,testids),'trace':trace,'steps':steps}
                torch.save(shared,sharedpath)
            traces={}; baselines={}
            for branch in ['X','Y']:
                previous=shared
                for task in range(2,3 if role=='fit' else 5):
                    path=folder/f'{branch}_task{task}.pt'
                    if path.exists(): record=f.load(path)
                    else:
                        net.load_state_dict(previous['state'])
                        classes=TASKS['B'+branch] if task==2 else TASKS['C' if task==3 else 'D']
                        memclasses=TASKS['A'] if task<4 else TASKS['A']+TASKS['C']
                        seen=sorted(TASKS['A']+classes) if task==2 else list(range(10))
                        _,trace,steps=train(net,trainset,evalset,current(classes),memory(memclasses),seen,900000+seed*100+task*10)
                        assert steps==145
                        zfit=reps(net,evalset,fitids); zeval=reps(net,evalset,testids)
                        record={'state':state(net),'fit':zfit,'eval':zeval,'trace':trace,'steps':steps,
                            'task':task,'branch':branch,'seed':seed,'memory_ids':memory(memclasses),'current_ids':current(classes),'seen':seen,
                            'drift_eval':f.center(zeval-previous['eval'])}
                        if task==2:
                            record['drift_fit']=f.center(zfit-shared['fit']); record['basis']=f.basis(record['drift_fit'],8)
                        torch.save(record,path)
                    traces[(branch,task)]=record['trace']; previous=record
                    print(role,seed,branch,'task',task,'seconds',round(time.perf_counter()-t,1),flush=True)
            if role=='test':
                for task in [3,4]: assert traces[('X',task)]==traces[('Y',task)]
            write(folder/'complete.json',{'role':role,'seed':seed,'seconds':time.perf_counter()-t,'common_streams_identical':role=='test'})
            del net,previous,record,shared; torch.cuda.empty_cache()
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest()==source_hash
    protocol['status']='training_complete'; write(OUT/'protocol.json',protocol)
    analyze(); write(OUT/'completion.json',{'status':'complete','seconds':time.perf_counter()-started})

def analyze():
    torch.set_num_threads(2)
    source={b:[f.load(OUT/f'fit_seed{s}'/f'{b}_task2.pt')['basis'] for s in FIT_SEEDS] for b in ['X','Y']}
    bases={b:f.pool(source[b],8) for b in ['X','Y']}; torch.save(bases,OUT/'scaffolds.pt')
    overlap=f.overlap(bases['X'],bases['Y'])
    rng=np.random.default_rng(20260912); draws=rng.integers(0,6,(2000,6)); cache={}
    for draw in draws:
        counts=tuple(np.bincount(draw,minlength=6))
        if counts not in cache:
            bb={b:f.pool([u*np.sqrt(n) for u,n in zip(source[b],counts) if n],8) for b in source}
            cache[counts]=f.overlap(bb['X'],bb['Y'])
    overlap_ci=np.quantile([cache[tuple(np.bincount(x,minlength=6))] for x in draws],[.025,.975]).tolist()
    similarity={'overlap':overlap,'ci95_source_bootstrap':overlap_ci,'principal_angle_degrees':torch.rad2deg(torch.acos(torch.linalg.svdvals(bases['X'].T@bases['Y']).clamp(0,1))).tolist(),
        'uniform_expected_overlap':8/512,'definition':'||U_X^T U_Y||_F^2 / 8','source_networks_per_scaffold':6}
    write(OUT/'similarity.json',similarity)
    rows=[]; fields=[]; checks=[]
    for seed in TEST_SEEDS:
        for task in [3,4]:
            rec={b:f.load(OUT/f'test_seed{seed}'/f'{b}_task{task}.pt') for b in ['X','Y']}
            assert rec['X']['trace']==rec['Y']['trace']
            for b in rec:
                assert rec[b]['steps']==145 and rec[b]['seen']==list(range(10))
                z=rec[b]['drift_eval']; assert torch.isfinite(z).all() and z.square().sum()>0
                assert float(z.mean(0).abs().max())<1e-10
                for v in bases.values():
                    e=abs(float((z@v).square().sum()-((z@v)@v.T).square().sum())/float(z.square().sum())); checks.append(e)
            row={'seed':seed,'task':task}
            for branch in ['X','Y']:
                z=rec[branch]['drift_eval']
                for scaffold in ['X','Y']: row[f'{branch}_using_{scaffold}']=f.coverage(z,bases[scaffold])
                row[branch+'_drift_energy']=float(z.square().sum())
                fields.append({'seed':seed,'task':task,'branch':branch,'field':z})
            row['X_own_minus_cross']=row['X_using_X']-row['X_using_Y']
            row['Y_own_minus_cross']=row['Y_using_Y']-row['Y_using_X']
            row['paired_own_minus_cross']=(row['X_own_minus_cross']+row['Y_own_minus_cross'])/2
            rows.append(row)
    assert max(checks)<1e-10
    csvsave('per_test_seed.csv',rows); torch.save(fields,OUT/'test_drift_fields.pt')
    summaries=[]; matrix=[]
    for task in [3,4]:
        rr=[r for r in rows if r['task']==task]; ids=np.random.default_rng(20260912).integers(0,6,(10000,6))
        row={'task':task,'positive_pairs':sum(r['paired_own_minus_cross']>0 for r in rr),
            'X_positive_runs':sum(r['X_own_minus_cross']>0 for r in rr),'Y_positive_runs':sum(r['Y_own_minus_cross']>0 for r in rr)}
        for key in ['X_using_X','X_using_Y','Y_using_X','Y_using_Y','X_own_minus_cross','Y_own_minus_cross','paired_own_minus_cross']:
            values=np.array([r[key] for r in rr]); lo,hi=np.quantile(values[ids].mean(1),[.025,.975])
            row.update({key:float(values.mean()),key+'_low':float(lo),key+'_high':float(hi)})
        summaries.append(row)
        for branch in ['X','Y']:
            for scaffold in ['X','Y']:
                key=f'{branch}_using_{scaffold}'
                matrix.append({'task':task,'drift_branch':branch,'scaffold':scaffold,'coverage':row[key],'ci_low':row[key+'_low'],'ci_high':row[key+'_high']})
    csvsave('summary.csv',summaries); csvsave('cross_coverage.csv',matrix)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    allcov=[r['coverage']*100 for r in matrix]
    for ax,task in zip(axes[:2],[3,4]):
        vals=np.array([[next(r['coverage']*100 for r in matrix if r['task']==task and r['drift_branch']==b and r['scaffold']==s) for s in ['X','Y']] for b in ['X','Y']])
        ax.imshow(vals,cmap='Blues',vmin=0,vmax=max(allcov)*1.3)
        for i in range(2):
            for j in range(2): ax.text(j,i,f'{vals[i,j]:.2f}%',ha='center',va='center',color='black')
        ax.set_xticks([0,1],['Scaffold X','Scaffold Y']); ax.set_yticks([0,1],['Drift X','Drift Y']); ax.set_title(f'Common task {task}')
    ax=axes[2]
    for offset,key,label in [(-.12,'X_own_minus_cross','Branch X'),(0,'Y_own_minus_cross','Branch Y'),(.12,'paired_own_minus_cross','Paired mean')]:
        m=np.array([r[key] for r in summaries])*100; lo=np.array([r[key+'_low'] for r in summaries])*100; hi=np.array([r[key+'_high'] for r in summaries])*100
        ax.errorbar(np.arange(2)+offset,m,yerr=[m-lo,hi-m],fmt='o',capsize=3,label=label)
    ax.set_xticks([0,1],['Task 3','Task 4']); ax.set_ylabel('Own minus cross coverage (pp)'); ax.axhline(0,color='gray',ls='--'); ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'ResNet-18 / CIFAR-10: two pooled 1-to-2 scaffolds (overlap {100*overlap:.2f}%)')
    fig.savefig(OUT/'comparison.pdf'); fig.savefig(OUT/'comparison.png',dpi=180); plt.close(fig)
    lines=['# Early task content and source-pooled scaffolds: ResNet-18 / CIFAR-10','',
      '| Position | Branch X | Branch Y |','|---|---|---|','| 1 | deer, frog | deer, frog |','| 2 | truck, ship | horse, airplane |','| 3 | bird, automobile | bird, automobile |','| 4 | cat, dog | cat, dog |','',
      'The two rank-eight scaffolds use the original centered 1-to-2 drift definition, pooled over six estimation networks per branch. Six other paired test seeds complete four tasks and never enter fitting. Shared A uses the same complete initialization and is trained once per pair. Fixed fitting anchors are 500 training images from A; 100 held-out A images measure test drift.', '',
      f"Scaffold overlap: {100*overlap:.2f}%; paired source-bootstrap 95% interval [{100*overlap_ci[0]:.2f}, {100*overlap_ci[1]:.2f}]. Uniform independent rank-eight expectation: 1.5625%. Similarity alone does not establish historical specificity.", '',
      '## Cross-coverage matrices','', '| Task | Drift branch | Scaffold X coverage % [95% CI] | Scaffold Y coverage % [95% CI] |', '|---|---|---|---|']
    for r in summaries:
        for b in ['X','Y']:
            x=f'{b}_using_X'; y=f'{b}_using_Y'
            lines.append(f"| {r['task']} | {b} | {100*r[x]:.2f} [{100*r[x+'_low']:.2f}, {100*r[x+'_high']:.2f}] | {100*r[y]:.2f} [{100*r[y+'_low']:.2f}, {100*r[y+'_high']:.2f}] |")
    lines += ['', '| Task | X own minus cross pp [95% CI] | Y own minus cross pp [95% CI] | Paired mean pp [95% CI] | Positive pairs |','|---|---|---|---|---|']
    for r in summaries:
        cells=[f"{100*r[k]:+.2f} [{100*r[k+'_low']:+.2f}, {100*r[k+'_high']:+.2f}]" for k in ['X_own_minus_cross','Y_own_minus_cross','paired_own_minus_cross']]
        lines.append(f"| {r['task']} | {' | '.join(cells)} | {r['positive_pairs']}/6 |")
    lines += ['', '## Matching and scope','',
      'During task 3 both branches replay A only; during task 4 both replay A+C. Branch-specific second-task data are excluded from later replay. All common-task augmented images, labels and batch IDs match exactly by SHA-256. Both later tasks use the same ten-class output competition; the task-2 output masks contain the actual classes learned by each branch. This is a dedicated content-control experiment, not the original all-old-class replay protocol. Model parameters, head and normalization buffers may retain early content effects; AdamW is reset per task.', '',
      'Coverage intervals use 10,000 paired resamples of six test seeds, conditional on the fitted scaffolds; source-fit uncertainty is not included in these coverage intervals. The overlap interval separately refits both scaffold pools in 2,000 paired source-bootstrap draws. Only one class partition and shared pretrained initialization are tested. Estimation/test runs share training content but differ in optimization seeds. Opposite signs for the two branch contrasts must not be hidden by their paired mean. Neither similarity nor a null difference proves history-independence.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'verification.json',{'complete':True,'source_networks_per_scaffold':6,'test_pairs':6,'common_task_streams_exact':True,
        'source_test_seeds_disjoint':not bool(set(FIT_SEEDS)&set(TEST_SEEDS)),'rank':8,'projection_identity_max_error':max(checks),'test_drift_fields':24})
    print('ANALYSIS COMPLETE',json.dumps({'similarity':similarity,'summary':summaries}),flush=True)

if __name__=='__main__': generate()
