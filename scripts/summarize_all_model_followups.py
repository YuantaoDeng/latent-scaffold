"""Produce independently scoped residual and content follow-up reports."""
from package_paths import read_config_text
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_scaffold_followup as f
from all_model_followups_config import ROOT,conditions

def csvsave(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def readcsv(path):return list(csv.DictReader(path.open(encoding='utf-8')))
def load(path):return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
def residual():
    rows=[];primary=[];summary=[]
    fig,axes=plt.subplots(2,4,figsize=(13,6.5),layout='constrained')
    for c,ax in zip(conditions(),axes.T.flat):
        out=ROOT/'residual'/f"{c['short']}_{c['dataset']}";check=json.loads(read_config_text(out/'verification.json'));assert check['complete']
        rr=readcsv(out/'summary.csv');rows+=rr
        pp=[r for r in rr if r['pca_timing']=='start' and int(r['pca_rank'])==c['rank']];primary+=pp
        m=np.array([float(r['gain']) for r in pp])*100;lo=np.array([float(r['gain_low']) for r in pp])*100;hi=np.array([float(r['gain_high']) for r in pp])*100
        ax.errorbar(range(len(pp)),m,yerr=[m-lo,hi-m],fmt='o-',capsize=3)
        ax.axhline(0,color='gray',ls='--',lw=.8);ax.set_xticks(range(len(pp)),[r['transition'] for r in pp]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.set_title(c['model']+' / '+c['dataset'].replace('cifar','CIFAR-')+f"\nscaffold/PCA rank {c['rank']}",fontsize=9)
        sm={'model':c['model'],'dataset':c['dataset'],'rank':c['rank'],'primary_transitions':len(pp),
            'positive_primary_ci':sum(float(r['gain_low'])>0 for r in pp),'gain_min_pp':float(min(m)),'gain_max_pp':float(max(m)),
            'positive_history_windows':sum(int(r['positive_histories']) for r in pp),'history_windows':12*len(pp)}
        for k in [8,16,32]:
            kk=[r for r in rr if r['pca_timing']=='start' and int(r['pca_rank'])==k]
            sm[f'pca{k}_positive_ci']=sum(float(r['gain_low'])>0 for r in kk)
        summary.append(sm)
    fig.supylabel('Residual coverage gain over uniform random (pp)');fig.supxlabel('Task transition')
    fig.suptitle('Current PCA removed before each transition: all eight conditions',fontsize=13)
    fig.savefig(ROOT/'residual_all_models.pdf');fig.savefig(ROOT/'residual_all_models.png',dpi=180);plt.close(fig)
    csvsave(ROOT/'residual_full_summary.csv',rows);csvsave(ROOT/'residual_primary.csv',primary);csvsave(ROOT/'residual_condition_summary.csv',summary)
    count=sum(x['positive_primary_ci'] for x in summary);total=sum(x['primary_transitions'] for x in summary)
    lines=['# All-model residual PCA follow-up','',f'Completed: 8 model/dataset conditions, 96 test histories. Primary positive 95% gain intervals: {count}/{total} task transitions.', '',
        'Each condition retains its original scaffold rank (ResNet 8/8, DeiT 8/20, MobileNet 24/8, Mixer 8/8 for CIFAR-10/100). Primary removal uses the same number of leading local PCA directions. All conditions additionally remove 8, 16 and 32 directions. Test PCA uses first-task training anchors at the start of each transition; drift uses disjoint held-out anchors. End-of-transition PCA is separately recorded as retrospective sensitivity.', '',
        '| Model | Dataset | Rank | Primary positive intervals | Residual gain range pp | Positive history/transition windows | Top-8 / 16 / 32 positive intervals |','|---|---|---|---|---|---|---|']
    lines += [f"| {r['model']} | {r['dataset']} | {r['rank']} | {r['positive_primary_ci']}/{r['primary_transitions']} | {r['gain_min_pp']:.2f} to {r['gain_max_pp']:.2f} | {r['positive_history_windows']}/{r['history_windows']} | {r['pca8_positive_ci']} / {r['pca16_positive_ci']} / {r['pca32_positive_ci']} |" for r in summary]
    lines += ['', 'Residual scaffold is the orthonormal basis of the original scaffold projected out of PCA; it is not fitted to the future residual. Its rank is verified unchanged. Uniform rank-q random subspaces in the d-k complement have exact expected coverage q/(d-k), so Monte Carlo sampling is unnecessary for the population mean. These controls are not classifier matched. Both residual-normalized and original-total-motion-normalized increments are retained, together with raw fixed-scaffold coverage without reorthonormalization.', '',
        'Intervals use 10,000 paired hierarchical task-order/run draws, conditional on existing fitted scaffolds, without multiplicity adjustment. This establishes additional motion geometry beyond the removed leading variance directions in these conditions; it does not establish unique causal importance, all-PCA independence, or training-content-disjoint transfer. The ResNet/CIFAR-10 analysis is recomputed consistently and checked against the pilot. No new training was needed.']
    # Exact agreement with pilot's main and sensitivity measurements.
    pilot=readcsv(Path('results/residual_pca_pilot/resnet18_cifar10/summary.csv'))
    canonical=[r for r in rows if r['model']=='ResNet-18' and r['dataset']=='cifar10']
    error=max(abs(float(r[key])-float(next(x for x in pilot if x['transition']==r['transition'] and x['pca_timing']==r['pca_timing'] and x['pca_rank']==r['pca_rank'])[key])) for r in canonical for key in ['gain','extra_total_coverage','raw_gain'])
    assert error<1e-10,error
    (ROOT/'RESIDUAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'residual_verification.json').write_text(json.dumps({'complete':True,'conditions':8,'histories':96,'primary_positive_ci':count,'primary_transitions':total,'resnet_pilot_max_error':error},indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

def content_condition(c,out_override=None,task_indices=(3,4),expected_steps=145,common_seen=None):
    out=Path(out_override) if out_override else ROOT/'content'/f"{c['short']}_cifar10";r=c['rank']
    common_seen=list(range(10)) if common_seen is None else common_seen
    if not all((out/f'{role}_seed{s}'/'complete.json').exists() for role,seeds in [('fit',range(220,226)),('test',range(230,236))] for s in seeds):return False
    source={b:[load(out/f'fit_seed{s}'/f'{b}_task2.pt')['basis'] for s in range(220,226)] for b in ['X','Y']}
    bases={b:f.pool(source[b],r) for b in source};dim=bases['X'].shape[0];torch.save(bases,out/'scaffolds.pt')
    overlap=f.overlap(bases['X'],bases['Y']);cache={};draws=np.random.default_rng(20260912).integers(0,6,(2000,6))
    for draw in draws:
        counts=tuple(np.bincount(draw,minlength=6))
        if counts not in cache:
            bb={b:f.pool([u*np.sqrt(n) for u,n in zip(source[b],counts) if n],r) for b in source};cache[counts]=f.overlap(bb['X'],bb['Y'])
    ci=np.quantile([cache[tuple(np.bincount(x,minlength=6))] for x in draws],[.025,.975]).tolist()
    (out/'similarity.json').write_text(json.dumps({'overlap':overlap,'ci95_source_bootstrap':ci,'rank':r,'dimension':dim,'uniform_expected_overlap':r/dim,'source_networks_per_scaffold':6},indent=2),encoding='utf-8')
    rows=[];fields=[];maxerror=0.
    for seed in range(230,236):
        for task in task_indices:
            rec={b:load(out/f'test_seed{seed}'/f'{b}_task{task}.pt') for b in source}
            assert rec['X']['trace']==rec['Y']['trace']
            assert torch.equal(rec['X']['memory_ids'],rec['Y']['memory_ids']) and torch.equal(rec['X']['current_ids'],rec['Y']['current_ids'])
            row={'seed':seed,'task':task}
            for b in source:
                assert rec[b]['steps']==expected_steps and rec[b]['seen']==common_seen
                z=rec[b]['drift_eval'];assert torch.isfinite(z).all() and z.square().sum()>0 and float(z.mean(0).abs().max())<1e-10
                for s,u in bases.items():
                    assert u.shape[1]==r and float((u.T@u-torch.eye(r)).abs().max())<1e-10
                    row[f'{b}_using_{s}']=f.coverage(z,u)
                    err=abs(float((z@u).square().sum()-((z@u)@u.T).square().sum())/float(z.square().sum()));maxerror=max(maxerror,err)
                row[b+'_drift_energy']=float(z.square().sum());fields.append({'seed':seed,'task':task,'branch':b,'field':z.clone()})
            row['X_own_minus_cross']=row['X_using_X']-row['X_using_Y'];row['Y_own_minus_cross']=row['Y_using_Y']-row['Y_using_X'];row['paired_own_minus_cross']=(row['X_own_minus_cross']+row['Y_own_minus_cross'])/2;rows.append(row)
    assert maxerror<1e-10
    csvsave(out/'per_test_seed.csv',rows);torch.save(fields,out/'test_drift_fields.pt');summaries=[];matrix=[]
    for task in task_indices:
        rr=[x for x in rows if x['task']==task];ids=np.random.default_rng(20260912).integers(0,6,(10000,6))
        row={'task':task,'positive_pairs':sum(x['paired_own_minus_cross']>0 for x in rr),'X_positive_runs':sum(x['X_own_minus_cross']>0 for x in rr),'Y_positive_runs':sum(x['Y_own_minus_cross']>0 for x in rr)}
        for key in ['X_using_X','X_using_Y','Y_using_X','Y_using_Y','X_own_minus_cross','Y_own_minus_cross','paired_own_minus_cross']:
            v=np.array([x[key] for x in rr]);lo,hi=np.quantile(v[ids].mean(1),[.025,.975]);row.update({key:float(v.mean()),key+'_low':float(lo),key+'_high':float(hi)})
        summaries.append(row)
        for b in source:
            for s in source:
                k=f'{b}_using_{s}';matrix.append({'task':task,'drift_branch':b,'scaffold':s,'coverage':row[k],'ci_low':row[k+'_low'],'ci_high':row[k+'_high']})
    csvsave(out/'summary.csv',summaries);csvsave(out/'cross_coverage.csv',matrix)
    (out/'verification.json').write_text(json.dumps({'complete':True,'source_networks_per_scaffold':6,'test_pairs':6,'rank':r,'projection_identity_max_error':maxerror,'common_task_streams_exact':True,'source_test_seeds_disjoint':True,'test_drift_fields':12*len(task_indices)},indent=2),encoding='utf-8')
    return True

def content(allow_partial=False):
    allrows=[];similarities=[];matrix=[]
    cs=[c for c in conditions() if c['dataset']=='cifar10'];complete=[]
    for c in cs:
        out=Path('results/early_content_scaffold_pilot/resnet18_cifar10') if c['short']=='resnet18' else ROOT/'content'/f"{c['short']}_cifar10"
        if c['short']!='resnet18' and not (out/'verification.json').exists():
            if not content_condition(c):
                if allow_partial:continue
                raise RuntimeError('Incomplete '+c['short'])
        complete.append(c);sim=json.loads(read_config_text(out/'similarity.json'));similarities.append({'model':c['model'],'rank':c['rank'],'overlap':sim['overlap'],'overlap_low':sim['ci95_source_bootstrap'][0],'overlap_high':sim['ci95_source_bootstrap'][1]})
        allrows += [{'model':c['model'],'rank':c['rank'],**r} for r in readcsv(out/'summary.csv')]
        matrix += [{'model':c['model'],'rank':c['rank'],**r} for r in readcsv(out/'cross_coverage.csv')]
    if not complete:return
    csvsave(ROOT/'content_summary.csv',allrows);csvsave(ROOT/'content_similarity.csv',similarities);csvsave(ROOT/'content_cross_coverage.csv',matrix)
    fig,axes=plt.subplots(1,len(complete),figsize=(3.5*len(complete),3.7),layout='constrained',squeeze=False)
    for c,ax in zip(complete,axes.flat):
        rr=[r for r in allrows if r['model']==c['model']]
        for offset,key,label in [(-.12,'X_own_minus_cross','X'),(0,'Y_own_minus_cross','Y'),(.12,'paired_own_minus_cross','Paired')]:
            m=np.array([float(r[key]) for r in rr])*100;lo=np.array([float(r[key+'_low']) for r in rr])*100;hi=np.array([float(r[key+'_high']) for r in rr])*100
            ax.errorbar(np.arange(2)+offset,m,yerr=[m-lo,hi-m],fmt='o',capsize=3,label=label)
        ax.set_xticks([0,1],['Task 3','Task 4']);ax.axhline(0,color='gray',ls='--');ax.set_title(c['model']+f" (rank {c['rank']})",fontsize=10);ax.spines[['top','right']].set_visible(False)
    axes.flat[0].legend(fontsize=8);fig.supylabel('Own minus cross coverage (pp)');fig.suptitle('CIFAR-10: effects of early task content')
    fig.savefig(ROOT/'content_all_models.pdf');fig.savefig(ROOT/'content_all_models.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,len(complete),figsize=(3*len(complete),5.5),layout='constrained',squeeze=False)
    vmax=max(float(x['coverage'])*100 for x in matrix)
    for col,c in enumerate(complete):
        for row,task in enumerate([3,4]):
            ax=axes[row,col]
            cells=[x for x in matrix if x['model']==c['model'] and int(x['task'])==task]
            values=np.array([[100*float(next(x['coverage'] for x in cells if x['drift_branch']==b and x['scaffold']==s)) for s in ['X','Y']] for b in ['X','Y']])
            im=ax.imshow(values,vmin=0,vmax=vmax,cmap='Blues')
            for i in range(2):
                for j in range(2):ax.text(j,i,f'{values[i,j]:.2f}%',ha='center',va='center',color='white' if values[i,j]>.55*vmax else 'black',fontweight='bold' if i==j else 'normal')
            ax.set_xticks([0,1],['Scaffold X','Scaffold Y']);ax.set_yticks([0,1],['Drift X','Drift Y']);ax.set_title(c['model']+f' / task {task}',fontsize=10)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Drift coverage (%)',shrink=.8)
    fig.suptitle('Identical later tasks: full own/cross coverage matrices')
    fig.savefig(ROOT/'content_cross_matrices.pdf');fig.savefig(ROOT/'content_cross_matrices.png',dpi=180);plt.close(fig)
    lines=['# All-model early-content follow-up','',f'Completed models: {len(complete)}/4. Dataset: CIFAR-10.', '',
        'X: deer/frog -> truck/ship -> bird/automobile -> cat/dog. Y replaces only task 2 with horse/airplane. Each scaffold pools original 1-to-2 drift from six estimation runs. Six separate paired test seeds complete all four tasks. Existing manuscript ranks are fixed, equal across X/Y; no new test-based rank selection. ResNet reuses its completed pilot.', '',
        '| Model | Rank | Scaffold overlap % [source-resampling percentile range] |','|---|---|---|']
    lines += [f"| {r['model']} | {r['rank']} | {100*r['overlap']:.2f} [{100*r['overlap_low']:.2f}, {100*r['overlap_high']:.2f}] |" for r in similarities]
    lines += ['', '| Model | Task | X own gain pp [95% CI] | Y own gain pp [95% CI] | Paired gain pp [95% CI] | Positive pairs |','|---|---|---|---|---|---|']
    for r in allrows:
        cells=[f"{100*float(r[k]):+.2f} [{100*float(r[k+'_low']):+.2f}, {100*float(r[k+'_high']):+.2f}]" for k in ['X_own_minus_cross','Y_own_minus_cross','paired_own_minus_cross']]
        lines.append(f"| {r['model']} | {r['task']} | {' | '.join(cells)} | {r['positive_pairs']}/6 |")
    lines += ['', 'Full four-cell cross-coverage matrices are in content_cross_coverage.csv, and individual paired seeds are retained per model. Both branch contrasts must be inspected: a positive paired mean alone does not establish preference in both branches.', '',
        'Task 3 replays A only; task 4 replays A+C. Branch-specific task-2 images are excluded from later replay. Common-task augmented input hashes and batch/replay IDs must match exactly. Common output competition includes all ten classes; AdamW resets per task. This controlled replay/output protocol isolates stored early-content effects but differs from original all-old-class replay. Model states, heads and normalization buffers may mediate history effects.', '',
        'Coverage intervals use 10,000 paired test-seed draws conditional on fitted scaffolds. Similarity intervals separately refit source pools with 2,000 paired draws. One class partition per model, shared pretrained initialization within each model, and overlapping training content between estimation/test runs; results do not isolate architecture from pretraining or establish unique causal memory storage. Mixer uses previously validated activation recomputation without changing batch size or epochs.']
    lines += ['', 'MobileNet uses full FP32 training uniformly across all retained fitting and test runs. AMP attempts at initial scales 128 and 32 encountered gradient overflow; all AMP MobileNet runs, including completed runs, were excluded and archived before restarting the entire condition. Learning rate, optimizer, data, batch size and epochs were unchanged. Every retained task requires the full scheduled update count. The numerical policy is recorded in content/mobilenetv3_small_cifar10/numerical_policy.json. Other models retain AMP training.']
    lines += ['', 'The source-resampling overlap range is descriptive, not a calibrated confidence interval: with only six source runs, resampling repeats sources and changes the effective diversity of the pooled estimator. In MobileNet the original overlap lies above this percentile range. The point overlap and this fitting sensitivity are therefore reported separately from the held-out coverage contrasts; the latter are the primary evidence for history preference.']
    (ROOT/'CONTENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'content_verification.json').write_text(json.dumps({'complete':len(complete)==4,'models':len(complete),'test_pairs_per_model':6,'source_networks_per_scaffold':6},indent=2),encoding='utf-8')
    print('Content summaries complete:',len(complete),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--residual',action='store_true');p.add_argument('--content',action='store_true');p.add_argument('--allow-partial',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    if a.residual:residual()
    if a.content:content(a.allow_partial)
