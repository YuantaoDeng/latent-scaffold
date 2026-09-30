"""Summarize and audit the pre-specified CIFAR-100 three-task experiment."""
from package_paths import read_config_text
import argparse,json,pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from resnet_cifar100_content import TASKS
from all_model_followups_config import conditions
from summarize_all_model_followups import content_condition,readcsv,load

def main(short='resnet18'):
    torch.set_num_threads(2)
    OUT=Path('results/early_content_three_task')/f'{short}_cifar100'
    c=next(c for c in conditions() if c['short']==short and c['dataset']=='cifar100')
    common=sorted(set(sum(TASKS.values(),[])))
    assert len(common)==40 and len(set(TASKS['A']))==10
    assert content_condition(c,out_override=OUT,task_indices=(3,),expected_steps=705,common_seen=common)
    anchors=load(OUT/'anchors.pt')
    assert len(anchors['fit_ids'])==500 and len(anchors['test_ids'])==500
    assert not torch.isin(anchors['fit_ids'],anchors['test_ids']).any()
    checked=0
    for role,seeds in [('fit',range(220,226)),('test',range(230,236))]:
        for seed in seeds:
            folder=OUT/f'{role}_seed{seed}'
            done=json.loads(read_config_text(folder/'complete.json'))
            assert done['dataset']=='cifar100' and done['rank']==c['rank'] and done['final_task']==3
            if short=='mobilenetv3_small':assert done['amp_enabled'] is False
            if short=='mixer_b16':assert done['activation_recomputation'] is True
            shared=load(folder/'shared_A.pt');assert shared['steps']==355
            for b in ['X','Y']:
                previous=shared
                for task in range(2,3 if role=='fit' else 4):
                    rec=load(folder/f'{b}_task{task}.pt')
                    assert rec['steps']==705 and len(rec['current_ids'])==4500 and len(rec['memory_ids'])==500
                    assert not torch.isin(anchors['test_ids'],rec['current_ids']).any()
                    assert not torch.isin(anchors['test_ids'],rec['memory_ids']).any()
                    z=rec['eval']-previous['eval'];z=z-z.mean(0,keepdim=True)
                    assert torch.allclose(z,rec['drift_eval'],atol=1e-12,rtol=1e-12)
                    if task==3:assert rec['seen']==common
                    checked+=1;previous=rec
    assert checked==36
    row=readcsv(OUT/'summary.csv')[0];runs=readcsv(OUT/'per_test_seed.csv');sim=json.loads(read_config_text(OUT/'similarity.json'))
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.7),layout='constrained')
    values=np.array([[100*float(row[f'{b}_using_{s}']) for s in ['X','Y']] for b in ['X','Y']])
    axes[0].imshow(values,cmap='Blues',vmin=0,vmax=float(values.max())*1.1)
    for i in range(2):
        for j in range(2):axes[0].text(j,i,f'{values[i,j]:.2f}%',ha='center',va='center',color='white' if values[i,j]>.6*values.max() else 'black',fontweight='bold' if i==j else 'normal')
    axes[0].set_xticks([0,1],['Scaffold X','Scaffold Y']);axes[0].set_yticks([0,1],['Drift X','Drift Y']);axes[0].set_title('Task 3 drift coverage')
    keys=['X_own_minus_cross','Y_own_minus_cross','paired_own_minus_cross']
    for i,k in enumerate(keys):
        vals=np.array([float(x[k]) for x in runs])*100
        axes[1].scatter(i+np.linspace(-.08,.08,6),vals,color='gray',alpha=.55,s=18)
        m,lo,hi=[100*float(row[k+x]) for x in ['', '_low','_high']]
        axes[1].errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='C0')
    axes[1].axhline(0,ls='--',lw=1,color='gray');axes[1].set_xticks([0,1,2],['X','Y','Paired']);axes[1].set_ylabel('Own minus cross coverage (pp)');axes[1].set_title('Six paired test runs; 95% bootstrap CI');axes[1].spines[['top','right']].set_visible(False)
    fig.suptitle(c['model']+' / CIFAR-100: three-task early-content test')
    fig.savefig(OUT/'results.pdf');fig.savefig(OUT/'results.png',dpi=180);plt.close(fig)
    names=pickle.load(open('data/cifar-100-python/meta','rb'),encoding='latin1')['fine_label_names']
    lines=['# '+c['model']+' / CIFAR-100: three-task early-content experiment','',
      f"Completed: six paired estimation seeds and six separate paired test seeds. Every history has three tasks of ten classes. Scaffold rank is fixed at {c['rank']}, matching the existing manuscript condition. No test-based direction or rank fitting.", '',
      'X: A -> BX -> C. Y: A -> BY -> C. Task blocks were fixed from original order 1103 before new results. Each model starts from the original CIFAR-100 experiment initialization; A is trained once per paired seed and copied to both branches.', '']
    lines += [f"- {k}: "+', '.join(names[i] for i in ids)+f' (labels {ids})' for k,ids in TASKS.items()]
    lines += ['',f"Scaffold overlap: {100*sim['overlap']:.2f}%. Uniform same-rank overlap expectation: {100*sim['uniform_expected_overlap']:.2f}%.", '',
      '| Task 3 drift | Scaffold X coverage | Scaffold Y coverage |','|---|---|---|']
    for b in ['X','Y']:lines.append(f"| {b} | {100*float(row[b+'_using_X']):.2f}% | {100*float(row[b+'_using_Y']):.2f}% |")
    lines += ['', '| Contrast | Gain pp [95% CI] | Positive runs |','|---|---|---|']
    for k,count,label in zip(keys,['X_positive_runs','Y_positive_runs','positive_pairs'],['X own minus cross','Y own minus cross','Paired mean']):
        lines.append(f"| {label} | {100*float(row[k]):+.2f} [{100*float(row[k+'_low']):+.2f}, {100*float(row[k+'_high']):+.2f}] | {row[count]}/6 |")
    both=all(float(row[k+'_low'])>0 for k in keys[:2])
    lines += ['',('Both branches show positive mean history preference with intervals above zero.' if both else 'A robust positive preference in both branches is not established; inspect the separate branch estimates and intervals.'), '',
      'Task 3 replays A only, excluding branch-specific task-2 images. Both branches compete over the same union of 40 experimental classes. Common augmented input hashes and current/replay image IDs match exactly. AdamW resets at each task. Each class contributes 450 training images, five epochs, batch64 (32 current + 32 replay), learning rate 0.0003, weight decay 0.05, memory500. Each first task has 355 successful updates and each later task has 705; skipped AMP updates are rejected.', '',
      'Scaffolds pool 1-to-2 drift from six source runs using 500 fixed first-task training anchors. Drift evaluation uses 500 other first-task images, excluded from training/replay. Estimation and test histories still share training content. Coverage uncertainty uses 10,000 paired seed bootstrap draws conditional on the two fitted scaffolds; nominal intervals are not multiplicity adjusted. Per-seed results and all four coverage cells are saved as CSV. Source-resampling overlap ranges in similarity.json are descriptive and are not calibrated confidence intervals.', '',
      'One fixed class partition and one common later task are evaluated. This tests persistence of early-content effects into the next task, not their longer-term decay or unique causal storage in the scaffold. Classifier state and normalization buffers may also mediate the effect. The common-replay/output protocol differs from replaying all previous classes.', '',
      '[Result figure](results.png) · [Full cross-coverage](cross_coverage.csv) · [Per-seed data](per_test_seed.csv) · [Protocol](protocol.json)']
    lines += ['', 'Numerical execution: MobileNet uses full FP32 consistently with its CIFAR-10 follow-up; other models use AMP with initial scale128. Mixer uses activation recomputation, with unchanged batch size and epochs.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    check=json.loads(read_config_text(OUT/'verification.json'));check.update(training_metadata_tasks=checked,anchor_disjointness=True,endpoint_drift_recomputed=True)
    (OUT/'verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps({'scaffold_overlap':sim['overlap'],'summary':row}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='resnet18');a=p.parse_args();main(a.model)
