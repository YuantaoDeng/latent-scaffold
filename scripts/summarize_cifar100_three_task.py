"""Combine completed CIFAR-100 three-task content experiments without refitting."""
from package_paths import read_config_text
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('results/early_content_three_task')
MODELS={'resnet18':'ResNet-18','deit_tiny':'DeiT-Tiny','mobilenetv3_small':'MobileNetV3-S','mixer_b16':'MLP-Mixer-B/16'}
def read(p):return list(csv.DictReader(p.open(encoding='utf-8')))
def save(p,rows):
    with p.open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(partial=False):
    rows=[];matrix=[]
    for short,name in MODELS.items():
        out=ROOT/f'{short}_cifar100';vp=out/'verification.json'
        if not vp.exists():
            if partial:continue
            raise RuntimeError('Incomplete '+short)
        v=json.loads(read_config_text(vp));assert v['complete'] and v['training_metadata_tasks']==36
        sim=json.loads(read_config_text(out/'similarity.json'));s=read(out/'summary.csv')[0]
        rows.append({'model':name,'rank':v['rank'],'scaffold_overlap':sim['overlap'],**s})
        matrix += [{'model':name,**r} for r in read(out/'cross_coverage.csv')]
    save(ROOT/'cifar100_summary.csv',rows);save(ROOT/'cifar100_cross_coverage.csv',matrix)
    fig,axes=plt.subplots(1,len(rows),figsize=(3.6*len(rows),3.8),layout='constrained',squeeze=False)
    keys=['X_own_minus_cross','Y_own_minus_cross','paired_own_minus_cross']
    for r,ax in zip(rows,axes.flat):
        for i,k in enumerate(keys):
            m,lo,hi=[100*float(r[k+x]) for x in ['', '_low','_high']]
            ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=f'C{i}')
        ax.axhline(0,ls='--',color='gray',lw=1);ax.set_xticks([0,1,2],['X','Y','Paired']);ax.set_title(r['model']+f" (rank {r['rank']})",fontsize=10);ax.spines[['top','right']].set_visible(False)
    fig.supylabel('Own minus cross coverage (pp)');fig.suptitle('CIFAR-100: identical third task after different early content')
    fig.savefig(ROOT/'cifar100_all_models.pdf');fig.savefig(ROOT/'cifar100_all_models.png',dpi=180);plt.close(fig)
    lines=['# CIFAR-100: three-task early-content experiments','',f'Completed: {len(rows)}/4 models. Each condition uses six paired estimation runs and six separate paired test runs. Existing manuscript ranks are retained.', '',
      '| Model | Rank | Scaffold overlap | X own gain pp [95% CI] | Y own gain pp [95% CI] | Paired gain pp [95% CI] | Positive paired runs |','|---|---|---|---|---|---|---|']
    for r in rows:
        cells=[f"{100*float(r[k]):+.2f} [{100*float(r[k+'_low']):+.2f}, {100*float(r[k+'_high']):+.2f}]" for k in keys]
        lines.append(f"| {r['model']} | {r['rank']} | {100*r['scaffold_overlap']:.2f}% | "+' | '.join(cells)+f" | {r['positive_pairs']}/6 |")
    lines += ['', 'X: A -> BX -> C; Y: A -> BY -> C. Each task has ten classes and each history has thirty classes. A, BX, BY and C are disjoint blocks from original order 1103, fixed before the new experiments. Within a paired seed A is trained once and its complete state is copied to X/Y. Task C replays A only and both branches compete over the same union of forty classes. Common augmented input hashes and image IDs match exactly. Every condition retains five epochs, 450 training images per class, AdamW reset per task, batch64 and memory500.', '',
      'Source 1-to-2 drift on 500 training anchors fits the scaffold. Subsequent drift is measured on 500 held-out anchors excluded from training and replay. Estimation/test histories share training content; this is not content-disjoint transfer. Paired bootstrap intervals use 10,000 test-seed draws, conditional on fitted scaffolds, without multiplicity adjustment. Similarity resampling ranges in per-model files are descriptive. Both branch contrasts must be inspected; a positive paired average alone is insufficient for bilateral preference.', '',
      'MobileNet uses FP32, consistent with the earlier CIFAR-10 follow-up. DeiT, ResNet and Mixer use AMP, with every planned update required. Mixer uses activation recomputation. Different model ranks and pretrained initializations preclude interpreting numerical differences as isolated architecture effects.', '',
      'One fixed class partition and one common later task are evaluated. The experiment tests whether early-content effects persist into the next common task; it cannot establish longer-term persistence, decay, or exclusive causal storage in the scaffold.', '',
      'Per-model reports contain full four-cell coverage matrices, per-seed data, task labels, initialization provenance and verification files.']
    (ROOT/'CIFAR100_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'cifar100_verification.json').write_text(json.dumps({'complete':len(rows)==4,'models':len(rows),'test_pairs_per_model':6,'source_pairs_per_model':6,'tasks_per_history':3},indent=2))
    print('Summarized',len(rows),'models',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');a=p.parse_args();main(a.allow_partial)
