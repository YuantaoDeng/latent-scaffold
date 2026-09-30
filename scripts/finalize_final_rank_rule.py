"""Verify reconstructed projectors and scalar metrics; write current experiment report."""
from package_paths import read_config_text
from pathlib import Path
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import final_rank_experiments as q
import analyze_scaffold_followup as f
from deit_three_claims import json_save

ROOT=q.ROOT

def verify_new_target_states():
    evaluation=f.IndexedCifar(Path('data'),True,False,'cifar10'); device=torch.device('cuda')
    net=f.model('deit_tiny_patch16_224',False,num_classes=10).to(device); max_error=0.
    for path in q.targets():
        t=f.load(q.OUT/'targets/artifacts'/path.name)
        for cp in (4,5):
            net.load_state_dict(t['states'][cp])
            _,z,labels=f.infer(net,evaluation,t['heldout_ids'],t['manifest']['batch_size'],device)
            assert torch.equal(labels,t['heldout_labels'])
            max_error=max(max_error,float((z-t['eval_reps'][cp-1]).abs().max()))
    assert max_error<=1e-5,max_error
    del net; torch.cuda.empty_cache()
    return {'pre_final_and_final_checkpoint_count':24,'saved_representation_max_error':max_error}

def verify_metrics(folder,target_paths,target_root,fields_root):
    bank=f.load(folder/'predictors.pt'); u=bank['scaffold'].double(); projector=u@u.T
    tables={name:{(int(x['order_seed']),int(x['seed'])):x for x in csv.DictReader((folder/f'{name}.csv').open())}
            for name in ('transfer','updates','functional')}
    max_error=0.
    for path in target_paths:
        a=f.load(path); key=(a['order_seed'],a['seed']); t=f.load(target_root/path.name)
        assert t['actual_update']['optimizer_step_before']==1 and t['actual_update']['optimizer_step_after']==2
        assert not set(map(int,t['fit_ids'])).intersection(map(int,t['heldout_ids']))
        delta=(t['eval_reps'][-1]-t['eval_reps'][-2]).double(); centered=delta-delta.mean(0)
        def coverage(field):
            field=field.double()
            return float((field*(field@projector)).sum()/field.square().sum())
        max_error=max(max_error,abs(coverage(centered)-float(tables['transfer'][key]['scaffold'])))
        for name,field in f.load(fields_root/path.name)['fields'].items():
            max_error=max(max_error,abs(coverage(field)-float(tables['updates'][key][name+'_scaffold'])))
        state=t['states'][len(t['eval_reps'])-1]; classes=sum(t['manifest']['order'][:-1],[])
        weight,bias=f.classifier_tensors(state); weight=weight[classes]; bias=bias[classes]
        start=t['eval_reps'][-2].double()+delta.mean(0)
        logits=[z@weight.T+bias for z in (start,start+centered,start+centered@projector)]
        labels=torch.tensor([classes.index(int(v)) for v in t['heldout_labels']])
        losses=[-torch.log_softmax(v,1)[torch.arange(len(labels)),labels] for v in logits]
        for endpoint,values in [('class_centered_logits',[v-v.mean(1,keepdim=True) for v in logits]),('loss',losses)]:
            change=values[1]-values[0]; predicted=values[2]-values[0]
            score=1-float((change-predicted).square().sum()/change.square().sum())
            stored=float(tables['functional'][key]['scaffold_centered_conditional_'+endpoint+'_reconstruction'])
            max_error=max(max_error,abs(score-stored))
    assert max_error<1e-7,max_error
    return {'target_count':len(target_paths),'independent_scalar_metric_max_error':max_error}

def verify_geometry(folder,source_paths,response_root):
    g=json.loads(read_config_text(folder/'geometry.json')); rank=g['r']; histories=[]; responses={cp:[] for cp in g['checkpoints']}
    def top(matrix):
        return torch.linalg.eigh(matrix.double()).eigenvectors[:,-rank:]
    def pool(bases):
        return top(sum(u@u.T for u in bases))
    for path in source_paths:
        a=f.load(path); h=a['historical_drift_centered'].double()
        histories.append(torch.linalg.svd(h,full_matrices=False).Vh[:rank].T)
        rec=f.load(response_root/path.name)
        for cp in responses:
            covariance=rec['states'][cp]['covariance']
            assert torch.allclose(covariance,covariance.T,atol=1e-12,rtol=1e-10)
            responses[cp].append(top(covariance))
    scaffold=pool(histories); bank=f.load(folder/'predictors.pt')
    projector_error=float((scaffold@scaffold.T-bank['scaffold']@bank['scaffold'].T).abs().max())
    values=[float(torch.trace((scaffold@scaffold.T)@(pool(responses[cp])@pool(responses[cp]).T))/rank) for cp in g['checkpoints']]
    error=float(np.max(np.abs(np.array(values)-g['overlap'])))
    assert projector_error<1e-7 and error<1e-7,(projector_error,error)
    assert g['response_count']==16
    return {'independent_projector_error':projector_error,'independent_overlap_error':error,'checkpoints':g['checkpoints']}

def main():
    import deit_three_claims as d
    settings=[('DeiT-Tiny / CIFAR-100',Path('results/deit_pca_dimension/cifar100/rank20'),
               Path('results/deit_rank20_existing/cifar100/rank20/evaluation.json'),
               [p for p,_ in d.inventory('source')],Path('results/deit_pca_dimension/cifar100/responses/artifacts')),
              ('DeiT-Tiny / CIFAR-10',q.OUT/'rank4',q.OUT/'rank4/evaluation.json',q.c.artifacts(),q.OUT/'responses/artifacts')]
    for ds in ('cifar100','cifar10'):
        out=ROOT/'resnet'/ds; selection=json.loads(read_config_text(out/'selection.json')); folder=out/f"rank{selection['selected']}"
        inv=json.loads(read_config_text(q.DATA[ds]/'scaffold_structure/inventory.json'))
        settings.append((f'ResNet-18 / {ds}',folder,folder/'evaluation.json',[Path(e['path']) for e in inv if e['role']=='source'],out/'responses/artifacts'))
    lines=['# 当前维度规则下的实验结果','',
           '固定 k=r；后续规则为 8–32 维内源端留任务顺序验证中平均覆盖增益高于匹配 Representation PCA 的最小维度，不使用 one-SE。本轮 DeiT/CIFAR-10 四维是用户明确指定的范围例外，CIFAR-100 固定二十维。所有响应空间使用 16 次参数扰动。',
           '', 'CIFAR-100 的既有二十维目标实验复用；DeiT/CIFAR-10 使用六种新目标顺序、每种两个种子。ResNet 从已有完整轨迹与响应协方差重新计算。均值为正不等于置信区间排除零。','']
    lines.extend(['| 模型 / 数据 | k=r | 源端选维度依据 |','|---|---:|---|'])
    c10selection=json.loads(read_config_text(q.c.ROOT/'selection.json'))
    c10gain=next(x['mean_score'] for x in c10selection['curves'] if x['r']==4)
    lines.append(f'| DeiT / CIFAR-10 | 4 | 用户指定；源端平均 PCA 覆盖增益 {100*c10gain:.4f} 个百分点 |')
    lines.append('| DeiT / CIFAR-100 | 20 | 用户已指定，保留此前完成的二十维目标实验 |')
    for ds in ('cifar100','cifar10'):
        sel=json.loads(read_config_text(ROOT/'resnet'/ds/'selection.json')); n=sel['selected']
        gain=next(x['mean_score'] for x in sel['curves'] if x['r']==n)
        lines.append(f'| ResNet / {ds} | {n} | 8–32 内最小正增益维度；平均增益 {100*gain:.4f} 个百分点 |')
    lines.append('')
    fig,axes=plt.subplots(2,2,figsize=(10,7)); checks={}
    for ax,(name,folder,evalpath,sources,responses) in zip(axes.flat,settings):
        checks[name]=verify_geometry(folder,sources,responses)
        q.all_overlaps(folder)
        g=json.loads(read_config_text(folder/'geometry.json')); e=json.loads(read_config_text(evalpath))
        lines.extend([f'## {name}：{g["r"]} 维','', '| checkpoint | scaffold–response overlap | 95% CI | 响应空间 8/16 样本稳定性 |','|---|---:|---|---:|'])
        for cp,value,ci in zip(g['checkpoints'],g['overlap'],g['ci95']):
            stability=g['response_sampling_pooled_overlap'].get(str(cp),float('nan'))
            lines.append(f'| {cp} | {100*value:.3f}% | [{100*ci[0]:.3f}, {100*ci[1]:.3f}]% | {stability:.4f} |')
        final=sorted(e['temporal'])[-1]
        metrics=[]
        for transition,values in sorted(e['temporal'].items()):
            metrics.append((f'观点一：漂移 {transition} 超过匹配随机（百分点）',values['gain_random']))
        metrics.extend([('最后任务漂移超过匹配 PCA（百分点）',e['temporal'][final]['gain_static']),
                        ('学习后响应空间相对初始响应空间的目标漂移覆盖增益（百分点）',e['transfer']['response_2_minus_initial']),
                        ('观点三：实际 AdamW 更新超过匹配随机子空间（百分点）',e['updates']['adamw_full_gain_random']),
                        ('实际 AdamW 更新超过匹配 PCA（百分点）',e['updates']['adamw_full_gain_static'])])
        for key in ('centered_conditional_class_centered_logits_reconstruction','centered_conditional_loss_reconstruction'):
            metrics.append((f'冻结读出 {key} 超过匹配随机（百分点）',e['functional']['gain_random_'+key]))
            metrics.append((f'冻结读出 {key} 超过匹配 PCA（百分点）',e['functional']['gain_static_'+key]))
        lines.extend(['','| 指标 | 均值 | 95% CI | 正增益任务顺序 |','|---|---:|---|---:|'])
        for label,v in metrics:
            lines.append(f'| {label} | {100*v["mean"]:.3f} | [{100*v["ci95"][0]:.3f}, {100*v["ci95"][1]:.3f}] | {v["positive_orders"]}/6 |')
        delta=g['change_from_initial'][-1]; ci=g['change_from_initial_ci95'][-1]
        def evidence(v):
            return '正增益区间排除零' if v['ci95'][0]>0 else '负增益区间排除零' if v['ci95'][1]<0 else '区间跨零，证据尚不确定'
        lines.extend(['',f'目标端判读：最后任务迁移{evidence(e["temporal"][final]["gain_random"])}；'
                      f'实际更新{evidence(e["updates"]["adamw_full_gain_random"])}。'])
        lines.extend(['',f'观点二：最终相对初始 overlap 改变 {delta:.6f}，95% CI [{ci[0]:.6f}, {ci[1]:.6f}]。是否持续提升应以完整曲线及相邻差值为准，不能只比较 checkpoint 0 和 2。',
                      '',f'完整指标与 PCA 对照：`{evalpath.as_posix()}`；全部两两 overlap：`{(folder/"all_pooled_overlaps.csv").as_posix()}`。',''])
        y=np.array(g['overlap']); bounds=np.array(g['ci95'])
        ax.plot(g['checkpoints'],y,'o-',label='Scaffold-response'); ax.fill_between(g['checkpoints'],bounds[:,0],bounds[:,1],alpha=.18,label='95% bootstrap CI')
        ax.axhline(g['isotropic_overlap'],color='gray',linestyle='--',label='Isotropic expectation'); ax.set_title(f'{name}; k=r={g["r"]}')
        ax.set_xlabel('Checkpoint'); ax.set_ylabel('Overlap'); ax.set_ylim(0,1); ax.set_xticks(g['checkpoints'])
        ax.legend(fontsize=7,loc='upper right')
    old=json.loads(read_config_text(Path('results/deit_rank20_existing/cifar100/rank20/geometry.json')))
    new=json.loads(read_config_text(settings[0][1].joinpath('geometry.json')))
    checks['cifar100_original_0_2_parity']=max(abs(new['overlap'][cp]-old['overlap'][i]) for i,cp in enumerate(old['checkpoints']))
    assert checks['cifar100_original_0_2_parity']<1e-10
    checks['deit_cifar10_metrics']=verify_metrics(q.OUT/'rank4',q.targets(),q.OUT/'targets/artifacts',q.OUT/'update_fields')
    checks['deit_cifar100_metrics']=verify_metrics(settings[0][2].parent,[p for p,_ in d.inventory('target')],d.ROOT/'targets/artifacts',d.ROOT/'update_fields')
    for path in q.target_manifests():
        baseline=json.loads(read_config_text(path)); current=json.loads(read_config_text(q.OUT/path.parent.name/'manifest.json'))
        for key in ('dataset','tasks','classes_per_task','seeds','epochs','train_per_class','probe_per_class',
                    'memory','batch_size','lr','continual_method','order_seed','order'):
            assert current[key]==baseline[key],(path,key)
        assert current['model']=='deit_tiny_patch16_224' and current['pretrained'] and current['status']=='complete'
    checks['deit_cifar10_target_protocol_matches_resnet']=True
    import hashlib
    for item in json.loads(read_config_text(q.OUT/'source_hashes.json')):
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    checks['deit_cifar10_sources_unchanged']=True
    checks['deit_cifar10_saved_target_states']=verify_new_target_states()
    for ds in ('cifar100','cifar10'):
        out=ROOT/'resnet'/ds; rank=json.loads(read_config_text(out/'selection.json'))['selected']
        inv=json.loads(read_config_text(q.DATA[ds]/'scaffold_structure/inventory.json'))
        checks['resnet_'+ds+'_metrics']=verify_metrics(out/f'rank{rank}',[Path(x['path']) for x in inv if x['role']=='target'],
            q.DATA[ds]/'scaffold_followup/target_recovery/artifacts',q.DATA[ds]/'scaffold_followup/analysis/update_fields')
        reference={ (int(x['heldout_order']),int(x['seed']),int(x['rank'])):float(x['excess'])
                    for x in csv.DictReader(Path('results/dimension_validation/validation.csv').open())
                    if x['construction']=='joint' and x['dataset']==('CIFAR-100' if ds=='cifar100' else 'CIFAR-10')}
        selection_rows=list(csv.DictReader((out/'selection_folds.csv').open()))
        selection_error=max(abs(float(x['score'])-reference[(int(x['order_seed']),int(x['seed']),int(x['r']))]) for x in selection_rows)
        assert selection_error<1e-6,selection_error
        eligible=[n for n in range(8,33) if np.mean([float(x['score']) for x in selection_rows if int(x['r'])==n])>0]
        assert rank==min(eligible)
        checks['resnet_'+ds+'_rank_selection']={'legacy_joint_scan_max_error':selection_error,'selected':rank,'one_se_used':False}
    fig.tight_layout()
    for suffix in ('png','pdf','svg'): fig.savefig(ROOT/f'overlap.{suffix}',dpi=180)
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8'); json_save(ROOT/'verification.json',checks)
    print('FINAL REPORT AND INDEPENDENT GEOMETRY VERIFICATION COMPLETE',flush=True)

if __name__=='__main__':
    torch.set_num_threads(2); main()
