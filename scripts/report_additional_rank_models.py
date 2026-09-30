"""Report every attempted model, with optional independent numerical verification."""
from package_paths import read_config_text
from pathlib import Path
import argparse,csv,json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import additional_rank_models as a
import analyze_scaffold_followup as f
from deit_three_claims import json_save

def verify_checkpoint_links(ex):
    initial=None; source_count=0
    keys=('dataset','tasks','classes_per_task','epochs','train_per_class','probe_per_class','memory','batch_size','lr','continual_method','order_seed','order')
    for role in ('source','target'):
        for entry in a.reference(ex.ds,role):
            path=ex.path(entry); raw=f.load(path)
            cfg=json.loads(read_config_text(path.parent.parent/'manifest.json')); baseline=json.loads(read_config_text(Path(entry['manifest_path'])))
            for key in keys: assert cfg[key]==baseline[key],(path,key)
            assert cfg['model']==ex.name and cfg['pretrained'] and raw['seed']==entry['seed']
            if role=='source':
                rec=f.load(ex.out/'sources/artifacts'/path.name); state=rec['states'][0]
                prefix=next(n for n in ('fc','head','classifier','head.fc') if n+'.weight' in state and n+'.bias' in state)
                encoder={k:v for k,v in state.items() if k not in (prefix+'.weight',prefix+'.bias')}
                if initial is None: initial=encoder
                else: assert all(torch.equal(encoder[k],initial[k]) for k in initial)
                saved=rec['states'][cfg['tasks']-1]
                assert all(torch.equal(saved[k],v) for k,v in raw['pre_future_state_dict'].items())
                source_count+=1
    evaluation=f.IndexedCifar(Path('data'),True,False,ex.ds); device=torch.device('cuda')
    net=f.model(ex.name,False,num_classes=evaluation.num_classes).to(device); error=0.
    for path in ex.paths('target'):
        rec=f.load(ex.out/'targets/artifacts'/path.name)
        for cp,state in rec['states'].items():
            net.load_state_dict(state)
            _,z,labels=f.infer(net,evaluation,rec['heldout_ids'],rec['manifest']['batch_size'],device)
            assert torch.equal(labels,rec['heldout_labels'])
            error=max(error,float((z-rec['eval_reps'][cp-1]).abs().max()))
    assert source_count==6 and error<=1e-5,error
    del net; torch.cuda.empty_cache()
    return dict(source_count=source_count,common_initial_encoder=True,source_pre_final_states_equal=True,
                matched_training_config=True,target_checkpoint_count=24,target_representation_max_error=error)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--verify',action='store_true'); args=p.parse_args()
    torch.set_num_threads(2); f.classifier_tensors=a.classifier_tensors
    lines=['# MobileNetV3-Small：完整实验结果','',
           '按用户最终指示，本轮完成 MobileNetV3-Small 的两个数据集后结束。RegNetX-002 与 EfficientNet-B0 已停止，保留已有源端准备数据；不据此对这两个模型作结论。', '',
           '## 三个观点与结论边界', '',
           '两个数据集均支持：早期 scaffold 跨目标学习历史解释后续漂移；学习后 checkpoint 2 的局部响应空间与 scaffold 对齐增强且更能覆盖目标漂移；实际 AdamW 更新复用 scaffold，并在冻结 logit 读出中可见。这里的支持是相对匹配随机对照及学习前响应空间而言，不要求所有指标优于 Representation PCA。', '',
           'CIFAR-100 的 loss 读出增益区间跨零；CIFAR-10 最终 checkpoint 相对初始的 overlap 增强区间跨零。两个数据集的 overlap 都不随学习单调增加。因此，三个核心观点有支持，但不能宣称完整复现了更严格的 ResNet 证据模式，或持续增强直至最后。下表保留原严格判据的 False，不将它误读为三个核心观点全部失败。', '',
           '维度选择：CIFAR-100 的正增益候选 N={8,9}，取 8；CIFAR-10 的 N={24,31,32}，取 24。CIFAR-10 在 24 维的源端平均 PCA 增益仅 0.036 个百分点，且仅 1/3 源任务顺序为正，选维度优势较弱；该规则仅要求均值为正。', '',
           '全部实验每个数据集包含 6 条源轨迹和 12 条目标轨迹。独立重算了投影、overlap、目标指标，并核对 24 个目标 checkpoint 的表示与保存值一致；详见 [独立核验](independent_verification.json)。Overlap 是子空间投影重叠，响应能量覆盖率是扰动协方差能量比例，两者不同；冻结 logit/loss 数值为中心化条件重构分数，并非分类准确率。', '',
           '固定 k=r，选择 8–32 维内源端留任务顺序验证中平均覆盖率高于匹配 Representation PCA 的最小维度；不使用 one-SE。全部响应测量使用 16 次参数扰动。完整评估顺序为 MobileNetV3-Small、RegNetX-002、EfficientNet-B0；后两者因运行时间调整顺序，调整发生在查看它们的目标端结果之前。EfficientNet 的已准备源轨迹保留。',
           '', '本轮始于探索性模型搜索，最终按用户指示在完成 MobileNet 后停止；下列 95% bootstrap 区间未对模型筛选作校正。所有尝试均保留。PCA 用于选维度及解释性对照，不作为三个观点的统一判据。',
           '', '| 模型 | 数据集 | 维度 | 状态 | 与预先固定的 ResNet 证据模式一致 |','|---|---|---:|---|---|']
    completed=[]
    for name in a.MODELS:
        for ds in ('cifar100','cifar10'):
            out=a.ROOT/name/ds
            sel=json.loads(read_config_text(out/'selection.json')) if (out/'selection.json').exists() else {}
            status=json.loads(read_config_text(out/'status.json')) if (out/'status.json').exists() else {}
            state=status.get('status','running' if out.exists() else 'not_attempted')
            lines.append(f'| {name} | {ds} | {sel.get("selected","—")} | {state} | {status.get("compatible","—")} |')
            if state=='complete': completed.append((name,ds,out,sel,status))
    checks={}
    for name,ds,out,sel,status in completed:
        rank=sel['selected']; folder=out/f'rank{rank}'; e=json.loads(read_config_text(folder/'evaluation.json')); g=json.loads(read_config_text(folder/'geometry.json'))
        if args.verify:
            import finalize_final_rank_rule as v
            ex=a.Experiment(name,ds)
            checks[name+'/'+ds]=dict(geometry=v.verify_geometry(folder,ex.paths('source'),out/'responses/artifacts'),
                 metrics=v.verify_metrics(folder,ex.paths('target'),out/'targets/artifacts',out/'update_fields'),
                 checkpoint_links=verify_checkpoint_links(ex))
        dimension=round(rank/g['isotropic_overlap']); source_gain=next(x['mean_score'] for x in sel['curves'] if x['r']==rank)
        lines.extend(['',f'## {name} / {ds}，{rank} 维','',
                      f'表示总维度 {dimension}，scaffold 占 {100*rank/dimension:.3f}%；源端选维度平均 PCA 覆盖增益 {100*source_gain:.3f} 个百分点。不同模型的覆盖率应结合维度比例和同模型对照解读。','',
                      '| Checkpoint | Scaffold–response overlap | 95% CI | 响应能量覆盖率 |','|---|---:|---|---:|'])
        cov=list(csv.DictReader((folder/'response_energy_coverage.csv').open()))
        for cp,value,ci in zip(g['checkpoints'],g['overlap'],g['ci95']):
            energy=np.mean([float(x['scaffold']) for x in cov if int(x['checkpoint'])==cp])
            lines.append(f'| {cp} | {100*value:.3f}% | [{100*ci[0]:.3f}, {100*ci[1]:.3f}]% | {100*energy:.3f}% |')
        lines.extend(['','| 指标 | Scaffold | 匹配随机 | 匹配 PCA | 相对随机增益及95%区间（百分点） |','|---|---:|---:|---:|---|'])
        metrics=[]
        for t,values in e['temporal'].items():
            metrics.append((f'漂移 {t}',values['scaffold'],values['matched_random'],values['matched_static'],values['gain_random']))
        for key in ('adamw_full','adamw_parameters','norm_matched_training_gradient'):
            values=e['updates']; metrics.append((key,values[key+'_scaffold'],values[key+'_random'],values[key+'_matched_static'],values[key+'_gain_random']))
        for key in ('centered_conditional_class_centered_logits_reconstruction','centered_conditional_loss_reconstruction'):
            values=e['functional']; metrics.append((key,values['scaffold_'+key],values['random_'+key],values['matched_static_'+key],values['gain_random_'+key]))
        for label,s,rand,pca,gain in metrics:
            lines.append(f'| {label} | {100*s["mean"]:.3f}% | {100*rand["mean"]:.3f}% | {100*pca["mean"]:.3f}% | {100*gain["mean"]:.3f} [{100*gain["ci95"][0]:.3f}, {100*gain["ci95"][1]:.3f}] |')
        transfer=e['transfer']['response_2_minus_initial']
        lines.extend(['',f'学习后响应空间对目标最终漂移覆盖率相对初始响应空间提高 {100*transfer["mean"]:.3f} 个百分点，95% CI [{100*transfer["ci95"][0]:.3f}, {100*transfer["ci95"][1]:.3f}]。',
                      '',f'逐项判据：`{json.dumps(status["gates"],ensure_ascii=False)}`。',
                      '', '这里的未通过表示未满足预先固定的证据要求，不等于证明该现象不存在。'])
        fig,ax=plt.subplots(figsize=(6,4)); ci=np.array(g['ci95'])
        ax.plot(g['checkpoints'],g['overlap'],'o-',label='Scaffold-response overlap')
        ax.fill_between(g['checkpoints'],ci[:,0],ci[:,1],alpha=.18,label='95% bootstrap CI')
        ax.axhline(g['isotropic_overlap'],ls='--',color='gray',label='Isotropic expectation')
        ax.set(title=f'{name}\n{ds}; k=r={rank}; 16 perturbations',xlabel='Checkpoint',ylabel='Overlap',ylim=(0,1),xticks=g['checkpoints'])
        ax.legend(fontsize=8); fig.tight_layout()
        for ext in ('png','pdf','svg'): fig.savefig(folder/f'overlap.{ext}',dpi=180)
        plt.close(fig)
    a.ROOT.mkdir(parents=True,exist_ok=True); (a.ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    if args.verify: json_save(a.ROOT/'independent_verification.json',checks)
    print('REPORT UPDATED; completed conditions',len(completed),flush=True)

if __name__=='__main__': main()
