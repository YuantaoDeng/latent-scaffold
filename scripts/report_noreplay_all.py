"""Per-model and four-model reporting for the no-replay replication."""
from package_paths import read_config_text
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from noreplay_all_common import ROOT,configure

def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def save(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def per_model(short):
    cfg=configure(short);out=cfg['OUT'];c=cfg['C'];rank=cfg['RANK'];dimension=cfg['DIM']
    assert json.loads(read_config_text(out/'verification.json'))['complete']
    c['status']='complete'
    c['state_storage']='Lossless byte-plane integer-delta; use lossless_checkpoint.load; bitwise roundtrip checked'
    (out/'protocol.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    pooled=read(out/'pooled_overlap.csv');motion=read(out/'motion_summary.csv');geo=read(out/'geometry_summary.csv');updates=read(out/'updates_summary.csv')
    def get(rows,metric,t,**kw):return next(r for r in rows if r['metric']==metric and int(r['task'])==t and all(str(r[k])==str(v) for k,v in kw.items()))
    def val(rows,m,t,**kw):return 100*float(get(rows,m,t,**kw)['mean'])
    lines=[f"# {c['model']} / CIFAR-10：无 replay 检验",'',
        f'五任务，每任务两类、每类450张训练图片、五轮。6条估计历史和12条独立测试历史。scaffold固定为{rank}维，仅由估计历史的1→2漂移构建。表征维度为{dimension}。',
        f"训练精度：{c['precision']}；梯度检查点：{c['activation_recomputation']}。每批64张当前图片，每任务75次成功更新，旧图片从不回放。",'',
        '| 已完成任务 | 汇聚响应 overlap % | 逐历史响应 overlap % | 能量加权比例 % | 当前PCA–响应 overlap % |','|---|---:|---:|---:|---:|']
    for r in pooled:
        t=int(r['task']);lines.append(f"| {t} | {100*float(r['scaffold_response_overlap']):.2f} | {val(geo,'overlap',t):.2f} | {val(geo,'energy_ratio',t):.2f} | {100*float(r['current_pca_response_overlap']):.2f} |")
    lines+=['','| 转移 | scaffold覆盖 % | 匹配随机覆盖 % | 增益 pp [95%区间] | 当前PCA覆盖 % | PCA残差额外覆盖 pp [95%区间] |','|---|---:|---:|---|---:|---|']
    for t in range(2,6):
        def ci(m):
            r=get(motion,m,t);return f"{100*float(r['mean']):.2f} [{100*float(r['low']):.2f}, {100*float(r['high']):.2f}]"
        lines.append(f"| {t-1}→{t} | {val(motion,'scaffold',t):.2f} | {val(motion,'matched_random',t):.2f} | {ci('gain_matched')} | {val(motion,'local_start_pca',t):.2f} | {ci('residual_gain')} |")
    lines+=['','| 转移 | scaffold logit重建 % | scaffold损失变化重建 % | 当前PCA logit重建 % |','|---|---:|---:|---:|']
    for t in range(2,6):lines.append(f'| {t-1}→{t} | '+' | '.join(f'{val(motion,m,t):.2f}' for m in ['frozen_logit_reconstruction','frozen_loss_reconstruction','pca_frozen_logit_reconstruction'])+' |')
    lines+=['','| 任务 | 更新位置 | 完整更新覆盖增益 pp | 参数更新覆盖增益 pp | 范数匹配梯度覆盖增益 pp |','|---|---:|---:|---:|---:|']
    for t in range(2,6):
        for step in (2,38,73):lines.append(f'| {t} | {step} | '+' | '.join(f"{val(updates,'gain',t,step=step,method=m):.2f}" for m in ['adamw_full','adamw_parameters','norm_matched_training_gradient'])+' |')
    lines+=['','## 口径与限制','',
        f'- 响应使用16次成对对称、幅度为编码器参数范数1e-4的各向同性扰动，FP32推理并固定当期缓冲区。能量比例相对于前{rank}个响应特征值之和；各次扰动的前8次/全部16次稳定性与eigengap保存在CSV。',
        f'- 当前PCA在转移起点第一任务的500张训练anchors上拟合，去除{rank}个方向；未来漂移在100张不参与训练的anchors上测量。残差随机期望为{rank}/{dimension-rank}，以剩余漂移能量为分母；不重新归一化的结果也保留。',
        '- 分类器匹配随机控制逐转移/更新构建，使用该随机分布投影能量的精确期望。',
        '- 输出重建固定转移起点分类器，对当期全部已见类别读出，以均值变化为已知条件。它不是准确率或性能提升。',
        '- 区间为10000次任务顺序/种子分层bootstrap，条件于已拟合scaffold，未作多重比较校正。源端只有3个顺序，几何区间按描述性理解。',
        '- 初始化在同一模型的所有历史间完全相同；源/测试顺序与种子不同，但训练内容重叠。',
        '- 不同模型的预训练、维度与秩不同，不据此排名架构优劣。与原replay的batch/更新预算不同，不是严格配对的replay因果消融。',
        '- 本次没有无replay方向抑制干预。所有原始状态、更新场、扰动协方差均保留。',
        '- 状态使用无损差分存储，逐张量验证恢复后比特完全相同。用项目根目录的 lossless_checkpoint.load(path) 读取；initial_state.pt 和历史中的前序任务文件须一起保留。','',
        '[结果图](results.png) · [漂移区间](motion_summary.csv) · [实际更新区间](updates_summary.csv)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        def curve(ax,rows,m,tasks,label,color,**kw):
            rr=[get(rows,m,t,**kw) for t in tasks];ax.plot(tasks,[100*float(r['mean']) for r in rr],'o-',label=label,color=color)
            ax.fill_between(tasks,[100*float(r['low']) for r in rr],[100*float(r['high']) for r in rr],color=color,alpha=.15)
        ax=axes[0,0]
        for key,label,color in [('scaffold_response_overlap','Scaffold','#167D8D'),('current_pca_response_overlap','Current PCA','#D87525')]:ax.plot(range(6),[100*float(r[key]) for r in pooled],'o-',label=label,color=color)
        ax.axhline(100*rank/dimension,color='gray',ls=':',label='Uniform reference');ax.set(title='Pooled response overlap',ylabel='Percent');ax.legend(fontsize=8)
        ax=axes[0,1];curve(ax,geo,'energy_ratio',list(range(6)),'Scaffold','#167D8D');ax.set(title='Energy-weighted sensitivity',ylabel='Percent')
        ax=axes[0,2]
        for m,l,color in [('scaffold','Scaffold','#167D8D'),('matched_random','Matched random','#777777'),('local_start_pca','Current PCA','#D87525')]:curve(ax,motion,m,[2,3,4,5],l,color)
        ax.set(title='Held-out drift coverage',ylabel='Percent');ax.legend(fontsize=8)
        ax=axes[1,0];curve(ax,motion,'residual_gain',[2,3,4,5],'Residual scaffold','#167D8D');ax.axhline(0,color='gray',ls=':');ax.set(title='Extra coverage after removing PCA',ylabel='Percentage points')
        ax=axes[1,1]
        for step,color in [(2,'#167D8D'),(38,'#D87525'),(73,'#8056A0')]:curve(ax,updates,'gain',[2,3,4,5],f'Step {step}',color,step=step,method='adamw_full')
        ax.axhline(0,color='gray',ls=':');ax.set(title='Actual update gain above matched random',ylabel='Percentage points');ax.legend(fontsize=8)
        ax=axes[1,2]
        for m,l,color in [('frozen_logit_reconstruction','Scaffold','#167D8D'),('pca_frozen_logit_reconstruction','Current PCA','#D87525')]:curve(ax,motion,m,[2,3,4,5],l,color)
        ax.set(title='Frozen-classifier logit reconstruction',ylabel='Score ×100');ax.legend(fontsize=8)
        for i,ax in enumerate(axes.flat):ax.set(xlabel='Completed tasks / transition endpoint',xticks=list(range(6)) if i<2 else [2,3,4,5])
        fig.suptitle(f"{c['model']} / CIFAR-10 without replay | rank {rank}\nSix estimation histories; twelve independent test histories")
        for ext in ('png','pdf'):fig.savefig(out/f'results.{ext}',dpi=170)
        plt.close(fig)
    print(json.dumps(dict(model=short,report=str(out/'REPORT.md'),complete=True)),flush=True)
    aggregate()

def aggregate():
    locations=[('ResNet-18',Path('results/noreplay_resnet_cifar10'))]+[(m,ROOT/f'{s}_cifar10') for s,m in [('deit_tiny','DeiT-Tiny'),('mobilenetv3_small','MobileNetV3-S'),('mixer_b16','MLP-Mixer-B/16')]]
    summary=[];all_motion=[];all_updates=[];all_geometry=[]
    for model,out in locations:
        if not (out/'verification.json').exists():continue
        v=json.loads(read_config_text(out/'verification.json'));assert v['complete']
        motion=read(out/'motion_summary.csv');updates=read(out/'updates_summary.csv');geometry=read(out/'pooled_overlap.csv')
        for rr,target in [(motion,all_motion),(updates,all_updates),(geometry,all_geometry)]:target.extend([dict(model=model,rank=v['rank'],**r) for r in rr])
        mg=[r for r in motion if r['metric']=='gain_matched'];pg=[r for r in motion if r['metric']=='residual_gain'];ug=[r for r in updates if r['metric']=='gain' and r['method']=='adamw_full']
        summary.append(dict(model=model,rank=v['rank'],motion_gain_min=min(float(r['mean']) for r in mg),motion_gain_max=max(float(r['mean']) for r in mg),
            motion_positive_ci=sum(float(r['low'])>0 for r in mg),residual_gain_min=min(float(r['mean']) for r in pg),residual_gain_max=max(float(r['mean']) for r in pg),
            residual_positive_ci=sum(float(r['low'])>0 for r in pg),update_positive_ci=sum(float(r['low'])>0 for r in ug),
            positive_update_runs=sum(int(r['positive_runs']) for r in ug),update_runs=sum(int(r['runs']) for r in ug),
            initial_overlap=float(geometry[0]['scaffold_response_overlap']),task2_overlap=float(geometry[2]['scaffold_response_overlap']),final_overlap=float(geometry[-1]['scaffold_response_overlap']),
            replay_scaffold_overlap=float(json.loads(read_config_text(out/'replay_comparison.json'))['scaffold_overlap'])))
    for name,rr in [('summary.csv',summary),('all_motion_summary.csv',all_motion),('all_updates_summary.csv',all_updates),('all_overlap.csv',all_geometry)]:save(ROOT/name,rr)
    lines=['# 四模型 CIFAR-10 无 replay 汇总','',f'已完成 {len(summary)}/4 个模型。各模型原始协议、状态、数值和图表保存在相应目录。','',
        '| 模型 | 秩 | 漂移增益范围 pp | PCA残差增益范围 pp | 正区间：漂移/残差/更新位置 | 正更新观测 |','|---|---:|---:|---:|---|---|']
    for r in summary:lines.append(f"| {r['model']} | {r['rank']} | {100*r['motion_gain_min']:.2f}–{100*r['motion_gain_max']:.2f} | {100*r['residual_gain_min']:.2f}–{100*r['residual_gain_max']:.2f} | {r['motion_positive_ci']}/4、{r['residual_positive_ci']}/4、{r['update_positive_ci']}/12 | {r['positive_update_runs']}/{r['update_runs']} |")
    names=[r['model'] for r in summary]
    lines+=['','有、无 replay 条件下估计的 scaffold 相似度（描述性比较）：','',
        '| 模型 | scaffold overlap % |','|---|---:|']
    for r in summary:lines.append(f"| {r['model']} | {100*r['replay_scaffold_overlap']:.2f} |")
    lines+=['','两套条件的初始化安排和更新预算并非严格配对，因此这些差异不能单独归因于 replay。']
    lines+=['','汇聚响应空间 overlap（百分比）：','', '| 已完成任务 | '+' | '.join(names)+' |', '|---|'+ '---:|'*len(names)]
    for task in range(6):
        vals=[next(r for r in all_geometry if r['model']==name and int(r['task'])==task) for name in names]
        lines.append(f'| {task} | '+' | '.join(f"{100*float(r['scaffold_response_overlap']):.2f}" for r in vals)+' |')
    lines+=['','各条件完整报告：','']
    for name,out in locations:
        if (out/'REPORT.md').exists():lines.append(f'- [{name}]({out.resolve().as_posix()}/REPORT.md)')
    lines+=['','结果回答无replay条件下是否仍有复现及额外覆盖；覆盖率不是准确率或因果重要性。区间条件于已拟合scaffold，源与测试共享训练内容。没有新增方向抑制干预。']
    if (ROOT/'INTERPRETATION.md').exists():lines+=['','[简要解释](INTERPRETATION.md)']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'verification.json').write_text(json.dumps(dict(complete=len(summary)==4,models=len(summary)),indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model');a=p.parse_args()
    if a.model:per_model(a.model)
    else:aggregate()
