"""Task-by-checkpoint accuracy, with explicit provenance and reproducibility checks."""
from package_paths import read_config_text
import argparse,csv,gc,json,os,re,time
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('HF_HOME',str(Path('data/hf_cache').resolve()))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import experiment_control as control
control.install_loader_hook()
import drift_experiment as d
import analyze_scaffold_followup as f
from all_model_followups_config import conditions
from lossless_checkpoint import load
OUT=control.OUT/'accuracy'

def inventory():
    jobs=[]
    for c in conditions():
        for path in c['records']:
            jobs.append(dict(family='baseline',model=c['name'],short=c['short'],dataset=c['dataset'],path=path,
                key=f"baseline_{c['short']}_{c['dataset']}_{Path(path).stem}"))
        if c['dataset']=='cifar100': root=Path('results/early_content_three_task')/f"{c['short']}_cifar100"
        elif c['short']=='resnet18':root=Path('results/early_content_scaffold_pilot/resnet18_cifar10')
        else:root=Path('results/all_model_followups/content')/f"{c['short']}_cifar10"
        for folder in sorted(root.glob('*_seed*')):
            if not folder.is_dir():continue
            jobs.append(dict(family='content',model=c['name'],short=c['short'],dataset=c['dataset'],path=str(folder),root=str(root),
                key=f"content_{c['short']}_{c['dataset']}_{folder.name}"))
        if c['dataset']=='cifar10':
            root=Path('results/noreplay_resnet_cifar10') if c['short']=='resnet18' else Path('results/noreplay_all_models')/f"{c['short']}_cifar10"
            for folder in sorted(root.glob('*/order*_seed*')):
                if not folder.is_dir() or folder.parent.name not in ('source','target'):continue
                jobs.append(dict(family='noreplay',model=c['name'],short=c['short'],dataset=c['dataset'],path=str(folder),root=str(root),
                    key=f"noreplay_{c['short']}_{folder.parent.name}_{folder.name}"))
    # Estimation histories also belong to the paper's training inventory.
    for c in conditions():
        if c['initial_is_raw_state']:
            root=Path('results/deit_cifar10_dimension')
            for folder in sorted((root/'states').glob('order*_seed*')):
                if not folder.is_dir():continue
                orderseed=int(folder.name.split('_')[0][5:]);seed=int(folder.name.split('_seed')[1])
                manifest=json.loads(read_config_text(root/f'source_order{orderseed}'/'manifest.json'))
                synthetic=OUT/'source_inputs'/f'{c["short"]}_{c["dataset"]}_{folder.name}.json'
                control.write(synthetic,dict(manifest=manifest,seed=seed,states_folder=str(folder)))
                jobs.append(dict(family='source_states',model=c['name'],short=c['short'],dataset=c['dataset'],path=str(synthetic),
                    key=f"source_{c['short']}_{c['dataset']}_{folder.name}"))
        else:
            for path in sorted(Path(c['initial_source']).parent.glob('*.pt')):
                jobs.append(dict(family='baseline',model=c['name'],short=c['short'],dataset=c['dataset'],path=str(path),role='source',
                    key=f"source_{c['short']}_{c['dataset']}_{path.stem}"))
    OUT.mkdir(parents=True,exist_ok=True)
    control.write(OUT/'inventory.json',jobs)
    return jobs

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as out:
        w=csv.DictWriter(out,keys);w.writeheader();w.writerows(rows)

@torch.inference_mode()
def evaluate(net,data,order,checkpoint,seen,amp,output,meta):
    ids=d.ids_by_class(data.targets,sum(order[:checkpoint],[]),50,tail=True)
    net.eval();logits=[];labels=[]
    for _,x,y in d.loader(data,ids,64):
        with torch.amp.autocast('cuda',enabled=amp): z=net(d.prepare_images(x,'cuda'))
        logits.append(z.float().cpu());labels.append(y)
    logits=torch.cat(logits);labels=torch.cat(labels)
    assert torch.isfinite(logits).all()
    torch.save(dict(logits=logits,labels=labels,ids=ids,seen=seen,order=order,checkpoint=checkpoint,**meta),output)
    return rows_from_predictions(load(output),str(output))

def rows_from_predictions(v,path):
    rows=[]
    for task,classes in enumerate(v['order'][:v['checkpoint']],1):
        mask=torch.isin(v['labels'],torch.tensor(classes));n=int(mask.sum())
        allowed=torch.tensor(v['seen']);correct=int((allowed[v['logits'][mask][:,allowed].argmax(1)]==v['labels'][mask]).sum())
        rows.append(dict(history=v['history'],family=v['family'],model=v['model'],dataset=v['dataset'],branch=v.get('branch',''),
            checkpoint=v['checkpoint'],task=task,correct=correct,n=n,accuracy=correct/n,
            competition_classes=json.dumps(v['seen']),task_classes=json.dumps(classes),
            evaluation_precision=v['evaluation_precision'],provenance=v['provenance'],source=v['source'],predictions=path))
    return rows

def execute(job,prefix=False):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    dest=OUT/job['key'];dest.mkdir(parents=True,exist_ok=True)
    marker=dest/('prefix_complete.json' if prefix else 'complete.json')
    if marker.exists():return
    data=d.IndexedCifar(Path('data'),True,False,job['dataset']);nclasses=data.num_classes
    net=d.model(job['model'],False,num_classes=nclasses).cuda()
    amp=job['family'] in ('baseline','source_states');rows=[];path=Path(job['path'])
    meta=dict(history=job['key'],family=job['family'],model=job['model'],dataset=job['dataset'],
        evaluation_precision='AMP' if amp else 'FP32',provenance='saved_checkpoint_evaluation',source=str(path))
    def eval_state(state,order,cp,seen,label='',provenance='saved_checkpoint_evaluation'):
        target=dest/f'{label}checkpoint{cp}.pt'
        if target.exists(): rows.extend(rows_from_predictions(load(target),str(target)));return
        control.poll(stage='accuracy',history=job['key'],checkpoint=cp)
        net.load_state_dict(state)
        rows.extend(evaluate(net,data,order,cp,seen,amp,target,{**meta,'branch':label.rstrip('_'),'provenance':provenance}))
        write_csv(dest/('prefix.csv' if prefix else 'matrix.csv'),rows)
        print(json.dumps(dict(stage='accuracy',history=job['key'],checkpoint=cp,branch=label,rows=len(rows))),flush=True)
    if job['family']=='baseline':
        rec=load(path);order=rec['manifest']['order']
        if prefix:
            missing=sorted(set(range(1,len(order)+1))-set(rec['states']))
            if not missing:
                control.write(marker,dict(complete=True,job=job['key'],prefix=True,missing_checkpoints=0));return
            cfg=SimpleNamespace(**rec['manifest']);seed=rec['seed'];train=d.IndexedCifar(Path('data'),True,True,job['dataset'])
            d.seed_all(seed);del net;net=d.model(job['model'],True,num_classes=nclasses).cuda()
            if getattr(cfg,'initial_checkpoint',None):
                initial=load(cfg.initial_checkpoint);net.load_state_dict(initial.get('state_dict',initial));d.seed_all(seed)
            ids={c:d.ids_by_class(train.targets,[c],cfg.train_per_class) for c in sum(order,[])}
            priority=torch.rand(len(train),generator=torch.Generator().manual_seed(seed+9000))
            end=max(missing);resume=dest/'prefix_resume.pt';start=0;seen=[]
            expected=rec.get('eval_reps',[])
            if len(expected)<end and rec.get('original_path'):
                expected=load(rec['original_path']).get('eval_representations',expected)
            if resume.exists():
                v=load(resume);net.load_state_dict(v['state']);f.restore_rng(v['rng']);start=v['task'];seen=sum(order[:start],[])
            for index in range(start,end):
                old=list(seen);seen+=order[index]
                memory=d.balanced_memory(ids,old,cfg.memory,priority) if old else torch.empty(0,dtype=torch.long)
                current=torch.cat([ids[c] for c in order[index]])
                control.poll(stage='accuracy_prefix_recovery',history=job['key'],task=index+1)
                d.train_task(net,'naive' if index==0 else cfg.continual_method,train,current,memory,{},seen,old,cfg,torch.device('cuda'),seed*100+index)
                rng=f.rng_state()
                _,observed,_=d.infer(net,data,rec['heldout_ids'],64,torch.device('cuda'))
                if index>=len(expected):raise RuntimeError('No saved feature reference for requested recovery checkpoint')
                error=float((observed-expected[index].float()).abs().max())
                if error>1e-5:
                    control.write(dest/'prefix_mismatch.json',dict(task=index+1,max_abs_feature_error=error,accepted=False))
                    raise RuntimeError(f'Reconstructed history mismatch: {job["key"]} checkpoint {index+1}: {error}')
                state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                eval_state(state,order,index+1,seen,provenance='reconstructed_prefix_verified_features')
                f.restore_rng(rng)
                tmp=resume.with_suffix('.writing.pt');torch.save(dict(state=state,rng=rng,task=index+1,error=error),tmp);tmp.replace(resume)
            resume.unlink(missing_ok=True)
        else:
            for cp,state in sorted(rec['states'].items()):
                if cp>0:eval_state(state,order,cp,sum(order[:cp],[]))
    elif job['family']=='source_states':
        rec=json.loads(read_config_text(path));order=rec['manifest']['order']
        for file in sorted(Path(rec['states_folder']).glob('checkpoint*.pt')):
            cp=int(file.stem.removeprefix('checkpoint'))
            if cp>0:eval_state(load(file),order,cp,sum(order[:cp],[]))
    elif job['family']=='content':
        tasks=load(Path(job['root'])/'anchors.pt')['tasks']
        for branch in ('X','Y'):
            order=[tasks['A'],tasks['B'+branch],tasks['C']]+([tasks['D']] if 'D' in tasks else [])
            for cp in range(1,len(order)+1):
                file=path/('shared_A.pt' if cp==1 else f'{branch}_task{cp}.pt')
                if not file.exists():continue
                rec=load(file);seen=rec.get('seen',sorted(sum(order[:cp],[])))
                eval_state(rec['state'],order,cp,seen,branch+'_')
    else:
        anchors=load(path/'anchors.pt');order=anchors['order']
        for cp in range(1,len(order)+1):
            rec=load(path/f'task{cp}.pt');eval_state(rec['state'],order,cp,rec.get('seen',sum(order[:cp],[])))
            del rec;gc.collect()
    control.write(marker,dict(complete=True,job=job['key'],prefix=prefix))

def collect():
    rows=[]
    for p in sorted(OUT.glob('*/checkpoint*.pt'))+sorted(OUT.glob('*/[XY]_checkpoint*.pt')):
        rows.extend(rows_from_predictions(load(p),str(p)))
    write_csv(OUT/'task_accuracy_long.csv',rows)
    grouped={}
    for r in rows:
        key=(r['history'],r['branch'],r['checkpoint'])
        item=grouped.setdefault(key,{k:r[k] for k in ('history','family','model','dataset','branch','checkpoint','provenance')})
        item[f'task{r["task"]}_accuracy']=r['accuracy']
    write_csv(OUT/'task_accuracy_matrix.csv',list(grouped.values()))
    jobs=json.loads(read_config_text(OUT/'inventory.json')) if (OUT/'inventory.json').exists() else []
    complete=sum((OUT/j['key']/'complete.json').exists() for j in jobs)
    prefixes=[j for j in jobs if j['family']=='baseline']
    control.write(OUT/'summary.json',dict(jobs=len(jobs),evaluations_complete=complete,baseline_prefixes=len(prefixes),
        prefixes_complete=sum((OUT/j['key']/'prefix_complete.json').exists() for j in prefixes),rows=len(rows),
        definition='Accuracy among actual competing classes, for every learned task after each checkpoint; 50 excluded training-split images per class. Blank means unavailable, never zero-filled.'))
    print(json.dumps(dict(accuracy_rows=len(rows),evaluation_jobs_complete=complete)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=int);p.add_argument('--prefix',action='store_true');p.add_argument('--collect',action='store_true');p.add_argument('--inventory',action='store_true')
    a=p.parse_args()
    if a.inventory:print(len(inventory()))
    if a.job is not None:execute(json.loads(read_config_text(OUT/'inventory.json'))[a.job],a.prefix)
    if a.collect:collect()
