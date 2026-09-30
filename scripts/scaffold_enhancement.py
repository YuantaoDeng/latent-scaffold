"""Directional feature-gradient enhancement with paired continuation controls."""
from package_paths import read_config_text
import argparse,hashlib,json,os,time
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('HF_HOME',str(Path('data/hf_cache').resolve()))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import experiment_control as ctl
ctl.install_loader_hook()
import drift_experiment as d
import analyze_scaffold_followup as f
import scaffold_intervention as s

OUT=ctl.OUT/'enhancement'
DEVICE=torch.device('cuda')

def save(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.writing.pt');torch.save(value,temp);temp.replace(path)

def enhance(g,basis,scaffold,gain,random_control=False):
    if gain==1:return g
    v=g.float();sc=(v@scaffold)@scaffold.T
    if random_control:
        selected=(v@basis)@basis.T
        a=sc.norm();b=selected.norm()
        if b.item()<1e-20:
            if a.item()>1e-15:raise RuntimeError('Degenerate random gradient projection')
            return g
        selected=selected*(a/b)
    else:selected=sc
    return (v+(gain-1)*selected).to(g.dtype)

@torch.inference_mode()
def observe(net,data,ids):
    mode=net.training;rng=f.rng_state();net.eval();zs=[];ls=[];ys=[]
    try:
        for _,x,y in d.loader(data,ids,64):
            # FP32 observation resolves the small changes made by one update.
            z=d.features(net,d.prepare_images(x,DEVICE));logits=net.get_classifier()(z)
            zs.append(z.cpu());ls.append(logits.cpu());ys.append(y)
        return dict(features=torch.cat(zs),logits=torch.cat(ls),labels=torch.cat(ys))
    finally:net.train(mode);f.restore_rng(rng)

def geometry(delta,u):
    v=delta.double();v=v-v.mean(0);total=float(v.square().sum());inside=float((v@u.double()).square().sum())
    return dict(total_energy=total/max(1,len(v)),scaffold_energy=inside/max(1,len(v)),scaffold_coverage=inside/max(total,1e-30))

def metrics(obs,order,task,reference,u):
    seen=sum(order[:task],[]);result={}
    for j,classes in enumerate(order,1):
        if j>task:
            result[f'task{j}_accuracy']=None;result[f'task{j}_loss']=None;continue
        mask=torch.isin(obs['labels'],torch.tensor(classes));logits=obs['logits'][mask];labels=obs['labels'][mask]
        result[f'task{j}_accuracy']=d.accuracy(logits,labels,seen)
        result[f'task{j}_loss']=float(d.masked_ce(logits,labels,seen))
    first=torch.isin(obs['labels'],torch.tensor(order[0]))
    result.update({'first_'+k:v for k,v in geometry(obs['features'][first]-reference,u).items()})
    result['current_accuracy']=result[f'task{task}_accuracy']
    result['old_mean_accuracy']=sum(result[f'task{j}_accuracy'] for j in range(1,task))/(task-1)
    return result

def start_state(order_seed,seed,train,evaluation,order,cfg,ids):
    folder=OUT/f'order{order_seed}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'task2_start.pt'
    if dest.exists():return s.load(dest)
    rec=s.load(Path('results/cifar10_confirmation/scaffold_followup/target_recovery/artifacts')/f'order{order_seed}_seed{seed}.pt')
    d.seed_all(seed);net=d.model('resnet18',True,num_classes=10).cuda()
    priority=torch.rand(len(train),generator=torch.Generator().manual_seed(seed+9000))
    errors=[];seen=[]
    for i in range(2):
        old=list(seen);seen+=order[i]
        memory=d.balanced_memory(ids,old,500,priority) if old else torch.empty(0,dtype=torch.long)
        d.train_task(net,'naive' if i==0 else 'er',train,torch.cat([ids[c] for c in order[i]]),memory,{},seen,old,cfg,DEVICE,seed*100+i)
        rng=f.rng_state();_,z,_=d.infer(net,evaluation,rec['heldout_ids'],64,DEVICE);f.restore_rng(rng)
        error=float((z-rec['eval_reps'][i].float()).abs().max());errors.append(error)
        if error>1e-5:raise RuntimeError(f'Start history mismatch task{i+1}: {error}')
    heldout=d.ids_by_class(evaluation.targets,sum(order,[]),50,True)
    obs=observe(net,evaluation,heldout)
    value=dict(state=s.cpu_state(net),observations=obs,heldout_ids=heldout,order=order,feature_recovery_errors=errors)
    save(value,dest);del rec,net;torch.cuda.empty_cache()
    print(json.dumps(dict(stage='verified_start',order=order_seed,seed=seed,errors=errors)),flush=True)
    return value

def checks(state,data,current,memory,seen,bases):
    torch.manual_seed(9231);g=torch.randn(64,512);u=bases['scaffold'];r=bases['random1']
    for basis in bases.values():assert (basis.T@basis-torch.eye(8)).abs().max()<2e-5
    for gain in (1.25,1.5,2.):
        a=enhance(g,u,u,gain);b=enhance(g,r,u,gain,True)
        assert torch.allclose(a@u,gain*(g@u),atol=1e-5,rtol=1e-5)
        assert abs(float((a-g).norm()/(b-g).norm())-1)<1e-5
    d.seed_all(81923)
    _,cx,cy=next(iter(d.loader(data,current[:32],32)))
    _,rx,ry=next(iter(d.loader(data,memory[:32],32)))
    x=d.prepare_images(torch.cat([cx,rx]),DEVICE);y=torch.cat([cy,ry]).cuda()
    net=d.model('resnet18',False,num_classes=10).cuda();states=[];head_grads=[]
    for gain,ordinary in [(1.,True),(1.,False),(1.5,False)]:
        net.load_state_dict(state);net.train();opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.05)
        scaler=torch.amp.GradScaler('cuda',init_scale=128.);opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            if ordinary:logits=net(x)
            else:
                z=d.features(net,x)
                if gain!=1:z.register_hook(lambda grad:enhance(grad,u.cuda(),u.cuda(),gain))
                logits=net.get_classifier()(z)
            loss=d.masked_ce(logits[:32],y[:32],seen)+d.masked_ce(logits[32:],y[32:],seen)
        scaler.scale(loss).backward();head_grads.append(net.get_classifier().weight.grad.detach().clone())
        scaler.step(opt);scaler.update();states.append(s.cpu_state(net))
    error=max(float((states[0][k].double()-states[1][k].double()).abs().max()) for k in states[0])
    assert error==0,error
    assert torch.equal(head_grads[0],head_grads[2]),'Classifier gradient altered'
    ctl.write(OUT/'implementation_checks.json',dict(gain_one_state_parity_error=error,classifier_gradient_identical=True,algebraic_gain_and_random_norm_match=True))
    del net,states;torch.cuda.empty_cache()

def epoch_train(net,opt,scaler,train,current,memory,seen,basis,u,gain,name,epoch_seed,evaluation,first_ids,task,epoch):
    d.seed_all(epoch_seed);net.train();cur=d.loader(train,current,32,True,epoch_seed);rep=d.loader(train,memory,32,True,epoch_seed+100)
    replay=iter(rep);digest=hashlib.sha256();updates=[];losses=[];skips=0
    for i,(ci,cx,cy) in enumerate(cur):
        try:ri,rx,ry=next(replay)
        except StopIteration:replay=iter(rep);ri,rx,ry=next(replay)
        ids=torch.cat([ci,ri]);images=torch.cat([cx,rx]);labels=torch.cat([cy,ry])
        for v in (ids,images,labels):digest.update(v.numpy().tobytes())
        step=(epoch-1)*len(cur)+i+1;capture=step in (2,73,145)
        if capture:before=observe(net,evaluation,first_ids)['features']
        net.train();x=d.prepare_images(images,DEVICE);y=labels.cuda();opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            z=d.features(net,x)
            if gain!=1:z.register_hook(lambda grad:enhance(grad,basis,u,gain,name.startswith('random')))
            logits=net.get_classifier()(z)
            loss=d.masked_ce(logits[:len(ci)],y[:len(ci)],seen)+d.masked_ce(logits[len(ci):],y[len(ci):],seen)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scale=scaler.get_scale();scaler.step(opt);scaler.update()
        skips+=scaler.get_scale()<scale
        if skips:raise RuntimeError(f'AMP skipped update at task{task},step{step}; reject trajectory')
        losses.append(float(loss.detach()))
        if capture:
            after=observe(net,evaluation,first_ids)['features']
            updates.append(dict(task=task,step=step,before=before,after=after,**geometry(after-before,u.cpu())))
    return dict(training_loss=sum(losses)/len(losses),steps=len(losses),skipped_steps=skips,input_hash=digest.hexdigest()),updates

def run(order_seed,seed,benchmark=False):
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    cfg=SimpleNamespace(**json.loads(read_config_text(s.ROOT/f'target_order{order_seed}/manifest.json')));order=cfg.order
    train=d.IndexedCifar(Path('data'),True,True,'cifar10');evaluation=d.IndexedCifar(Path('data'),True,False,'cifar10')
    ids={c:d.ids_by_class(train.targets,[c],450) for c in sum(order,[])}
    bases={k:v.float() for k,v in s.load(Path('results/scaffold_intervention/bases.pt')).items() if k!='pca'}
    start=start_state(order_seed,seed,train,evaluation,order,cfg,ids)
    folder=OUT/f'order{order_seed}_seed{seed}'
    if (folder/'complete.json').exists():return
    priority=torch.rand(len(train),generator=torch.Generator().manual_seed(seed+9000))
    firstids=d.ids_by_class(evaluation.targets,order[0],50,True)
    assert not torch.isin(start['heldout_ids'],torch.cat(list(ids.values()))).any()
    firstmask=torch.isin(start['observations']['labels'],torch.tensor(order[0]));reference=start['observations']['features'][firstmask]
    if not (OUT/'implementation_checks.json').exists():
        checks(start['state'],train,torch.cat([ids[c] for c in order[2]]),d.balanced_memory(ids,sum(order[:2],[]),500,priority),sum(order[:3],[]),bases)
    net=d.model('resnet18',False,num_classes=10).cuda();u=bases['scaffold'].cuda()
    branches=[('baseline',1.)]+[(n,g) for g in (1.25,1.5,2.) for n in ('scaffold','random1','random2')]
    for name,gain in branches:
        label=f'{name}_gain{gain:g}';branch=folder/label;branch.mkdir(exist_ok=True)
        if (branch/'complete.json').exists():continue
        previous=start['state'];rows=[]
        for task in (3,4,5):
            dest=branch/f'task{task}.pt';resume=branch/f'task{task}_resume.pt'
            if dest.exists():
                value=s.load(dest);previous=value['state'];rows=value['rows'];continue
            net.load_state_dict(previous);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.05)
            scaler=torch.amp.GradScaler('cuda',init_scale=128.);updates=[];next_epoch=1
            if resume.exists():
                value=s.load(resume);net.load_state_dict(value['state']);opt.load_state_dict(value['optimizer']);scaler.load_state_dict(value['scaler'])
                rows=value['rows'];updates=value['updates'];next_epoch=value['epoch']+1;del value
            current=torch.cat([ids[c] for c in order[task-1]]);seen=sum(order[:task],[])
            memory=d.balanced_memory(ids,sum(order[:task-1],[]),500,priority)
            for epoch in range(next_epoch,6):
                ctl.poll(stage='enhancement',order=order_seed,seed=seed,condition=label,task=task,epoch=epoch)
                began=time.perf_counter()
                stats,fields=epoch_train(net,opt,scaler,train,current,memory,seen,bases.get(name,bases['scaffold']).cuda(),u,gain,name,
                    8100000+order_seed*1000+seed*30+task*5+epoch,evaluation,firstids,task,epoch)
                updates.extend(fields);obs=observe(net,evaluation,start['heldout_ids'])
                row=dict(order=order_seed,seed=seed,condition=name,gain=gain,task=task,epoch=epoch,**metrics(obs,order,task,reference,bases['scaffold']),**stats)
                baseline_csv=folder/'baseline_gain1'/'curve.csv'
                if name!='baseline':
                    import csv
                    b=next(r for r in csv.DictReader(baseline_csv.open()) if int(r['task'])==task and int(r['epoch'])==epoch)
                    assert row['input_hash']==b['input_hash'],'Paired input streams differ'
                rows.append(row);s.write_rows(branch/'curve.csv',rows)
                save(dict(state=s.cpu_state(net),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rows=rows,updates=updates,epoch=epoch),resume)
                elapsed=time.perf_counter()-began
                print(json.dumps(dict(stage='enhancement',order=order_seed,seed=seed,condition=label,task=task,epoch=epoch,first_accuracy=row['task1_accuracy'],new_accuracy=row['current_accuracy'],seconds=elapsed)),flush=True)
                if benchmark:
                    ctl.write(OUT/'benchmark.json',dict(epoch_seconds=elapsed,epochs_total=900,naive_hours=elapsed*900/3600,
                        note='First normal-training epoch including measurement and checkpoint I/O; enhancement and setup overhead remain.'))
                    return
            obs=observe(net,evaluation,start['heldout_ids']);previous=s.cpu_state(net)
            save(dict(state=previous,observations=obs,rows=rows,updates=updates,order=order,seen=seen,heldout_ids=start['heldout_ids']),dest)
            resume.unlink(missing_ok=True);del opt,scaler
        ctl.write(branch/'complete.json',dict(complete=True,condition=name,gain=gain,tasks=[3,4,5]))
    ctl.write(folder/'complete.json',dict(complete=True,order=order_seed,seed=seed,branches=10,tasks_per_branch=3))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--order',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--benchmark',action='store_true')
    a=p.parse_args();run(a.order,a.seed,a.benchmark)
