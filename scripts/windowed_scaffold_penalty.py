"""Equal-duration final-task penalty windows, ResNet18/CIFAR10."""
from package_paths import read_config_text
import argparse,csv,hashlib,json,time
from pathlib import Path
import torch
import scaffold_enhancement as e
from all_model_causal import observe,sha
d,s,ctl=e.d,e.s,e.ctl
OUT=ctl.OUT/'windowed_penalty'
WINDOWS={'early':(1,48),'middle':(49,96),'late':(97,144),'full':(1,145),'none':(0,0)}
def active(window,step):
    lo,hi=WINDOWS[window];return lo<=step<=hi and window!='none'
def run(order_seed,seed,model_name='resnet18'):
    global OUT
    mobile=model_name=='mobilenetv3_small'
    additional=model_name!='resnet18'
    if additional:OUT=ctl.OUT/('windowed_penalty_mobilenet_cifar10' if mobile else 'windowed_penalty_deit_cifar10')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    folder=OUT/f'order{order_seed}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    for w in ('early','middle','late'):assert sum(active(w,i) for i in range(1,146))==48
    source=Path(f'results/cifar10_confirmation/target_order{order_seed}/artifacts/order{order_seed}_seed{seed}.pt')
    if additional:
        c=next(c for c in json.loads(read_config_text(Path('results/all_model_followups/inventory.json'))) if c['short']==model_name and c['dataset']=='cifar10')
        source=next(Path(p) for p in c['paths'] if Path(p).stem==f'order{order_seed}_seed{seed}')
    state=torch.load(source,map_location='cpu',weights_only=False,mmap=True)['pre_future_state_dict']
    cfg=json.loads(read_config_text(source.parent.parent/'manifest.json'));order=cfg['order'];seen=sum(order,[]);old=sum(order[:-1],[])
    bases={k:v.float() for k,v in s.load(Path('results/scaffold_intervention/bases.pt')).items() if k!='pca'}
    basis_path='results/scaffold_intervention/bases.pt'
    if additional:
        basis_path=c['scaffold'];u=s.load(Path(basis_path))['scaffold'].float();dim,rank=u.shape
        assert rank==c['rank']==(24 if mobile else 8)
        bases={'scaffold':u,**{f'random{i}':torch.linalg.qr(torch.randn(dim,rank,generator=torch.Generator().manual_seed(91000+i))).Q for i in (1,2)}}
        for b in bases.values():assert float((b.T@b-torch.eye(rank)).abs().max())<2e-5
    train=d.IndexedCifar(Path('data'),True,True,'cifar10');evaluation=d.IndexedCifar(Path('data'),True,False,'cifar10')
    ids={c:d.ids_by_class(train.targets,[c],450) for c in seen};priority=torch.rand(len(train),generator=torch.Generator().manual_seed(seed+9000))
    memory=d.balanced_memory(ids,old,500,priority);current=torch.cat([ids[c] for c in order[-1]])
    heldout=d.ids_by_class(train.targets,seen,50,True);assert not torch.isin(heldout,torch.cat(list(ids.values()))).any()
    images=torch.stack([evaluation[int(i)][1] for i in memory]);positions=torch.full((len(train),),-1,dtype=torch.long);positions[memory]=torch.arange(len(memory))
    net=d.model(c['name'] if additional else 'resnet18',False,num_classes=10).cuda();net.load_state_dict(state)
    initial=observe(net,evaluation,heldout);mi=observe(net,evaluation,memory);reference=mi['features'].cuda()
    net.eval()
    with torch.no_grad():err=float((d.features(net,d.prepare_images(images[:32],'cuda'))-reference[:32]).abs().max())
    assert err<1e-4,err
    ctl.write(folder/'provenance.json',dict(source=str(source),sha256=sha(source),basis_sha256=sha(basis_path),reference_error=err,model=model_name,rank=bases['scaffold'].shape[1],amp_enabled=not mobile))
    def measure(step,epoch,window,name):
        obs=observe(net,evaluation,heldout);mem=observe(net,evaluation,memory);r=dict(step=step,epoch=epoch,window=window,direction=name)
        for j,classes in enumerate(order,1):
            mask=torch.isin(obs['labels'],torch.tensor(classes));r[f'task{j}_accuracy']=d.accuracy(obs['logits'][mask],obs['labels'][mask],seen)
            r[f'task{j}_loss']=float(d.masked_ce(obs['logits'][mask],obs['labels'][mask],seen))
        for label,z,z0 in [('memory',mem['features'],mi['features']),('first',obs['features'][:100],initial['features'][:100])]:
            r.update({label+'_'+k:v for k,v in e.geometry(z-z0,bases['scaffold']).items()})
        return r,obs
    branches=[('baseline','none')]+[(n,w) for w in ('early','middle','late','full') for n in ('scaffold','random1','random2')]
    for name,window in branches:
        label=name+'_'+window;dest=folder/f'{label}.pt';resume=folder/f'{label}_resume.pt'
        if dest.exists():continue
        net.load_state_dict(state);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.05);scaler=torch.amp.GradScaler('cuda',init_scale=128.,enabled=not mobile)
        rows=[];epochs=[];next_epoch=1;on_steps=0
        if resume.exists():
            v=s.load(resume);net.load_state_dict(v['state']);opt.load_state_dict(v['optimizer']);scaler.load_state_dict(v['scaler']);rows=v['rows'];epochs=v['epochs'];next_epoch=v['epoch']+1;on_steps=v['on_steps'];del v
        else:rows.append(measure(0,0,window,name)[0])
        for epoch in range(next_epoch,6):
            began=time.perf_counter();es=700000+order_seed*1000+seed*10+epoch;d.seed_all(es)
            cur=d.loader(train,current,32,True,es);rep=d.loader(train,memory,32,True,es+100);replay=iter(rep);trace=hashlib.sha256();pen=[];losses=[]
            assert len(cur)==29
            for batch,(ci,cx,cy) in enumerate(cur,1):
                step=(epoch-1)*29+batch;ctl.poll(stage='windowed_penalty',order=order_seed,seed=seed,branch=label,step=step)
                try:ri,rx,ry=next(replay)
                except StopIteration:replay=iter(rep);ri,rx,ry=next(replay)
                raw=torch.cat([cx,rx]);ys=torch.cat([cy,ry])
                for x in (torch.cat([ci,ri]),raw,ys):trace.update(x.numpy().tobytes())
                opt.zero_grad(set_to_none=True)
                if active(window,step):
                    net.eval();z=d.features(net,d.prepare_images(images[positions[ri]],'cuda'));reg=s.penalty(z-reference[positions[ri].cuda()],bases[name].cuda())
                    assert torch.isfinite(reg);scaler.scale(10*reg).backward();pen.append(float(reg.detach()));on_steps+=1
                net.train()
                with torch.amp.autocast('cuda',enabled=not mobile):
                    logits=net(d.prepare_images(raw,'cuda'));y=ys.cuda();loss=d.masked_ce(logits[:len(ci)],y[:len(ci)],seen)+d.masked_ce(logits[len(ci):],y[len(ci):],seen)
                assert torch.isfinite(loss);scaler.scale(loss).backward();scale=scaler.get_scale();scaler.step(opt);scaler.update();assert scaler.get_scale()>=scale,'Skipped AMP update'
                losses.append(float(loss.detach()))
                if step in (48,96,144):rows.append(measure(step,epoch,window,name)[0])
            row,obs=measure(epoch*29,epoch,window,name);rows.append(row)
            ep=dict(epoch=epoch,input_hash=trace.hexdigest(),penalty_steps=sum(active(window,(epoch-1)*29+i) for i in range(1,30)),penalty_mean=sum(pen)/max(1,len(pen)),loss=sum(losses)/len(losses),seconds=time.perf_counter()-began)
            if name!='baseline':
                with (folder/'baseline_none_epochs.csv').open() as f:b=next(r for r in csv.DictReader(f) if int(r['epoch'])==epoch)
                assert b['input_hash']==ep['input_hash']
            epochs.append(ep);s.write_rows(folder/f'{label}.csv',rows);s.write_rows(folder/f'{label}_epochs.csv',epochs)
            e.save(dict(state=s.cpu_state(net),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rows=rows,epochs=epochs,epoch=epoch,on_steps=on_steps),resume)
            print(json.dumps(dict(order=order_seed,seed=seed,branch=label,**ep)),flush=True)
        assert on_steps==(0 if window=='none' else 145 if window=='full' else 48)
        obs=observe(net,evaluation,heldout)
        e.save(dict(rows=rows,epochs=epochs,observations=obs,on_steps=on_steps,order=order),dest);resume.unlink(missing_ok=True)
        del opt,scaler
    ctl.write(folder/'complete.json',dict(complete=True,branches=13,order=order_seed,seed=seed))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--order',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--model',choices=['resnet18','mobilenetv3_small','deit_tiny'],default='resnet18');a=p.parse_args();run(a.order,a.seed,a.model)
