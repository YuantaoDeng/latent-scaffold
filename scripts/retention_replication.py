"""Held-out replication: suppress scaffold-directed plasticity, measure task one."""
from package_paths import read_config_text
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
import experiment_control as control
control.install_loader_hook()
import scaffold_intervention as s
import drift_experiment as d

OUT=control.OUT/'retention_replication'

def run(order_seed,seed):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    folder=OUT/f'order{order_seed}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    device=torch.device('cuda')
    bases={k:v for k,v in s.load(Path('results/scaffold_intervention/bases.pt')).items() if k!='pca'}
    cfg=SimpleNamespace(**json.loads(read_config_text(s.ROOT/f'target_order{order_seed}/manifest.json')))
    order=cfg.order;old=sum(order[:-1],[]);seen=sum(order,[])
    original=s.load(s.ROOT/f'target_order{order_seed}/artifacts/order{order_seed}_seed{seed}.pt')
    state=original['pre_future_state_dict'];del original
    train=d.IndexedCifar(Path('data'),True,True,'cifar10');evaluation=d.IndexedCifar(Path('data'),True,False,'cifar10')
    trainids={c:d.ids_by_class(train.targets,[c],450) for c in seen}
    priority=torch.rand(len(train),generator=torch.Generator().manual_seed(seed+9000))
    memory=d.balanced_memory(trainids,old,500,priority);current=torch.cat([trainids[c] for c in order[-1]])
    heldout=d.ids_by_class(train.targets,seen,50,True)
    assert not torch.isin(heldout,torch.cat([memory,current])).any()
    positions=torch.full((len(train),),-1,dtype=torch.long);positions[memory]=torch.arange(len(memory))
    images=torch.stack([evaluation[int(i)][1] for i in memory]).to(device)
    net=d.model('resnet18',False,num_classes=10).to(device);net.load_state_dict(state)
    initial=s.observe(net,evaluation,heldout,device);mi=s.observe(net,evaluation,memory,device)
    with torch.no_grad(): reference=torch.cat([d.features(net,d.prepare_images(x,device)) for x in images.split(32)])
    torch.save(dict(initial=initial,memory_initial=mi,heldout_ids=heldout,memory_ids=memory,current_ids=current,order=order),folder/'initial.pt')
    def measure(obs,mem):
        result=s.measure(obs,initial,mem,mi,order,bases)
        for i,classes in enumerate(order,1):
            mask=torch.isin(obs['labels'],torch.tensor(classes))
            result[f'task{i}_accuracy']=d.accuracy(obs['logits'][mask],obs['labels'][mask],seen)
        return result
    branches=[('baseline',0.)]+[(name,lam) for lam in (1.,10.,100.) for name in ('scaffold','random1','random2')]
    for name,lam in branches:
        label=f'{name}_lambda{lam:g}'; dest=folder/f'{label}.pt'; resume=folder/f'{label}_resume.pt'
        if dest.exists():continue
        control.poll(stage='retention',order=order_seed,seed=seed,condition=label)
        net.load_state_dict(state);opt=torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=.05)
        scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        rows=[dict(order=order_seed,seed=seed,condition=name,coefficient=lam,epoch=0,**measure(initial,mi),training_loss=0.,training_penalty=0.,steps=0,skipped_steps=0)]
        if resume.exists():
            v=s.load(resume);net.load_state_dict(v['state']);opt.load_state_dict(v['optimizer']);scaler.load_state_dict(v['scaler']);rows=v['rows'];del v
        for epoch in range(len(rows),6):
            started=time.monotonic()
            stats=s.train_epoch(net,opt,scaler,train,current,memory,seen,cfg,device,
                700000+order_seed*1000+seed*10+epoch,bases[name].to(device) if lam else None,lam,reference,images,positions)
            assert stats['skipped_steps']==0,stats
            obs=s.observe(net,evaluation,heldout,device);mem=s.observe(net,evaluation,memory,device)
            rows.append(dict(order=order_seed,seed=seed,condition=name,coefficient=lam,epoch=epoch,**measure(obs,mem),**stats))
            s.write_rows(folder/f'{label}.csv',rows)
            tmp=resume.with_suffix('.writing.pt');torch.save(dict(state=s.cpu_state(net),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rows=rows),tmp);tmp.replace(resume)
            print(json.dumps(dict(stage='retention',order=order_seed,seed=seed,condition=label,epoch=epoch,first_accuracy=rows[-1]['first_accuracy'],seconds=time.monotonic()-started)),flush=True)
        if len(rows)==6:
            obs=s.observe(net,evaluation,heldout,device);mem=s.observe(net,evaluation,memory,device)
        torch.save(dict(state=s.cpu_state(net),observations=obs,memory=mem,rows=rows),dest)
        resume.unlink(missing_ok=True)
        del opt,scaler
    control.write(folder/'complete.json',dict(complete=True,branches=len(branches),order=order_seed,seed=seed))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--order',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();run(a.order,a.seed)
