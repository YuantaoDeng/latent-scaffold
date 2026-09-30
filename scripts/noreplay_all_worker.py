"""Five-task no-replay ResNet experiment; isolated, resumable history workers."""
from package_paths import read_config_text
import argparse, json, subprocess, sys, time
from pathlib import Path
import torch
import drift_experiment as d
import analyze_scaffold_followup as f

from noreplay_all_common import configure,save_checkpoint
BASE=Path('results/cifar10_confirmation')
INITIAL=Path('results/resnet_temporal_dimension/cifar10/sources/artifacts/order2003_seed160.pt')

def write(p,obj): p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def state(net): return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
def representations(net,batches):
    mode=net.training; rng=f.rng_state()
    recomputation=getattr(net,'grad_checkpointing',False)
    try:
        if recomputation:net.set_grad_checkpointing(False)
        return f.cached_representations(net,batches).double()
    finally:
        if recomputation:net.set_grad_checkpointing(True)
        net.train(mode);f.restore_rng(rng)

def worker(role,order_seed,seed):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    folder=OUT/role/f'order{order_seed}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    cfg=json.loads(read_config_text(BASE/f'{role}_order{order_seed}/manifest.json'));order=cfg['order']
    train=d.IndexedCifar(Path('data'),True,True,'cifar10');evaluation=d.IndexedCifar(Path('data'),True,False,'cifar10')
    ids={c:d.ids_by_class(train.targets,[c],450) for c in range(10)}
    priority=torch.rand(len(train),generator=torch.Generator().manual_seed(seed+9000))
    fitids=d.balanced_memory(ids,order[0],500,priority)
    testids=d.ids_by_class(evaluation.targets,order[0],50,tail=True)
    assert not torch.isin(testids,torch.cat(list(ids.values()))).any()
    torch.save(dict(fit_ids=fitids,test_ids=testids,labels=evaluation.targets[testids],order=order),folder/'anchors.pt')
    net=d.model(MODEL_NAME,False,num_classes=10).cuda()
    if C['activation_recomputation']:net.set_grad_checkpointing(True)
    net.load_state_dict(f.load(OUT/'initial_state.pt'))
    fitbatch=f.image_batches(net,evaluation,fitids,'cuda');testbatch=f.image_batches(net,evaluation,testids,'cuda')
    if not (folder/'task0.pt').exists():
        save_checkpoint(dict(state=state(net),fit=representations(net,fitbatch),eval=representations(net,testbatch)),folder/'task0.pt',OUT/'initial_state.pt')
    started=time.perf_counter()
    for task in range(1,6):
        dest=folder/f'task{task}.pt'
        if dest.exists():continue
        net.load_state_dict(f.load(folder/f'task{task-1}.pt')['state'])
        d.seed_all(seed*100+task-1);net.train()
        opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.05)
        scaler=torch.amp.GradScaler('cuda',init_scale=128.,enabled=AMP)
        current=torch.cat([ids[c] for c in order[task-1]]);seen=sum(order[:task],[])
        captured=[];step=0
        for epoch in range(5):
            for batchids,x,y in d.loader(train,current,64,True,seed*100+task-1+epoch):
                assert torch.isin(y,torch.tensor(order[task-1])).all()
                observe=role=='target' and task>=2 and step+1 in (2,38,73)
                if observe:before=state(net)
                x=d.prepare_images(x,'cuda');y=y.cuda();opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=AMP):
                    loss=d.masked_ce(net(x),y,seen)
                assert torch.isfinite(loss).item(), "Nonfinite training loss"
                scaler.scale(loss).backward();oldscale=scaler.get_scale();scaler.step(opt);scaler.update()
                assert scaler.get_scale()>=oldscale,'Skipped update: do not accept incomplete trajectory'
                step+=1
                if observe:
                    after=state(net);gradient={n:p.grad.detach().cpu().clone() for n,p in net.named_parameters() if p.grad is not None}
                    rng=f.rng_state()
                    record=dict(manifest={'model':MODEL_NAME},heldout_ids=testids,
                        actual_update=dict(before_state=before,after_state=after,gradient=gradient))
                    if C['activation_recomputation']:
                        from mixer_update_measurement import measure
                        fields,norm=measure(net,record,evaluation)
                    else:fields,norm=f.actual_update_fields(record,evaluation,torch.device('cuda'))
                    f.restore_rng(rng)
                    captured.append(dict(step=step,fields=fields,relative_norm=norm,classifier=before[HEAD_WEIGHT],seen=seen))
                    del before,after,gradient;torch.cuda.empty_cache()
            write(folder/'progress.json',dict(task=task,epoch=epoch+1,steps=step,seconds=time.perf_counter()-started))
        assert step==75
        result=dict(state=state(net),fit=representations(net,fitbatch),eval=representations(net,testbatch),
                    steps=step,seen=seen,current_ids=current,updates=captured)
        assert torch.isfinite(result["fit"]).all() and torch.isfinite(result["eval"]).all(), "Nonfinite endpoint features"
        save_checkpoint(result,dest,folder/f'task{task-1}.pt')
        print(json.dumps(dict(role=role,order=order_seed,seed=seed,task=task,seconds=round(time.perf_counter()-started,1))),flush=True)
        del result,opt,scaler;torch.cuda.empty_cache()
    write(folder/'complete.json',dict(complete=True,seconds=time.perf_counter()-started,tasks=5,replay_images=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--role',required=True);p.add_argument('--order',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();globals().update(configure(a.model));worker(a.role,a.order,a.seed)
