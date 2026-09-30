"""Five-task no-replay ResNet experiment; isolated, resumable history workers."""
from package_paths import read_config_text
import argparse, json, subprocess, sys, time
from pathlib import Path
import torch
import drift_experiment as d
import analyze_scaffold_followup as f

OUT=Path('results/noreplay_resnet_cifar10')
BASE=Path('results/cifar10_confirmation')
INITIAL=Path('results/resnet_temporal_dimension/cifar10/sources/artifacts/order2003_seed160.pt')

def write(p,obj): p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def state(net): return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
def representations(net,batches):
    mode=net.training; rng=f.rng_state()
    try:return f.cached_representations(net,batches).double()
    finally:net.train(mode);f.restore_rng(rng)

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
    net=d.model('resnet18',False,num_classes=10).cuda()
    net.load_state_dict(f.load(OUT/'initial_state.pt'))
    fitbatch=f.image_batches(net,evaluation,fitids,'cuda');testbatch=f.image_batches(net,evaluation,testids,'cuda')
    if not (folder/'task0.pt').exists():
        torch.save(dict(state=state(net),fit=representations(net,fitbatch),eval=representations(net,testbatch)),folder/'task0.pt')
    started=time.perf_counter()
    for task in range(1,6):
        dest=folder/f'task{task}.pt'
        if dest.exists():continue
        net.load_state_dict(f.load(folder/f'task{task-1}.pt')['state'])
        d.seed_all(seed*100+task-1);net.train()
        opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.05)
        scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        current=torch.cat([ids[c] for c in order[task-1]]);seen=sum(order[:task],[])
        captured=[];step=0
        for epoch in range(5):
            for batchids,x,y in d.loader(train,current,64,True,seed*100+task-1+epoch):
                assert torch.isin(y,torch.tensor(order[task-1])).all()
                observe=role=='target' and task>=2 and step+1 in (2,38,73)
                if observe:before=state(net)
                x=d.prepare_images(x,'cuda');y=y.cuda();opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    loss=d.masked_ce(net(x),y,seen)
                scaler.scale(loss).backward();oldscale=scaler.get_scale();scaler.step(opt);scaler.update()
                assert scaler.get_scale()>=oldscale,'Skipped update: do not accept incomplete trajectory'
                step+=1
                if observe:
                    after=state(net);gradient={n:p.grad.detach().cpu().clone() for n,p in net.named_parameters() if p.grad is not None}
                    rng=f.rng_state()
                    fields,norm=f.actual_update_fields(dict(manifest={'model':'resnet18'},heldout_ids=testids,
                        actual_update=dict(before_state=before,after_state=after,gradient=gradient)),evaluation,torch.device('cuda'))
                    f.restore_rng(rng)
                    captured.append(dict(step=step,fields=fields,relative_norm=norm,classifier=before['fc.weight'],seen=seen))
                    del before,after,gradient;torch.cuda.empty_cache()
            write(folder/'progress.json',dict(task=task,epoch=epoch+1,steps=step,seconds=time.perf_counter()-started))
        assert step==75
        result=dict(state=state(net),fit=representations(net,fitbatch),eval=representations(net,testbatch),
                    steps=step,seen=seen,current_ids=current,updates=captured)
        torch.save(result,dest)
        print(json.dumps(dict(role=role,order=order_seed,seed=seed,task=task,seconds=round(time.perf_counter()-started,1))),flush=True)
        del result,opt,scaler;torch.cuda.empty_cache()
    write(folder/'complete.json',dict(complete=True,seconds=time.perf_counter()-started,tasks=5,replay_images=0))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'initial_state.pt').exists():torch.save(f.load(INITIAL)['states'][0],OUT/'initial_state.pt')
    write(OUT/'protocol.json',dict(model='ResNet-18',dataset='CIFAR-10',tasks=5,classes_per_task=2,
        replay=False,memory_images=0,training_images_per_class=450,epochs=5,current_batch_size=64,
        updates_per_task=75,optimizer='AdamW lr=0.0003 wd=0.05; reset per task; AMP initial scale128, zero skipped steps',
        scaffold_rank=8,rank_selection='Fixed original manuscript rank, never selected on new test outcomes',
        initial_state=str(INITIAL),initial_state_index=0,initialization='Identical full pretrained initial state across all histories',
        anchors='500 first-task training images for fitting, 100 disjoint first-task images for testing; anchors are never replayed',
        source_orders=[2003,2011,2017],source_seeds=[160,161],target_orders=[2101,2111,2129,2131,2137,2141],target_seeds=[170,171],
        update_positions=[2,38,73],update_tasks=[2,3,4,5],
        comparison_scope='Same task orders, data counts and epochs as paper; no-replay batch64 gives75 updates vs replay145 after task1. No pure replay-only causal contrast.',
        statistics='Hierarchical order/run bootstrap conditional on fitted scaffold; shared training content across source/test',status='running'))
    started=time.perf_counter()
    for role,orders,seeds in [('source',[2003,2011,2017],[160,161]),('target',[2101,2111,2129,2131,2137,2141],[170,171])]:
        for order in orders:
            for seed in seeds:
                write(OUT/'progress.json',dict(status='training',role=role,order=order,seed=seed,seconds=time.perf_counter()-started))
                subprocess.run([sys.executable,__file__,'--worker','--role',role,'--order',str(order),'--seed',str(seed)],check=True)
    write(OUT/'progress.json',dict(status='training_complete',seconds=time.perf_counter()-started))
    print('ALL_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--role');p.add_argument('--order',type=int);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.worker:worker(a.role,a.order,a.seed)
    else:run()
