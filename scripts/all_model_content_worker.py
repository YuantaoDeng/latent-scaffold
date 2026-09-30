"""One isolated paired run of the fixed four-task early-content experiment."""
import argparse, gc, hashlib, json, time, math
from pathlib import Path
import torch
import drift_experiment as d
import analyze_scaffold_followup as f
from all_model_followups_config import ROOT,conditions
from early_content_scaffold_pilot import TASKS,FIT_SEEDS,TEST_SEEDS

def load(path): return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def state(net): return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
@torch.no_grad()
def reps(net,data,ids):
    rng=f.rng_state(); mode=net.training; checkpointing=getattr(net,'grad_checkpointing',False)
    try:
        net.eval()
        if checkpointing: net.set_grad_checkpointing(False)
        return torch.cat([d.features(net,d.prepare_images(x,'cuda')).cpu().double() for _,x,_ in d.loader(data,ids,64)])
    finally:
        if checkpointing: net.set_grad_checkpointing(True)
        net.train(mode); f.restore_rng(rng)
def train(net,data,current,memory,seen,seed,progress,identity):
    d.seed_all(seed); net.train(); opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.05)
    amp_enabled=identity['model']!='mobilenetv3_small'
    initial_scale=128.
    scaler=torch.amp.GradScaler('cuda',init_scale=initial_scale,enabled=amp_enabled)
    trace=hashlib.sha256(); steps=0; started=time.perf_counter()
    for epoch in range(5):
        current_loader=d.loader(data,current,32 if len(memory) else 64,True,seed+epoch)
        replay_loader=d.loader(data,memory,32,True,seed+100+epoch) if len(memory) else None
        replay=iter(replay_loader) if replay_loader else None
        for ci,cx,cy in current_loader:
            if replay is not None:
                try:ri,rx,ry=next(replay)
                except StopIteration:replay=iter(replay_loader);ri,rx,ry=next(replay)
                ids=torch.cat([ci,ri]); x=torch.cat([cx,rx]); y=torch.cat([cy,ry])
            else:ids,x,y=ci,cx,cy
            trace.update(ids.numpy().tobytes());trace.update(x.numpy().tobytes());trace.update(y.numpy().tobytes())
            x=d.prepare_images(x,'cuda');y=y.cuda();opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=amp_enabled):
                logits=net(x);loss=d.masked_ce(logits[:len(ci)],y[:len(ci)],seen)
                if replay is not None:loss+=d.masked_ce(logits[len(ci):],y[len(ci):],seen)
            scaler.scale(loss).backward(); scale=scaler.get_scale();scaler.step(opt);scaler.update()
            if scaler.get_scale()<scale:
                bad=[n for n,p in net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                write(progress.parent/f"overflow_{identity['branch']}_task{identity['task']}_scale{initial_scale:g}.json",{**identity,'epoch':epoch+1,'completed_steps':steps,'initial_scale':initial_scale,'loss':float(loss.detach()),'nonfinite_gradients':bad})
                raise RuntimeError('Skipped optimizer update; diagnostic saved, condition must be audited')
            steps+=1
        status={**identity,'epoch':epoch+1,'epochs':5,'successful_steps':steps,'seconds':time.perf_counter()-started,'last_loss':float(loss.detach())}
        write(progress,status);print(json.dumps(status),flush=True)
    opt.zero_grad(set_to_none=True);del opt,scaler;torch.cuda.empty_cache()
    return trace.hexdigest(),steps
def run(short,role,seed,dataset='cifar10',tasks=None,final_task=4,output_root=None):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    taskset=tasks or TASKS
    num_classes=10 if dataset=='cifar10' else 100
    common_seen=sorted(set(sum(taskset.values(),[])))
    c=next(c for c in conditions() if c['short']==short and c['dataset']==dataset)
    out=Path(output_root) if output_root else ROOT/'content'/f'{short}_{dataset}';out.mkdir(parents=True,exist_ok=True)
    folder=out/f'{role}_seed{seed}';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    started=time.perf_counter(); initpath=out/'initial_state.pt'
    if not initpath.exists():
        source=load(c['initial_source']); init=source if c['initial_is_raw_state'] else source['states'][0]
        torch.save(init,initpath);write(out/'initialization.json',{'source':c['initial_source'],'state_index':None if c['initial_is_raw_state'] else 0,'model':c['name']})
        del init,source;gc.collect()
    trainset=d.IndexedCifar(Path('data'),True,True,dataset);evalset=d.IndexedCifar(Path('data'),True,False,dataset)
    ids={k:d.ids_by_class(trainset.targets,[k],450) for k in range(num_classes)}
    fitpriority=torch.rand(len(trainset),generator=torch.Generator().manual_seed(820012))
    fitids=d.balanced_memory(ids,sorted(taskset['A']),500,fitpriority).sort().values
    testids=d.ids_by_class(evalset.targets,sorted(taskset['A']),50,tail=True)
    assert not torch.isin(testids,torch.cat(list(ids.values()))).any()
    priority=torch.rand(len(trainset),generator=torch.Generator().manual_seed(seed+9000))
    current=lambda classes:torch.cat([ids[k] for k in sorted(classes)])
    memory=lambda classes:d.balanced_memory(ids,sorted(classes),500,priority).sort().values
    if not (out/'anchors.pt').exists():torch.save({'fit_ids':fitids,'test_ids':testids,'tasks':taskset},out/'anchors.pt')
    net=d.model(c['name'],False,num_classes=num_classes).cuda()
    if short=='mixer_b16':net.set_grad_checkpointing(True)
    sharedpath=folder/'shared_A.pt'
    if not sharedpath.exists():
        net.load_state_dict(load(initpath))
        trace,steps=train(net,trainset,current(taskset['A']),torch.empty(0,dtype=torch.long),sorted(taskset['A']),900000+seed*100,folder/'progress.json',{'model':short,'role':role,'seed':seed,'branch':'shared','task':1})
        assert steps==5*math.ceil(len(current(taskset['A']))/64)
        shared={'state':state(net),'fit':reps(net,evalset,fitids),'eval':reps(net,evalset,testids),'trace':trace,'steps':steps}
        torch.save(shared,sharedpath);del shared;gc.collect()
    shared=load(sharedpath);traces={}
    for branch in ['X','Y']:
        previous=shared
        for task in range(2,3 if role=='fit' else final_task+1):
            path=folder/f'{branch}_task{task}.pt'
            if path.exists():record=load(path)
            else:
                net.load_state_dict(previous['state'])
                classes=taskset['B'+branch] if task==2 else taskset['C' if task==3 else 'D']
                memclasses=taskset['A'] if task<4 else taskset['A']+taskset['C']; seen=sorted(taskset['A']+classes) if task==2 else common_seen
                trace,steps=train(net,trainset,current(classes),memory(memclasses),seen,900000+seed*100+task*10,folder/'progress.json',{'model':short,'role':role,'seed':seed,'branch':branch,'task':task})
                assert steps==5*math.ceil(len(current(classes))/32)
                zfit=reps(net,evalset,fitids);zeval=reps(net,evalset,testids)
                record={'state':state(net),'fit':zfit,'eval':zeval,'trace':trace,'steps':steps,'task':task,'branch':branch,'seed':seed,
                    'memory_ids':memory(memclasses),'current_ids':current(classes),'seen':seen,'drift_eval':f.center(zeval-previous['eval'])}
                if task==2:record['drift_fit']=f.center(zfit-shared['fit']);record['basis']=f.basis(record['drift_fit'],c['rank'])
                torch.save(record,path)
            traces[(branch,task)]=record['trace'];previous=record
            torch.cuda.empty_cache();gc.collect()
            print(short,role,seed,branch,'task',task,'SAVED',flush=True)
    if role=='test':
        for task in range(3,final_task+1):assert traces[('X',task)]==traces[('Y',task)]
    write(folder/'complete.json',{'model':short,'dataset':dataset,'final_task':final_task,'role':role,'seed':seed,'rank':c['rank'],'seconds':time.perf_counter()-started,'common_streams_identical':role=='test','activation_recomputation':short=='mixer_b16','amp_enabled':short!='mobilenetv3_small','amp_initial_scale':None if short=='mobilenetv3_small' else 128})
    print('PAIR COMPLETE',short,role,seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('role');p.add_argument('seed',type=int);a=p.parse_args();run(a.model,a.role,a.seed)
