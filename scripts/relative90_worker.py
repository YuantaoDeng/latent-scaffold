"""Matched-suppression worker adapted from all_model_causal.run.
Preserves original training, FP32 observation, AMP policy, input hashes and resume.
The queue installs its own batch-level duty-cycle poll in experiment_control.
"""
from package_paths import read_config_text
from all_model_causal import *
OUT=Path('results/relative90_all_models')

def folder_for(case, mode='suppress'):
    return OUT/f"{case['short']}_{case['dataset']}"/f"order{case['order']}_seed{case['seed']}"

def run_branch(index, selected_name, coefficient):
    mode='suppress'; preflight=False; smoke=False; strength=coefficient
    requested_branches=[('baseline',0.)] if selected_name=='baseline' else [('baseline',0.),(selected_name,coefficient)]
    case=cases()[index];c=conditions()[case['condition']];folder=folder_for(case,mode);folder.mkdir(parents=True,exist_ok=True)
    marker=f'complete_{selected_name}_{coefficient:.12g}.json'
    if (folder/marker).exists():return
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    source=Path(case['source']);cfg=json.loads(read_config_text(source.parent.parent/'manifest.json'));order=cfg['order']
    original=load(source);recovery=None
    if 'pre_future_state_dict' in original:state=original['pre_future_state_dict']
    else:
        recovery=source.parent.parent.parent/'scaffold_followup/target_recovery/artifacts'/source.name
        recovered=load(recovery);state=recovered['states'][len(order)-1]
    u=load(c['scaffold'])['scaffold'].float();rank=u.shape[1];dim=u.shape[0]
    bases={'scaffold':u}
    for j in (1,2):bases[f'random{j}']=torch.linalg.qr(torch.randn(dim,rank,generator=torch.Generator().manual_seed(91000+j))).Q
    if (folder/'active_bases.pt').exists(): bases.update(load(folder/'active_bases.pt'))
    assert rank==c['rank']
    for b in bases.values():assert float((b.T@b-torch.eye(rank)).abs().max())<2e-5
    # Actual hook parity and added-norm matching, independent of learned outcomes.
    g=torch.randn(64,dim,generator=torch.Generator().manual_seed(1717))
    assert torch.equal(e.enhance(g,u,u,1),g)
    for a in (1.25,1.5,2.):
        z=e.enhance(g,u,u,a);r=e.enhance(g,bases['random1'],u,a,True)
        assert torch.allclose(z@u,a*(g@u),atol=2e-5,rtol=2e-5)
        assert abs(float((z-g).norm()/(r-g).norm())-1)<2e-5
    ds=c['dataset'];train=d.IndexedCifar(Path('data'),True,True,ds);evaluation=d.IndexedCifar(Path('data'),True,False,ds)
    seen=sum(order,[]);old=sum(order[:-1],[])
    ids={cl:d.ids_by_class(train.targets,[cl],cfg['train_per_class']) for cl in seen}
    priority=torch.rand(len(train),generator=torch.Generator().manual_seed(case['seed']+9000))
    memory=d.balanced_memory(ids,old,cfg['memory'],priority);current=torch.cat([ids[cl] for cl in order[-1]])
    heldout=d.ids_by_class(train.targets,seen,cfg['probe_per_class'],True)
    assert not torch.isin(heldout,torch.cat(list(ids.values()))).any()
    net=d.model(c['name'],False,num_classes=10 if ds=='cifar10' else 100).cuda();net.load_state_dict(state)
    if c['short']=='mixer_b16':net.set_grad_checkpointing(True)
    initial=observe(net,evaluation,heldout)
    mi=observe(net,evaluation,memory)
    images=torch.stack([evaluation[int(i)][1] for i in memory])
    positions=torch.full((len(train),),-1,dtype=torch.long);positions[memory]=torch.arange(len(memory))
    reference=mi['features'].cuda()
    amp=c['short']!='mobilenetv3_small'
    checks=dict(orthogonality_passed=True,gradient_algebra_passed=True,excluded_eval_disjoint=True)
    # FP32 reference evaluated with different batching, must agree within float tolerance.
    net.eval()
    with torch.no_grad():check=d.features(net,d.prepare_images(images[:32],'cuda'))
    checks['reference_batch_max_error']=float((check-reference[:32]).abs().max())
    assert checks['reference_batch_max_error']<1e-4,checks
    protocol_name=f'protocol_{selected_name}_{coefficient:.12g}.json'
    ctl.write(folder/protocol_name,dict(**case,mode=mode,model=c['name'],rank=rank,task=len(order),epochs=5,
        source_sha256=sha(source),scaffold_source=c['scaffold'],scaffold_sha256=sha(c['scaffold']),
        recovered_source=str(recovery) if recovery else None,checks=checks,amp_enabled=amp,
        classifier='trainable; all seen classes compete',initialization='saved pre-final-task; fresh paired continuation, not original trajectory replay',
        branches=requested_branches,storage='epoch-resume model retained until branch completed; final per-example predictions/features and all-task curves retained; no redundant final weights',
        precision='FP32 penalty/observation; AMP classification except MobileNet; scaler128'))
    e.save(dict(initial=initial,memory_initial=mi,order=order,heldout_ids=heldout,memory_ids=memory),folder/'initial.pt')
    def metrics(obs,mem):
        result={}
        for j,classes in enumerate(order,1):
            mask=torch.isin(obs['labels'],torch.tensor(classes))
            result[f'task{j}_accuracy']=d.accuracy(obs['logits'][mask],obs['labels'][mask],seen)
            result[f'task{j}_loss']=float(d.masked_ce(obs['logits'][mask],obs['labels'][mask],seen))
        result['first_accuracy']=result['task1_accuracy'];result['new_accuracy']=result[f'task{len(order)}_accuracy']
        mask=torch.isin(obs['labels'],torch.tensor(old))
        result['old_accuracy']=d.accuracy(obs['logits'][mask],obs['labels'][mask],seen)
        for label,delta in [('memory',mem['features']-mi['features']),('first',(obs['features']-initial['features'])[torch.isin(obs['labels'],torch.tensor(order[0]))])]:
            v=delta.double();v=v-v.mean(0);total=float(v.square().sum(1).mean())
            result[label+'_total_energy']=total
            for name,b in bases.items():result[f'{label}_{name}_energy']=float((v@b.double()).square().sum(1).mean())
        return result
    for name,a in requested_branches:
        label=f'{name}_{a:.12g}';dest=folder/f'{label}.pt';resume=folder/f'{label}_resume.pt';curve=folder/f'{label}.csv'
        if dest.exists():continue
        net.load_state_dict(state);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.05)
        scaler=torch.amp.GradScaler('cuda',init_scale=128.,enabled=amp)
        rows=[dict(epoch=0,condition=name,strength=a,**metrics(initial,mi),training_loss=0.,training_penalty=0.,steps=0,input_hash='',seconds=0.)]
        if resume.exists():
            value=load(resume);net.load_state_dict(value['state']);opt.load_state_dict(value['optimizer']);scaler.load_state_dict(value['scaler']);rows=value['rows'];del value
        for epoch in range(len(rows),6):
            ctl.poll(stage='all_model_causal',model=c['short'],dataset=ds,mode=mode,condition=label,epoch=epoch,case=index)
            began=time.perf_counter();seed=700000+case['order']*1000+case['seed']*10+epoch;d.seed_all(seed)
            cur=d.loader(train,current,32,True,seed);rep=d.loader(train,memory,32,True,seed+100);replay=iter(rep)
            trace=hashlib.sha256();losses=[];penalties=[]
            for ci,cx,cy in cur:
                try:ri,rx,ry=next(replay)
                except StopIteration:replay=iter(rep);ri,rx,ry=next(replay)
                raw=torch.cat([cx,rx]);labels=torch.cat([cy,ry])
                for v in (torch.cat([ci,ri]),raw,labels):trace.update(v.numpy().tobytes())
                x=d.prepare_images(raw,'cuda');y=labels.cuda();opt.zero_grad(set_to_none=True)
                if mode=='suppress' and a:
                    net.eval();z=d.features(net,d.prepare_images(images[positions[ri]],'cuda'))
                    reg=s.penalty(z-reference[positions[ri].cuda()],bases[name].cuda())
                    assert torch.isfinite(reg);scaler.scale(a*reg).backward();penalties.append(float(reg.detach()))
                net.train()
                with torch.amp.autocast('cuda',enabled=amp):
                    if mode=='enhance':
                        z=d.features(net,x)
                        if a!=1:z.register_hook(lambda grad:e.enhance(grad,bases[name].cuda(),u.cuda(),a,name.startswith('random')))
                        logits=net.get_classifier()(z)
                    else:logits=net(x)
                    loss=d.masked_ce(logits[:len(ci)],y[:len(ci)],seen)+d.masked_ce(logits[len(ci):],y[len(ci):],seen)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward();scale=scaler.get_scale();scaler.step(opt);scaler.update()
                if scaler.get_scale()<scale:raise RuntimeError(f'AMP skipped update: {label}, epoch{epoch}, step{len(losses)+1}')
                losses.append(float(loss.detach()))
            obs=observe(net,evaluation,heldout);mem=observe(net,evaluation,memory)
            row=dict(epoch=epoch,condition=name,strength=a,**metrics(obs,mem),training_loss=sum(losses)/len(losses),training_penalty=sum(penalties)/max(1,len(penalties)),steps=len(losses),input_hash=trace.hexdigest(),seconds=time.perf_counter()-began)
            if name!='baseline':
                baseline=folder/f"baseline_{0 if mode=='suppress' else 1}.csv"
                with baseline.open() as file:expected=next(r for r in csv.DictReader(file) if int(r['epoch'])==epoch)
                assert row['input_hash']==expected['input_hash'],'paired data mismatch'
            rows.append(row);s.write_rows(curve,rows)
            e.save(dict(state=s.cpu_state(net),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rows=rows),resume)
            print(json.dumps(dict(case=index,model=c['short'],dataset=ds,mode=mode,branch=label,**row)),flush=True)
            if preflight or (smoke and name=='scaffold'):
                ctl.write(folder/('smoke.json' if smoke else 'preflight.json'),dict(passed=True,branch=label,epoch_seconds=row['seconds'],steps=row['steps'],checks=checks))
                return
        if len(rows)==6:
            obs=observe(net,evaluation,heldout);mem=observe(net,evaluation,memory)
            e.save(dict(rows=rows,observations=obs,memory=mem,order=order,seen=seen),dest)
            resume.unlink(missing_ok=True)
        del opt,scaler;gc.collect();torch.cuda.empty_cache()
    ctl.write(folder/marker,dict(complete=True,branches=len(requested_branches),strength=strength,**case,mode=mode))
