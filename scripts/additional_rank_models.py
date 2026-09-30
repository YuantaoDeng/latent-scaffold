"""Prospective source-rank selection and complete cross-model geometry experiments."""
from package_paths import read_config_text
from pathlib import Path
from types import SimpleNamespace as NS
import argparse, hashlib, json, time
import numpy as np
import torch
import analyze_scaffold_followup as f
import sequence_drift_experiment as seq
import resnet_temporal_dimension as r
from analyze_simple_sources_function import advance_augmentation_rng
from deit_three_claims import json_save,csv_save

ROOT=Path('results/additional_rank_rule')
MODELS=['mobilenetv3_small_100.lamb_in1k','regnetx_002.pycls_in1k','efficientnet_b0.ra_in1k']
DATA={'cifar100':Path('results/confirmatory'),'cifar10':Path('results/cifar10_confirmation')}
BASE_TRAIN=seq.train_task

def classifier_tensors(state):
    names=[n for n in ('fc','head','classifier','head.fc') if n+'.weight' in state and n+'.bias' in state]
    assert len(names)==1,names
    return state[names[0]+'.weight'].double(),state[names[0]+'.bias'].double()

def reference(ds,role):
    entries=[e for e in json.loads(read_config_text(DATA[ds]/'scaffold_structure/inventory.json')) if e['role']==role]
    if ds=='cifar100':
        base=Path('results/backbone_initialization_validation/pretrained_deit_tiny')
        entries=[{**e,'path':str(base/f"{role}_order{e['order_seed']}"/'artifacts'/Path(e['path']).name),
                  'manifest_path':str(base/f"{role}_order{e['order_seed']}"/'manifest.json')} for e in entries]
    return entries

class Experiment:
    def __init__(self,name,ds):
        self.name,self.ds=name,ds; self.out=ROOT/name/ds
        self.legacy=Path('results/cross_model_extension')/name if ds=='cifar100' else None
    def path(self,e):
        original=Path(e['path']); new=self.out/original.parent.parent.name/'artifacts'/original.name
        if new.exists(): return new
        old=self.legacy/original.parent.parent.name/'artifacts'/original.name if self.legacy else None
        return old if old and old.exists() else new
    def paths(self,role): return [self.path(e) for e in reference(self.ds,role)]
    def configure(self):
        r.ROOT=ROOT/self.name; r.entries=lambda ds,role='source':[{'path':str(p)} for p in self.paths(role)]
        f.classifier_tensors=classifier_tensors
    def tracked_train(self,*args,**kwargs):
        config,seed=args[8],args[10]; cp=seed%100+1; net=args[0]
        folder=self.out/'captures'/f'order{config.order_seed}_seed{seed//100}'
        role='source' if 'source_order' in str(config.output) else 'target'
        if role=='source' and cp==1: f.save(f.state_dict_cpu(net),folder/'checkpoint0.pt')
        observer=f.CaptureUpdate() if role=='target' and cp==config.tasks else None
        if observer: kwargs['step_observer']=observer
        result=BASE_TRAIN(*args,**kwargs)
        if role=='source' or cp==config.tasks: f.save(f.state_dict_cpu(net),folder/f'checkpoint{cp}.pt')
        if observer:
            assert observer.record['optimizer_step_after']==2
            f.save(observer.record,folder/'actual_update.pt')
        return result
    def train_missing(self,role):
        seq.train_task=self.tracked_train
        for e in reference(self.ds,role):
            if self.path(e).exists(): continue
            baseline=json.loads(read_config_text(Path(e['manifest_path']))); original=Path(e['path'])
            cfg={**baseline,'model':self.name,'pretrained':True,'initial_checkpoint':None,
                 'seeds':[e['seed']],'rank':8,'save_trajectory':True,'save_layers':False,
                 'save_pre_future_checkpoint':True,'force':False,'data':Path('data'),
                 'output':self.out/original.parent.parent.name,'validation_suite':False}
            seq.run(NS(**cfg)); torch.cuda.empty_cache()
        seq.train_task=BASE_TRAIN
    def select(self):
        self.configure(); records=r.source_data(self.ds); orders=sorted({a['order'] for a in records}); rows=[]
        for rank in range(8,33):
            for held in orders:
                fit=[a for a in records if a['order']!=held]
                h=f.pool([a['history'][:,:rank] for a in fit],rank); p=f.pool([a['static'][:,:rank] for a in fit],rank)
                for a in records:
                    if a['order']!=held: continue
                    hc=f.coverage(a['future'],h); pc=f.coverage(a['future'],p)
                    rows.append(dict(order_seed=held,seed=a['seed'],k=rank,r=rank,scaffold=hc,matched_static=pc,score=hc-pc))
        curves=[]
        for rank in range(8,33):
            scores={str(o):float(np.mean([x['score'] for x in rows if x['r']==rank and x['order_seed']==o])) for o in orders}
            curves.append(dict(k=rank,r=rank,mean_score=float(np.mean(list(scores.values()))),order_scores=scores))
        positive=[x['r'] for x in curves if x['mean_score']>0]; chosen=min(positive) if positive else None
        result=dict(selected=chosen,N=positive,curves=curves,source_only=True,one_se_used=False,
                    rule='minimum positive matched-projector PCA gain; joint k=r; 8..32')
        csv_save(self.out/'selection_folds.csv',rows); json_save(self.out/'selection.json',result)
        json_save(self.out/'source_selection_hashes.json',[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.paths('source')])
        print('DIMENSION SELECTED',self.name,self.ds,chosen,'N',positive,flush=True)
        return chosen
    def source_states(self):
        missing=[]
        for path in self.paths('source'):
            dst=self.out/'sources/artifacts'/path.name
            if dst.exists() and f.load(dst)['status']=='complete': continue
            a=f.load(path); cfg=json.loads(read_config_text(path.parent.parent/'manifest.json'))
            cap=self.out/'captures'/path.stem
            if (cap/'checkpoint0.pt').exists():
                rec=dict(status='complete',dataset=self.ds,manifest=cfg,original_path=str(path),
                         order_seed=a['order_seed'],seed=a['seed'],fit_ids=a['fit_ids'],heldout_ids=a['eval_ids'],
                         heldout_labels=a['eval_labels'],fit_reps=a['fit_representations'],eval_reps=a['eval_representations'],
                         states={cp:f.load(cap/f'checkpoint{cp}.pt') for cp in range(cfg['tasks']+1)},
                         history_max_abs_error=0.,future_max_abs_error=0.,old_accuracy=a['task1_accuracies'])
                f.save(rec,dst)
            else: missing.append(path.parent.parent)
        if missing:
            f.recover(NS(dataset=self.ds,role='source',roots=sorted(set(missing)),output=self.out/'early_sources'))
            self.configure()
            r.recover(self.ds,prior_sources=self.out/'early_sources/artifacts')
    def responses(self):
        evaluation=f.IndexedCifar(Path('data'),True,False,self.ds)
        for path in self.paths('source'):
            dst=self.out/'responses/artifacts'/path.name; rec=f.load(self.out/'sources/artifacts'/path.name)
            result=f.load(dst) if dst.exists() else dict(directions=16,relative_step=1e-4,order_seed=rec['order_seed'],seed=rec['seed'],states={})
            if len(result['states'])==rec['manifest']['tasks']+1: continue
            net=f.model(self.name,False,num_classes=evaluation.num_classes).cuda()
            batches=f.image_batches(net,evaluation,rec['fit_ids'],'cuda')
            head={id(p) for p in net.get_classifier().parameters()}; params=dict(net.named_parameters())
            names=[n for n,p in params.items() if id(p) not in head]; dimension=net.get_classifier().weight.shape[1]
            for cp,state in rec['states'].items():
                if cp in result['states']: continue
                norm=torch.sqrt(sum(state[n].double().square().sum() for n in names))
                gen=torch.Generator().manual_seed(rec['order_seed']*100003+rec['seed']+701)
                cov=torch.zeros(dimension,dimension,dtype=torch.float64); prefixes={}; started=time.monotonic()
                for draw in range(16):
                    direction={n:torch.randn(state[n].shape,generator=gen) for n in names}
                    length=torch.sqrt(sum(v.double().square().sum() for v in direction.values()))
                    scale=float(1e-4*norm/length); values=[]
                    for sign in (1,-1):
                        net.load_state_dict(state)
                        with torch.no_grad():
                            for n in names: params[n].add_(direction[n].cuda(),alpha=sign*scale)
                        values.append(f.cached_representations(net,batches))
                    field=f.center((values[0].double()-values[1].double())/2); cov+=field.T@field
                    if draw+1 in (8,16):
                        eig,v=torch.linalg.eigh(cov/(draw+1))
                        prefixes[draw+1]=dict(covariance=cov.clone()/(draw+1),basis=v[:,-8:],spectrum=eig.flip(0))
                result['states'][cp]=dict(covariance=prefixes[16]['covariance'],prefixes=prefixes)
                f.save(result,dst); print('RESPONSE16',self.name,self.ds,path.stem,cp,round(time.monotonic()-started,1),flush=True)
            del net,rec,batches; torch.cuda.empty_cache()
    def target_records(self):
        train=f.IndexedCifar(Path('data'),True,True,self.ds); evaluation=f.IndexedCifar(Path('data'),True,False,self.ds)
        device=torch.device('cuda')
        for path in self.paths('target'):
            dst=self.out/'targets/artifacts'/path.name
            if not dst.exists():
                a=f.load(path); cfg=json.loads(read_config_text(path.parent.parent/'manifest.json')); config=NS(**cfg)
                cap=self.out/'captures'/path.stem; end=cfg['tasks']; errors={}
                if (cap/'actual_update.pt').exists():
                    update=f.load(cap/'actual_update.pt'); final=f.load(cap/f'checkpoint{end}.pt')
                else:
                    seed=a['seed']; f.seed_all(seed); net=f.model(self.name,True,num_classes=evaluation.num_classes).cuda()
                    for m in net.modules():
                        if isinstance(m,torch.nn.Dropout): assert m.p==0
                        assert getattr(m,'drop_prob',0)==0
                    order=cfg['order']; train_ids={c:f.ids_by_class(train.targets,[c],config.train_per_class) for c in sum(order,[])}
                    priority=torch.rand(len(train),generator=torch.Generator().manual_seed(seed+9000))
                    seen,memory=advance_augmentation_rng(train,train_ids,order,priority,config,seed)
                    net.load_state_dict(a['pre_future_state_dict'])
                    _,pre,_=f.infer(net,evaluation,a['eval_ids'],config.batch_size,device)
                    observer=f.CaptureUpdate(); current=torch.cat([train_ids[c] for c in order[-1]])
                    f.train_task(net,config.continual_method,train,current,memory,{},seen+order[-1],seen,config,device,seed*100+end-1,step_observer=observer)
                    _,post,_=f.infer(net,evaluation,a['eval_ids'],config.batch_size,device)
                    errors=dict(pre_error=float((pre-a['eval_representations'][-2]).abs().max()),
                                final_error=float((post-a['eval_representations'][-1]).abs().max()))
                    assert max(errors.values())<=1e-5,(path,errors)
                    update=observer.record; final=f.state_dict_cpu(net); del net; torch.cuda.empty_cache()
                rec=dict(status='complete',dataset=self.ds,manifest=cfg,original_path=str(path),
                         order_seed=a['order_seed'],seed=a['seed'],fit_ids=a['fit_ids'],heldout_ids=a['eval_ids'],
                         heldout_labels=a['eval_labels'],eval_reps=a['eval_representations'],fit_reps=a['fit_representations'],
                         states={end-1:a['pre_future_state_dict'],end:final},actual_update=update,replay_errors=errors)
                f.save(rec,dst)
            cache=self.out/'update_fields'/path.name
            if not cache.exists():
                fields,norm=f.actual_update_fields(f.load(dst),evaluation,device)
                f.save(dict(fields=fields,relative_step=norm),cache)
            print('TARGET READY',self.name,self.ds,path.stem,flush=True)
    def supplementary(self,label):
        folder=self.out/label; bank=f.load(folder/'predictors.pt'); u=bank['scaffold']; rows=[]; coverage=[]
        for n,v in bank.items():
            for m,w in bank.items(): rows.append(dict(left=n,right=m,overlap=f.overlap(v,w),response_draws=16))
        csv_save(folder/'all_pooled_overlaps.csv',rows)
        for path in self.paths('source'):
            rec=f.load(self.out/'responses/artifacts'/path.name)
            for cp,state in rec['states'].items():
                cov=state['covariance']; row=dict(order_seed=rec['order_seed'],seed=rec['seed'],checkpoint=cp,response_draws=16)
                for name in ('scaffold','matched_static'):
                    v=bank[name]; row[name]=float(torch.trace(v.T@cov@v)/torch.trace(cov))
                coverage.append(row)
        csv_save(folder/'response_energy_coverage.csv',coverage)
        # Independently rebuild the scaffold from raw drift SVDs and averaged projectors.
        source_bases=[]
        for p in self.paths('source'):
            h=f.load(p)['historical_drift_centered'].double(); source_bases.append(torch.linalg.svd(h,full_matrices=False).Vh[:u.shape[1]].T)
        eig,v=torch.linalg.eigh(sum(b@b.T for b in source_bases)); other=v[:,-u.shape[1]:]
        error=float((other@other.T-u@u.T).abs().max()); assert error<1e-7
        for item in json.loads(read_config_text(self.out/'source_selection_hashes.json')):
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
        json_save(folder/'verification.json',dict(independent_scaffold_projector_error=error,source_hashes_unchanged=True,
                   target_count=len(self.paths('target')),response_draws=16))
    def run(self):
        self.configure(); self.train_missing('source')
        selection=self.out/'selection.json'
        rank=json.loads(read_config_text(selection))['selected'] if selection.exists() else self.select()
        if rank is None:
            json_save(self.out/'status.json',dict(status='no_eligible_dimension',reason='No positive mean source PCA gain in 8..32'))
            return False
        if not (self.out/'source_selection_hashes.json').exists():
            json_save(self.out/'source_selection_hashes.json',[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.paths('source')])
        label=f'rank{rank}'; self.source_states(); self.responses(); self.configure()
        if not (self.out/label/'geometry.json').exists(): r.geometry(self.ds,rank,rank,label,response_count=16)
        self.train_missing('target'); self.target_records(); self.configure()
        if not (self.out/label/'evaluation.json').exists():
            r.evaluate(self.ds,label,target_root=self.out/'targets/artifacts',fields_root=self.out/'update_fields')
        self.supplementary(label)
        e=json.loads(read_config_text(self.out/label/'evaluation.json')); g=json.loads(read_config_text(self.out/label/'geometry.json'))
        gates=dict(temporal=all(v['gain_random']['ci95'][0]>0 for v in e['temporal'].values()),
                   response_alignment=g['change_from_initial_ci95'][2][0]>0 and g['change_from_initial_ci95'][-1][0]>0,
                   response_transfer=e['transfer']['response_2_minus_initial']['ci95'][0]>0,
                   actual_update=e['updates']['adamw_full_gain_random']['ci95'][0]>0,
                   logit_readout=e['functional']['gain_random_centered_conditional_class_centered_logits_reconstruction']['ci95'][0]>0,
                   loss_readout=e['functional']['gain_random_centered_conditional_loss_reconstruction']['ci95'][0]>0)
        required=[k for k in gates if k!='loss_readout' or self.ds=='cifar100']
        compatible=all(gates[k] for k in required)
        json_save(self.out/'status.json',dict(status='complete',rank=rank,gates=gates,required_for_resnet_pattern=required,compatible=compatible))
        print('CONDITION COMPLETE',self.name,self.ds,'rank',rank,gates,'compatible',compatible,flush=True)
        return compatible

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=MODELS); p.add_argument('--dataset',choices=list(DATA)); a=p.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol=dict(models=MODELS,datasets=['cifar100','cifar10'],rule='joint k=r; smallest mean-positive source-held-out matched-PCA gain in 8..32',
                  response_draws=16,one_se_used=False,stop='first model compatible on both datasets; retain every attempted result',
                  compatible='all temporal random-gain CI lower bounds >0; cp2 and final overlap minus initial lower bounds >0; response transfer and actual update random-gain lower bounds >0; logit readout random-gain lower bound >0; additionally CIFAR100 loss readout lower bound >0',
                  inference='exploratory sequential model search; nominal 95% intervals do not adjust for model selection')
    dest=ROOT/'protocol.json'
    if dest.exists(): assert json.loads(read_config_text(dest))==protocol
    else: json_save(dest,protocol)
    for name in ([a.model] if a.model else MODELS):
        outcomes=[]
        for ds in ([a.dataset] if a.dataset else ['cifar100','cifar10']): outcomes.append(Experiment(name,ds).run())
        if not a.dataset and all(outcomes):
            json_save(ROOT/'matching_model.json',dict(model=name,exploratory=True)); print('MATCHING MODEL FOUND',name,flush=True); break

if __name__=='__main__': main()
