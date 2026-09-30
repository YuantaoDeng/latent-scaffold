"""Background 90%-of-own-energy calibration, 45-minute work / 15-minute rest."""
from __future__ import annotations
from package_paths import read_config_text
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/relative90_all_models'
RUN_SECONDS=2700
REST_SECONDS=900
TARGET=.90
TOLERANCE=.01
MAX_TRIALS=24
MAX_COEFFICIENT=1000.
NAMES=('scaffold','active_random1','active_random2')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)


def read(path,default=None):
    return json.loads(read_config_text(path, encoding='utf-8')) if path.exists() else default


def curve(path):
    with path.open(encoding='utf-8') as f:
        return [{k:(v if k in ('condition','input_hash') else float(v)) for k,v in row.items()}
                for row in csv.DictReader(f)]


def duty_phase(elapsed):
    phase=elapsed%(RUN_SECONDS+REST_SECONDS)
    return phase>=RUN_SECONDS,(RUN_SECONDS+REST_SECONDS if phase>=RUN_SECONDS else RUN_SECONDS)-phase


class Duty:
    def __init__(self,torch):
        self.torch=torch
        schedule=read(OUT/'schedule.json')
        if schedule is None:
            schedule={'started_at':time.time(),'run_seconds':RUN_SECONDS,'rest_seconds':REST_SECONDS}
            write(OUT/'schedule.json',schedule)
        self.started=time.monotonic()-max(0.,time.time()-schedule['started_at'])
        self.last=0.
        self.progress={}
        self.was_resting=False

    def poll(self,**progress):
        self.progress.update(progress)
        while True:
            now=time.monotonic()
            command=read(OUT/'control.json',{'command':'run'})['command']
            resting,remaining=duty_phase(now-self.started)
            status='scheduled_rest' if resting and command=='run' else command
            if command=='stop':raise SystemExit(75)
            if resting and not self.was_resting:
                if self.torch.cuda.is_initialized():self.torch.cuda.synchronize()
            self.was_resting=resting
            if now-self.last>1 or command!='run':
                write(OUT/'status.json',dict(pid=os.getpid(),updated_at=time.time(),status=status,
                    seconds_to_schedule_transition=remaining,run_minutes=45,rest_minutes=15,**self.progress))
                self.last=now
            if command=='run' and not resting:return
            time.sleep(.5)


def make_controls(w,case,folder,torch):
    dest=folder/'active_bases.pt'
    if dest.exists():return w.load(dest),read(folder/'basis_construction.json')
    initial=w.load(folder/'initial.pt')
    baseline=w.load(folder/'baseline_0.pt')
    c=w.conditions()[case['condition']]
    u=w.load(c['scaffold'])['scaffold'].float()
    delta=(baseline['memory']['features']-initial['memory_initial']['features']).double()
    delta-=delta.mean(0)
    covariance=delta.T@delta/len(delta)
    eigenvalues,vectors=torch.linalg.eigh(covariance)
    total=float(eigenvalues.sum())
    scales=(eigenvalues/eigenvalues.max()).clamp_min(1e-12)
    def energy(b):return float((delta@b.double()).square().sum(1).mean())
    target_energy=energy(u)
    result={}
    diagnostics={'baseline_total_energy':total,'scaffold_energy':target_energy,
        'rank':u.shape[1],'source':'unconstrained final-task OLD TRAINING memory drift only',
        'retrospective_control':True,'controls':{}}
    for j in (1,2):
        g=torch.randn(u.shape[0],u.shape[1],generator=torch.Generator().manual_seed(91000+j))
        g=torch.linalg.qr(g).Q.double()
        coords=vectors.T@g
        def at(alpha):return torch.linalg.qr(vectors@(scales.pow(alpha)[:,None]*coords)).Q
        # Permit negative weighting if a particular scaffold has below-random coverage.
        lo,hi=-1.,1.
        while energy(at(lo))>target_energy and lo>-8:lo*=2
        while energy(at(hi))<target_energy and hi<8:hi*=2
        assert energy(at(lo))<=target_energy<=energy(at(hi)), 'Activity matching has no bracket'
        for _ in range(48):
            mid=(lo+hi)/2
            if energy(at(mid))<target_energy:lo=mid
            else:hi=mid
        alpha=(lo+hi)/2;b=at(alpha).float();name=f'active_random{j}'
        assert float((b.T@b-torch.eye(b.shape[1])).abs().max())<2e-5
        assert abs(energy(b)/target_energy-1)<1e-5
        result[name]=b
        diagnostics['controls'][name]=dict(alpha=alpha,baseline_energy=energy(b),
            scaffold_overlap=float((u.T@b).square().sum()/u.shape[1]))
    w.e.save(result,dest)
    write(folder/'basis_construction.json',diagnostics)
    return result,diagnostics


def points(folder,name,reference):
    result=[dict(coefficient=0.,dose=0.,path=str(folder/'baseline_0.csv'))]
    for path in sorted(folder.glob(f'{name}_*.csv')):
        if not path.with_suffix('.pt').exists():continue
        rows=curve(path)
        assert len(rows)==6
        result.append(dict(coefficient=rows[0]['strength'],
            dose=1-rows[-1][f'memory_{name}_energy']/reference,path=str(path)))
    return result


def next_coefficient(values):
    ordered=sorted(values,key=lambda p:p['coefficient'])
    brackets=[(a,b) for a,b in zip(ordered[:-1],ordered[1:]) if (a['dose']-TARGET)*(b['dose']-TARGET)<0]
    if brackets:
        a,b=min(brackets,key=lambda x:x[1]['coefficient']-x[0]['coefficient'])
        fraction=min(.85,max(.15,(TARGET-a['dose'])/(b['dose']-a['dose'])))
        # Obtain a positive lower bracket promptly; repeated 0.85 reductions
        # from a saturated upper point can exhaust the trial budget.
        value=math.exp(math.log(a['coefficient'])+fraction*math.log(b['coefficient']/a['coefficient'])) if a['coefficient'] else b['coefficient']/10
    else:
        maximum=max(p['coefficient'] for p in values)
        value=1. if maximum==0 else maximum*10
        if value>MAX_COEFFICIENT:return None
    return float(f'{value:.10g}')


def verify_selected(w,folder,item,torch):
    saved=w.load(Path(item['path']).with_suffix('.pt'))
    rows=curve(Path(item['path']));baseline=curve(folder/'baseline_0.csv')
    initial=w.load(folder/'initial.pt')
    assert len(rows)==6
    expected_steps=sum(r['steps'] for r in baseline)
    assert sum(r['steps'] for r in rows)==expected_steps
    assert all(r['input_hash']==b['input_hash'] for r,b in zip(rows[1:],baseline[1:]))
    obs=saved['observations'];labels=obs['labels'];seen=torch.tensor(sum(initial['order'],[]))
    predicted=seen[obs['logits'][:,seen].argmax(1)]
    maximum=0.
    for label,classes in [('new',initial['order'][-1]),('old',sum(initial['order'][:-1],[])),('first',initial['order'][0])]:
        mask=torch.isin(labels,torch.tensor(classes))
        acc=float((predicted[mask]==labels[mask]).float().mean())
        maximum=max(maximum,abs(acc-rows[-1][label+'_accuracy']))
    delta=(saved['memory']['features']-initial['memory_initial']['features']).double()
    delta-=delta.mean(0)
    total=float(delta.square().sum(1).mean())
    maximum=max(maximum,abs(total-rows[-1]['memory_total_energy']))
    if item['condition']=='scaffold':
        worker_protocol=read(folder/f"protocol_scaffold_{item['coefficient']:.12g}.json")
        # Match the exact FP32 basis used in training and control construction.
        basis=w.load(worker_protocol['scaffold_source'])['scaffold'].float()
    else:
        basis=w.load(folder/'active_bases.pt')[item['condition']]
    projected=float((delta@basis.double()).square().sum(1).mean())
    maximum=max(maximum,abs(projected-rows[-1][f"memory_{item['condition']}_energy"]))
    baseline_saved=w.load(folder/'baseline_0.pt')
    base_delta=(baseline_saved['memory']['features']-initial['memory_initial']['features']).double()
    base_delta-=base_delta.mean(0)
    reference=float((base_delta@basis.double()).square().sum(1).mean())
    assert abs((1-projected/reference)-item['dose'])<1e-10
    assert item['matched']==(abs(item['dose']-TARGET)<=TOLERANCE)
    assert maximum<1e-5
    return dict(**{k:v for k,v in item.items() if k!='candidates'},
        new_mean_diff_pp=100*sum(r['new_accuracy']-b['new_accuracy'] for r,b in zip(rows[1:],baseline[1:]))/5,
        old_final_diff_pp=100*(rows[-1]['old_accuracy']-baseline[-1]['old_accuracy']),
        first_final_diff_pp=100*(rows[-1]['first_accuracy']-baseline[-1]['first_accuracy']),
        total_energy_reduction_pct=100*(1-total/baseline[-1]['memory_total_energy']),
        maximum_recomputed_error=maximum,steps=expected_steps)


def summarize():
    records=[]
    fully_matched=0
    for path in sorted(OUT.glob('*_cifar*/order*/result.json')):
        results=read(path)['results']
        records.extend(results)
        fully_matched+=len(results)==3 and all(r['matched'] for r in results)
    if records:
        with (OUT/'results.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(OUT/'summary.json',dict(completed_histories=fully_matched,
        verified_histories=len(list(OUT.glob('*_cifar*/order*/result.json'))),
        selected_branches=len(records),matched_branches=sum(r['matched'] for r in records),results=records))


def run():
    os.chdir(ROOT)
    import torch
    import relative90_worker as w
    torch.set_num_threads(2)
    duty=Duty(torch)
    w.ctl.poll=duty.poll  # existing loader hooks now check OUR controls at each batch
    jobs=[(i,c) for i,c in enumerate(w.cases()) if c['short']!='mixer_b16']
    paths=[Path(__file__),ROOT/'scripts/relative90_worker.py',ROOT/'scripts/all_model_causal.py']
    protocol=dict(target=TARGET,tolerance=TOLERANCE,metric='1 - constrained memory projected energy / baseline SAME-subspace energy',
        excluded=['Mixer','PCA'],conditions=list(NAMES),cases=[c for _,c in jobs],
        histories=36,comparisons=108,run_minutes=45,rest_minutes=15,
        polling='before each training/evaluation batch, CUDA synchronized on rest entry',
        fixed_basis='original paper ranks; active random baseline energy matched to scaffold within each history',
        selection='memory-only target calibration; no accuracy-based selection',
        max_trials=MAX_TRIALS,maximum_coefficient=MAX_COEFFICIENT,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    prior=read(OUT/'protocol.json')
    if prior:assert prior==protocol, 'Protocol changed; do not silently resume'
    else:write(OUT/'protocol.json',protocol)
    priority=read(OUT/'priority.json',{}).get('condition_order',[])
    priority_rank={key:rank for rank,key in enumerate(priority)}
    jobs.sort(key=lambda job:(priority_rank.get(f"{job[1]['short']}_{job[1]['dataset']}",len(priority)),job[0]))
    write(OUT/'execution_order.json',[case for _,case in jobs])
    errors=[]
    for job_number,(index,case) in enumerate(jobs):
        folder=w.folder_for(case,'suppress')
        previous=read(folder/'result.json')
        if previous and all(r['matched'] for r in previous['results']):continue
        duty.poll(job=job_number+1,jobs=len(jobs),model=case['short'],dataset=case['dataset'],order=case['order'],seed=case['seed'],stage='baseline')
        try:
            w.run_branch(index,'baseline',0.)
            active,construction=make_controls(w,case,folder,torch)
            references={'scaffold':construction['scaffold_energy'],
                **{k:v['baseline_energy'] for k,v in construction['controls'].items()}}
            selected=[]
            for name in NAMES:
                for attempt in range(MAX_TRIALS+1):
                    values=points(folder,name,references[name]);positive=[p for p in values if p['coefficient']>0]
                    best=min(positive,key=lambda p:abs(p['dose']-TARGET)) if positive else None
                    matched=best is not None and abs(best['dose']-TARGET)<=TOLERANCE
                    coefficient=None if matched else next_coefficient(values)
                    if matched or len(positive)>=MAX_TRIALS or coefficient is None:
                        assert best is not None
                        item=dict(model=case['short'],dataset=case['dataset'],order=case['order'],seed=case['seed'],
                            condition=name,target=TARGET,**best,matched=matched,candidates=values)
                        selected.append(item);write(folder/'selection.json',selected)
                        print('SELECT '+json.dumps({k:v for k,v in item.items() if k!='candidates'}),flush=True)
                        break
                    duty.poll(stage='calibration',direction=name,trial=attempt+1,coefficient=coefficient)
                    w.run_branch(index,name,coefficient)
            results=[verify_selected(w,folder,item,torch) for item in selected]
            write(folder/'result.json',dict(complete=True,results=results,construction=construction))
            summarize()
        except Exception:
            detail=dict(case=case,error=traceback.format_exc())
            errors.append(detail);write(folder/'error.json',detail);print(detail['error'],flush=True)
        torch.cuda.empty_cache()
    summarize()
    summary=read(OUT/'summary.json',{})
    final_status='complete_with_errors' if errors else ('complete' if summary.get('completed_histories')==len(jobs) else 'complete_with_unmatched')
    write(OUT/'status.json',dict(status=final_status,
        errors=errors,updated_at=time.time(),**{k:v for k,v in read(OUT/'summary.json',{}).items() if k!='results'}))


def start():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'queue.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    except FileExistsError:
        raise SystemExit('Queue lock exists; check status before starting another worker')
    write(OUT/'control.json',dict(command='run'))
    with (OUT/'queue.log').open('a',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    write(OUT/'supervisor.json',dict(pid=proc.pid,started=time.time(),run_minutes=45,rest_minutes=15))
    print(json.dumps(read(OUT/'supervisor.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--start',action='store_true')
    args=p.parse_args()
    if args.start:start()
    elif args.worker:
        try:run()
        except BaseException:
            write(OUT/'fatal.json',dict(error=traceback.format_exc(),time=time.time()))
            raise
    else:print(json.dumps(read(OUT/'status.json',{}),indent=2))
