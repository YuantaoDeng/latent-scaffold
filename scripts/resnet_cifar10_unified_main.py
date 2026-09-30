"""Re-run ResNet/CIFAR10 paper intervention with the shared worker."""
import sys,json,time,traceback,hashlib
from pathlib import Path
import torch
import all_model_causal as w
import relative90_queue as q
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/resnet_cifar10_unified_main_20260921'
q.OUT=OUT
w.OUT=OUT/'all_model_causal'
torch.set_num_threads(2)
duty=q.Duty(torch);w.ctl.poll=duty.poll
jobs=[(i,c) for i,c in enumerate(w.cases()) if c['short']=='resnet18' and c['dataset']=='cifar10']
q.write(OUT/'protocol.json',dict(worker='scripts/all_model_causal.py',worker_sha256=hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest(),histories=6,branches=60,strengths=[1,10,100],controls=['random1','random2'],cases=[c for _,c in jobs]))
try:
 for j,(i,c) in enumerate(jobs):
  duty.poll(job=j+1,jobs=6,order=c['order'],seed=c['seed'])
  w.run(i,'suppress')
 q.write(OUT/'status.json',dict(status='complete',histories=6,branches=60,updated_at=time.time()))
except BaseException:
 q.write(OUT/'fatal.json',dict(error=traceback.format_exc(),time=time.time()))
 raise
