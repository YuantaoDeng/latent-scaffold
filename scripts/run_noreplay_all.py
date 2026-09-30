"""Finish the remaining three CIFAR-10 models, using fresh processes."""
import json,subprocess,sys,time
from pathlib import Path
from noreplay_all_common import ROOT,prepare
def main():
    ROOT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    def status(**kw):(ROOT/'progress.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,**kw),indent=2),encoding='utf-8')
    for model in ('deit_tiny','mobilenetv3_small','mixer_b16'):
        prepare(model)
        for role,orders,seeds in [('source',[2003,2011,2017],[160,161]),('target',[2101,2111,2129,2131,2137,2141],[170,171])]:
            for order in orders:
                for seed in seeds:
                    status(model=model,stage='training',role=role,order=order,seed=seed)
                    subprocess.run([sys.executable,'scripts/noreplay_all_worker.py','--model',model,'--role',role,'--order',str(order),'--seed',str(seed)],check=True)
        status(model=model,stage='analysis')
        subprocess.run([sys.executable,'scripts/analyze_noreplay_all.py','--model',model],check=True)
        status(model=model,stage='report')
        subprocess.run([sys.executable,'scripts/report_noreplay_all.py','--model',model],check=True)
    status(stage='complete')
    print('THREE_MODELS_COMPLETE',flush=True)
if __name__=='__main__':main()
