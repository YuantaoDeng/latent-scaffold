"""Complete the three remaining CIFAR-100 three-task conditions, one GPU worker at a time."""
from package_paths import read_config_text
import argparse,json,subprocess,sys,time
from pathlib import Path
from resnet_cifar100_content import TASKS
ROOT=Path('results/early_content_three_task')
MODELS={'deit_tiny':('DeiT-Tiny',20),'mobilenetv3_small':('MobileNetV3-S',8),'mixer_b16':('MLP-Mixer-B/16',8)}

def write(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def main():
    start=time.perf_counter();ROOT.mkdir(parents=True,exist_ok=True)
    master={'models':list(MODELS),'dataset':'cifar100','tasks_per_history':3,'status':'running','resnet_reused':True}
    write(ROOT/'remaining_progress.json',master)
    for short,(name,rank) in MODELS.items():
        out=ROOT/f'{short}_cifar100';out.mkdir(exist_ok=True)
        protocol=json.loads(read_config_text(ROOT/'resnet18_cifar100/protocol.json'))
        for k in ['invocation_seconds','total_seconds']:protocol.pop(k,None)
        protocol.update(model=name,rank=rank,status='running',amp_initial_scale=None if short=='mobilenetv3_small' else 128,
          training_precision='FP32' if short=='mobilenetv3_small' else 'AMP',activation_recomputation=short=='mixer_b16',
          rank_policy='Existing manuscript CIFAR-100 rank, equal across X/Y, never fitted on new test results.')
        write(out/'protocol.json',protocol)
        for role,seeds in [('fit',range(220,226)),('test',range(230,236))]:
            for seed in seeds:
                if (out/f'{role}_seed{seed}'/'complete.json').exists():continue
                current={'model':short,'role':role,'seed':seed,'started':time.time()}
                master.update(current=current);write(ROOT/'remaining_progress.json',master);write(out/'current.json',current)
                with (out/'training.log').open('a',encoding='utf-8') as log:
                    result=subprocess.run([sys.executable,__file__,'--worker',short,role,str(seed)],stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    protocol.update(status='failed',failed=current);write(out/'protocol.json',protocol)
                    master.update(status='failed',failed=current);write(ROOT/'remaining_progress.json',master)
                    raise RuntimeError(current)
                print('COMPLETE',short,role,seed,flush=True)
        protocol.update(status='training_complete');write(out/'protocol.json',protocol)
        with (out/'analysis.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'scripts/analyze_resnet_cifar100_content.py','--model',short],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            master.update(status='analysis_failed',model=short);write(ROOT/'remaining_progress.json',master);raise RuntimeError(master)
        protocol.update(status='complete');write(out/'protocol.json',protocol)
    subprocess.run([sys.executable,'scripts/summarize_cifar100_three_task.py'],check=True)
    master.update(status='complete',invocation_seconds=time.perf_counter()-start);write(ROOT/'remaining_progress.json',master)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',nargs=3);a=p.parse_args()
    if a.worker:
        from all_model_content_worker import run
        short,role,seed=a.worker
        run(short,role,int(seed),dataset='cifar100',tasks=TASKS,final_task=3,output_root=ROOT/f'{short}_cifar100')
    else:main()
