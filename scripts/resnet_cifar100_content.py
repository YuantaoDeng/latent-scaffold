"""Three-task CIFAR-100 early-content experiment; isolated paired workers."""
import argparse,json,subprocess,sys,time
from pathlib import Path
OUT=Path('results/early_content_three_task/resnet18_cifar100')
TASKS={'A':[77,55,7,78,87,38,94,24,92,23],
       'BX':[12,44,42,71,20,22,99,81,75,32],
       'BY':[37,28,48,69,10,74,51,8,47,64],
       'C':[34,39,62,72,65,93,30,95,90,19]}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    protocol={'model':'ResNet-18','dataset':'cifar100','tasks_per_history':3,'classes_per_task':10,'tasks':TASKS,
      'partition':'Original target order 1103: A=block1, BX=block2, C=block3, BY=block4; fixed before new results.',
      'rank':8,'fit_seeds':list(range(220,226)),'test_seeds':list(range(230,236)),
      'train_per_class':450,'epochs':5,'batch_size':64,'memory':500,'fit_anchors':500,'heldout_anchors':500,
      'optimizer':'AdamW reset per task','lr':.0003,'weight_decay':.05,'amp_initial_scale':128,
      'shared_task_updates':355,'subsequent_task_updates':705,
      'controls':'Same original pretrained initialization, shared A state within each pair; task3 replays A only. Common task3 output competition is the same union of all 40 experimental classes. Augmented input streams must match exactly.',
      'scope':'One class partition; estimation and test runs share training content. Only one common later task is measured.',
      'status':'running'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for role,seeds in [('fit',range(220,226)),('test',range(230,236))]:
        for seed in seeds:
            if (OUT/f'{role}_seed{seed}'/'complete.json').exists():continue
            (OUT/'current.json').write_text(json.dumps({'role':role,'seed':seed,'started':time.time()}))
            with (OUT/'training.log').open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,__file__,'--worker',role,str(seed)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                protocol.update(status='failed',failed={'role':role,'seed':seed,'returncode':result.returncode})
                (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));raise RuntimeError(protocol['failed'])
            print('COMPLETE',role,seed,flush=True)
    protocol.update(status='training_complete',invocation_seconds=time.perf_counter()-started)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,'scripts/analyze_resnet_cifar100_content.py'],check=True)
    protocol.update(status='complete',total_seconds=time.perf_counter()-started)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',nargs=2);a=p.parse_args()
    if a.worker:
        from all_model_content_worker import run
        run('resnet18',a.worker[0],int(a.worker[1]),dataset='cifar100',tasks=TASKS,final_task=3,output_root=OUT)
    else:main()
