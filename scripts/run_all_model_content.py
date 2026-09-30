"""Standard-library coordinator: one GPU paired worker per fresh process."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path('results/all_model_followups')
def main():
    start=time.perf_counter();models=['deit_tiny','mobilenetv3_small','mixer_b16']
    protocol={'models':['resnet18']+models,'dataset':'cifar10','fit_seeds':list(range(220,226)),'test_seeds':list(range(230,236)),
        'resnet_reused':'results/early_content_scaffold_pilot/resnet18_cifar10','status':'running',
        'rank_policy':'Use existing manuscript condition rank, same in X/Y; no selection on new test results.',
        'tasks':{'A':[4,6],'BX':[9,8],'BY':[7,0],'C':[2,1],'D':[3,5]},
        'training':'Same four-task controlled content protocol as ResNet pilot; fit 1->2 pools from six sources, evaluate complete task3/task4 2x2 coverage in six separate paired seeds.',
        'controls':'Replay A at task3 and A+C at task4; common all-ten-class output mask, exact common augmented input hashes, AdamW reset each task.',
        'worker':'Fresh process per paired run; persisted task states allow restart. Mixer activation recomputation retains original batch64 and five epochs.'}
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'content_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for model in models:
        for role,seeds in [('fit',range(220,226)),('test',range(230,236))]:
            for seed in seeds:
                folder=ROOT/'content'/f'{model}_cifar10'/f'{role}_seed{seed}'
                if (folder/'complete.json').exists():continue
                (ROOT/'content_progress.json').write_text(json.dumps({'model':model,'role':role,'seed':seed,'started_utc':time.time()},indent=2))
                log=ROOT/f'{model}_content.log'
                with log.open('a',encoding='utf-8') as stream:
                    stream.write(f'\nSTART {model} {role} {seed}\n');stream.flush()
                    result=subprocess.run([sys.executable,'scripts/all_model_content_worker.py',model,role,str(seed)],stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:
                    protocol.update(status='failed',failed={'model':model,'role':role,'seed':seed,'returncode':result.returncode})
                    (ROOT/'content_protocol.json').write_text(json.dumps(protocol,indent=2));raise RuntimeError(protocol['failed'])
                print('COMPLETE',model,role,seed,flush=True)
    protocol.update(status='training_complete',invocation_seconds=time.perf_counter()-start)
    (ROOT/'content_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('ALL CONTENT TRAINING COMPLETE',flush=True)
if __name__=='__main__':main()
