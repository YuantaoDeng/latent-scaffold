"""One GPU worker at a time; paused/stopped through experiment_control.py."""
from package_paths import read_config_text
import json,os,subprocess,sys,time,traceback
from pathlib import Path
import experiment_control as c

def main():
    c.OUT.mkdir(parents=True,exist_ok=True);lock=c.OUT/'queue.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise SystemExit('Another queue owns queue.lock')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    failures=[]
    try:
        inventory=json.loads(read_config_text(c.OUT/'accuracy/inventory.json'))
        jobs=[];causal_markers={}
        causal_inventory=c.OUT/'all_model_causal/jobs.json'
        if causal_inventory.exists():
            for job in json.loads(read_config_text(causal_inventory))['jobs']:
                args=['scripts/all_model_causal.py','--case',str(job['case']),'--mode',job['mode']]
                jobs.append(args);causal_markers[tuple(args)]=c.ROOT/job['folder']/'complete.json'
        if (c.ROOT/'archive/cleanup_20260918/SCAFFOLD_ENHANCEMENT_PROTOCOL.md').exists():
            jobs.extend([['scripts/scaffold_enhancement.py','--order',str(o),'--seed',str(s)] for o in (2131,2137,2141) for s in (170,171)])
        # Evaluate a small saved-state condition first to validate the pipeline.
        jobs.append(['scripts/accuracy_backfill.py','--job','0'])
        jobs.append(['scripts/retention_replication.py','--order','2131','--seed','170'])
        jobs.append(['scripts/accuracy_backfill.py','--job','0','--prefix'])
        jobs.extend([['scripts/accuracy_backfill.py','--job',str(i)] for i in range(1,len(inventory))])
        jobs.extend([['scripts/retention_replication.py','--order',str(o),'--seed',str(s)] for o in (2131,2137,2141) for s in (170,171) if (o,s)!=(2131,170)])
        jobs.extend([['scripts/accuracy_backfill.py','--job',str(i),'--prefix'] for i,j in enumerate(inventory) if j['family']=='baseline'])
        scope_file=c.OUT/'queue_scope.json'
        if scope_file.exists():
            scope=json.loads(read_config_text(scope_file))
            if scope.get('only')=='windowed_penalty':
                jobs=[['scripts/windowed_scaffold_penalty.py','--order',str(o),'--seed',str(s),'--model',m] for m in scope.get('models',['resnet18']) for o in (2131,2137,2141) for s in (170,171)]
            if scope.get('only')=='causal_suppression':
                jobs=[args for args in jobs if args[0]=='scripts/all_model_causal.py' and args[-1]=='suppress']
                if scope.get('models'):
                    model_by_case={str(j['case']):j['short'] for j in json.loads(read_config_text(causal_inventory))['jobs']}
                    allowed={key for key,model in model_by_case.items() if model in scope['models']}
                    jobs=[args for args in jobs if args[2] in allowed]
                    jobs.sort(key=lambda args:scope['models'].index(model_by_case[args[2]]))
        for i,args in enumerate(jobs):
            # Restarts skip completed jobs without reimporting torch hundreds of times.
            if args[0]=='scripts/windowed_scaffold_penalty.py':
                window_model=args[-1] if '--model' in args else 'resnet18'
                window_directory={'resnet18':'windowed_penalty','mobilenetv3_small':'windowed_penalty_mobilenet_cifar10','deit_tiny':'windowed_penalty_deit_cifar10'}[window_model]
                if (c.OUT/window_directory/f'order{args[2]}_seed{args[4]}'/'complete.json').exists():continue
            elif args[0]=='scripts/all_model_causal.py':
                if causal_markers[tuple(args)].exists():continue
            elif args[0]=='scripts/accuracy_backfill.py':
                j=inventory[int(args[2])];marker='prefix_complete.json' if '--prefix' in args else 'complete.json'
                if (c.OUT/'accuracy'/j['key']/marker).exists():continue
            elif args[0] in ('scripts/retention_replication.py','scripts/scaffold_enhancement.py'):
                directory='retention_replication' if args[0]=='scripts/retention_replication.py' else 'enhancement'
                if (c.OUT/directory/f'order{args[2]}_seed{args[4]}'/'complete.json').exists():continue
            c.poll(stage='queue_boundary',job_index=i,total_jobs=len(jobs))
            c.write(c.OUT/'queue_status.json',dict(status='running',job_index=i,total_jobs=len(jobs),command=args,failures=failures))
            print('START '+json.dumps(args),flush=True)
            logpath=c.OUT/'logs'/('job_'+str(i)+'.log');logpath.parent.mkdir(exist_ok=True)
            with logpath.open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-X','utf8','-u',*args],cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if result.returncode==75:
                c.write(c.OUT/'queue_status.json',dict(status='stopped',job_index=i,total_jobs=len(jobs),failures=failures));return
            if result.returncode:
                failures.append(dict(command=args,returncode=result.returncode,log=str(logpath)))
                if args[0] in ('scripts/all_model_causal.py','scripts/windowed_scaffold_penalty.py') or result.returncode>255 or result.returncode<0:
                    c.write(c.OUT/'queue_status.json',dict(status='failed',job_index=i,total_jobs=len(jobs),failures=failures))
                    return
                # Continue independent cases, never silently omit a failed case.
            if args[0]=='scripts/windowed_scaffold_penalty.py':
                analyzed=subprocess.run([sys.executable,'scripts/summarize_windowed_penalty.py','--model',window_model],cwd=c.ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if analyzed.returncode:
                    c.write(c.OUT/'queue_status.json',dict(status='analysis_failed',job_index=i,failures=[dict(returncode=analyzed.returncode)]));return
                continue
            if args[0]=='scripts/all_model_causal.py':
                analyzed=subprocess.run([sys.executable,'scripts/summarize_all_model_causal.py'],cwd=c.ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if analyzed.returncode:
                    c.write(c.OUT/'queue_status.json',dict(status='analysis_failed',job_index=i,total_jobs=len(jobs),failures=[dict(command=['scripts/summarize_all_model_causal.py'],returncode=analyzed.returncode)]));return
                continue
            subprocess.run([sys.executable,'scripts/accuracy_backfill.py','--collect'],cwd=c.ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            subprocess.run([sys.executable,'scripts/summarize_retention_replication.py'],cwd=c.ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if args[0]=='scripts/scaffold_enhancement.py':
                with (c.OUT/'logs/enhancement_summary.log').open('a',encoding='utf-8') as log:
                    analyzed=subprocess.run([sys.executable,'-X','utf8','scripts/summarize_scaffold_enhancement.py'],cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if analyzed.returncode:failures.append(dict(command=['scripts/summarize_scaffold_enhancement.py'],returncode=analyzed.returncode))
        c.write(c.OUT/'queue_status.json',dict(status='complete_with_failures' if failures else 'complete',total_jobs=len(jobs),failures=failures))
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
