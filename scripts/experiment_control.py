"""Cooperative, persistent controls for this revision's experiment queue only."""
from package_paths import read_config_text
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_20260913'
CONTROL = OUT / 'control.json'
STATUS = OUT / 'status.json'
_last_status_write = 0.

def duty_phase(now, started, run_seconds=3600., rest_seconds=600.):
    elapsed = max(0., now-started)
    phase = elapsed % (run_seconds+rest_seconds)
    resting = phase >= run_seconds
    remaining = (run_seconds+rest_seconds if resting else run_seconds)-phase
    return resting, remaining

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    for attempt in range(30):
        try:tmp.replace(path);break
        except PermissionError:
            if attempt==29:raise
            time.sleep(.02)

def read(path):
    return json.loads(read_config_text(path, encoding='utf-8')) if path.exists() else {}

def poll(**progress):
    global _last_status_write
    while True:
        command = read(CONTROL).get('command', 'run')
        now=time.time()
        duty_start = os.environ.get('ORC_MIXER_DUTY_START')
        resting, remaining = duty_phase(now, float(duty_start), float(os.environ.get('ORC_MIXER_RUN_SECONDS', '3600')), float(os.environ.get('ORC_MIXER_REST_SECONDS', '600'))) if duty_start else (False, 0.)
        effective = 'scheduled_pause' if command == 'run' and resting else command
        if now-_last_status_write>.5 or command!='run':
            write(STATUS, dict(pid=os.getpid(), time=now, status=effective,
                              duty_seconds_to_transition=remaining if duty_start else None, **progress))
            _last_status_write=now
        if command == 'stop':
            raise SystemExit(75)
        if command != 'pause' and not resting:
            return
        time.sleep(.5)

def install_loader_hook():
    import drift_experiment as d
    original = d.loader
    class ControlledLoader:
        def __init__(self, source): self.source=source
        def __len__(self): return len(self.source)
        def __iter__(self):
            for batch in self.source:
                poll(stage=os.environ.get('REVISION_STAGE', 'evaluation_or_training'))
                yield batch
        def __getattr__(self, name): return getattr(self.source,name)
    def controlled(*args, **kwargs):
        return ControlledLoader(original(*args, **kwargs))
    d.loader = controlled

def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['start','pause','resume','stop','status'])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.command == 'start':
        lock = OUT / 'queue.lock'
        if lock.exists():
            print('Queue lock exists; use status/resume. After an abnormal exit verify that its PID is gone before removing this task-specific lock.'); return
        write(CONTROL, dict(command='run', time=time.time()))
        log = (OUT/'queue.log').open('a', encoding='utf-8')
        proc = subprocess.Popen([sys.executable, str(ROOT/'scripts/revision_jobs.py')], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write(OUT/'supervisor.json', dict(pid=proc.pid, started=time.time()))
        print(json.dumps(dict(started=True,pid=proc.pid)))
    elif a.command in ('pause','resume','stop'):
        write(CONTROL, dict(command='run' if a.command=='resume' else a.command, time=time.time()))
        print('Command saved: '+a.command)
    else:
        print(json.dumps(dict(control=read(CONTROL),worker=read(STATUS),queue=read(OUT/'queue_status.json')), indent=2, ensure_ascii=False))

if __name__ == '__main__': main()
