"""One fresh process per unfinished history; no PyTorch in the coordinator."""
from package_paths import read_config_text
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('results/multistep_update_all')
protocol=json.loads(read_config_text(root/'protocol.json'))
started=time.monotonic()
for index,c in enumerate(protocol['conditions']):
    for source in c['paths']:
        path=Path(source)
        out=Path(c['output'])/path.stem
        if (out/'summary.json').exists():continue
        print('START ISOLATED',c['model'],c['dataset'],path.stem,flush=True)
        subprocess.run([sys.executable,'scripts/run_multistep_case.py','--condition',str(index),'--source',source],check=True)
    print('COMPLETE',c['model'],c['dataset'],flush=True)
protocol['status']='complete'
(root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
(root/'completion.json').write_text(json.dumps({'status':'complete','conditions':8,'histories':96,'updates':864,'isolated_invocation_seconds':time.monotonic()-started},indent=2),encoding='utf-8')
