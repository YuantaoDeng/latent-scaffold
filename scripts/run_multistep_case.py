from package_paths import read_config_text
import argparse
import json
from pathlib import Path
import torch
import pilot_multistep_updates as p

parser=argparse.ArgumentParser()
parser.add_argument('--condition',type=int,required=True)
parser.add_argument('--source',required=True)
args=parser.parse_args()
c=json.loads(read_config_text(Path('results/multistep_update_all/protocol.json')))['conditions'][args.condition]
source=Path(args.source)
assert str(source) in c['paths']
order=int(source.stem.split('_')[0].replace('order',''))
seed=int(source.stem.split('_')[1].replace('seed',''))
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
p.run(order,seed,dataset=c['dataset'],source_path=source,scaffold_path=Path(c['scaffold']),output_root=Path(c['output']),stop_after_last=True)
