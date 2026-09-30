"""Run the same predeclared multi-position measurement for all eight paper conditions."""
from package_paths import read_config_text
import json
from pathlib import Path
import time
import torch
import additional_rank_models as a
import pilot_multistep_updates as p

ROOT = Path('results/multistep_update_all')
def conditions():
    snapshot=json.loads(read_config_text(Path('data/measurements/baseline.json')))
    records=[]
    for row in snapshot:
        ds=row['dataset']; label=row['model']; rank=row['rank']
        if label=='ResNet-18':
            base=Path('results/cifar10_confirmation' if ds=='cifar10' else 'results/confirmatory')
            inventory=json.loads(read_config_text(base/'scaffold_structure/inventory.json'))
            paths=[Path(e['path']) for e in inventory if e['role']=='target']
            key='resnet18'
        elif label=='DeiT-Tiny':
            base=Path('results/final_rank_rule/deit/cifar10' if ds=='cifar10' else 'results/backbone_initialization_validation/pretrained_deit_tiny')
            paths=sorted(base.glob('target_order*/artifacts/*.pt')); key='deit_tiny'
        elif label=='MobileNetV3-S':
            paths=a.Experiment('mobilenetv3_small_100.lamb_in1k',ds).paths('target'); key='mobilenetv3_small'
        elif label=='MLP-Mixer-B/16':
            base=Path('results/mlp_rnn_extension/mixer_b16_224.goog_in21k_ft_in1k')/ds
            paths=sorted(base.glob('target_order*/artifacts/*.pt')); key='mixer_b16'
        else:
            raise ValueError(label)
        assert len(paths)==12,(label,ds,len(paths))
        scaffold=Path(row['folder'])/'predictors.pt'
        assert scaffold.exists(),scaffold
        for path in paths:
            assert path.exists(),path
        output=p.OUT if (key,ds)==('resnet18','cifar10') else ROOT/f'{key}_{ds}'
        records.append(dict(model=label,dataset=ds,rank=rank,paths=[str(x) for x in paths],scaffold=str(scaffold),output=str(output)))
    return records

if __name__=='__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    records=conditions()
    p.write(ROOT/'protocol.json',{'conditions':records,'positions':'early 2–4; middle round(N*65/145)..+2; late round(N*125/145)..+2; N=planned attempted updates','random_controls':128,'status':'running'})
    started=time.perf_counter()
    for c in records:
        for path in map(Path,c['paths']):
            stem=path.stem; order=int(stem.split('_')[0].replace('order','')); seed=int(stem.split('_')[1].replace('seed',''))
            print('START',c['model'],c['dataset'],order,seed,flush=True)
            p.run(order,seed,dataset=c['dataset'],source_path=path,scaffold_path=Path(c['scaffold']),output_root=Path(c['output']))
            torch.cuda.empty_cache()
        print('CONDITION COMPLETE',c['model'],c['dataset'],flush=True)
    p.write(ROOT/'completion.json',{'status':'complete','conditions':8,'histories':96,'updates':864,'invocation_seconds':time.perf_counter()-started})
