"""Frozen input inventory for the two all-model follow-up analyses."""
from package_paths import read_config_text
import json
from pathlib import Path

ROOT=Path('results/all_model_followups')
def conditions():
    old=json.loads(read_config_text(Path('results/multistep_update_all/protocol.json')))['conditions']
    result=[]
    for i,c in enumerate(old):
        ds=c['dataset']; model=c['model']; base=Path(c['scaffold']).parent.parent
        if model=='ResNet-18':
            short='resnet18'; name='resnet18'
            recovery=Path('results/cifar10_confirmation' if ds=='cifar10' else 'results/confirmatory')/'scaffold_followup/target_recovery/artifacts'
            initial=next((Path('results/resnet_temporal_dimension')/ds/'sources/artifacts').glob('*.pt')); raw_initial=False
        elif model=='DeiT-Tiny':
            short='deit_tiny'; name='deit_tiny_patch16_224'
            recovery=(Path('results/final_rank_rule/deit/cifar10') if ds=='cifar10' else Path('results/deit_three_claims'))/'targets/artifacts'
            if ds=='cifar10': initial=Path('results/deit_cifar10_dimension/states/order2003_seed160/checkpoint0.pt'); raw_initial=True
            else: initial=next(Path('results/deit_three_claims/sources/artifacts').glob('*.pt')); raw_initial=False
        else:
            short='mobilenetv3_small' if model=='MobileNetV3-S' else 'mixer_b16'
            name='mobilenetv3_small_100.lamb_in1k' if model=='MobileNetV3-S' else 'mixer_b16_224.goog_in21k_ft_in1k'
            recovery=base/'targets/artifacts'; initial=next((base/'sources/artifacts').glob('*.pt')); raw_initial=False
        paths=[str(recovery/Path(p).name) for p in c['paths']]
        assert all(Path(p).exists() for p in paths),(model,ds,paths)
        result.append({**c,'index':i,'short':short,'name':name,'records':paths,'initial_source':str(initial),'initial_is_raw_state':raw_initial})
    return result
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'inventory.json').write_text(json.dumps(conditions(),indent=2),encoding='utf-8')
    print('Eight conditions inventoried')
