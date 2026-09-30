"""Shared configuration and dimension-generic response measurement."""
from package_paths import read_config_text
import json
from pathlib import Path
import torch
import drift_experiment as d
import analyze_scaffold_followup as f
from all_model_followups_config import conditions
from lossless_checkpoint import load as checkpoint_load,save as save_checkpoint,save_value

ROOT=Path('results/noreplay_all_models')
def load(path):return checkpoint_load(path)
def load_record(path,keys):return checkpoint_load(path,state_keys=keys)
def prepare(short):
    c=next(x for x in conditions() if x['short']==short and x['dataset']=='cifar10')
    out=ROOT/f'{short}_cifar10';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():return
    initial=load(c['initial_source']);initial=initial if c['initial_is_raw_state'] else initial['states'][0]
    torch.save(initial,out/'initial_state.pt')
    net=d.model(c['name'],False,num_classes=10);net.load_state_dict(initial)
    head=net.get_classifier();params=dict(net.named_parameters())
    weight=next(n for n,p in params.items() if p is head.weight);bias=next(n for n,p in params.items() if p is head.bias)
    protocol=json.loads(read_config_text(Path('results/noreplay_resnet_cifar10/protocol.json')))
    protocol.update(model=c['model'],model_name=c['name'],short=short,scaffold_rank=c['rank'],feature_dimension=head.in_features,
        initial_state=c['initial_source'],initial_state_index=None if c['initial_is_raw_state'] else 0,
        replay_predictors=c['scaffold'],head_weight=weight,head_bias=bias,
        amp=short!='mobilenetv3_small',activation_recomputation=short=='mixer_b16',
        optimizer='AdamW lr=0.0003 wd=0.05; reset per task; zero skipped steps',
        precision='FP32' if short=='mobilenetv3_small' else 'AMP with initial scale128',
        state_storage='Lossless byte-plane integer-delta; use lossless_checkpoint.load; bitwise roundtrip checked',status='prepared')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
def configure(short):
    out=ROOT/f'{short}_cifar10';c=json.loads(read_config_text(out/'protocol.json'))
    f.load=load
    return dict(OUT=out,C=c,MODEL_NAME=c['model_name'],RANK=c['scaffold_rank'],DIM=c['feature_dimension'],
                HEAD_WEIGHT=c['head_weight'],HEAD_BIAS=c['head_bias'],AMP=c['amp'],REPLAY_PATH=Path(c['replay_predictors']))
def response(net,batches,state,seed):
    net.load_state_dict(state)
    checkpointing=getattr(net,'grad_checkpointing',False)
    if checkpointing:net.set_grad_checkpointing(False)
    head={id(p) for p in net.get_classifier().parameters()};params=dict(net.named_parameters())
    names=[n for n,p in params.items() if id(p) not in head]
    norm=torch.sqrt(sum(state[n].double().square().sum() for n in names))
    generator=torch.Generator().manual_seed(920000+seed)
    dimension=net.get_classifier().in_features;cov=torch.zeros(dimension,dimension,dtype=torch.double);prefixes={}
    for draw in range(16):
        direction={n:torch.randn(state[n].shape,generator=generator) for n in names}
        length=torch.sqrt(sum(v.double().square().sum() for v in direction.values()));scale=float(1e-4*norm/length);values=[]
        for sign in (1,-1):
            net.load_state_dict(state)
            with torch.no_grad():
                for n in names:params[n].add_(direction[n].cuda(),alpha=sign*scale)
            values.append(f.cached_representations(net,batches).double())
        field=f.center((values[0]-values[1])/2);cov+=field.T@field
        if draw+1 in (8,16):prefixes[draw+1]=cov.clone()/(draw+1)
    if checkpointing:net.set_grad_checkpointing(True)
    return prefixes
