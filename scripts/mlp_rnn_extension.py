"""Fixed two-model extension, preserving the existing source-only rank rule."""
from package_paths import read_config_text
import os
from pathlib import Path
os.environ['HF_HOME']=str(Path('data/hf_cache').resolve())
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
    os.environ.pop(key,None)
import argparse,json,traceback
import torch
import additional_rank_models as a
import analyze_scaffold_followup as f
from deit_three_claims import json_save

ROOT=Path('results/mlp_rnn_extension')
MODELS=['mixer_b16_224.goog_in21k_ft_in1k','sequencer2d_s.in1k']
PROTOCOL=dict(models=MODELS,datasets=['cifar100','cifar10'],
    selection='joint k=r; smallest mean-positive source-held-out matched-projector Representation PCA coverage gain in 8..32',
    no_eligible_rank='report failure to select; do not expand range or tune on targets',
    one_se=False,response_draws=16,response_relative_step=1e-4,scaffold='fixed source drift 1->2',
    checkpoints=dict(cifar100=[0,1,2,3,4],cifar10=[0,1,2,3,4,5]),
    trajectories_per_dataset=dict(source=6,target=12),
    training='same baseline data, orders, seeds, epochs, AdamW, ER memory and batch64; AMP training and saved trajectory inference; FP32 perturbation and actual-update field inference',
    preprocessing='same baseline 224 resize and ImageNet mean/std; retained also for Mixer although its pretrained cfg recommends mean/std 0.5',
    initializations=dict(mixer='ImageNet21k then ImageNet1k',sequencer='ImageNet1k'),
    representation='standard global pre-logit vector; Sequencer is a visual bidirectional LSTM, not a temporal-task test',
    core_claims=dict(temporal='all target temporal random-gain CI lower bounds >0',
        response='checkpoint2-minus-initial overlap and response transfer CI lower bounds >0',
        reuse='actual AdamW update and frozen logit reconstruction random-gain CI lower bounds >0'),
    extensions='report final-minus-initial alignment and loss readout separately; do not require monotonic overlap',
    stop='complete both specified models on both datasets, regardless of positive or negative results',
    inference='exploratory nominal bootstrap intervals; architecture and pretraining differences are not isolated',
    storage='remove only newly generated duplicate capture files after exact equality checks against retained full records')

class Experiment(a.Experiment):
    def discard_verified_captures(self,path,rec):
        cap=(self.out/'captures'/path.stem).resolve()
        root=(ROOT/self.name/self.ds/'captures').resolve()
        assert cap.is_relative_to(root)
        removed=[]
        for file in cap.glob('*.pt'):
            if file.stem.startswith('checkpoint'):
                cp=int(file.stem.removeprefix('checkpoint')); expected=rec['states'][cp]
            elif file.name=='actual_update.pt': expected=rec['actual_update']
            else: raise ValueError(file)
            saved=f.load(file)
            def equal(x,y):
                if isinstance(x,torch.Tensor): return isinstance(y,torch.Tensor) and torch.equal(x,y)
                if isinstance(x,dict): return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
                if isinstance(x,(list,tuple)): return len(x)==len(y) and all(equal(v,w) for v,w in zip(x,y))
                return x==y
            assert equal(saved,expected),file
            removed.append(dict(path=str(file),bytes=file.stat().st_size,exact_equal=True))
            del saved
            file.unlink()
        if removed: json_save(cap/'deduplicated.json',dict(retained=str(path),removed=removed))

    def source_states(self):
        super().source_states()
        for raw in self.paths('source'):
            path=self.out/'sources/artifacts'/raw.name
            self.discard_verified_captures(path,f.load(path))

    def target_records(self):
        evaluation=f.IndexedCifar(Path('data'),True,False,self.ds)
        for path in self.paths('target'):
            dst=self.out/'targets/artifacts'/path.name
            if not dst.exists():
                raw=f.load(path); cfg=json.loads(read_config_text(path.parent.parent/'manifest.json'))
                cap=self.out/'captures'/path.stem; end=cfg['tasks']
                update=f.load(cap/'actual_update.pt'); final=f.load(cap/f'checkpoint{end}.pt')
                assert update['optimizer_step_after']==2
                rec=dict(status='complete',dataset=self.ds,manifest=cfg,original_path=str(path),
                    order_seed=raw['order_seed'],seed=raw['seed'],fit_ids=raw['fit_ids'],heldout_ids=raw['eval_ids'],
                    heldout_labels=raw['eval_labels'],eval_reps=raw['eval_representations'],fit_reps=raw['fit_representations'],
                    states={end-1:raw['pre_future_state_dict'],end:final},actual_update=update,replay_errors={})
                f.save(rec,dst)
                del rec,raw,update,final
            rec=f.load(dst)
            self.discard_verified_captures(dst,rec)
            cache=self.out/'update_fields'/path.name
            if not cache.exists():
                fields,norm=f.actual_update_fields(rec,evaluation,torch.device('cuda'))
                f.save(dict(fields=fields,relative_step=norm),cache)
            del rec; torch.cuda.empty_cache()
            print('TARGET READY',self.name,self.ds,path.stem,flush=True)

    def finish(self):
        status=json.loads(read_config_text(self.out/'status.json'))
        if status['status']!='complete': return
        rank=status['rank']; folder=self.out/f'rank{rank}'
        g=json.loads(read_config_text(folder/'geometry.json'))
        gates=status['gates']
        core=dict(temporal=gates['temporal'],response=g['change_from_initial_ci95'][2][0]>0 and gates['response_transfer'],
                  reuse=gates['actual_update'] and gates['logit_readout'])
        status['core_claim_support']=core; status['all_core_claims_supported']=all(core.values())
        json_save(self.out/'status.json',status)
        verification=self.out/'independent_verification.json'
        if not verification.exists():
            import finalize_final_rank_rule as v
            import report_additional_rank_models as report
            self.configure()
            checks=dict(geometry=v.verify_geometry(folder,self.paths('source'),self.out/'responses/artifacts'),
                metrics=v.verify_metrics(folder,self.paths('target'),self.out/'targets/artifacts',self.out/'update_fields'),
                checkpoint_links=report.verify_checkpoint_links(self))
            json_save(verification,checks)
        print('VERIFIED',self.name,self.ds,'core',core,flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=MODELS); p.add_argument('--dataset',choices=['cifar100','cifar10']); args=p.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    a.ROOT=ROOT
    dest=ROOT/'protocol.json'
    if dest.exists(): assert json.loads(read_config_text(dest))==PROTOCOL
    else: json_save(dest,PROTOCOL)
    for name in ([args.model] if args.model else MODELS):
        for ds in ([args.dataset] if args.dataset else PROTOCOL['datasets']):
            ex=Experiment(name,ds)
            try:
                ex.run(); ex.finish()
            except Exception:
                json_save(ex.out/'execution_error.json',dict(traceback=traceback.format_exc()))
                raise
    json_save(ROOT/'requested_batch_finished.json',dict(models=[args.model] if args.model else MODELS,datasets=[args.dataset] if args.dataset else PROTOCOL['datasets']))

if __name__=='__main__': main()
