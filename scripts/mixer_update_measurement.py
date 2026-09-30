"""Memory-efficient measurement with exact restoration of the live training model."""
import torch
import analyze_scaffold_followup as f

def measure(net, record, evaluation):
    capture=record['actual_update']
    before,after=capture['before_state'],capture['after_state']
    modes=[(module,module.training) for module in net.modules()]
    recomputation=net.grad_checkpointing
    head_ids={id(p) for p in net.get_classifier().parameters()}
    names=[n for n,p in net.named_parameters() if id(p) not in head_ids]
    buffers=set(dict(net.named_buffers()))
    # Mixer has no changing running-statistic buffers, so the two AdamW fields coincide.
    assert all(torch.equal(before[n],after[n]) for n in buffers)
    delta_norm=torch.sqrt(sum((after[n].double()-before[n].double()).square().sum() for n in names))
    grad_norm=torch.sqrt(sum(capture['gradient'][n].double().square().sum() for n in names))
    relative=float(delta_norm/torch.sqrt(sum(before[n].double().square().sum() for n in names)))
    batches=None
    try:
        net.set_grad_checkpointing(False)
        batches=f.image_batches(net,evaluation,record['heldout_ids'],torch.device('cuda'))
        net.load_state_dict(before)
        base=f.cached_representations(net,batches)
        net.load_state_dict(after)
        full=f.cached_representations(net,batches)-base
        net.load_state_dict(before)
        with torch.no_grad():
            for n,p in net.named_parameters():
                if n in names:
                    p.add_(capture['gradient'][n].to(p.device),alpha=-float(delta_norm/grad_norm))
        gradient=f.cached_representations(net,batches)-base
        fields={'adamw_full':f.center(full.double()),'adamw_parameters':f.center(full.double()),
                'norm_matched_training_gradient':f.center(gradient.double())}
    finally:
        net.load_state_dict(after)
        net.set_grad_checkpointing(recomputation)
        for module,mode in modes: module.training=mode
        del batches
    return fields,relative
