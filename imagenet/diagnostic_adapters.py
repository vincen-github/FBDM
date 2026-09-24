"""Velocity-only optimizer scale and FM-off implementation gates."""
import torch
import torch.nn.functional as F
from optimizer_recipe import make_optimizer,CompatibleLARS

def make_diagnostic_optimizer(model,cfg):
    original=make_optimizer(model,cfg)
    if not getattr(cfg,'dispatch_module_lrs',False):return original
    velocity={id(p) for p in model.v_net.parameters()}
    encoder={id(p) for p in model.model.parameters()}
    groups=[]
    for old in original.param_groups:
        for module in ('encoder','head','velocity'):
            def belongs(p):
                return ('velocity' if id(p) in velocity else ('encoder' if id(p) in encoder else 'head'))==module
            params=[p for p in old['params'] if belongs(p)]
            if not params:continue
            factor=float(getattr(cfg,'dispatch_'+module+'_lr',1.0))
            group={k:v for k,v in old.items() if k!='params'}
            group.update(params=params,module_lr_multiplier=factor,module_name=module,lr=cfg.lr*factor)
            if module == 'velocity' and group.get('layer_adaptation'):
                group['weight_decay'] = float(getattr(cfg, 'velocity_weight_decay', group['weight_decay']))
            groups.append(group)
    return CompatibleLARS(torch.optim.SGD(groups,lr=cfg.lr,momentum=.9),trust_coefficient=.001)

def assert_smoke_updates(model,cfg,ratios):
    assert ratios['model.conv1.weight']>0 and ratios['head.0.weight']>0
    if cfg.fm_weight:
        assert ratios['v_net.v.0.weight']>0
    else:
        assert cfg.endpoint_weight in (.65,1.3,2.6) and cfg.lambda_param==1.3
        assert ratios['v_net.v.0.weight']==0
        assert all(not p.requires_grad and p.grad is None for p in model.v_net.parameters())

def assert_loss_semantics(model,cfg,loss):
    if not cfg.fm_weight:
        expected=cfg.endpoint_weight*model.last_losses['endpoint']+cfg.lambda_param*model.last_losses['align']
        assert bool(torch.allclose(loss.detach().float(),expected.float(),rtol=1e-5,atol=1e-7))
        assert float(model.last_losses['fm'])==0

def checked_forward(model,ddp,views,cfg,smoke):
    if not (smoke and cfg.fm_weight==0):return ddp(views)
    from methods.dm import sample_time
    planned=model._logical_preassigned_targets
    assert planned is not None
    outputs=[]
    hook=model.head.register_forward_hook(lambda module,inputs,output:outputs.append(F.normalize(output,p=2,dim=1).detach()))
    before=torch.cuda.get_rng_state()
    try:loss=ddp(views)
    finally:hook.remove()
    after=torch.cuda.get_rng_state()
    assert len(outputs)==2
    r1,r2,_=planned
    endpoint=.5*((outputs[0]-r1).square().mean()+(outputs[1]-r2).square().mean())
    assert bool(torch.allclose(endpoint.float(),model.last_losses['endpoint'].float(),rtol=1e-5,atol=1e-7))
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.set_rng_state(before)
        for z in outputs:sample_time(len(z),z.device,z.dtype,cfg.time_sampling,cfg.time_map,cfg.time_map_param)
        assert torch.equal(torch.cuda.get_rng_state(),after),'FM-off time RNG consumption mismatch'
    model._endpoint_semantic_checks=getattr(model,'_endpoint_semantic_checks',0)+1
    return loss
