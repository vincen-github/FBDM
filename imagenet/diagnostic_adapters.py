"""Optimizer parameter groups and training-objective checks."""
import torch
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
    assert ratios['model.conv1.weight'] > 0 and ratios['head.0.weight'] > 0
    assert ratios['v_net.v.0.weight'] > 0


def assert_loss_semantics(model,cfg,loss):
    expected = cfg.fm_weight * model.last_losses['fm'] + cfg.lambda_param * model.last_losses['align']
    assert bool(torch.allclose(loss.detach().float(), expected.float(), rtol=1e-5, atol=1e-7))
