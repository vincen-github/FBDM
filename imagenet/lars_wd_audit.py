"""Read-only guards for the existing LARS implementation; no new update rule."""
import math
import torch


def configured_weight_decay(cfg):
    value = float(getattr(cfg, 'recipe_weight_decay', 1e-6))
    if not math.isfinite(value) or value < 0:
        raise ValueError('recipe_weight_decay must be finite and nonnegative')
    return value


def audit_optimizer(optimizer, model, cfg):
    if cfg.optimizer_name != 'lars':
        raise ValueError('This diagnostic is only for LARS')
    expected = configured_weight_decay(cfg)
    actual = getattr(optimizer, 'optim', None)
    if actual is None or optimizer.param_groups is not actual.param_groups:
        raise ValueError('Expected native LARS wrapper with aliased param_groups')
    if optimizer.trust_coefficient != .001:
        raise ValueError('Unexpected LARS trust coefficient')
    names = {id(p): n for n, p in model.named_parameters()}
    excluded = {id(p) for m in model.modules()
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                for p in m.parameters(recurse=False)}
    excluded.update(id(p) for n, p in model.named_parameters() if n.endswith('bias'))
    seen, groups = set(), []
    for g in actual.param_groups:
        members = []
        for p in g['params']:
            pid = id(p)
            if pid not in names or pid in seen:
                raise ValueError('Unknown or duplicate optimizer parameter')
            seen.add(pid)
            adapted = pid not in excluded
            if g.get('layer_adaptation') != adapted:
                raise ValueError('Layer-adaptation membership changed: ' + names[pid])
            group_expected = float(getattr(cfg, 'velocity_weight_decay', expected)) if names[pid].startswith('v_net.') else expected
            if g['weight_decay'] != (group_expected if adapted else 0.):
                raise ValueError('Effective weight decay mismatch: ' + names[pid])
            members.append(dict(name=names[pid],trainable=p.requires_grad))
        if g.get('momentum') != .9:
            raise ValueError('Momentum changed')
        groups.append(dict(lr=g['lr'],weight_decay=g['weight_decay'],
                           layer_adaptation=g['layer_adaptation'],members=members))
    required = {id(p) for p in model.parameters() if p.requires_grad}
    if not required <= seen:
        raise ValueError('Trainable parameters missing from optimizer')
    return dict(status='ok',configured_weight_decay=expected,trust_coefficient=.001,
                frozen_parameters_in_optimizer=sum(not p.requires_grad for g in actual.param_groups for p in g['params']),
                groups=groups)


@torch.no_grad()
def decay_diagnostics(optimizer, watched):
    """Call after AMP unscale/DDP and BEFORE LARS.step modifies gradients."""
    by_id = {id(p): g for g in optimizer.param_groups for p in g['params']}
    result = {}
    for name, p in watched.items():
        group = by_id.get(id(p))
        if group is None or p.grad is None:
            result[name] = dict(status='no_optimizer_group_or_gradient')
            continue
        gradient = p.grad.detach().float()
        weight = p.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            result[name] = dict(status='nonfinite_gradient')
            continue
        decay = weight * group['weight_decay']
        gn, dn = float(gradient.norm()), float(decay.norm())
        result[name] = dict(status='ok',weight_decay=group['weight_decay'],
            weight_norm=float(weight.norm()),data_gradient_norm=gn,decay_norm=dn,
            decay_to_data_gradient=dn/gn if gn else None,
            decay_data_cosine=float((decay*gradient).sum())/(dn*gn) if dn and gn else None,
            layer_adaptation=group['layer_adaptation'])
    return result
