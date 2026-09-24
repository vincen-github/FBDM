"""Native FBDM Adam or documented SimCLR training-recipe adaptation."""
import torch
import math
from upstream_lars import LARS
from lars_wd_audit import configured_weight_decay

class CompatibleLARS(LARS):
    def zero_grad(self,set_to_none=True):self.optim.zero_grad(set_to_none=set_to_none)

def learning_rate(cfg,step,total_steps):
    warm=cfg.warmup_steps
    if cfg.optimizer_name=='lars':
        return cfg.lr*((step+1)/(warm+1) if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/(total_steps-warm))))
    return cfg.lr*((step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/(total_steps-warm))))

def make_optimizer(model,cfg):
    if cfg.optimizer_name=='lars':
        excluded={id(p) for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)
                  for p in m.parameters(recurse=False)}
        excluded.update(id(p) for n,p in model.named_parameters() if n.endswith('bias'))
        groups=[dict(params=[p for p in model.parameters() if id(p) not in excluded],
                     weight_decay=configured_weight_decay(cfg),layer_adaptation=True),
                dict(params=[p for p in model.parameters() if id(p) in excluded],
                     weight_decay=0.,layer_adaptation=False)]
        return CompatibleLARS(torch.optim.SGD(groups,lr=cfg.lr,momentum=.9),trust_coefficient=.001)
    assert cfg.optimizer_name=='adam'
    decay={id(m.weight) for m in model.modules() if isinstance(m,(torch.nn.Linear,torch.nn.modules.conv._ConvNd))}
    return torch.optim.Adam([
        dict(params=[p for p in model.parameters() if id(p) in decay],weight_decay=cfg.adam_l2),
        dict(params=[p for p in model.parameters() if id(p) not in decay],weight_decay=0.)
    ],lr=cfg.lr,betas=(cfg.adam_beta1,cfg.adam_beta2),eps=1e-8)
