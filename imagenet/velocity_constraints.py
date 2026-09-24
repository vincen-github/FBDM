"""Isolated velocity constraints; default settings preserve the native objective."""
import math
import torch
import torch.nn.functional as F


def validate(model, cfg):
    beta = float(getattr(cfg, 'velocity_energy_beta', 0.))
    cap = float(getattr(cfg, 'velocity_norm_cap', 0.))
    wd = float(getattr(cfg, 'velocity_weight_decay', cfg.recipe_weight_decay))
    assert all(math.isfinite(x) and x >= 0 for x in (beta, cap, wd))
    assert cfg.velocity_mode == 'standard' and cfg.flow_path_mode == 'single'
    assert cfg.fm_time_samples == 1 and cfg.alignment_space == 'z0'
    assert cfg.fm_encoder_grad_scale == 1.
    assert (cfg.fm_weight == 1.3 and cfg.endpoint_weight == 0.) or (cfg.fm_weight == 0. and cfg.endpoint_weight == 1.3 and beta == cap == 0.)
    assert all(isinstance(m, (torch.nn.Sequential, torch.nn.Linear, torch.nn.ReLU))
               or m is model.v_net for m in model.v_net.modules())
    assert not list(model.v_net.buffers())


def bound_velocity(v, cap):
    """Smooth L2 bound; cap=0 is exact identity. Calculate norm in FP32 under AMP."""
    if not cap:
        return v
    x = v.float()
    return x / torch.sqrt(1. + x.square().sum(dim=1, keepdim=True) / (cap * cap))


def energy_penalty(model, z1, t1, z2, t2):
    v1 = model._velocity(z1.detach(), t1.detach())[0].float()
    v2 = model._velocity(z2.detach(), t2.detach())[0].float()
    if model.cfg.velocity_tangent_projection:
        a, b = F.normalize(z1.detach().float(), dim=1), F.normalize(z2.detach().float(), dim=1)
        v1 = v1 - (v1 * a).sum(1, keepdim=True) * a
        v2 = v2 - (v2 * b).sum(1, keepdim=True) * b
    return .5 * (v1.square().mean() + v2.square().mean())


@torch.no_grad()
def transport_probe(model, views, steps=16):
    """Small eval-mode Euler readout, not a training loss or endpoint evaluation."""
    modes = [(m, m.training) for m in model.modules()]
    device = next(model.parameters()).device
    try:
        model.eval()
        z = F.normalize(model.head(model.model(views[0][:16].to(device))).float(), dim=1)
        start = z.clone()
        path = torch.zeros(len(z), device=device)
        speed = torch.zeros_like(path)
        for i in range(steps):
            t = z.new_full((len(z), 1), i / steps)
            v = model._velocity(z, t)[0].float()
            if model.cfg.path_geometry == 'spherical':
                v = v - (v * z).sum(1, keepdim=True) * z
                nxt = F.normalize(z + v / steps, dim=1)
            else:
                nxt = z + v / steps
            speed += v.norm(dim=1) / steps
            path += (nxt - z).norm(dim=1)
            z = nxt
        return dict(samples=len(z), euler_steps=steps,
                    path_chord_sum_mean=float(path.mean()),
                    endpoint_chord_mean=float((z-start).norm(dim=1).mean()),
                    integrated_speed_mean=float(speed.mean()))
    finally:
        for module, mode in modes:
            module.training = mode
