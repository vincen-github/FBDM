"""Joint target assignment over several physical ImageNet minibatches.

The encoder/head snapshot is used without gradients to obtain assignment
features for a logical batch.  One joint capacitated matching is then fixed
while the original physical minibatches are replayed through the normal FBDM
forward/backward path.  Only those replayed forwards carry autograd graphs.
"""
from __future__ import annotations

import itertools
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

from methods.dm import DM


def install_preassigned_assignment(model) -> None:
    """Let the normal forward consume one externally planned target slice."""
    if hasattr(model, "_logical_native_assign_targets"):
        raise RuntimeError("logical assignment hook installed twice")
    native = model.assign_targets
    model._logical_native_assign_targets = native
    model._logical_preassigned_targets = None

    def assign_targets(self, z1, z2):
        planned = self._logical_preassigned_targets
        if planned is None:
            return self._logical_native_assign_targets(z1, z2)
        self._logical_preassigned_targets = None
        if len(planned) != 3:
            raise RuntimeError("logical assignment must provide r1/r2/center")
        if any(x.shape != z1.shape for x in planned):
            raise RuntimeError("logical assignment slice shape mismatch")
        return tuple(x.to(device=z1.device, dtype=torch.float32) for x in planned)

    model.assign_targets = types.MethodType(assign_targets, model)


def _bn_snapshot(model):
    rows = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            rows.append(
                (
                    module,
                    module.running_mean.detach().clone()
                    if module.running_mean is not None else None,
                    module.running_var.detach().clone()
                    if module.running_var is not None else None,
                    module.num_batches_tracked.detach().clone()
                    if module.num_batches_tracked is not None else None,
                )
            )
    return rows


def _restore_bn(rows) -> None:
    for module, mean, var, tracked in rows:
        if mean is not None:
            module.running_mean.copy_(mean)
        if var is not None:
            module.running_var.copy_(var)
        if tracked is not None:
            module.num_batches_tracked.copy_(tracked)


@torch.no_grad()
def _encode_assignment_features(model, views):
    if model.assignment_teacher is not None:
        raise RuntimeError("logical assignment is initially restricted to online features")
    device = next(model.parameters()).device
    x1, x2 = (x.to(device, non_blocking=True) for x in views)
    with torch.autocast("cuda"):
        if getattr(model.cfg, "joint_two_views", False):
            backbone = model.model(torch.cat((x1, x2), dim=0))
            z = F.normalize(model.head(backbone), p=2, dim=1)
            z1, z2 = z.chunk(2, dim=0)
        else:
            z1 = F.normalize(model.head(model.model(x1)), p=2, dim=1)
            z2 = F.normalize(model.head(model.model(x2)), p=2, dim=1)
    return z1.float(), z2.float()


@torch.no_grad()
def plan_logical_assignment(model, batches):
    """Return one target tuple per local physical minibatch."""
    if not dist.is_initialized():
        raise RuntimeError("logical assignment requires initialized distributed training")
    if model._logical_preassigned_targets is not None:
        raise RuntimeError("previous logical targets were not consumed")
    bn = _bn_snapshot(model)
    try:
        local = [_encode_assignment_features(model, batch[0]) for batch in batches]
    finally:
        _restore_bn(bn)

    local_sizes = [a.shape[0] for a, _ in local]
    local_z1 = torch.cat([a for a, _ in local], dim=0).contiguous()
    local_z2 = torch.cat([b for _, b in local], dim=0).contiguous()
    world, rank = dist.get_world_size(), dist.get_rank()
    gathered1 = [torch.empty_like(local_z1) for _ in range(world)]
    gathered2 = [torch.empty_like(local_z2) for _ in range(world)]
    dist.all_gather(gathered1, local_z1)
    dist.all_gather(gathered2, local_z2)
    global_z1, global_z2 = torch.cat(gathered1), torch.cat(gathered2)

    if rank == 0:
        with torch.autocast("cuda", enabled=False):
            targets = torch.stack(
                DM.assign_targets(model, global_z1.float(), global_z2.float())
            ).contiguous()
    else:
        targets = torch.empty(
            (3, global_z1.shape[0], model.cfg.emb),
            dtype=torch.float32,
            device=local_z1.device,
        )
    dist.broadcast(targets, src=0)

    rank_start = rank * local_z1.shape[0]
    planned = []
    offset = 0
    for size in local_sizes:
        start = rank_start + offset
        planned.append(tuple(x[start:start + size] for x in targets))
        offset += size
    return planned


def logical_batches(loader, model, group_batches: int, max_steps: int):
    """Yield physical batches after jointly planning each bounded group."""
    group_batches = int(group_batches)
    if group_batches < 1:
        raise ValueError("logical assignment group must be positive")
    iterator = iter(loader)
    emitted = 0
    while emitted < max_steps:
        group = list(itertools.islice(iterator, min(group_batches, max_steps - emitted)))
        if not group:
            break
        if group_batches == 1:
            plans = [None] * len(group)
        else:
            plans = plan_logical_assignment(model, group)
        for batch, planned in zip(group, plans):
            if planned is not None:
                if model._logical_preassigned_targets is not None:
                    raise RuntimeError("logical target overwrite")
                model._logical_preassigned_targets = planned
            yield batch
            if planned is not None and model._logical_preassigned_targets is not None:
                raise RuntimeError("logical target was not consumed by model forward")
            emitted += 1
    if emitted != max_steps:
        raise RuntimeError(f"logical iterator emitted {emitted}, expected {max_steps}")

