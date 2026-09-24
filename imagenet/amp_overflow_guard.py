"""DDP-safe AMP overflow handling for the ImageNet FBDM training loop.

The caller must invoke ``scaler.unscale_(optimizer)`` before this helper.  The
normal CUDA GradScaler records per-device ``found_inf`` tensors during that
call, but it does not synchronize those tensors across independent DDP ranks.
We synchronize them here so every rank either performs or skips the optimizer
step together.  A non-finite forward loss remains fatal; a finite loss with an
AMP gradient overflow is handled by GradScaler's normal backoff path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class AmpStepResult:
    optimizer_step: bool
    overflow: bool
    scale_before: float
    scale_after: float


def _synchronize_found_inf(scaler, optimizer) -> bool:
    """Synchronize GradScaler's recorded overflow decision across DDP ranks."""

    found_inf_per_device = scaler._found_inf_per_device(optimizer)
    if not found_inf_per_device:
        raise RuntimeError("GradScaler recorded no inf check after unscale_")
    for found_inf in found_inf_per_device.values():
        dist.all_reduce(found_inf, op=dist.ReduceOp.MAX)
    return any(bool(found_inf.item()) for found_inf in found_inf_per_device.values())


def distributed_amp_step(*, scaler, optimizer, loss: torch.Tensor) -> AmpStepResult:
    """Take one synchronized AMP step, recovering from gradient overflow.

    ``loss`` non-finiteness is a model/forward failure and is never suppressed.
    Gradient overflow with a finite loss is recoverable: all DDP ranks skip the
    update and GradScaler lowers its scale using its configured backoff factor.
    """

    loss_bad = (~torch.isfinite(loss.detach())).to(dtype=torch.int32)
    dist.all_reduce(loss_bad, op=dist.ReduceOp.MAX)
    if bool(loss_bad.item()):
        raise FloatingPointError("nonfinite forward loss")

    overflow = _synchronize_found_inf(scaler, optimizer)
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if overflow and not scale_after < scale_before:
        raise RuntimeError(
            f"GradScaler did not back off after overflow: {scale_before} -> {scale_after}"
        )
    return AmpStepResult(
        optimizer_step=not overflow,
        overflow=overflow,
        scale_before=scale_before,
        scale_after=scale_after,
    )
