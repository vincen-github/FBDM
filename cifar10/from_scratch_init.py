"""Strict, paired initialization policies that never load pretrained weights.

Every optional reinitialization saves and restores the CPU RNG state.  The
control and candidate therefore consume identical construction randomness;
only tensors explicitly named by the selected policy may differ.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable

import torch
import torch.nn as nn


CONTROL = "incumbent_control"
ZERO_INIT_RESIDUAL = "zero_init_residual"
HEAD_KAIMING = "head_kaiming_normal"
VELOCITY_KAIMING = "velocity_kaiming_normal"
BACKBONE_KAIMING_FAN_IN = "backbone_kaiming_fan_in"
BACKBONE_KAIMING_UNIFORM_FAN_IN = "backbone_kaiming_uniform_fan_in"
HEAD_ORTHOGONAL = "head_orthogonal"
BACKBONE_ORTHOGONAL = "backbone_orthogonal"
HEAD_XAVIER_UNIFORM = "head_xavier_uniform"
VELOCITY_XAVIER_UNIFORM = "velocity_xavier_uniform"
BACKBONE_XAVIER_UNIFORM = "backbone_xavier_uniform"
HEAD_LECUN_NORMAL = "head_lecun_normal"
VELOCITY_LECUN_NORMAL = "velocity_lecun_normal"
BACKBONE_LECUN_NORMAL = "backbone_lecun_normal"
BN_GAMMA_090 = "bn_gamma_090"
VELOCITY_ORTHOGONAL = "velocity_orthogonal"

CANDIDATES = (
    HEAD_KAIMING,
    VELOCITY_KAIMING,
    BACKBONE_KAIMING_FAN_IN,
    BACKBONE_KAIMING_UNIFORM_FAN_IN,
    HEAD_ORTHOGONAL,
    BACKBONE_ORTHOGONAL,
    HEAD_XAVIER_UNIFORM,
    VELOCITY_XAVIER_UNIFORM,
    BACKBONE_XAVIER_UNIFORM,
    HEAD_LECUN_NORMAL,
    VELOCITY_LECUN_NORMAL,
    BACKBONE_LECUN_NORMAL,
    BN_GAMMA_090,
    VELOCITY_ORTHOGONAL,
)
VALID_MODES = (CONTROL, *CANDIDATES)

CONV1_DEFAULT = "current_default"
CONV1_TORCHVISION = "torchvision_kaiming_fan_out"
VALID_BASE_CONV1_MODES = (CONV1_DEFAULT, CONV1_TORCHVISION)

MODE_SCOPE = {
    CONTROL: None,
    ZERO_INIT_RESIDUAL: "backbone",
    HEAD_KAIMING: "head",
    VELOCITY_KAIMING: "velocity",
    BACKBONE_KAIMING_FAN_IN: "backbone",
    BACKBONE_KAIMING_UNIFORM_FAN_IN: "backbone",
    HEAD_ORTHOGONAL: "head",
    BACKBONE_ORTHOGONAL: "backbone",
    HEAD_XAVIER_UNIFORM: "head",
    VELOCITY_XAVIER_UNIFORM: "velocity",
    BACKBONE_XAVIER_UNIFORM: "backbone",
    HEAD_LECUN_NORMAL: "head",
    VELOCITY_LECUN_NORMAL: "velocity",
    BACKBONE_LECUN_NORMAL: "backbone",
    BN_GAMMA_090: "backbone",
    VELOCITY_ORTHOGONAL: "velocity",
}


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def rng_state_sha256(state: torch.Tensor) -> str:
    return tensor_sha256(state)


def tensor_hashes(module: nn.Module) -> dict[str, str]:
    return {
        name: tensor_sha256(tensor)
        for name, tensor in sorted(module.state_dict().items())
    }


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _backbone(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def _linear_modules(module: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, value) for name, value in module.named_modules() if isinstance(value, nn.Linear)]


def _conv_modules(module: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    return [(name, value) for name, value in module.named_modules() if isinstance(value, nn.Conv2d)]


def _initialize_backbone(module: nn.Module, mode: str) -> list[str]:
    backbone = _backbone(module)
    targets: list[str] = []
    if mode == ZERO_INIT_RESIDUAL:
        for name, block in backbone.named_modules():
            if re.fullmatch(r"layer[1-4]\.\d+", name) and hasattr(block, "bn2"):
                if not isinstance(block.bn2, nn.BatchNorm2d) or block.bn2.weight is None:
                    raise RuntimeError(f"unexpected residual block at {name}")
                nn.init.zeros_(block.bn2.weight)
                targets.append(f"module.{name}.bn2.weight")
    elif mode in {
        BACKBONE_KAIMING_FAN_IN, BACKBONE_KAIMING_UNIFORM_FAN_IN,
        BACKBONE_XAVIER_UNIFORM, BACKBONE_LECUN_NORMAL,
    }:
        for name, conv in _conv_modules(backbone):
            if mode == BACKBONE_KAIMING_FAN_IN:
                nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            elif mode == BACKBONE_KAIMING_UNIFORM_FAN_IN:
                nn.init.kaiming_uniform_(conv.weight, mode="fan_in", nonlinearity="relu")
            elif mode == BACKBONE_XAVIER_UNIFORM:
                nn.init.xavier_uniform_(conv.weight, gain=math.sqrt(2.0))
            else:
                nn.init.normal_(conv.weight, mean=0.0, std=1.0 / math.sqrt(conv.weight[0].numel()))
            targets.append(f"module.{name}.weight")
    elif mode == BACKBONE_ORTHOGONAL:
        for name, conv in _conv_modules(backbone):
            nn.init.orthogonal_(conv.weight, gain=math.sqrt(2.0))
            targets.append(f"module.{name}.weight")
    elif mode == BN_GAMMA_090:
        for name, bn in backbone.named_modules():
            if isinstance(bn, nn.BatchNorm2d) and bn.weight is not None:
                nn.init.constant_(bn.weight, 0.9)
                targets.append(f"module.{name}.weight")
    return sorted(targets)


def _initialize_linears(module: nn.Module, mode: str, scope: str) -> list[str]:
    linears = _linear_modules(module)
    targets: list[str] = []
    if mode not in {
        HEAD_KAIMING, VELOCITY_KAIMING, HEAD_ORTHOGONAL, VELOCITY_ORTHOGONAL,
        HEAD_XAVIER_UNIFORM, VELOCITY_XAVIER_UNIFORM,
        HEAD_LECUN_NORMAL, VELOCITY_LECUN_NORMAL,
    }:
        return targets
    for index, (name, linear) in enumerate(linears):
        if mode in {HEAD_KAIMING, VELOCITY_KAIMING}:
            nonlinearity = "linear" if index == len(linears) - 1 else "relu"
            nn.init.kaiming_normal_(
                linear.weight, mode="fan_in", nonlinearity=nonlinearity
            )
        elif mode in {HEAD_ORTHOGONAL, VELOCITY_ORTHOGONAL}:
            gain = 1.0 if index == len(linears) - 1 else math.sqrt(2.0)
            nn.init.orthogonal_(linear.weight, gain=gain)
        elif mode in {HEAD_XAVIER_UNIFORM, VELOCITY_XAVIER_UNIFORM}:
            gain = 1.0 if index == len(linears) - 1 else math.sqrt(2.0)
            nn.init.xavier_uniform_(linear.weight, gain=gain)
        else:
            nn.init.normal_(linear.weight, mean=0.0, std=1.0 / math.sqrt(linear.in_features))
        targets.append(f"{name}.weight")
    if not targets:
        raise RuntimeError(f"{scope} initialization found no Linear tensors")
    return sorted(targets)


def _apply_policy_preserving_rng(module: nn.Module, mode: str, scope: str) -> dict:
    before_hashes = tensor_hashes(module)
    rng_before = torch.random.get_rng_state().clone()
    applies = MODE_SCOPE[mode] == scope
    declared_targets: list[str] = []
    try:
        if applies and scope == "backbone":
            declared_targets = _initialize_backbone(module, mode)
        elif applies and scope in {"head", "velocity"}:
            declared_targets = _initialize_linears(module, mode, scope)
    finally:
        torch.random.set_rng_state(rng_before)
    rng_after = torch.random.get_rng_state()
    if not torch.equal(rng_before, rng_after):
        raise RuntimeError(f"{scope} initialization advanced CPU RNG")

    after_hashes = tensor_hashes(module)
    changed = sorted(name for name in before_hashes if before_hashes[name] != after_hashes[name])
    if changed != declared_targets:
        raise RuntimeError(
            f"{mode} changed {scope} tensors outside its declaration: "
            f"declared={declared_targets}, observed={changed}"
        )
    return {
        "event": "from_scratch_initialization",
        "scope": scope,
        "mode": mode,
        "candidate_applies_to_scope": applies,
        "declared_target_tensor_names": declared_targets,
        "changed_tensor_names": changed,
        "changed_tensor_count": len(changed),
        "rng_state_preserved": True,
        "rng_state_sha256_before": rng_state_sha256(rng_before),
        "rng_state_sha256_after": rng_state_sha256(rng_after),
        "module_state_sha256": module_state_sha256(module),
    }


def apply_backbone_initialization(
    wrapped_model: nn.Module,
    *,
    dataset: str,
    base_conv1_mode: str,
    mode: str,
) -> dict:
    if dataset != "cifar10":
        raise ValueError("this initialization suite is restricted to CIFAR-10")
    if mode not in VALID_MODES:
        raise ValueError(f"unknown initialization mode: {mode}")
    if base_conv1_mode not in VALID_BASE_CONV1_MODES:
        raise ValueError(f"unknown base conv1 mode: {base_conv1_mode}")

    backbone = _backbone(wrapped_model)
    conv1 = backbone.conv1
    if not isinstance(conv1, nn.Conv2d) or tuple(conv1.weight.shape) != (64, 3, 3, 3):
        raise RuntimeError("backbone does not have the frozen CIFAR replacement conv1")
    if tuple(conv1.stride) != (1, 1) or tuple(conv1.padding) != (1, 1):
        raise RuntimeError("replacement conv1 is not the frozen CIFAR stem")
    if conv1.bias is not None or not isinstance(backbone.maxpool, nn.Identity):
        raise RuntimeError("frozen CIFAR stem structure changed")

    base_before = tensor_sha256(conv1.weight)
    rng_before = torch.random.get_rng_state().clone()
    try:
        if base_conv1_mode == CONV1_TORCHVISION:
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
    finally:
        torch.random.set_rng_state(rng_before)
    if not torch.equal(rng_before, torch.random.get_rng_state()):
        raise RuntimeError("base conv1 initialization advanced CPU RNG")
    base_after = tensor_sha256(conv1.weight)

    event = _apply_policy_preserving_rng(wrapped_model, mode, "backbone")
    event.update(
        {
            "base_conv1_mode": base_conv1_mode,
            "base_conv1_reinitialized": base_conv1_mode == CONV1_TORCHVISION,
            "base_conv1_sha256_before": base_before,
            "base_conv1_sha256_after": base_after,
            "base_conv1_weight_shape": [64, 3, 3, 3],
            "torchvision_weights_argument": getattr(wrapped_model, "init_weights_argument", None),
            "init_mode": getattr(wrapped_model, "init_mode", None),
            "effective_conv1_policy": (
                "overridden_by_candidate"
                if mode in {
                    BACKBONE_KAIMING_FAN_IN, BACKBONE_KAIMING_UNIFORM_FAN_IN,
                    BACKBONE_ORTHOGONAL,
                    BACKBONE_XAVIER_UNIFORM, BACKBONE_LECUN_NORMAL,
                }
                else base_conv1_mode
            ),
        }
    )
    wrapped_model.from_scratch_init_metadata = event
    return event


def apply_head_initialization(module: nn.Module, *, mode: str) -> dict:
    event = _apply_policy_preserving_rng(module, mode, "head")
    module.from_scratch_init_metadata = event
    return event


def apply_velocity_initialization(module: nn.Module, *, mode: str) -> dict:
    event = _apply_policy_preserving_rng(module, mode, "velocity")
    module.from_scratch_init_metadata = event
    return event


def make_get_model(
    original: Callable,
    *,
    mode: str,
    base_conv1_mode: str,
    emit_event: bool = True,
) -> Callable:
    def wrapped(arch, dataset, init_mode="pretrained"):
        if init_mode != "random":
            raise ValueError("from-scratch initialization requires --init-mode random")
        model, out_size = original(arch, dataset, init_mode)
        if getattr(model, "init_weights_argument", "not-none") is not None:
            raise RuntimeError("torchvision pretrained weights were requested")
        event = apply_backbone_initialization(
            model, dataset=dataset, base_conv1_mode=base_conv1_mode, mode=mode
        )
        if emit_event:
            print(event, flush=True)
        return model, out_size

    wrapped.__name__ = f"get_model_{base_conv1_mode}_{mode}"
    return wrapped


def make_get_head(original: Callable, *, mode: str, emit_event: bool = True) -> Callable:
    def wrapped(out_size, cfg):
        head = original(out_size, cfg)
        event = apply_head_initialization(head, mode=mode)
        if emit_event:
            print(event, flush=True)
        return head

    wrapped.__name__ = f"get_head_{mode}"
    return wrapped


def make_velocity_net(original: Callable, *, mode: str, emit_event: bool = True) -> Callable:
    def wrapped(emb_dim, *args, **kwargs):
        velocity = original(emb_dim, *args, **kwargs)
        event = apply_velocity_initialization(velocity, mode=mode)
        if emit_event:
            print(event, flush=True)
        return velocity

    wrapped.__name__ = f"VelocityNet_{mode}"
    return wrapped
