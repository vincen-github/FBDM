"""Frozen-trainer launcher for strict paired from-scratch initialization."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from from_scratch_init import (
    VALID_BASE_CONV1_MODES,
    VALID_MODES,
    make_get_head,
    make_get_model,
    make_velocity_net,
)


def pop_single_value(argv: list[str], option: str) -> str | None:
    values: list[str] = []
    retained = [argv[0]]
    index = 1
    while index < len(argv):
        value = argv[index]
        if value == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} requires a value")
            values.append(argv[index + 1])
            index += 2
            continue
        if value.startswith(f"{option}="):
            values.append(value.split("=", 1)[1])
            index += 1
            continue
        retained.append(value)
        index += 1
    if len(values) > 1:
        raise SystemExit(f"provide {option} at most once")
    argv[:] = retained
    return values[0] if values else None


def parse_custom_options(argv: list[str]) -> tuple[str, str, int]:
    mode = pop_single_value(argv, "--scratch-init-mode")
    if mode is None:
        raise SystemExit("provide --scratch-init-mode exactly once")
    if mode not in VALID_MODES:
        raise SystemExit(f"invalid --scratch-init-mode {mode!r}; choose one of {VALID_MODES}")
    base_conv1_mode = pop_single_value(argv, "--base-conv1-mode")
    if base_conv1_mode is None:
        raise SystemExit("provide --base-conv1-mode exactly once")
    if base_conv1_mode not in VALID_BASE_CONV1_MODES:
        raise SystemExit(
            f"invalid --base-conv1-mode {base_conv1_mode!r}; "
            f"choose one of {VALID_BASE_CONV1_MODES}"
        )
    interval_text = pop_single_value(argv, "--diag-component-grad-every-epochs")
    try:
        interval = 0 if interval_text is None else int(interval_text)
    except ValueError as error:
        raise SystemExit("--diag-component-grad-every-epochs must be an integer") from error
    if interval < 0:
        raise SystemExit("--diag-component-grad-every-epochs must be >= 0")
    return mode, base_conv1_mode, interval


def encoder_group(parameter_name: str) -> str:
    name = parameter_name.removeprefix("module.")
    if name.startswith(("conv1.", "bn1.")):
        return "stem"
    for layer in ("layer1", "layer2", "layer3", "layer4"):
        if name.startswith(f"{layer}."):
            return layer
    return "other"


def gradient_triplet_stats(named_parameters, total_loss, align_loss) -> dict:
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    total_grads = torch.autograd.grad(
        total_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    align_grads = torch.autograd.grad(
        align_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    accumulators: dict[str, dict[str, torch.Tensor]] = {}
    device = total_loss.device
    for name, parameter, total_grad, align_grad in zip(
        names, parameters, total_grads, align_grads
    ):
        group = encoder_group(name)
        if group not in accumulators:
            accumulators[group] = {
                key: torch.zeros((), device=device, dtype=torch.float64)
                for key in (
                    "non_alignment_sq",
                    "align_sq",
                    "total_sq",
                    "non_alignment_align_dot",
                )
            }
        total_value = (
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if total_grad is None
            else total_grad
        )
        align_value = (
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if align_grad is None
            else align_grad
        )
        non_alignment_value = total_value - align_value
        target = accumulators[group]
        target["non_alignment_sq"] += non_alignment_value.double().square().sum()
        target["align_sq"] += align_value.double().square().sum()
        target["total_sq"] += total_value.double().square().sum()
        target["non_alignment_align_dot"] += (
            non_alignment_value.double() * align_value.double()
        ).sum()

    for key in ("all",):
        accumulators[key] = {
            metric: sum((value[metric] for group, value in accumulators.items() if group != key), torch.zeros((), device=device, dtype=torch.float64))
            for metric in (
                "non_alignment_sq",
                "align_sq",
                "total_sq",
                "non_alignment_align_dot",
            )
        }

    result = {}
    for group, values in sorted(accumulators.items()):
        non_alignment_norm = values["non_alignment_sq"].sqrt()
        align_norm = values["align_sq"].sqrt()
        denominator = (non_alignment_norm * align_norm).clamp_min(1e-300)
        result[group] = {
            "weighted_non_alignment_norm": float(non_alignment_norm.detach()),
            "weighted_align_norm": float(align_norm.detach()),
            "total_norm": float(values["total_sq"].sqrt().detach()),
            "non_alignment_align_cosine": float(
                (values["non_alignment_align_dot"] / denominator).detach()
            ),
        }
    return result


def install_component_gradient_diagnostic(dm_class, interval: int) -> None:
    original_forward = dm_class.forward

    def forward_with_component_diagnostic(self, samples):
        head_outputs = []

        def capture_head_output(_module, _inputs, output):
            head_outputs.append(output)

        hook = self.head.register_forward_hook(capture_head_output)
        try:
            total = original_forward(self, samples)
        finally:
            hook.remove()

        epoch = int(getattr(self.instrumentation, "epoch", -1))
        already_sampled = getattr(self, "_component_grad_sampled_epoch", None) == epoch
        should_sample = (
            interval > 0
            and epoch >= 0
            and not already_sampled
            and (epoch == 0 or (epoch + 1) % interval == 0)
        )
        if should_sample:
            if self.cfg.alignment_space != "z0":
                raise RuntimeError("component diagnostic currently requires z0 alignment")
            if self.cfg.velocity_mode != "standard":
                raise RuntimeError("component diagnostic requires the standard velocity objective")
            if len(head_outputs) != 2:
                raise RuntimeError(f"expected exactly two online head outputs, got {len(head_outputs)}")
            z0_1 = F.normalize(head_outputs[0], p=2, dim=1)
            z0_2 = F.normalize(head_outputs[1], p=2, dim=1)
            align_weighted = self.cfg.lambda_param * (z0_1 - z0_2).square().mean()
            named_encoder_parameters = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            groups = gradient_triplet_stats(
                named_encoder_parameters,
                total,
                align_weighted,
            )
            print(
                {
                    "event": "component_gradient_diagnostic",
                    "epoch": epoch,
                    "sample_index_within_epoch": 0,
                    "objective_identity": "weighted_non_alignment=total-weighted_alignment",
                    "fm_weight": float(self.cfg.fm_weight),
                    "lambda_param": float(self.cfg.lambda_param),
                    "groups": groups,
                },
                flush=True,
            )
            self._component_grad_sampled_epoch = epoch
        return total

    dm_class.forward = forward_with_component_diagnostic


def main() -> None:
    mode, base_conv1_mode, component_grad_interval = parse_custom_options(sys.argv)

    import cfg as cfg_module
    import methods.base as base_module
    import methods.dm as dm_module
    import model as model_module

    original_get_cfg = cfg_module.get_cfg
    original_get_model = model_module.get_model
    original_get_head = model_module.get_head
    original_velocity_net = model_module.VelocityNet

    def get_cfg_with_initialization():
        parsed = original_get_cfg()
        parsed.scratch_init_mode = mode
        parsed.base_conv1_mode = base_conv1_mode
        parsed.diag_component_grad_every_epochs = component_grad_interval
        return parsed

    paired_get_model = make_get_model(
        original_get_model,
        mode=mode,
        base_conv1_mode=base_conv1_mode,
        emit_event=True,
    )
    paired_get_head = make_get_head(original_get_head, mode=mode, emit_event=True)
    paired_velocity_net = make_velocity_net(
        original_velocity_net, mode=mode, emit_event=True
    )
    cfg_module.get_cfg = get_cfg_with_initialization
    model_module.get_model = paired_get_model
    model_module.get_head = paired_get_head
    model_module.VelocityNet = paired_velocity_net
    base_module.get_model = paired_get_model
    base_module.get_head = paired_get_head
    dm_module.VelocityNet = paired_velocity_net
    install_component_gradient_diagnostic(dm_module.DM, component_grad_interval)

    import train as train_module

    train_module.get_cfg = get_cfg_with_initialization
    print(
        {
            "event": "from_scratch_init_launcher",
            "status": "ok",
            "scratch_init_mode": mode,
            "base_conv1_mode": base_conv1_mode,
            "diag_component_grad_every_epochs": component_grad_interval,
            "frozen_trainer_module": str(train_module.__file__),
        },
        flush=True,
    )
    train_module.main()


if __name__ == "__main__":
    main()
