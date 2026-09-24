import json
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import mean
from torch import load, save
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from cfg import get_cfg
from instrumentation import EpochInstrumentation
from datasets import get_ds
from methods import get_method
from methods.utils import gen_signed_basis, gen_simplex_projection_centers


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parse_capacity_schedule(values, batch_size, num_centers):
    """Parse an epoch-to-slots schedule without changing the static code path."""
    if not values:
        return []
    minimum_slots = math.ceil(batch_size / num_centers)
    schedule = []
    for value in values:
        try:
            epoch_text, slots_text = value.split(":", 1)
            epoch = int(epoch_text)
            slots = int(slots_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid capacity schedule entry {value!r}; expected EPOCH:SLOTS"
            ) from error
        if epoch < 0:
            raise ValueError("capacity schedule epochs must be non-negative")
        if slots < minimum_slots:
            raise ValueError(
                f"capacity schedule slots must be >= ceil(bs/Kprime)={minimum_slots}"
            )
        if schedule and epoch <= schedule[-1][0]:
            raise ValueError("capacity schedule epochs must be strictly increasing")
        schedule.append((epoch, slots))
    if schedule[0][0] != 0:
        raise ValueError("capacity schedule must start at epoch 0")
    return schedule


def capacity_for_epoch(schedule, epoch):
    active = schedule[0][1]
    for milestone, slots in schedule[1:]:
        if epoch < milestone:
            break
        active = slots
    return active


def build_optimizer_groups(model, cfg):
    """Partition encoder and auxiliary parameters while preserving control numerics."""
    effective_lrs = {
        "encoder": cfg.lr * cfg.encoder_lr_multiplier,
        "head_velocity": cfg.lr * cfg.head_velocity_lr_multiplier,
    }
    if any(value < 0 for value in effective_lrs.values()):
        raise ValueError("optimizer LR multipliers must be non-negative")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder = [parameter for parameter in model.model.parameters() if parameter.requires_grad]
    encoder_ids = {id(parameter) for parameter in encoder}
    head_velocity = [parameter for parameter in trainable if id(parameter) not in encoder_ids]
    if not encoder or not head_velocity:
        raise RuntimeError("expected non-empty encoder and head/velocity parameter groups")
    if len(encoder) + len(head_velocity) != len(trainable):
        raise RuntimeError("optimizer parameter partition is incomplete or overlapping")

    parameter_counts = {
        "encoder": sum(parameter.numel() for parameter in encoder),
        "head_velocity": sum(parameter.numel() for parameter in head_velocity),
    }
    if effective_lrs["encoder"] == cfg.lr and effective_lrs["head_velocity"] == cfg.lr:
        return trainable, [cfg.lr], ["all"], parameter_counts, effective_lrs

    groups = [
        {"params": encoder, "lr": effective_lrs["encoder"]},
        {"params": head_velocity, "lr": effective_lrs["head_velocity"]},
    ]
    return (
        groups,
        [effective_lrs["encoder"], effective_lrs["head_velocity"]],
        ["encoder", "head_velocity"],
        parameter_counts,
        effective_lrs,
    )


def build_selective_decay_groups(model, cfg):
    """Apply coupled L2 only to convolution and linear weight tensors."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if cfg.adam_decay_scope == "all_modules":
        roots = [model]
    elif cfg.adam_decay_scope == "encoder_head":
        roots = [model.model, model.head]
    elif cfg.adam_decay_scope == "encoder_only":
        roots = [model.model]
    else:
        raise ValueError(f"unknown selective decay scope: {cfg.adam_decay_scope}")
    decay_ids = set()
    for root in roots:
        for module in root.modules():
            if isinstance(module, (_ConvNd, nn.Linear)):
                weight = getattr(module, "weight", None)
                if weight is not None and weight.requires_grad:
                    decay_ids.add(id(weight))

    decay = [parameter for parameter in trainable if id(parameter) in decay_ids]
    no_decay = [parameter for parameter in trainable if id(parameter) not in decay_ids]
    if not decay or not no_decay:
        raise RuntimeError("selective decay requires non-empty decay and no-decay groups")
    if len(decay) + len(no_decay) != len(trainable):
        raise RuntimeError("selective decay parameter partition is incomplete")
    if {id(parameter) for parameter in decay}.intersection(
        id(parameter) for parameter in no_decay
    ):
        raise RuntimeError("selective decay parameter partition overlaps")

    groups = [
        {"params": decay, "lr": cfg.lr, "weight_decay": cfg.adam_l2},
        {"params": no_decay, "lr": cfg.lr, "weight_decay": 0.0},
    ]
    parameter_counts = {
        "decay": sum(parameter.numel() for parameter in decay),
        "no_decay": sum(parameter.numel() for parameter in no_decay),
    }
    effective_lrs = {"decay": cfg.lr, "no_decay": cfg.lr}
    return groups, [cfg.lr, cfg.lr], ["decay", "no_decay"], parameter_counts, effective_lrs


def configure_batchnorm_momentum(model, momentum):
    if not 0.0 < momentum <= 1.0:
        raise ValueError("bn_momentum must be in (0, 1]")
    modules = [module for module in model.modules() if isinstance(module, _BatchNorm)]
    if not modules:
        raise RuntimeError("expected at least one BatchNorm module")
    for module in modules:
        module.momentum = momentum
    return modules


def centralize_gradients(parameters):
    """Center output-channel gradients for matrix/convolution parameters."""
    centralized = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None or gradient.ndim <= 1:
            continue
        dimensions = tuple(range(1, gradient.ndim))
        gradient.sub_(gradient.mean(dim=dimensions, keepdim=True))
        centralized += 1
    return centralized


def main():
    cfg = get_cfg()
    capacity_schedule = parse_capacity_schedule(
        cfg.capacity_schedule,
        cfg.bs,
        cfg.Kprime,
    )
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers)
    device = torch.device("cuda")
    if cfg.centers_file:
        centers_payload = load(cfg.centers_file, map_location="cpu")
        centers = (
            centers_payload["centers"]
            if isinstance(centers_payload, dict)
            else centers_payload
        )
        if not isinstance(centers, torch.Tensor):
            raise TypeError("centers-file must contain a tensor or a dict with key 'centers'")
        if tuple(centers.shape) != (cfg.Kprime, cfg.emb):
            raise ValueError(
                f"centers-file shape {tuple(centers.shape)} does not match "
                f"(Kprime, emb)=({cfg.Kprime}, {cfg.emb})"
            )
        if not torch.isfinite(centers).all():
            raise ValueError("centers-file contains non-finite values")
        centers = centers.float().contiguous().to(device)
        centers_source = str(Path(cfg.centers_file).resolve())
    else:
        centers = (
            gen_simplex_projection_centers(cfg.Kprime, cfg.emb, device)
            if cfg.prior == "etf"
            else gen_signed_basis(cfg.Kprime, cfg.emb, device)
        )
        centers_source = "generated"
    centers_sha256 = hashlib.sha256(
        centers.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    model = get_method(cfg.method)(cfg, centers).cuda().train()
    if cfg.fname:
        checkpoint = load(cfg.fname, map_location="cuda")
        state_dict = checkpoint.get("model", checkpoint)
        if hasattr(model, "load_checkpoint_state"):
            model.load_checkpoint_state(state_dict)
        else:
            model.load_state_dict(state_dict)
    seed_everything(cfg.seed)
    batchnorm_modules = configure_batchnorm_momentum(model, cfg.bn_momentum)
    if not 0.0 <= cfg.adam_beta1 < 1.0:
        raise ValueError("adam_beta1 must be in [0, 1)")
    if not 0.0 <= cfg.adam_beta2 < 1.0:
        raise ValueError("adam_beta2 must be in [0, 1)")
    if cfg.adam_amsgrad and cfg.optimizer_name != "adam":
        raise ValueError("AMSGrad is supported only by the Adam policy")
    if cfg.adam_decay_mode == "selective":
        optimizer_groups = build_selective_decay_groups(model, cfg)
    else:
        optimizer_groups = build_optimizer_groups(model, cfg)
    optimizer_input, nominal_group_lrs, optimizer_group_names, parameter_counts, effective_lrs = (
        optimizer_groups
    )
    optimizer_class = {
        "adam": optim.Adam,
        "radam": optim.RAdam,
        "nadam": optim.NAdam,
    }[cfg.optimizer_name]
    optimizer_kwargs = {
        "lr": cfg.lr,
        "betas": (cfg.adam_beta1, cfg.adam_beta2),
        "eps": 1e-8,
        "weight_decay": cfg.adam_l2 if cfg.adam_decay_mode == "all" else 0.0,
    }
    if cfg.optimizer_name == "adam":
        optimizer_kwargs["amsgrad"] = cfg.adam_amsgrad
    optimizer = optimizer_class(optimizer_input, **optimizer_kwargs)
    centralized_parameter_count = sum(
        parameter.requires_grad and parameter.ndim > 1
        for parameter in model.parameters()
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.T0 or cfg.epoch,
        T_mult=cfg.Tmult,
        eta_min=cfg.eta_min,
    ) if cfg.lr_step == "cos" else None
    instrumentation = EpochInstrumentation(
        device,
        grad_every_steps=cfg.diag_grad_every_steps,
        grad_every_epochs=cfg.diag_grad_every_epochs,
        natural_every_epochs=cfg.diag_natural_every,
        natural_batches=cfg.diag_natural_batches,
        natural_ode_steps=cfg.diag_natural_ode_steps,
    )
    model.instrumentation = instrumentation
    run_dir = Path("runs") / cfg.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(cfg))
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print({"event": "config", **config}, flush=True)
    print(
        {
            "event": "optimizer_configuration",
            "name": optimizer.__class__.__name__,
            "optimizer_name": cfg.optimizer_name,
            "beta1": cfg.adam_beta1,
            "beta2": cfg.adam_beta2,
            "eps": 1e-8,
            "amsgrad": cfg.adam_amsgrad,
            "gradient_centralization": cfg.gradient_centralization,
            "centralized_parameter_count": centralized_parameter_count,
            "decay_mode": cfg.adam_decay_mode,
            "decay_scope": cfg.adam_decay_scope,
            "groups": [
                {
                    "name": name,
                    "parameter_count": sum(
                        parameter.numel() for parameter in group["params"]
                    ),
                    "weight_decay": group["weight_decay"],
                    "nominal_lr": nominal_lr,
                }
                for name, nominal_lr, group in zip(
                    optimizer_group_names,
                    nominal_group_lrs,
                    optimizer.param_groups,
                )
            ],
        },
        flush=True,
    )
    print(
        {
            "event": "batchnorm_configuration",
            "module_count": len(batchnorm_modules),
            "momentum": cfg.bn_momentum,
            "observed_momenta": sorted(
                {float(module.momentum) for module in batchnorm_modules}
            ),
        },
        flush=True,
    )
    print(
        {
            "event": "centers_source",
            "source": centers_source,
            "tensor_sha256": centers_sha256,
            "shape": list(centers.shape),
        },
        flush=True,
    )
    print({"event": "prototype_stats", **model.prototype_stats()}, flush=True)
    print(
        {
            "event": "backbone_initialization",
            "init_mode": cfg.init_mode,
            "torchvision_weights_argument": getattr(model.model, "init_weights_argument", None),
            "state_sha256": tensor_state_sha256(model.model),
            "random_has_no_pretrained_state": cfg.init_mode == "random",
        },
        flush=True,
    )
    if cfg.eval_at_start:
        instrumentation.begin_eval()
        message = {"event": "eval", "epoch": -1, "stage": "initialization"}
        for representation_index, representation in enumerate(cfg.eval_representations):
            seed_everything(cfg.seed + 900_000_000 + representation_index)
            knn, linear = model.get_acc(ds.clf, ds.test, representation)
            message[f"{representation}_linear"] = linear[1]
            message[f"{representation}_knn"] = knn
        print(message, flush=True)
        instrumentation.end_eval()
    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    warmup = 0
    previous_capacity = None
    for epoch in range(cfg.epoch):
        active_capacity = None
        if capacity_schedule:
            active_capacity = capacity_for_epoch(capacity_schedule, epoch)
            base_repeats = math.ceil(cfg.bs / cfg.Kprime)
            cfg.target_redundancy = 1
            cfg.pool_extra_repeats = active_capacity - base_repeats
            if active_capacity != previous_capacity:
                print(
                    {
                        "event": "capacity_schedule",
                        "epoch": epoch,
                        "slots_per_center": active_capacity,
                        "target_redundancy": cfg.target_redundancy,
                        "pool_extra_repeats": cfg.pool_extra_repeats,
                    },
                    flush=True,
                )
                previous_capacity = active_capacity
        instrumentation.begin_epoch(epoch)
        instrumentation.begin_train()
        losses = []
        num_iterations = len(ds.train)
        model.reset_epoch_stats(num_iterations)
        for iteration, (samples, _) in enumerate(tqdm(ds.train, desc=f"epoch {epoch}")):
            if warmup < cfg.warmup_steps:
                if optimizer_group_names == ["all"]:
                    optimizer.param_groups[0]["lr"] = (
                        cfg.lr * (warmup + 1) / cfg.warmup_steps
                    )
                else:
                    fraction = (warmup + 1) / cfg.warmup_steps
                    for group, nominal_lr in zip(optimizer.param_groups, nominal_group_lrs):
                        group["lr"] = nominal_lr * fraction
                warmup += 1
            elif scheduler:
                scheduler.step(epoch + iteration / len(ds.train))
                cosine_drops = sum(
                    epoch >= milestone for milestone in cfg.cos_step_milestones
                )
                if cosine_drops:
                    factor = cfg.drop_gamma ** cosine_drops
                    for group in optimizer.param_groups:
                        group["lr"] *= factor
            elif cfg.lr_step == "step":
                drops = sum(epoch >= milestone for milestone in cfg.step_milestones)
                if optimizer_group_names == ["all"]:
                    optimizer.param_groups[0]["lr"] = cfg.lr * (cfg.drop_gamma ** drops)
                else:
                    for group, nominal_lr in zip(optimizer.param_groups, nominal_group_lrs):
                        group["lr"] = nominal_lr * (cfg.drop_gamma ** drops)
            optimizer.zero_grad(set_to_none=True)
            loss = model(samples)
            loss.backward()
            if cfg.gradient_centralization:
                observed_count = centralize_gradients(model.parameters())
                if observed_count != centralized_parameter_count:
                    raise RuntimeError("gradient centralization parameter count drifted")
            if instrumentation.should_sample_grad(iteration, num_iterations):
                parameter_groups = {
                    "encoder": model.model.parameters(),
                    "head": model.head.parameters(),
                }
                if hasattr(model, "v_net"):
                    parameter_groups["v_net"] = model.v_net.parameters()
                if getattr(model, "delta_v_net", None) is not None:
                    parameter_groups["delta_v_net"] = model.delta_v_net.parameters()
                instrumentation.observe_gradients(parameter_groups)
            optimizer.step()
            if hasattr(model, "update_assignment_teacher"):
                model.update_assignment_teacher()
            loss_value = loss.item()
            instrumentation.observe_loss(loss_value)
            losses.append(loss_value)
        instrumentation.end_train()
        print(
            {
                "event": "train",
                "epoch": epoch,
                "loss": mean(losses),
                "lr": optimizer.param_groups[0]["lr"],
                "lr_groups": {
                    name: group["lr"]
                    for name, group in zip(optimizer_group_names, optimizer.param_groups)
                },
                "optimizer_parameter_counts": parameter_counts,
                "optimizer_effective_lrs": effective_lrs,
                "target_capacity": active_capacity,
                **{k: float(v) for k, v in model.last_losses.items()},
                **model.assignment_stats(),
            },
            flush=True,
        )
        should_eval = (
            (epoch + 1) % cfg.eval_every == 0
            or (epoch + 1) in cfg.extra_eval_epochs
        )
        if should_eval:
            instrumentation.begin_eval()
            message = {"event": "eval", "epoch": epoch}
            for representation_index, representation in enumerate(cfg.eval_representations):
                seed_everything(cfg.seed + 10_000 * (epoch + 1) + representation_index)
                knn, linear = model.get_acc(ds.clf, ds.test, representation)
                message[f"{representation}_linear"] = linear[1]
                message[f"{representation}_knn"] = knn
            print(message, flush=True)
            instrumentation.end_eval()
        should_save = (
            (cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0)
            or (epoch + 1) in cfg.extra_save_epochs
        )
        if should_save:
            save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                },
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
            )
        timing, stability = instrumentation.end_epoch()
        print({"event": "timing", "epoch": epoch, **timing}, flush=True)
        print({"event": "stability", "epoch": epoch, **stability}, flush=True)


if __name__ == "__main__":
    main()
