"""Isolated ImageNet-1K FBDM DDP smoke/throughput gate; NOT a formal trainer.

Use frozen_source and imagenet_hdf5 on PYTHONPATH. One torchrun process/GPU.
Global assignment is performed once on rank zero over the global batch.
This intentionally cannot launch long training or produce accuracy records.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import transforms as T

from imagenet_hdf5.dataset import ImageNetHDF5
from methods.dm import DM
from methods.utils import gen_simplex_projection_centers
from blur_equivalent_20260910 import replace_blur


class TwoViews:
    def __init__(self, cfg):
        self.transform = T.Compose([
            T.RandomResizedCrop(224, scale=(cfg.crop_s0, cfg.crop_s1)),
            T.RandomHorizontalFlip(cfg.hf_prob),
            T.RandomApply([T.ColorJitter(cfg.cj_bright, cfg.cj_contrast,
                                      cfg.cj_sat, cfg.cj_hue)], p=cfg.cj_prob),
            T.RandomGrayscale(cfg.gs_prob),
            T.RandomApply([T.GaussianBlur(23, sigma=(getattr(cfg, 'blur_sigma_min', 0.1), 2.0))], p=cfg.blur_prob),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if getattr(cfg, 'input_normalization', True) else T.Lambda(lambda x:x),
        ])
        replace_blur(self.transform, getattr(cfg, 'gaussian_blur_impl', 'original'))

    def __call__(self, image):
        return [self.transform(image), self.transform(image)]


class GlobalAssignmentDM(DM):
    @torch.no_grad()
    def assign_targets(self, z1, z2):
        world, rank = dist.get_world_size(), dist.get_rank()
        assert z1.shape[0] * world == self.cfg.bs
        parts1 = [torch.empty_like(z1) for _ in range(world)]
        parts2 = [torch.empty_like(z2) for _ in range(world)]
        dist.all_gather(parts1, z1.detach().contiguous())
        dist.all_gather(parts2, z2.detach().contiguous())
        if rank == 0:
            with torch.autocast("cuda", enabled=False):
                targets = super().assign_targets(torch.cat(parts1).float(),
                                                 torch.cat(parts2).float())
            targets = torch.stack(targets).contiguous()
        else:
            targets = torch.empty((3, self.cfg.bs, self.cfg.emb),
                                  dtype=torch.float32, device=z1.device)
        dist.broadcast(targets, src=0)
        start = rank * z1.shape[0]
        return tuple(t[start:start + z1.shape[0]] for t in targets)


def finite(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def code_hashes():
    root = Path(__file__).resolve().parent
    files = [Path(__file__).resolve()]
    files.append(root/'blur_equivalent_20260910.py')
    files.append(root/'training_monitor.py')
    files.extend([root/'optimizer_recipe.py',root/'upstream_lars.py'])
    files.extend(p for p in (root/'imagenet_formal100_20260909.py', root/'imagenet_linear_eval_20260909.py') if p.exists())
    for folder in ('frozen_source', 'imagenet_hdf5'):
        files.extend(sorted((root/folder).rglob('*.py')))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def prepare_ddp(model):
    model._ddp_params_and_buffers_to_ignore = {
        name for name, buf in model.named_buffers() if buf.numel() == 0}
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=320)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    p.add_argument('--gaussian-blur-impl', choices=('original','separable'), default='original')
    p.add_argument('--test-inline-eval', action='store_true')
    p.add_argument('--val-manifest')
    p.add_argument('--optimizer-recipe',choices=('native','simclr_lars'),default='native')
    p.add_argument('--lr-stress',action='store_true')
    p.add_argument("--steps-per-epoch", type=int, default=16)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--global-batch", type=int, default=512)
    p.add_argument("--time-map", choices=("frontload_rational", "frontload_exponential"), default="frontload_rational")
    p.add_argument("--time-map-param", type=float, default=0.5)
    args = p.parse_args()
    assert 2 <= args.steps_per_epoch <= 128, "smoke only, not a formal trainer"
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    assert world in (1, 2, 4, 8, 16) and args.global_batch % world == 0
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    cfg = json.loads(Path(args.baseline).read_text())
    capacity_scale = max(1, args.global_batch/512)
    capacities = [math.ceil(c*capacity_scale) for c in (5,7,9)]
    cfg.update(dataset="imagenet", arch="resnet50", init_mode="random",
               Kprime=args.k, emb=args.dim, bs=args.global_batch, centers_file=None,
               blur_prob=0.5, blur_kernel_size=23, gaussian_blur_impl=args.gaussian_blur_impl, seed=0,
               crop_s0=0.08, cj_bright=0.8, cj_contrast=0.8, cj_sat=0.8,
               cj_hue=0.2, gs_prob=0.2, epoch=100, T0=100,
               capacity_schedule=[f'{e}:{c}' for e,c in zip((0,10,25),capacities)],
               time_map=args.time_map, time_map_param=args.time_map_param, time_sampling='path_uniform',
               cos_step_milestones=[], warmup_steps=10*(1281167//args.global_batch),
               target_redundancy=1, pool_extra_repeats=capacities[0]-math.ceil(args.global_batch/args.k))
    if args.optimizer_recipe=='simclr_lars':
        cfg.update(optimizer_name='lars',lr=.6,recipe_weight_decay=1e-6,recipe_momentum=.9,
                   recipe_trust_coefficient=.001,recipe_source='AndrewAtanov/simclr-pytorch@6493ecc3d0512b171d14de45b14dd3d8b726a5c4')
    cfg = SimpleNamespace(**cfg)
    assert cfg.pool_extra_repeats >= 0
    assert not cfg.assignment_epoch_global and cfg.assignment_queue_size == 0
    assert cfg.assignment_feature_source == "online" and cfg.velocity_mode == "standard"
    assert all(getattr(cfg, key) == 0 for key in (
        "soft_occupancy_weight", "z0_variance_weight", "z0_covariance_weight",
        "z0_uniformity_weight", "z0_crosscorr_weight", "backbone_covariance_weight",
        "backbone_variance_weight", "backbone_alignment_weight"))
    out = Path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260909)
        centers = gen_simplex_projection_centers(args.k, args.dim, "cpu")
    model = GlobalAssignmentDM(cfg, centers)
    assert model.model.init_weights_argument is None
    assert model.model.init_mode == "random"
    model.model = model.model.module
    assert tuple(model.model.conv1.weight.shape) == (64, 3, 7, 7)
    assert tuple(model.model.conv1.stride) == (2, 2)
    assert model.out_size == 2048
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(local_rank)
    model.train()
    ddp = DistributedDataParallel(prepare_ddp(model), device_ids=[local_rank], broadcast_buffers=False)
    decay_ids = {id(m.weight) for m in model.modules()
                 if isinstance(m, (torch.nn.Linear, torch.nn.modules.conv._ConvNd))}
    groups = [dict(params=[v for v in model.parameters() if id(v) in decay_ids],
                   weight_decay=cfg.adam_l2),
              dict(params=[v for v in model.parameters() if id(v) not in decay_ids],
                   weight_decay=0.0)]
    from optimizer_recipe import make_optimizer,learning_rate
    optimizer = make_optimizer(model,cfg)
    metadata = json.loads(Path(args.manifest).read_text())
    assert metadata["num_samples"] == 1281167
    assert len(metadata["class_to_idx"]) == 1000
    dataset = ImageNetHDF5(args.manifest, transform=TwoViews(cfg))
    generator = torch.Generator().manual_seed(20260909)
    indices = torch.randperm(len(dataset), generator=generator)[:cfg.bs*args.steps_per_epoch]
    subset = Subset(dataset, indices.tolist())
    sampler = DistributedSampler(subset, shuffle=True, seed=0, drop_last=True)
    loader = DataLoader(subset, batch_size=cfg.bs//world, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        worker_init_fn=seed_worker, persistent_workers=args.workers > 0)
    evidence = dict(config=vars(cfg), code_hashes=code_hashes(), weights=None, pretrained=False,
                    method="FBDM (native class named DM)", encoder="ResNet-50",
                    world_size=world, global_batch=cfg.bs, bn="SyncBatchNorm",
                    assignment="rank-zero global batch native Hungarian",
                    precision=args.precision, full_dataset_samples=len(dataset),
                    smoke_subset_samples=len(subset), epoch_kind="subset smoke, not full epoch",
                    manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                    torch_version=torch.__version__, adam_eps=1e-8,
                    transform=repr(dataset.transform.transform))
    if rank == 0:
        torch.save(centers, out / "centers.pt")
        evidence["centers_sha256"] = hashlib.sha256((out / "centers.pt").read_bytes()).hexdigest()
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "amp", init_scale=128.0)
    torch.manual_seed(1000 + rank)
    measurements = []
    for epoch in range(2):
        sampler.set_epoch(epoch)
        model.reset_epoch_stats(len(loader))
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        total_loss = 0.0
        data_wait = compute_time = 0.0
        step_end = start
        for step, (views, _) in enumerate(loader):
            step_start = time.perf_counter()
            data_wait += step_start - step_end
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr if args.lr_stress else learning_rate(cfg,epoch*len(loader)+step,100*(1281167//cfg.bs))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=args.precision == "amp"):
                loss = ddp(views)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            ok = ((~torch.stack([(~torch.isfinite(p.grad)).any() for p in model.parameters()
                                if p.grad is not None]).any()) & torch.isfinite(loss)).int()
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if not bool(ok):
                raise RuntimeError("nonfinite gradients on at least one rank")
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            torch.cuda.synchronize()
            step_end = time.perf_counter()
            compute_time += step_end - step_start
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        values = torch.tensor([duration, torch.cuda.max_memory_allocated()/2**30,
                               torch.cuda.max_memory_reserved()/2**30], device=local_rank)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        loss_value = torch.tensor(total_loss / len(loader), device=local_rank)
        dist.all_reduce(loss_value)
        record = dict(epoch=epoch, subset_steps=len(loader), seconds=float(values[0]),
                      peak_allocated_GiB=float(values[1]), peak_reserved_GiB=float(values[2]),
                      loss=float(loss_value)/world,lr=optimizer.param_groups[0]['lr'],lr_stress=args.lr_stress)
        components = torch.tensor([data_wait, compute_time], device=local_rank)
        dist.all_reduce(components, op=dist.ReduceOp.MAX)
        record.update(max_rank_data_wait_seconds=float(components[0]),
                      max_rank_training_seconds=float(components[1]),
                      note='Training includes H2D, forward/backward, assignment and communication; component maxima may come from different ranks')
        measurements.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scaler=scaler.state_dict(), centers=centers, config=vars(cfg), epoch=1)
    ok = torch.tensor(int(finite(state)), device=local_rank)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    assert bool(ok), "nonfinite final checkpoint state"
    if rank == 0:
        torch.save(state, out / "smoke_checkpoint.pt")
        assert finite(torch.load(out / "smoke_checkpoint.pt", map_location="cpu"))
        (out / "completion.json").write_text(json.dumps(dict(status="ok", measurements=measurements,
            checkpoint_sha256=hashlib.sha256((out / "smoke_checkpoint.pt").read_bytes()).hexdigest(),
            warning="Two subset smoke epochs only; not full-epoch benchmark or formal result"), indent=2))
    if args.test_inline_eval:
        from training_monitor import evaluate
        assert args.val_manifest
        evaluate(model.model,args.manifest,args.val_manifest,out/'eval_gate',epoch=0,
                 workers=args.workers,per_class=4,probe_epochs=2)
        assert all(m.training for m in model.model.modules())
        optimizer.zero_grad(set_to_none=True)
        views,_=next(iter(loader))
        with torch.autocast('cuda',enabled=args.precision=='amp'):
            loss=ddp(views)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        ok=((~torch.stack([(~torch.isfinite(p.grad)).any() for p in model.parameters()
                          if p.grad is not None]).any()) & torch.isfinite(loss)).int()
        dist.all_reduce(ok,op=dist.ReduceOp.MIN)
        assert bool(ok),'nonfinite post-eval training gradient'
        scaler.step(optimizer);scaler.update()
        ok=torch.tensor(int(finite(model.state_dict()) and finite(optimizer.state_dict())),device=local_rank)
        dist.all_reduce(ok,op=dist.ReduceOp.MIN)
        assert bool(ok),'nonfinite post-eval training state'
        if rank==0:(out/'post_eval_train_gate.json').write_text(json.dumps(dict(status='ok',loss=float(loss))))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
