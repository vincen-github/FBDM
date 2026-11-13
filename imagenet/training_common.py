"""Distributed model setup and data augmentation for ImageNet training."""
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
from gaussian_blur import replace_blur


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


def prepare_ddp(model):
    model._ddp_params_and_buffers_to_ignore = {
        name for name, buf in model.named_buffers() if buf.numel() == 0}
    return model

