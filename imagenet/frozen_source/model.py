import torch
import torch.nn as nn
from torchvision import models


def get_head(out_size, cfg):
    """ creates projection head g() from config """
    x = []
    in_size = out_size
    for _ in range(cfg.head_layers - 1):
        hidden = nn.Linear(in_size, cfg.head_size, bias=True)
        if cfg.head_hidden_bias == "disabled":
            hidden.register_parameter("bias", None)
        x.append(hidden)
        if cfg.add_bn:
            x.append(nn.BatchNorm1d(cfg.head_size))
        x.append(nn.ReLU())
        in_size = cfg.head_size
    output = nn.Linear(in_size, cfg.emb, bias=True)
    if cfg.head_output_bias == "disabled":
        output.register_parameter("bias", None)
    x.append(output)
    if cfg.head_output_norm in {"bn_affine", "bn_noaffine"}:
        x.append(
            nn.BatchNorm1d(
                cfg.emb,
                affine=cfg.head_output_norm == "bn_affine",
            )
        )
    elif cfg.head_output_norm == "ln_noaffine":
        x.append(nn.LayerNorm(cfg.emb, elementwise_affine=False))
    elif cfg.head_output_norm != "none":
        raise ValueError(f"unknown projector output normalization: {cfg.head_output_norm}")
    return nn.Sequential(*x)


def get_model(arch, dataset, init_mode="pretrained", stl_maxpool="enabled"):
    """ creates encoder E() by name and modifies it for dataset """
    if init_mode == "pretrained":
        weights = "IMAGENET1K_V1"
    elif init_mode == "random":
        weights = None
    else:
        raise ValueError(f"unknown init mode: {init_mode}")
    model = getattr(models, arch)(weights=weights)
    if dataset != "imagenet":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if dataset == "cifar10" or dataset == "cifar100":
        model.maxpool = nn.Identity()
    if dataset == "stl10" and stl_maxpool == "disabled":
        model.maxpool = nn.Identity()
    out_size = model.fc.in_features
    model.fc = nn.Identity()

    model = nn.DataParallel(model)
    model.init_mode = init_mode
    model.init_weights_argument = weights
    return model, out_size

class VelocityNet(nn.Module):
    """Flow Matching 必须的速度网络，预测在时间 t 时点 z 的速度向量"""
    def __init__(
        self,
        emb_dim,
        hidden=2048,
        layers=8,
        time_feature_mode="raw",
        hidden_bias="enabled",
        output_bias="enabled",
    ):
        super().__init__()
        if hidden <= 0:
            raise ValueError("velocity hidden size must be positive")
        if layers < 2:
            raise ValueError("velocity network needs at least two linear layers")
        if time_feature_mode not in {"raw", "centered"}:
            raise ValueError(f"unknown time feature mode: {time_feature_mode}")
        if hidden_bias not in {"enabled", "disabled"}:
            raise ValueError(f"unknown velocity hidden bias mode: {hidden_bias}")
        if output_bias not in {"enabled", "disabled"}:
            raise ValueError(f"unknown velocity output bias mode: {output_bias}")
        self.hidden = hidden
        self.layers = layers
        self.time_feature_mode = time_feature_mode

        def make_hidden(in_features):
            linear = nn.Linear(in_features, hidden, bias=True)
            if hidden_bias == "disabled":
                linear.register_parameter("bias", None)
            return linear

        modules = [make_hidden(emb_dim + 1), nn.ReLU()]
        for _ in range(layers - 2):
            modules.extend((make_hidden(hidden), nn.ReLU()))
        output = nn.Linear(hidden, emb_dim, bias=True)
        if output_bias == "disabled":
            output.register_parameter("bias", None)
        modules.append(output)
        self.v = nn.Sequential(*modules)

    def forward(self, z, t):
        if self.time_feature_mode == "centered":
            t = 2.0 * t - 1.0
        return self.v(torch.cat([z, t], dim=-1))
