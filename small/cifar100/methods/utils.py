import math

import torch
import torch.nn.functional as F


def gen_signed_basis(num_centers, dim, device):
    if num_centers > dim:
        raise ValueError("signed_basis requires Kprime <= emb")
    centers = torch.zeros(num_centers, dim, device=device)
    indices = torch.randperm(dim, device=device)[:num_centers]
    centers[torch.arange(num_centers, device=device), indices] = (
        torch.randint(0, 2, (num_centers,), device=device, dtype=torch.float32) * 2 - 1
    )
    return centers


def gen_simplex_projection_centers(num_centers, dim, device):
    """Unit-norm simplex centers, spectrally projected to ``dim`` dimensions.

    For K > dim + 1 this is a tight-frame-style projection, not an exact ETF claim.
    """
    if num_centers < 2:
        raise ValueError("Kprime must be at least 2")
    k = num_centers
    gram = torch.full((k, k), -1.0 / (k - 1), dtype=torch.float32)
    gram.fill_diagonal_(1.0)
    values, vectors = torch.linalg.eigh(gram)
    d_eff = min(dim, k - 1)
    coordinates = vectors[:, -d_eff:] * values[-d_eff:].clamp_min(0).sqrt().unsqueeze(0)
    if d_eff < dim:
        rotation, _ = torch.linalg.qr(torch.randn(dim, d_eff))
        coordinates = coordinates @ rotation.T
    return F.normalize(coordinates, p=2, dim=1).to(device)


def perturbation(centers, eps):
    noise = F.normalize(torch.randn_like(centers), p=2, dim=1)
    return F.normalize(centers + eps * noise, p=2, dim=1)


def repeated_center_pool(centers, batch_size, redundancy=1, extra_repeats=0):
    if redundancy < 1 or extra_repeats < 0:
        raise ValueError("target redundancy must be >= 1 and extra repeats >= 0")
    repeats = math.ceil(batch_size / centers.shape[0]) * redundancy + extra_repeats
    return centers.repeat(repeats, 1)
