import torch
from torch import randn, cat, float32
from torch.nn.functional import normalize
from torch.cuda import is_available
from numpy import full
from numpy.random import choice

device = "cuda" if is_available() else "cpu"


def gen_etf_centers(num_centers, dim):
    """
    生成 ETF（等角紧框架）中心，应在训练开始时调用一次后固定。

    K <= dim+1 : 精确单纯形 ETF，任意两中心余弦 = -1/(K-1)
    K >  dim+1 : 精确 ETF 理论上不存在，
                 退化为最优紧框架近似（两两余弦不再完全相等，但尽量均匀分散）

    Args:
        num_centers (int): 中心数量 K，可以远大于 dim
        dim         (int): 特征空间维度 d
    Returns:
        Tensor [K, dim]，每行为单位球面上一个中心
    """
    K = num_centers

    G = torch.full((K, K), fill_value=-1.0 / (K - 1))
    G.fill_diagonal_(1.0)                                   

    eigenvalues, eigenvectors = torch.linalg.eigh(G)

    d_eff = min(dim, K - 1)
    vals = eigenvalues[-d_eff:]                              
    vecs = eigenvectors[:, -d_eff:]                             

    M = vecs * vals.sqrt().unsqueeze(0)                         

    if d_eff < dim:
        Q, _ = torch.linalg.qr(randn(dim, d_eff))                    
        M = M @ Q.T                                           

    centers = normalize(M, dim=1).to(device)
    return centers


def perturbation(template, n, eps):
    dim = template.size(-1)
    perturbation_dir = normalize(randn((n, dim)), dim=1).to(device)
    return normalize(template + eps * perturbation_dir)


def gen_reference(centers, n, eps):
    """
    根据 ETF 中心生成 Batch 规模的参考点池。
    """
    num_centers = centers.size(0)

    base = n // num_centers
    remainder = n % num_centers
    parts = full(shape=num_centers, fill_value=base)
    if remainder > 0:
        for i in choice(a=num_centers, size=remainder, replace=False):
            parts[i] += 1

    reference = []
    for i in range(num_centers):
        num_samples = int(parts[i])
        if num_samples > 0:
            reference.append(perturbation(centers[i], num_samples, eps))

    return cat(reference, dim=0)              
