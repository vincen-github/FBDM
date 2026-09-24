import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from eval.get_data import get_data
from eval.knn import eval_knn
from eval.sgd import eval_sgd
from model import VelocityNet
from velocity_constraints import bound_velocity, energy_penalty

from .base import BaseMethod
from .utils import perturbation, repeated_center_pool


EPS = 1e-6


class AssignmentTeacher(torch.nn.Module):
    """EMA encoder/head used only to produce stop-gradient assignment features."""

    def __init__(self, online_model, online_head):
        super().__init__()
        self.model = copy.deepcopy(online_model)
        self.head = copy.deepcopy(online_head)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        return F.normalize(self.head(self.model(x)), p=2, dim=1)


def slerp(z0, z1, s):
    cosine = (z0 * z1).sum(dim=1, keepdim=True)
    small = cosine > 1.0 - 1e-5
    omega = torch.acos(cosine.clamp(-1.0 + EPS, 1.0 - EPS))
    sine = torch.sin(omega).clamp_min(EPS)
    geodesic = (torch.sin((1.0 - s) * omega) / sine) * z0 + (torch.sin(s * omega) / sine) * z1
    nlerp = F.normalize((1.0 - s) * z0 + s * z1, p=2, dim=1)
    return torch.where(small, nlerp, geodesic)


def slerp_velocity(z0, z1, s, s_dot):
    cosine = (z0 * z1).sum(dim=1, keepdim=True)
    small = cosine > 1.0 - 1e-5
    omega = torch.acos(cosine.clamp(-1.0 + EPS, 1.0 - EPS))
    sine = torch.sin(omega).clamp_min(EPS)
    tangent = (-omega * torch.cos((1.0 - s) * omega) / sine) * z0
    tangent = tangent + (omega * torch.cos(s * omega) / sine) * z1

    y = (1.0 - s) * z0 + s * z1
    y_norm = y.norm(p=2, dim=1, keepdim=True).clamp_min(EPS)
    nlerp = y / y_norm
    direction = z1 - z0
    nlerp_tangent = (direction - (nlerp * direction).sum(dim=1, keepdim=True) * nlerp) / y_norm
    velocity = torch.where(small, nlerp_tangent, tangent)
    return s_dot * velocity


def tangent_projection(velocity, position):
    """Project a velocity vector onto the tangent plane at a unit position."""
    return velocity - (velocity * position).sum(dim=1, keepdim=True) * position


def radial_velocity_mse(velocity, position):
    """Mean squared radial vector, using the same per-coordinate scale as FM MSE."""
    radial = (velocity * position).sum(dim=1, keepdim=True) * position
    return radial.square().mean()


def path_coordinate(t, time_map, time_map_param=2.0):
    if time_map == "linear":
        return t, torch.ones_like(t)
    if time_map == "frontload_quadratic":
        return 1.0 - (1.0 - t).square(), 2.0 * (1.0 - t)
    if time_map == "frontload_power":
        one_minus_t = (1.0 - t).clamp_min(0.0)
        return (
            1.0 - one_minus_t.pow(time_map_param),
            time_map_param * one_minus_t.pow(time_map_param - 1.0),
        )
    if time_map == "frontload_exponential":
        denominator = -math.expm1(-time_map_param)
        return (
            -torch.expm1(-time_map_param * t) / denominator,
            time_map_param * torch.exp(-time_map_param * t) / denominator,
        )
    if time_map == "frontload_rational":
        denominator = 1.0 + time_map_param * t
        return (
            (1.0 + time_map_param) * t / denominator,
            (1.0 + time_map_param) / denominator.square(),
        )
    if time_map == "frontload_sine":
        sine_alpha = math.sin(time_map_param)
        return (
            torch.sin(time_map_param * t) / sine_alpha,
            (
                time_map_param
                * torch.cos(time_map_param * t)
                / sine_alpha
            ).clamp_min(0.0),
        )
    raise ValueError(f"unknown time map: {time_map}")


def invert_path_coordinate(s, time_map, time_map_param=2.0):
    if time_map == "linear":
        return s
    if time_map == "frontload_quadratic":
        return 1.0 - torch.sqrt((1.0 - s).clamp_min(0.0))
    if time_map == "frontload_power":
        return 1.0 - (1.0 - s).clamp_min(0.0).pow(1.0 / time_map_param)
    if time_map == "frontload_exponential":
        denominator = -math.expm1(-time_map_param)
        return -torch.log1p(-s * denominator) / time_map_param
    if time_map == "frontload_rational":
        return s / (1.0 + time_map_param - time_map_param * s)
    if time_map == "frontload_sine":
        return (
            torch.asin(
                (s * math.sin(time_map_param)).clamp(0.0, 1.0)
            )
            / time_map_param
        )
    raise ValueError(f"unknown time map: {time_map}")


def sample_time(
    batch_size,
    device,
    dtype,
    time_sampling,
    time_map,
    time_map_param=2.0,
    num_samples=1,
):
    if num_samples==1:
        u = torch.rand(batch_size, 1, device=device, dtype=dtype)
    else:
        assert num_samples==2
        u = ((torch.rand(2,batch_size,1,device=device,dtype=dtype)+torch.arange(2,device=device,dtype=dtype).view(2,1,1))/2).reshape(2*batch_size,1)
    if time_sampling == "time_uniform":
        return u
    if time_sampling == "path_uniform":
        return invert_path_coordinate(u, time_map, time_map_param)
    raise ValueError(
        f"unsupported time sampling/map pair: {time_sampling}/{time_map}"
    )


def rational_path_speed_squared_moment(gamma, time_map_param):
    """Return E_q[(s_dot^2)^gamma] for uniform path coordinate s.

    For s(t)=(1+a)t/(1+at), path-uniform sampling makes s uniform and
    s_dot=(1+a-a*s)^2/(1+a). The expectation is analytic, avoiding noisy
    minibatch renormalization.
    """
    gamma = float(gamma)
    a = float(time_map_param)
    if not math.isfinite(gamma) or not math.isfinite(a) or a <= 0.0:
        raise ValueError("finite gamma and positive rational time-map parameter required")
    exponent = 4.0 * gamma + 1.0
    if abs(exponent) < 1e-12:
        integral = math.log1p(a) / a
    else:
        integral = ((1.0 + a) ** exponent - 1.0) / (a * exponent)
    moment = integral / ((1.0 + a) ** (2.0 * gamma))
    if not math.isfinite(moment) or moment <= 0.0:
        raise ValueError("invalid analytic rational path-speed moment")
    return moment


def normalized_path_speed_weight(s_dot, gamma, time_map_param):
    gamma = float(gamma)
    if gamma == 0.0:
        return torch.ones_like(s_dot)
    moment = rational_path_speed_squared_moment(gamma, time_map_param)
    return s_dot.square().pow(gamma) / moment


def flow_matching_mse(prediction, target, s_dot, gamma, time_map_param):
    if float(gamma) == 0.0:
        return (prediction - target).square().mean()
    per_sample_error = (prediction - target).square().mean(dim=1)
    weight = normalized_path_speed_weight(
        s_dot,
        gamma,
        time_map_param,
    ).squeeze(1)
    return (weight * per_sample_error).mean()


def matched_huber(prediction, target, beta=0.1):
    """Huber tails with exactly the same local quadratic curvature as MSE."""
    return 2.0 * beta * F.smooth_l1_loss(
        prediction,
        target,
        reduction="mean",
        beta=beta,
    )


def flow_matching_loss(
    prediction,
    target,
    s_dot,
    gamma,
    time_map_param,
    mode,
):
    if mode == "mse":
        return flow_matching_mse(
            prediction,
            target,
            s_dot,
            gamma,
            time_map_param,
        )
    if mode == "huber_matched":
        if float(gamma) != 0.0:
            raise ValueError("matched Huber FM cannot be combined with time weighting")
        return matched_huber(prediction, target)
    raise ValueError(f"unknown FM loss mode: {mode}")


def scale_free_spherical_features(z):
    """Map unit vectors to per-coordinate unit variance under spherical isotropy."""
    return math.sqrt(z.shape[1]) * z


def variance_floor_loss(z):
    y = scale_free_spherical_features(z)
    std = torch.sqrt(y.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def covariance_loss(z):
    y = scale_free_spherical_features(z)
    y = y - y.mean(dim=0, keepdim=True)
    covariance = y.T @ y / max(1, y.shape[0] - 1)
    squared = covariance.square()
    return (squared.sum() - squared.diagonal().sum()) / covariance.shape[0]


def hyperspherical_uniformity_loss(z):
    distances = torch.pdist(z, p=2).square()
    if not distances.numel():
        return z.new_zeros(())
    return torch.logsumexp(-2.0 * distances, dim=0) - math.log(distances.numel())


def cross_correlation_offdiagonal_loss(z1, z2):
    y1 = (z1 - z1.mean(dim=0, keepdim=True)) / torch.sqrt(
        z1.var(dim=0, unbiased=False, keepdim=True) + 1e-4
    )
    y2 = (z2 - z2.mean(dim=0, keepdim=True)) / torch.sqrt(
        z2.var(dim=0, unbiased=False, keepdim=True) + 1e-4
    )
    correlation = y1.T @ y2 / y1.shape[0]
    squared = correlation.square()
    return (squared.sum() - squared.diagonal().sum()) / correlation.shape[0]


class DM(BaseMethod):
    def __init__(self, cfg, centers):
        super().__init__(cfg)
        self.cfg = cfg
        if cfg.fm_time_weight_gamma != 0.0:
            if cfg.flow_path_mode != "single":
                raise ValueError("FM time weighting is a single-path ablation")
            if cfg.time_sampling != "path_uniform":
                raise ValueError("FM time weighting requires path_uniform sampling")
            if cfg.time_map != "frontload_rational":
                raise ValueError("FM time weighting requires frontload_rational")
            rational_path_speed_squared_moment(
                cfg.fm_time_weight_gamma,
                cfg.time_map_param,
            )
        self.register_buffer("centers", F.normalize(centers.float(), p=2, dim=1), persistent=False)
        if cfg.assignment_queue_size < 0:
            raise ValueError("assignment_queue_size must be non-negative")
        if cfg.assignment_queue_size:
            if cfg.assignment_queue_size % cfg.bs:
                raise ValueError("assignment_queue_size must be a multiple of bs")
            if cfg.bs % self.centers.shape[0]:
                raise ValueError("bs must be divisible by Kprime for queued exact-capacity assignment")
            if cfg.target_redundancy != 1 or cfg.pool_extra_repeats != 0:
                raise ValueError(
                    "queued assignment currently requires target_redundancy=1 "
                    "and pool_extra_repeats=0"
                )
        if cfg.assignment_epoch_global:
            if cfg.assignment_queue_size:
                raise ValueError("epoch-global assignment and feature queue are mutually exclusive")
            if not 0.0 <= cfg.assignment_global_alpha <= 1.0:
                raise ValueError("assignment_global_alpha must be in [0, 1]")
            if cfg.assignment_global_cap < 0:
                raise ValueError("assignment_global_cap must be non-negative")
            if cfg.assignment_global_temperature <= 0:
                raise ValueError("assignment_global_temperature must be positive")
            if cfg.assignment_global_dual_lr < 0:
                raise ValueError("assignment_global_dual_lr must be non-negative")
            if cfg.assignment_global_dual_clip < 0:
                raise ValueError("assignment_global_dual_clip must be non-negative")
            if cfg.target_redundancy != 1 or cfg.pool_extra_repeats != 0:
                raise ValueError(
                    "epoch-global assignment currently requires target_redundancy=1 "
                    "and pool_extra_repeats=0"
                )
        if cfg.soft_occupancy_weight < 0:
            raise ValueError("soft occupancy weight must be non-negative")
        if cfg.soft_occupancy_temperature <= 0:
            raise ValueError("soft occupancy temperature must be positive")
        regularizer_weights = {
            "z0_variance_weight": cfg.z0_variance_weight,
            "z0_covariance_weight": cfg.z0_covariance_weight,
            "z0_uniformity_weight": cfg.z0_uniformity_weight,
            "z0_crosscorr_weight": cfg.z0_crosscorr_weight,
            "backbone_alignment_weight": cfg.backbone_alignment_weight,
            "backbone_variance_weight": cfg.backbone_variance_weight,
            "backbone_covariance_weight": cfg.backbone_covariance_weight,
            "velocity_radial_weight": cfg.velocity_radial_weight,
        }
        for name, value in regularizer_weights.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if cfg.fm_loss_mode not in {"mse", "huber_matched"}:
            raise ValueError(f"unknown FM loss mode: {cfg.fm_loss_mode}")
        if cfg.fm_loss_mode != "mse" and cfg.flow_path_mode != "single":
            raise ValueError("robust FM loss is restricted to the single-path objective")
        if (cfg.velocity_tangent_projection or cfg.velocity_radial_weight) and cfg.path_geometry != "spherical":
            raise ValueError("velocity tangent/radial policies require spherical path geometry")
        self.v_net = VelocityNet(
            cfg.emb,
            hidden=cfg.velocity_hidden,
            layers=cfg.velocity_layers,
            time_feature_mode=cfg.time_feature_mode,
            hidden_bias=cfg.velocity_hidden_bias,
            output_bias=cfg.velocity_output_bias,
        )
        if cfg.velocity_mode == "residual":
            if cfg.flow_path_mode != "single":
                raise ValueError("residual velocity and coarse-fine path are separate ablations")
            if cfg.residual_regularization < 0:
                raise ValueError("residual_regularization must be non-negative")
            self.v_net.requires_grad_(False)
            self.delta_v_net = VelocityNet(
                cfg.emb,
                hidden=cfg.velocity_hidden,
                layers=cfg.velocity_layers,
                time_feature_mode=cfg.time_feature_mode,
                hidden_bias=cfg.velocity_hidden_bias,
                output_bias=cfg.velocity_output_bias,
            )
            torch.nn.init.zeros_(self.delta_v_net.v[-1].weight)
            if self.delta_v_net.v[-1].bias is not None:
                torch.nn.init.zeros_(self.delta_v_net.v[-1].bias)
        else:
            self.delta_v_net = None
        if cfg.assignment_feature_source == "ema":
            if not 0.0 <= cfg.assignment_teacher_momentum < 1.0:
                raise ValueError("assignment_teacher_momentum must be in [0, 1)")
            if cfg.endpoint_weight != 0:
                raise ValueError("EMA assignment requires endpoint_weight=0")
            if cfg.alignment_space != "z0" or cfg.lambda_param != 50:
                raise ValueError("EMA assignment requires explicit z0 alignment with lambda=50")
            self.assignment_teacher = AssignmentTeacher(self.model, self.head)
        else:
            self.assignment_teacher = None
        if cfg.flow_path_mode == "coarse_fine":
            if not 0.0 < cfg.flow_stage_rho < 1.0:
                raise ValueError("flow_stage_rho must be in (0, 1)")
            if not 0.0 < cfg.flow_stage_split < 1.0:
                raise ValueError("flow_stage_split must be in (0, 1)")
        if cfg.time_map == "frontload_power" and cfg.time_map_param <= 1.0:
            raise ValueError(
                "frontload_power requires time_map_param > 1"
            )
        if cfg.time_map in {
            "frontload_exponential",
            "frontload_rational",
        } and cfg.time_map_param <= 0.0:
            raise ValueError(
                f"{cfg.time_map} requires time_map_param > 0"
            )
        if cfg.time_map == "frontload_sine" and not (
            0.0 < cfg.time_map_param <= 0.5 * math.pi
        ):
            raise ValueError(
                "frontload_sine requires time_map_param in (0, pi/2]"
            )
        self.last_losses = {}
        self.register_buffer(
            "assignment_counts",
            torch.zeros(self.centers.shape[0], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_queue",
            torch.empty(cfg.assignment_queue_size, self.centers.shape[1]),
            persistent=False,
        )
        self.register_buffer(
            "assignment_queue_center_ids",
            torch.empty(cfg.assignment_queue_size, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_queue_valid",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_queue_reassignments",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_queue_reassignment_total",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_batch_active_sum",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_batch_active_min",
            torch.full((), self.centers.shape[0], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_batch_active_max",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_batch_steps",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_global_remaining",
            torch.zeros(self.centers.shape[0], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_global_dual",
            torch.zeros(self.centers.shape[0]),
            persistent=False,
        )
        self.register_buffer(
            "assignment_global_quota_min",
            torch.full((), cfg.bs, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_global_quota_max",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_global_quota_cv_sum",
            torch.zeros(()),
            persistent=False,
        )
        self.register_buffer(
            "assignment_cosine_sum",
            torch.zeros(()),
            persistent=False,
        )
        self.register_buffer(
            "assignment_oracle_regret_sum",
            torch.zeros(()),
            persistent=False,
        )
        self.register_buffer(
            "assignment_quality_samples",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "assignment_late_cosine_sum",
            torch.zeros(()),
            persistent=False,
        )
        self.register_buffer(
            "assignment_late_oracle_regret_sum",
            torch.zeros(()),
            persistent=False,
        )
        self.register_buffer(
            "assignment_late_quality_samples",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.assignment_epoch_num_batches = 0
        self.assignment_match_seconds = 0.0
        self.instrumentation = None

    def load_checkpoint_state(self, state_dict):
        """Strictly load old or EMA-aware checkpoints without hiding corruption."""
        state_dict = dict(state_dict)
        teacher_prefix = "assignment_teacher."
        teacher_keys = [key for key in state_dict if key.startswith(teacher_prefix)]
        if self.assignment_teacher is not None and not teacher_keys:
            teacher_state = self.assignment_teacher.state_dict()
            for key in teacher_state:
                if key not in state_dict:
                    raise RuntimeError(
                        f"old checkpoint is missing online source for teacher key: {key}"
                    )
                state_dict[f"{teacher_prefix}{key}"] = state_dict[key]
        delta_prefix = "delta_v_net."
        delta_keys = [key for key in state_dict if key.startswith(delta_prefix)]
        if self.delta_v_net is not None and not delta_keys:
            current_state = self.state_dict()
            for key, value in current_state.items():
                if key.startswith(delta_prefix):
                    state_dict[key] = value
        return self.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def update_assignment_teacher(self):
        if self.assignment_teacher is None:
            return
        momentum = self.cfg.assignment_teacher_momentum
        module_pairs = (
            (self.model, self.assignment_teacher.model),
            (self.head, self.assignment_teacher.head),
        )
        for online, teacher in module_pairs:
            online_parameters = dict(online.named_parameters())
            for name, teacher_parameter in teacher.named_parameters():
                teacher_parameter.lerp_(
                    online_parameters[name].detach(),
                    1.0 - momentum,
                )
            online_buffers = dict(online.named_buffers())
            for name, teacher_buffer in teacher.named_buffers():
                online_buffer = online_buffers[name].detach()
                if teacher_buffer.is_floating_point():
                    teacher_buffer.lerp_(online_buffer, 1.0 - momentum)
                else:
                    teacher_buffer.copy_(online_buffer)
        self.assignment_teacher.eval()

    def _velocity(self, z, t):
        if self.delta_v_net is None:
            return bound_velocity(self.v_net(z, t), float(getattr(self.cfg, 'velocity_norm_cap', 0.))), None, None, None
        with torch.no_grad():
            base_velocity = self.v_net(z, t)
        delta_velocity = self.delta_v_net(z, t)
        if self.cfg.residual_alpha == "bell":
            alpha = 4.0 * t * (1.0 - t)
        else:
            alpha = torch.ones_like(t)
        return (
            base_velocity + alpha * delta_velocity,
            delta_velocity,
            base_velocity,
            alpha,
        )

    @staticmethod
    def _integer_quota_projection(target, lower, upper, total):
        """Exact bounded integer projection minimizing squared distance to target."""
        target = np.asarray(target, dtype=np.float64)
        lower = np.asarray(lower, dtype=np.int64)
        upper = np.asarray(upper, dtype=np.int64)
        if target.shape != lower.shape or lower.shape != upper.shape:
            raise ValueError("quota projection arrays must have identical shapes")
        if np.any(lower < 0) or np.any(upper < lower):
            raise RuntimeError("invalid global-assignment quota bounds")
        lower_sum, upper_sum = int(lower.sum()), int(upper.sum())
        if not lower_sum <= total <= upper_sum:
            raise RuntimeError(
                f"infeasible global-assignment quotas: lower={lower_sum} "
                f"total={total} upper={upper_sum}"
            )

        quota = lower.copy()
        for _ in range(total - lower_sum):
            marginal = 2.0 * (quota - target) + 1.0
            marginal[quota >= upper] = np.inf
            index = int(np.argmin(marginal))
            if not np.isfinite(marginal[index]):
                raise RuntimeError("quota projection exhausted all upper bounds")
            quota[index] += 1
        return quota

    def _epoch_global_pool(self, consensus):
        batch_size = consensus.shape[0]
        remaining = self.assignment_global_remaining
        remaining_total = int(remaining.sum())
        if remaining_total < batch_size or remaining_total % batch_size:
            raise RuntimeError(
                f"invalid epoch-global remaining mass: {remaining_total}"
            )

        future_total = remaining_total - batch_size
        k = self.centers.shape[0]
        future_floor = future_total // k
        future_ceil = math.ceil(future_total / k)
        cap = self.cfg.assignment_global_cap
        lower = torch.maximum(
            torch.zeros_like(remaining),
            remaining - (future_ceil + cap),
        )
        lower = torch.maximum(
            lower,
            remaining - future_total,
        )
        upper = torch.minimum(
            remaining,
            remaining - max(0, future_floor - cap),
        )

        center_score = consensus @ self.centers.T
        adjusted_score = center_score - self.assignment_global_dual.unsqueeze(0)
        adjusted_score[:, upper == 0] = -torch.inf
        demand = torch.softmax(
            adjusted_score / self.cfg.assignment_global_temperature,
            dim=1,
        ).sum(dim=0)
        pace = remaining.float() * (batch_size / remaining_total)
        alpha = self.cfg.assignment_global_alpha
        target = (1.0 - alpha) * pace + alpha * demand
        quota_np = self._integer_quota_projection(
            target.detach().cpu().numpy(),
            lower.cpu().numpy(),
            upper.cpu().numpy(),
            batch_size,
        )
        quota = torch.as_tensor(quota_np, device=consensus.device, dtype=torch.long)
        center_indices = torch.arange(k, device=consensus.device)
        pool_center_ids = torch.cat(
            [
                center_indices[quota > repeat_index]
                for repeat_index in range(int(quota.max()))
            ]
        )
        if pool_center_ids.numel() != batch_size:
            raise RuntimeError("epoch-global quota projection did not produce one target per sample")

        quota_float = quota.float()
        self.assignment_global_quota_min.copy_(
            torch.minimum(self.assignment_global_quota_min, quota.min())
        )
        self.assignment_global_quota_max.copy_(
            torch.maximum(self.assignment_global_quota_max, quota.max())
        )
        self.assignment_global_quota_cv_sum.add_(
            quota_float.std(unbiased=False) / quota_float.mean().clamp_min(EPS)
        )

        dual_lr = self.cfg.assignment_global_dual_lr
        if dual_lr:
            dual_update = dual_lr * (
                demand / batch_size - remaining.float() / remaining_total
            )
            self.assignment_global_dual.add_(dual_update)
            self.assignment_global_dual.sub_(self.assignment_global_dual.mean())
            clip = self.cfg.assignment_global_dual_clip
            if clip:
                self.assignment_global_dual.clamp_(-clip, clip)

        self.assignment_global_remaining.sub_(quota)
        if (self.assignment_global_remaining < 0).any():
            raise RuntimeError("epoch-global assignment consumed negative center capacity")
        return self.centers[pool_center_ids], pool_center_ids

    def ode_solve(self, z, steps):
        if steps < 1:
            return F.normalize(z, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((z.shape[0], 1), step * dt, device=z.device, dtype=z.dtype)
            velocity = self._velocity(z, t)[0]
            if self.cfg.path_geometry == "spherical":
                velocity = tangent_projection(velocity, z)
                z = F.normalize(z + dt * velocity, p=2, dim=1)
            else:
                z = z + dt * velocity
        return z

    def assign_targets(self, z0_1, z0_2):
        with torch.no_grad():
            match_start = time.perf_counter()
            consensus = 0.5 * (z0_1.detach() + z0_2.detach())
            if self.cfg.normalize_assignment_consensus:
                consensus = F.normalize(consensus, p=2, dim=1)

            queue_capacity = self.assignment_queue.shape[0]
            if self.cfg.assignment_epoch_global:
                pool, pool_center_ids = self._epoch_global_pool(consensus)
                score = consensus @ pool.T
                _, columns = linear_sum_assignment((-score).cpu().numpy())
                columns = torch.as_tensor(columns, device=z0_1.device, dtype=torch.long)
                center_ids = pool_center_ids[columns]
                center = pool[columns]
            elif queue_capacity == 0:
                pool = repeated_center_pool(
                    self.centers,
                    z0_1.shape[0],
                    self.cfg.target_redundancy,
                    self.cfg.pool_extra_repeats,
                )
                score = consensus @ pool.T
                _, columns = linear_sum_assignment((-score).cpu().numpy())
                columns = torch.as_tensor(columns, device=z0_1.device, dtype=torch.long)
                center_ids = columns.remainder(self.centers.shape[0])
                center = pool[columns]
            else:
                queue_valid = int(self.assignment_queue_valid)
                queued = self.assignment_queue[:queue_valid]
                joint_consensus = torch.cat((queued, consensus), dim=0)
                pool = repeated_center_pool(
                    self.centers,
                    joint_consensus.shape[0],
                    self.cfg.target_redundancy,
                    self.cfg.pool_extra_repeats,
                )
                score = joint_consensus @ pool.T
                _, columns = linear_sum_assignment((-score).cpu().numpy())
                columns = torch.as_tensor(columns, device=z0_1.device, dtype=torch.long)
                joint_center_ids = columns.remainder(self.centers.shape[0])

                if queue_valid:
                    previous_center_ids = self.assignment_queue_center_ids[:queue_valid]
                    self.assignment_queue_reassignments.add_(
                        (joint_center_ids[:queue_valid] != previous_center_ids).sum()
                    )
                    self.assignment_queue_reassignment_total.add_(queue_valid)

                center_ids = joint_center_ids[-z0_1.shape[0] :]
                center = self.centers[center_ids]

                retained = min(queue_capacity, joint_consensus.shape[0])
                self.assignment_queue[:retained].copy_(
                    joint_consensus[-retained:].detach()
                )
                self.assignment_queue_center_ids[:retained].copy_(
                    joint_center_ids[-retained:]
                )
                self.assignment_queue_valid.fill_(retained)

            self.assignment_match_seconds += time.perf_counter() - match_start
            center_score = consensus @ self.centers.T
            assigned_cosine = center_score.gather(1, center_ids[:, None]).squeeze(1)
            oracle_regret = center_score.max(dim=1).values - assigned_cosine
            self.assignment_cosine_sum.add_(assigned_cosine.sum())
            self.assignment_oracle_regret_sum.add_(oracle_regret.sum())
            self.assignment_quality_samples.add_(z0_1.shape[0])
            if self.cfg.assignment_epoch_global and int(
                self.assignment_batch_steps
            ) >= math.floor(0.9 * self.assignment_epoch_num_batches):
                self.assignment_late_cosine_sum.add_(assigned_cosine.sum())
                self.assignment_late_oracle_regret_sum.add_(oracle_regret.sum())
                self.assignment_late_quality_samples.add_(z0_1.shape[0])
            self.assignment_counts.add_(
                torch.bincount(center_ids, minlength=self.centers.shape[0])
            )
            batch_active = torch.unique(center_ids).numel()
            self.assignment_batch_active_sum.add_(batch_active)
            self.assignment_batch_active_min.copy_(
                torch.minimum(
                    self.assignment_batch_active_min,
                    self.assignment_batch_active_min.new_tensor(batch_active),
                )
            )
            self.assignment_batch_active_max.copy_(
                torch.maximum(
                    self.assignment_batch_active_max,
                    self.assignment_batch_active_max.new_tensor(batch_active),
                )
            )
            self.assignment_batch_steps.add_(1)
            target_1 = perturbation(center, self.cfg.eps)
            target_2 = target_1 if self.cfg.target_mode == "shared_point" else perturbation(center, self.cfg.eps)
        return target_1, target_2, center

    def _coarse_fine_sample(self, z0, center, target):
        split = self.cfg.flow_stage_split
        random_unit = torch.rand(
            z0.shape[0],
            1,
            device=z0.device,
            dtype=z0.dtype,
        )
        stage_zero = random_unit < split
        if self.cfg.time_sampling == "path_uniform":
            local_path_coordinate = torch.where(
                stage_zero,
                random_unit / split,
                (random_unit - split) / (1.0 - split),
            ).clamp(0.0, 1.0)
            local_t = invert_path_coordinate(
                local_path_coordinate,
                self.cfg.time_map,
                self.cfg.time_map_param,
            )
            local_s, local_s_dot = path_coordinate(
                local_t,
                self.cfg.time_map,
                self.cfg.time_map_param,
            )
            global_t = torch.where(
                stage_zero,
                split * local_t,
                split + (1.0 - split) * local_t,
            )
        else:
            global_t = random_unit
            local_t = torch.where(
                stage_zero,
                global_t / split,
                (global_t - split) / (1.0 - split),
            ).clamp(0.0, 1.0)
            local_s, local_s_dot = path_coordinate(
                local_t,
                self.cfg.time_map,
                self.cfg.time_map_param,
            )

        midpoint = slerp(
            z0,
            center,
            z0.new_full((z0.shape[0], 1), self.cfg.flow_stage_rho),
        )
        start = torch.where(stage_zero, z0, midpoint)
        end = torch.where(stage_zero, midpoint, target)
        duration = torch.where(
            stage_zero,
            z0.new_full((z0.shape[0], 1), split),
            z0.new_full((z0.shape[0], 1), 1.0 - split),
        )

        if self.cfg.path_geometry == "spherical":
            zt = slerp(start, end, local_s)
            target_velocity = slerp_velocity(
                start,
                end,
                local_s,
                local_s_dot,
            ) / duration
        else:
            zt = (1.0 - local_s) * start + local_s * end
            target_velocity = local_s_dot * (end - start) / duration
        return zt, global_t, target_velocity, duration.square(), stage_zero

    def reset_epoch_stats(self, num_batches=None):
        self.assignment_counts.zero_()
        self.assignment_queue_valid.zero_()
        self.assignment_queue_reassignments.zero_()
        self.assignment_queue_reassignment_total.zero_()
        self.assignment_batch_active_sum.zero_()
        self.assignment_batch_active_min.fill_(self.centers.shape[0])
        self.assignment_batch_active_max.zero_()
        self.assignment_batch_steps.zero_()
        self.assignment_global_quota_min.fill_(self.cfg.bs)
        self.assignment_global_quota_max.zero_()
        self.assignment_global_quota_cv_sum.zero_()
        self.assignment_cosine_sum.zero_()
        self.assignment_oracle_regret_sum.zero_()
        self.assignment_quality_samples.zero_()
        self.assignment_late_cosine_sum.zero_()
        self.assignment_late_oracle_regret_sum.zero_()
        self.assignment_late_quality_samples.zero_()
        if self.cfg.assignment_epoch_global:
            if not num_batches:
                raise ValueError("epoch-global assignment requires a positive num_batches")
            epoch_samples = num_batches * self.cfg.bs
            if epoch_samples % self.centers.shape[0]:
                raise ValueError(
                    "epoch-global assignment requires epoch samples divisible by Kprime"
                )
            self.assignment_global_remaining.fill_(
                epoch_samples // self.centers.shape[0]
            )
            self.assignment_epoch_num_batches = num_batches
        self.assignment_match_seconds = 0.0

    def assignment_stats(self):
        counts = self.assignment_counts.float()
        total = counts.sum()
        if total == 0:
            return {}
        probability = counts / total
        positive = probability > 0
        entropy = -(probability[positive] * probability[positive].log()).sum()
        stats = {
            "assignment_entropy": float(entropy / math.log(len(counts))),
            "assignment_active": int(positive.sum()),
            "assignment_min": int(counts.min()),
            "assignment_max": int(counts.max()),
            "assignment_cv": float(counts.std(unbiased=False) / counts.mean().clamp_min(EPS)),
        }
        batch_steps = int(self.assignment_batch_steps)
        if batch_steps:
            stats.update(
                {
                    "assignment_batch_active_mean": float(
                        self.assignment_batch_active_sum.float() / batch_steps
                    ),
                    "assignment_batch_active_min": int(self.assignment_batch_active_min),
                    "assignment_batch_active_max": int(self.assignment_batch_active_max),
                    "assignment_queue_valid": int(self.assignment_queue_valid),
                    "assignment_match_seconds": self.assignment_match_seconds,
                    "assignment_match_seconds_per_batch": (
                        self.assignment_match_seconds / batch_steps
                    ),
                }
            )
        reassignment_total = int(self.assignment_queue_reassignment_total)
        if reassignment_total:
            stats["assignment_queue_churn"] = float(
                self.assignment_queue_reassignments.float() / reassignment_total
            )
        if self.cfg.assignment_epoch_global and batch_steps:
            quality_samples = int(self.assignment_quality_samples)
            late_quality_samples = int(self.assignment_late_quality_samples)
            stats.update(
                {
                    "assignment_global_quota_min": int(self.assignment_global_quota_min),
                    "assignment_global_quota_max": int(self.assignment_global_quota_max),
                    "assignment_global_quota_cv_mean": float(
                        self.assignment_global_quota_cv_sum / batch_steps
                    ),
                    "assignment_global_remaining_total": int(
                        self.assignment_global_remaining.sum()
                    ),
                    "assignment_global_remaining_min": int(
                        self.assignment_global_remaining.min()
                    ),
                    "assignment_global_remaining_max": int(
                        self.assignment_global_remaining.max()
                    ),
                    "assignment_global_dual_absmax": float(
                        self.assignment_global_dual.abs().max()
                    ),
                    "assignment_global_dual_saturated_fraction": float(
                        (
                            self.assignment_global_dual.abs()
                            >= max(self.cfg.assignment_global_dual_clip - EPS, 0.0)
                        ).float().mean()
                    ) if self.cfg.assignment_global_dual_clip else 0.0,
                    "assignment_cosine_mean": float(
                        self.assignment_cosine_sum / max(quality_samples, 1)
                    ),
                    "assignment_oracle_regret_mean": float(
                        self.assignment_oracle_regret_sum / max(quality_samples, 1)
                    ),
                    "assignment_late_cosine_mean": float(
                        self.assignment_late_cosine_sum / max(late_quality_samples, 1)
                    ),
                    "assignment_late_oracle_regret_mean": float(
                        self.assignment_late_oracle_regret_sum
                        / max(late_quality_samples, 1)
                    ),
                }
            )
        return stats

    def prototype_stats(self):
        gram = self.centers @ self.centers.T
        mask = ~torch.eye(len(gram), dtype=torch.bool, device=gram.device)
        off_diagonal = gram[mask]
        singular_values = torch.linalg.svdvals(self.centers)
        energy = singular_values.square()
        probability = energy / energy.sum()
        effective_rank = torch.exp(
            -(probability * probability.clamp_min(EPS).log()).sum()
        )
        return {
            "prototype_offdiag_mean": float(off_diagonal.mean()),
            "prototype_offdiag_absmax": float(off_diagonal.abs().max()),
            "prototype_effective_rank": float(effective_rank),
            "prototype_matrix_rank": int(torch.linalg.matrix_rank(self.centers)),
        }

    def forward(self, samples):
        x1, x2 = samples
        device = next(self.parameters()).device
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        if getattr(self.cfg, "joint_two_views", False):
            both_backbone = self.model(torch.cat((x1, x2), dim=0))
            both_z0 = F.normalize(self.head(both_backbone), p=2, dim=1)
            backbone_1, backbone_2 = both_backbone.chunk(2, dim=0)
            z0_1, z0_2 = both_z0.chunk(2, dim=0)
        else:
            backbone_1 = self.model(x1)
            backbone_2 = self.model(x2)
            z0_1 = F.normalize(self.head(backbone_1), p=2, dim=1)
            z0_2 = F.normalize(self.head(backbone_2), p=2, dim=1)
        if self.cfg.z0_variance_weight:
            loss_z0_variance = 0.5 * (
                variance_floor_loss(z0_1) + variance_floor_loss(z0_2)
            )
        else:
            loss_z0_variance = z0_1.new_zeros(())
        if self.cfg.z0_covariance_weight:
            loss_z0_covariance = 0.5 * (
                covariance_loss(z0_1) + covariance_loss(z0_2)
            )
        else:
            loss_z0_covariance = z0_1.new_zeros(())
        if self.cfg.z0_uniformity_weight:
            loss_z0_uniformity = 0.5 * (
                hyperspherical_uniformity_loss(z0_1)
                + hyperspherical_uniformity_loss(z0_2)
            )
        else:
            loss_z0_uniformity = z0_1.new_zeros(())
        if self.cfg.z0_crosscorr_weight:
            loss_z0_crosscorr = cross_correlation_offdiagonal_loss(z0_1, z0_2)
        else:
            loss_z0_crosscorr = z0_1.new_zeros(())
        if any(
            (
                self.cfg.backbone_alignment_weight,
                self.cfg.backbone_variance_weight,
                self.cfg.backbone_covariance_weight,
            )
        ):
            normalized_backbone_1 = F.normalize(backbone_1, p=2, dim=1)
            normalized_backbone_2 = F.normalize(backbone_2, p=2, dim=1)
        if self.cfg.backbone_alignment_weight:
            loss_backbone_alignment = (
                normalized_backbone_1 - normalized_backbone_2
            ).square().mean()
        else:
            loss_backbone_alignment = z0_1.new_zeros(())
        if self.cfg.backbone_variance_weight:
            loss_backbone_variance = 0.5 * (
                variance_floor_loss(normalized_backbone_1)
                + variance_floor_loss(normalized_backbone_2)
            )
        else:
            loss_backbone_variance = z0_1.new_zeros(())
        if self.cfg.backbone_covariance_weight:
            loss_backbone_covariance = 0.5 * (
                covariance_loss(normalized_backbone_1)
                + covariance_loss(normalized_backbone_2)
            )
        else:
            loss_backbone_covariance = z0_1.new_zeros(())
        if self.cfg.soft_occupancy_weight:
            temperature = self.cfg.soft_occupancy_temperature
            probability_1 = torch.softmax(z0_1 @ self.centers.T / temperature, dim=1)
            probability_2 = torch.softmax(z0_2 @ self.centers.T / temperature, dim=1)
            marginal = 0.5 * (probability_1.mean(dim=0) + probability_2.mean(dim=0))
            loss_soft_occupancy = (
                marginal
                * (marginal.clamp_min(EPS).log() + math.log(self.centers.shape[0]))
            ).sum()
        else:
            loss_soft_occupancy = z0_1.new_zeros(())
        if self.instrumentation is not None:
            self.instrumentation.observe_natural_centers(
                z0_1,
                z0_2,
                self.centers,
                self.ode_solve,
            )
        if self.assignment_teacher is None:
            assignment_z0_1, assignment_z0_2 = z0_1, z0_2
        else:
            assignment_z0_1 = self.assignment_teacher(x1)
            assignment_z0_2 = self.assignment_teacher(x2)
        r1, r2, assigned_center = self.assign_targets(
            assignment_z0_1,
            assignment_z0_2,
        )
        fm_encoder_grad_scale = float(self.cfg.fm_encoder_grad_scale)
        if not 0.0 <= fm_encoder_grad_scale <= 2.0:
            raise ValueError(
                "fm_encoder_grad_scale must be in [0, 2], got "
                f"{fm_encoder_grad_scale}"
            )
        if self.cfg.detach_fm_encoder or fm_encoder_grad_scale == 0.0:
            fm_z0_1, fm_z0_2 = z0_1.detach(), z0_2.detach()
        elif fm_encoder_grad_scale == 1.0:
            fm_z0_1, fm_z0_2 = z0_1, z0_2
        else:
            fm_z0_1 = z0_1.detach() + fm_encoder_grad_scale * (
                z0_1 - z0_1.detach()
            )
            fm_z0_2 = z0_2.detach() + fm_encoder_grad_scale * (
                z0_2 - z0_2.detach()
            )
        if self.cfg.fm_weight:
            if self.cfg.flow_path_mode == "coarse_fine":
                zt1, t1, target_v1, weight1, stage_zero_1 = self._coarse_fine_sample(
                    fm_z0_1,
                    assigned_center,
                    r1,
                )
                zt2, t2, target_v2, weight2, stage_zero_2 = self._coarse_fine_sample(
                    fm_z0_2,
                    assigned_center,
                    r2,
                )
                prediction1, delta1, base1, residual_alpha1 = self._velocity(zt1, t1)
                prediction2, delta2, base2, residual_alpha2 = self._velocity(zt2, t2)
                if self.cfg.velocity_tangent_projection:
                    prediction1 = tangent_projection(prediction1, zt1)
                    prediction2 = tangent_projection(prediction2, zt2)
                error1 = (prediction1 - target_v1).square().mean(dim=1)
                error2 = (prediction2 - target_v2).square().mean(dim=1)
                weighted_error1 = weight1.squeeze(1) * error1
                weighted_error2 = weight2.squeeze(1) * error2
                loss_fm = 0.5 * (
                    weighted_error1.mean() + weighted_error2.mean()
                )
                stage_zero_errors = torch.cat(
                    (
                        weighted_error1[stage_zero_1.squeeze(1)],
                        weighted_error2[stage_zero_2.squeeze(1)],
                    )
                )
                stage_one_errors = torch.cat(
                    (
                        weighted_error1[~stage_zero_1.squeeze(1)],
                        weighted_error2[~stage_zero_2.squeeze(1)],
                    )
                )
                loss_fm_stage0 = (
                    stage_zero_errors.mean()
                    if stage_zero_errors.numel()
                    else loss_fm.detach().new_zeros(())
                )
                loss_fm_stage1 = (
                    stage_one_errors.mean()
                    if stage_one_errors.numel()
                    else loss_fm.detach().new_zeros(())
                )
            else:
                dispatch_saved_targets=(r1,r2)
                dispatch_samples=int(getattr(self.cfg,'fm_time_samples',1))
                if dispatch_samples==2:
                    fm_z0_1=fm_z0_1.repeat(2,1);fm_z0_2=fm_z0_2.repeat(2,1)
                    r1=r1.repeat(2,1);r2=r2.repeat(2,1)
                t1 = sample_time(
                    z0_1.shape[0],
                    device,
                    z0_1.dtype,
                    self.cfg.time_sampling,
                    self.cfg.time_map,
                    self.cfg.time_map_param,
                    dispatch_samples,
                )
                t2 = sample_time(
                    z0_2.shape[0],
                    device,
                    z0_2.dtype,
                    self.cfg.time_sampling,
                    self.cfg.time_map,
                    self.cfg.time_map_param,
                    dispatch_samples,
                )
                s1, sd1 = path_coordinate(
                    t1,
                    self.cfg.time_map,
                    self.cfg.time_map_param,
                )
                s2, sd2 = path_coordinate(
                    t2,
                    self.cfg.time_map,
                    self.cfg.time_map_param,
                )
                if self.cfg.path_geometry == "spherical":
                    zt1, zt2 = slerp(fm_z0_1, r1, s1), slerp(fm_z0_2, r2, s2)
                    target_v1 = slerp_velocity(fm_z0_1, r1, s1, sd1)
                    target_v2 = slerp_velocity(fm_z0_2, r2, s2, sd2)
                else:
                    zt1 = (1.0 - s1) * fm_z0_1 + s1 * r1
                    zt2 = (1.0 - s2) * fm_z0_2 + s2 * r2
                    target_v1 = sd1 * (r1 - fm_z0_1)
                    target_v2 = sd2 * (r2 - fm_z0_2)
                prediction1, delta1, base1, residual_alpha1 = self._velocity(zt1, t1)
                prediction2, delta2, base2, residual_alpha2 = self._velocity(zt2, t2)
                if self.cfg.velocity_tangent_projection:
                    prediction1 = tangent_projection(prediction1, zt1)
                    prediction2 = tangent_projection(prediction2, zt2)
                loss_fm = 0.5 * (
                    flow_matching_loss(
                        prediction1,
                        target_v1,
                        sd1,
                        self.cfg.fm_time_weight_gamma,
                        self.cfg.time_map_param,
                        self.cfg.fm_loss_mode,
                    )
                    + flow_matching_loss(
                        prediction2,
                        target_v2,
                        sd2,
                        self.cfg.fm_time_weight_gamma,
                        self.cfg.time_map_param,
                        self.cfg.fm_loss_mode,
                    )
                )
                loss_fm_stage0 = loss_fm.detach().new_zeros(())
                loss_fm_stage1 = loss_fm.detach().new_zeros(())
            if self.cfg.velocity_radial_weight:
                loss_velocity_radial = 0.5 * (
                    radial_velocity_mse(prediction1, zt1)
                    + radial_velocity_mse(prediction2, zt2)
                )
            else:
                loss_velocity_radial = loss_fm.detach().new_zeros(())
            if delta1 is not None:
                loss_residual_regularization = 0.5 * (
                    delta1.square().mean() + delta2.square().mean()
                )
                loss_residual_base_mse = 0.5 * (
                    (base1 - target_v1).square().mean()
                    + (base2 - target_v2).square().mean()
                )
                residual_delta_norm = torch.sqrt(
                    0.5 * (delta1.square().mean() + delta2.square().mean())
                )
                residual_base_norm = torch.sqrt(
                    0.5 * (base1.square().mean() + base2.square().mean())
                )
                residual_correction_norm = torch.sqrt(
                    0.5
                    * (
                        (residual_alpha1 * delta1).square().mean()
                        + (residual_alpha2 * delta2).square().mean()
                    )
                )
            else:
                loss_residual_regularization = loss_fm.detach().new_zeros(())
                loss_residual_base_mse = loss_fm.detach().new_zeros(())
                residual_delta_norm = loss_fm.detach().new_zeros(())
                residual_base_norm = loss_fm.detach().new_zeros(())
                residual_correction_norm = loss_fm.detach().new_zeros(())
        else:
            sample_time(z0_1.shape[0],device,z0_1.dtype,self.cfg.time_sampling,self.cfg.time_map,self.cfg.time_map_param)
            sample_time(z0_2.shape[0],device,z0_2.dtype,self.cfg.time_sampling,self.cfg.time_map,self.cfg.time_map_param)
            loss_fm = z0_1.new_zeros(())
            loss_fm_stage0 = z0_1.new_zeros(())
            loss_fm_stage1 = z0_1.new_zeros(())
            loss_velocity_radial = z0_1.new_zeros(())
            loss_residual_regularization = z0_1.new_zeros(())
            loss_residual_base_mse = z0_1.new_zeros(())
            residual_delta_norm = z0_1.new_zeros(())
            residual_base_norm = z0_1.new_zeros(())
            residual_correction_norm = z0_1.new_zeros(())
        if self.cfg.fm_weight and self.cfg.flow_path_mode=='single':r1,r2=dispatch_saved_targets
        if getattr(self.cfg, 'endpoint_loss_mode', 'chordal_mse') == 'geodesic_mse':
            cosine_1 = (z0_1 * F.normalize(r1, p=2, dim=1)).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            cosine_2 = (z0_2 * F.normalize(r2, p=2, dim=1)).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            loss_endpoint = 0.5 * (torch.acos(cosine_1).square().mean() + torch.acos(cosine_2).square().mean()) / z0_1.shape[1]
        else:
            loss_endpoint = 0.5 * ((z0_1 - r1).square().mean() + (z0_2 - r2).square().mean())
        if self.cfg.alignment_space == "z1":
            align_1, align_2 = self.ode_solve(z0_1, self.cfg.train_ode_steps), self.ode_solve(z0_2, self.cfg.train_ode_steps)
        else:
            align_1, align_2 = z0_1, z0_2
        loss_align = (align_1 - align_2).square().mean()
        beta = float(getattr(self.cfg, 'velocity_energy_beta', 0.))
        loss_velocity_energy = energy_penalty(self, zt1, t1, zt2, t2) if beta else loss_fm.detach().new_zeros(())
        total = (
            self.cfg.fm_weight * beta * loss_velocity_energy
            +             self.cfg.fm_weight * loss_fm
            + self.cfg.velocity_radial_weight * loss_velocity_radial
            + self.cfg.endpoint_weight * loss_endpoint
            + self.cfg.soft_occupancy_weight * loss_soft_occupancy
            + self.cfg.z0_variance_weight * loss_z0_variance
            + self.cfg.z0_covariance_weight * loss_z0_covariance
            + self.cfg.z0_uniformity_weight * loss_z0_uniformity
            + self.cfg.z0_crosscorr_weight * loss_z0_crosscorr
            + self.cfg.backbone_alignment_weight * loss_backbone_alignment
            + self.cfg.backbone_variance_weight * loss_backbone_variance
            + self.cfg.backbone_covariance_weight * loss_backbone_covariance
            + self.cfg.lambda_param * loss_align
            + self.cfg.residual_regularization * loss_residual_regularization
        )
        self.last_losses = {
            "velocity_energy": loss_velocity_energy.detach(),
            "velocity_effective_norm": .5 * (prediction1.detach().float().norm(dim=1).mean() + prediction2.detach().float().norm(dim=1).mean()) if self.cfg.fm_weight else loss_fm.detach().new_zeros(()),
            "target_velocity_norm": .5 * (target_v1.detach().float().norm(dim=1).mean() + target_v2.detach().float().norm(dim=1).mean()) if self.cfg.fm_weight else loss_fm.detach().new_zeros(()),
            "fm": loss_fm.detach(),
            "fm_stage0": loss_fm_stage0.detach(),
            "fm_stage1": loss_fm_stage1.detach(),
            "velocity_radial": loss_velocity_radial.detach(),
            "residual_regularization": loss_residual_regularization.detach(),
            "residual_base_mse": loss_residual_base_mse.detach(),
            "residual_delta_norm": residual_delta_norm.detach(),
            "residual_base_norm": residual_base_norm.detach(),
            "residual_correction_norm": residual_correction_norm.detach(),
            "endpoint": loss_endpoint.detach(),
            "soft_occupancy": loss_soft_occupancy.detach(),
            "z0_variance": loss_z0_variance.detach(),
            "z0_covariance": loss_z0_covariance.detach(),
            "z0_uniformity": loss_z0_uniformity.detach(),
            "z0_crosscorr": loss_z0_crosscorr.detach(),
            "backbone_alignment": loss_backbone_alignment.detach(),
            "backbone_variance": loss_backbone_variance.detach(),
            "backbone_covariance": loss_backbone_covariance.detach(),
            "align": loss_align.detach(),
        }
        return total

    def get_acc(self, ds_clf, ds_test, representation):
        was_training = self.training
        self.eval()

        def encode(x):
            with torch.no_grad():
                backbone = self.model(x.cuda())
                if representation == "backbone":
                    return backbone
                z0 = F.normalize(self.head(backbone), p=2, dim=1)
                if representation == "z0":
                    return z0
                if representation == "z1":
                    return self.ode_solve(z0, self.cfg.eval_ode_steps)
                raise ValueError(f"unknown representation: {representation}")

        output_size = self.out_size if representation == "backbone" else self.emb_size
        x_train, y_train = get_data(encode, ds_clf, output_size, "cuda")
        x_test, y_test = get_data(encode, ds_test, output_size, "cuda")
        knn = eval_knn(x_train, y_train, x_test, y_test, self.knn)
        linear = eval_sgd(
            x_train,
            y_train,
            x_test,
            y_test,
            epoch=self.cfg.linear_probe_epochs,
        )
        del x_train, y_train, x_test, y_test
        self.train(was_training)
        return knn, linear
