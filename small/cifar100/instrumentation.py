"""Low-overhead timing and stability diagnostics for FBDM training.

The monitor deliberately keeps expensive diagnostics sparse:

* loss finiteness is checked for every minibatch using the scalar that the
  training loop already copies to the CPU;
* gradient norms are sampled (normally first/last minibatch of an epoch);
* natural ETF occupancy is sampled on a small number of minibatches and only
  every ``natural_every_epochs`` epochs;
* CUDA synchronization happens only at train/eval/epoch timing boundaries.

Natural occupancy is *not* the Hungarian assignment histogram.  It assigns
each unbalanced model representation independently to its nearest ETF center.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F


_GIB = float(1024**3)


class EpochInstrumentation:
    """Collect epoch timing, numerical stability, gradients, and occupancy."""

    def __init__(
        self,
        device: Union[torch.device, str],
        *,
        grad_every_steps: int = 0,
        grad_every_epochs: int = 10,
        natural_every_epochs: int = 10,
        natural_batches: int = 1,
        natural_ode_steps: int = 8,
        expected_grad_groups: tuple[str, ...] = ("encoder", "head", "v_net"),
    ) -> None:
        if grad_every_steps < 0:
            raise ValueError("grad_every_steps must be >= 0")
        if grad_every_epochs < 0:
            raise ValueError("grad_every_epochs must be >= 0")
        if natural_every_epochs < 0:
            raise ValueError("natural_every_epochs must be >= 0")
        if natural_batches < 0:
            raise ValueError("natural_batches must be >= 0")
        if natural_ode_steps < 1:
            raise ValueError("natural_ode_steps must be >= 1")

        self.device = torch.device(device)
        self.grad_every_steps = grad_every_steps
        self.grad_every_epochs = grad_every_epochs
        self.natural_every_epochs = natural_every_epochs
        self.natural_batches = natural_batches
        self.natural_ode_steps = natural_ode_steps
        self.expected_grad_groups = expected_grad_groups

        self.epoch = -1
        self._epoch_started_at: Optional[float] = None
        self._train_started_at: Optional[float] = None
        self._eval_started_at: Optional[float] = None
        self._train_seconds = 0.0
        self._eval_seconds = 0.0
        self._train_peak_allocated = 0
        self._train_peak_reserved = 0
        self._eval_peak_allocated = 0
        self._eval_peak_reserved = 0
        self._loss_count = 0
        self._loss_nonfinite = 0
        self._grad_samples = 0
        self._grad_norm_sum: dict[str, float] = {}
        self._grad_norm_max: dict[str, float] = {}
        self._grad_finite: dict[str, bool] = {}
        self._grad_present: dict[str, bool] = {}
        self._natural_enabled = False
        self._natural_batches_seen = 0
        self._natural_z0_counts: Optional[torch.Tensor] = None
        self._natural_z1_counts: Optional[torch.Tensor] = None

    @property
    def _cuda_enabled(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _cuda_sync(self) -> None:
        if self._cuda_enabled:
            torch.cuda.synchronize(self.device)

    def _reset_cuda_peak(self) -> None:
        if self._cuda_enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _read_cuda_peak(self) -> tuple[int, int]:
        if not self._cuda_enabled:
            return 0, 0
        return (
            int(torch.cuda.max_memory_allocated(self.device)),
            int(torch.cuda.max_memory_reserved(self.device)),
        )

    def begin_epoch(self, epoch: int) -> None:
        """Reset diagnostics and start the full epoch wall clock."""
        self.epoch = epoch
        self._train_seconds = 0.0
        self._eval_seconds = 0.0
        self._train_peak_allocated = 0
        self._train_peak_reserved = 0
        self._eval_peak_allocated = 0
        self._eval_peak_reserved = 0
        self._loss_count = 0
        self._loss_nonfinite = 0
        self._grad_samples = 0
        self._grad_norm_sum = {name: 0.0 for name in self.expected_grad_groups}
        self._grad_norm_max = {name: 0.0 for name in self.expected_grad_groups}
        self._grad_finite = {name: True for name in self.expected_grad_groups}
        self._grad_present = {name: False for name in self.expected_grad_groups}
        self._natural_enabled = (
            self.natural_every_epochs > 0
            and self.natural_batches > 0
            and (epoch == 0 or (epoch + 1) % self.natural_every_epochs == 0)
        )
        self._natural_batches_seen = 0
        self._natural_z0_counts = None
        self._natural_z1_counts = None

        self._cuda_sync()
        self._reset_cuda_peak()
        self._epoch_started_at = time.perf_counter()
        self._train_started_at = None
        self._eval_started_at = None

    def begin_train(self) -> None:
        self._cuda_sync()
        self._reset_cuda_peak()
        self._train_started_at = time.perf_counter()

    def end_train(self) -> None:
        if self._train_started_at is None:
            raise RuntimeError("begin_train() must be called before end_train()")
        self._cuda_sync()
        self._train_seconds = time.perf_counter() - self._train_started_at
        self._train_peak_allocated, self._train_peak_reserved = self._read_cuda_peak()
        self._train_started_at = None

    def begin_eval(self) -> None:
        self._cuda_sync()
        self._reset_cuda_peak()
        self._eval_started_at = time.perf_counter()

    def end_eval(self) -> None:
        if self._eval_started_at is None:
            raise RuntimeError("begin_eval() must be called before end_eval()")
        self._cuda_sync()
        self._eval_seconds = time.perf_counter() - self._eval_started_at
        self._eval_peak_allocated, self._eval_peak_reserved = self._read_cuda_peak()
        self._eval_started_at = None

    def observe_loss(self, loss_value: float) -> None:
        """Record the already-materialized scalar loss without another sync."""
        self._loss_count += 1
        if not math.isfinite(loss_value):
            self._loss_nonfinite += 1

    def should_sample_grad(self, iteration: int, num_iterations: int) -> bool:
        """Sample first/last step, plus an optional sparse fixed interval."""
        if (
            num_iterations < 1
            or self.grad_every_epochs == 0
            or not (
                self.epoch == 0
                or (self.epoch + 1) % self.grad_every_epochs == 0
            )
        ):
            return False
        if iteration == 0 or iteration + 1 == num_iterations:
            return True
        return (
            self.grad_every_steps > 0
            and (iteration + 1) % self.grad_every_steps == 0
        )

    @staticmethod
    def _total_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> tuple[float, bool]:
        grads = [
            parameter.grad.detach()
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not grads:
            return 0.0, False

        squared_norm = torch.zeros((), device=grads[0].device, dtype=torch.float32)
        for grad in grads:
            tensor_norm = torch.linalg.vector_norm(grad.float())
            squared_norm.add_(tensor_norm.square())
        norm = float(torch.sqrt(squared_norm).detach())
        return norm, True

    def observe_gradients(
        self,
        parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
    ) -> None:
        """Measure groupwise total L2 gradient norms after backward()."""
        self._grad_samples += 1
        for name, parameters in parameter_groups.items():
            if name not in self._grad_norm_sum:
                self._grad_norm_sum[name] = 0.0
                self._grad_norm_max[name] = 0.0
                self._grad_finite[name] = True
                self._grad_present[name] = False
            norm, present = self._total_grad_norm(parameters)
            self._grad_present[name] = self._grad_present[name] or present
            if present:
                self._grad_norm_sum[name] += norm
                self._grad_norm_max[name] = max(self._grad_norm_max[name], norm)
                self._grad_finite[name] = self._grad_finite[name] and math.isfinite(norm)

    @staticmethod
    def _nearest_center_counts(
        representation: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        representation = F.normalize(representation.float(), p=2, dim=1)
        centers = F.normalize(centers.float(), p=2, dim=1)
        nearest = (representation @ centers.T).argmax(dim=1)
        return torch.bincount(nearest, minlength=centers.shape[0])

    def observe_natural_centers(
        self,
        z0_1: torch.Tensor,
        z0_2: torch.Tensor,
        centers: torch.Tensor,
        solve_z1: Callable[[torch.Tensor, int], torch.Tensor],
    ) -> None:
        """Sample unconstrained nearest-center occupancy for z0 and z1.

        This is intended to be called from ``DM.forward`` after z0 has already
        been computed.  It reuses detached z0 and incurs no extra encoder/head
        pass.  z1 integration is the only substantial extra computation.
        """
        if (
            not self._natural_enabled
            or self._natural_batches_seen >= self.natural_batches
        ):
            return

        with torch.inference_mode():
            z0 = F.normalize(
                torch.cat((z0_1.detach(), z0_2.detach()), dim=0),
                p=2,
                dim=1,
            )
            z0_counts = self._nearest_center_counts(z0, centers)
            z1 = solve_z1(z0, self.natural_ode_steps)
            z1_counts = self._nearest_center_counts(z1, centers)

        if self._natural_z0_counts is None:
            self._natural_z0_counts = torch.zeros_like(z0_counts)
            self._natural_z1_counts = torch.zeros_like(z1_counts)
        self._natural_z0_counts.add_(z0_counts)
        assert self._natural_z1_counts is not None
        self._natural_z1_counts.add_(z1_counts)
        self._natural_batches_seen += 1

    @staticmethod
    def _occupancy_stats(
        prefix: str,
        counts: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        empty = {
            f"{prefix}_active": None,
            f"{prefix}_active_fraction": None,
            f"{prefix}_entropy_nats": None,
            f"{prefix}_entropy_norm": None,
            f"{prefix}_effective_centers": None,
            f"{prefix}_max_share": None,
            f"{prefix}_samples": 0,
        }
        if counts is None:
            return empty

        counts_float = counts.float()
        total = counts_float.sum()
        if total <= 0:
            return empty
        probability = counts_float / total
        positive = probability > 0
        entropy = -(probability[positive] * probability[positive].log()).sum()
        k = counts.numel()
        entropy_norm = entropy / math.log(k) if k > 1 else entropy.new_zeros(())
        return {
            f"{prefix}_active": int(positive.sum()),
            f"{prefix}_active_fraction": float(positive.float().mean()),
            f"{prefix}_entropy_nats": float(entropy),
            f"{prefix}_entropy_norm": float(entropy_norm),
            f"{prefix}_effective_centers": float(entropy.exp()),
            f"{prefix}_max_share": float(probability.max()),
            f"{prefix}_samples": int(total),
        }

    def stability_stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "loss_finite": self._loss_count > 0 and self._loss_nonfinite == 0,
            "loss_nonfinite_steps": self._loss_nonfinite,
            "loss_steps": self._loss_count,
            "grad_finite": None,
            "grad_samples": self._grad_samples,
            "natural_occupancy_sampled": self._natural_batches_seen > 0,
            "natural_occupancy_batches": self._natural_batches_seen,
            "natural_z1_ode_steps": (
                self.natural_ode_steps if self._natural_batches_seen > 0 else None
            ),
        }

        observed_group_finite: list[bool] = []
        for name in sorted(self._grad_norm_sum):
            present = self._grad_present[name]
            result[f"{name}_grad_present"] = present
            result[f"{name}_grad_finite"] = self._grad_finite[name] if present else None
            if present and self._grad_samples:
                result[f"{name}_grad_norm"] = (
                    self._grad_norm_sum[name] / self._grad_samples
                )
                result[f"{name}_grad_norm_max"] = self._grad_norm_max[name]
                observed_group_finite.append(self._grad_finite[name])
            else:
                result[f"{name}_grad_norm"] = None
                result[f"{name}_grad_norm_max"] = None
        if observed_group_finite:
            result["grad_finite"] = all(observed_group_finite)

        result.update(
            self._occupancy_stats("natural_z0", self._natural_z0_counts)
        )
        result.update(
            self._occupancy_stats("natural_z1", self._natural_z1_counts)
        )
        return result

    def end_epoch(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Finish the full wall clock and return timing/stability records."""
        if self._epoch_started_at is None:
            raise RuntimeError("begin_epoch() must be called before end_epoch()")
        if self._train_started_at is not None or self._eval_started_at is not None:
            raise RuntimeError("train/eval timer is still running")
        self._cuda_sync()
        seconds_per_epoch = time.perf_counter() - self._epoch_started_at
        peak_allocated = max(
            self._train_peak_allocated,
            self._eval_peak_allocated,
        )
        peak_reserved = max(
            self._train_peak_reserved,
            self._eval_peak_reserved,
        )
        timing = {
            "train_seconds": self._train_seconds,
            "eval_seconds": self._eval_seconds,
            "seconds_per_epoch": seconds_per_epoch,
            "peak_cuda_memory_bytes": peak_allocated,
            "peak_cuda_memory_gib": peak_allocated / _GIB,
            "peak_cuda_reserved_bytes": peak_reserved,
            "peak_cuda_reserved_gib": peak_reserved / _GIB,
            "train_peak_cuda_memory_gib": self._train_peak_allocated / _GIB,
            "train_peak_cuda_reserved_gib": self._train_peak_reserved / _GIB,
            "eval_peak_cuda_memory_gib": self._eval_peak_allocated / _GIB,
            "eval_peak_cuda_reserved_gib": self._eval_peak_reserved / _GIB,
        }
        self._epoch_started_at = None
        return timing, self.stability_stats()
