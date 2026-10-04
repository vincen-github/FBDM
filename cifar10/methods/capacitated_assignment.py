"""Exact matching for repeated reference slots using a capacitated transport LP."""

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment


def solve_repeated_assignment(cost, num_centers):
    """Return row/slot indices with the original costs and capacity constraints."""
    cost = np.asarray(cost)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("finite 2D costs required")
    rows_count, slots_count = cost.shape
    if (
        int(num_centers) != num_centers or num_centers < 1
        or rows_count < 1 or slots_count < rows_count
        or slots_count % num_centers
    ):
        raise ValueError("invalid repeated-center shape or capacity")
    num_centers = int(num_centers)
    capacity = slots_count // num_centers
    first = cost[:, :num_centers]
    for start in range(num_centers, slots_count, num_centers):
        if not np.array_equal(cost[:, start:start + num_centers], first):
            return linear_sum_assignment(cost)

    base = np.ascontiguousarray(first, dtype=np.float64)
    costs = np.empty((rows_count + 1, num_centers), dtype=np.float64)
    costs[:rows_count] = base
    costs[rows_count] = 0.0
    supply = np.ones(rows_count + 1, dtype=np.float64)
    supply[rows_count] = slots_count - rows_count
    demand = np.full(num_centers, capacity, dtype=np.float64)
    plan, info = ot.emd(
        supply, demand, costs, log=True, numItermax=1000000, numThreads=1,
    )
    if info.get("warning"):
        raise RuntimeError("network simplex failed: " + str(info["warning"]))
    rows = np.arange(rows_count)
    centers = plan[:rows_count].argmax(axis=1)
    if not np.allclose(plan[:rows_count].sum(1), 1, rtol=0, atol=1e-9):
        raise RuntimeError("transport row mass violated")
    if not np.allclose(plan[rows, centers], 1, rtol=0, atol=1e-9):
        raise RuntimeError("nonintegral real-image assignment")
    counts = np.bincount(centers, minlength=num_centers)
    if counts.max(initial=0) > capacity:
        raise RuntimeError("center capacity violated")
    if not np.allclose(plan.sum(0), demand, rtol=0, atol=1e-9):
        raise RuntimeError("transport column mass violated")
    residual = costs - info["u"][:, None] - info["v"][None, :]
    primal = float(base[rows, centers].sum())
    dual = float(supply @ info["u"] + demand @ info["v"])
    if residual.min() < -1e-8 or abs(primal - dual) > 1e-7 * max(1, abs(primal)):
        raise RuntimeError("transport primal/dual certificate failed")

    used = np.zeros(num_centers, dtype=np.int64)
    slots = np.empty(rows_count, dtype=np.int64)
    for row, center in enumerate(centers):
        slots[row] = center + used[center] * num_centers
        used[center] += 1
    return rows, slots
