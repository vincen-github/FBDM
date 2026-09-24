"""Exact capacitated matching with certified active-column screening.

Full dual feasibility is checked against every original cost column. No
approximate matching, stale embeddings, or changed capacity is accepted.
The public adapter still consumes native repeated-column costs, preserving
the original GPU score computation and its floating-point values.
"""
import json
import os
import time
from pathlib import Path
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment


def _restricted(base, active, cap):
    n = base.shape[0]
    costs = np.empty((n + 1, len(active)), dtype=np.float64)
    costs[:n] = base[:, active]
    costs[n] = 0.
    src = np.ones(n + 1, dtype=np.float64)
    src[n] = len(active) * cap - n
    dst = np.full(len(active), cap, dtype=np.float64)
    plan, info = ot.emd(src, dst, costs, log=True, numItermax=1000000, numThreads=1)
    if info.get('warning'):
        raise RuntimeError('network simplex: ' + str(info['warning']))
    local_ids = plan[:n].argmax(axis=1)
    rows = np.arange(n)
    if not np.allclose(plan[:n].sum(1), 1., rtol=0, atol=1e-9):
        raise RuntimeError('invalid row mass')
    if not np.allclose(plan[rows, local_ids], 1., rtol=0, atol=1e-9):
        raise RuntimeError('nonintegral assignment')
    if not np.allclose(plan.sum(0), dst, rtol=0, atol=1e-9):
        raise RuntimeError('invalid column mass')
    ids = active[local_ids]
    if np.bincount(ids, minlength=base.shape[1]).max() > cap:
        raise RuntimeError('capacity exceeded')
    shift = float(np.max(info['v']))
    u = np.asarray(info['u'][:n]) + shift
    v = np.asarray(info['v']) - shift
    primal = float(np.asarray(base[rows, ids], dtype=np.float64).sum())
    dual = float(u.sum() + cap * v.sum())
    if abs(primal-dual) > 1e-7 * max(1., abs(primal)):
        raise RuntimeError('restricted primal/dual gap')
    return ids, u, v, primal, dual


def _certificate(base, active, u, v):
    full_v = np.zeros(base.shape[1], dtype=np.float64)
    full_v[active] = v
    reduced_min = np.empty(base.shape[1], dtype=np.float64)
    for start in range(0, base.shape[1], 128):
        end = min(start+128, base.shape[1])
        block = np.asarray(base[:, start:end], dtype=np.float64)
        reduced_min[start:end] = (block-u[:, None]-full_v[None, start:end]).min(axis=0)
    return reduced_min


def _candidates(base, cap):
    n,k=base.shape
    width=min(k,max((n+cap-1)//cap+128,256))
    favored=np.argsort(base.mean(axis=0,dtype=np.float64),kind='stable')[:width]
    return np.union1d(favored,base.argmin(axis=1)).astype(np.int64)


def solve_base(base, cap, max_screen_rounds=2, initial=None):
    begin = time.perf_counter()
    base = np.asarray(base)
    if base.ndim != 2 or not np.isfinite(base).all():
        raise ValueError('finite 2D costs required')
    n, k = base.shape
    if not n or not k or int(cap) != cap or cap < 1 or n > k*cap:
        raise ValueError('invalid capacity or shape')
    cap = int(cap)
    active = _candidates(base,cap) if initial is None else initial
    if len(active) >= .85*k or k <= 256:
        active = np.arange(k)
    sizes = []
    for attempt in range(max_screen_rounds+1):
        if attempt == max_screen_rounds:
            active = np.arange(k)
        sizes.append(len(active))
        ids, u, v, primal, dual = _restricted(base, active, cap)
        reduced = _certificate(base, active, u, v)
        violated = np.flatnonzero(reduced < -1e-8)
        if not len(violated):
            return np.arange(n), ids, dict(
                solver='certified_active_columns', seconds=time.perf_counter()-begin,
                rows=n, centers=k, capacity=cap, candidate_sizes=sizes,
                active_centers=int(np.unique(ids).size), primal=primal,
                dual_gap=abs(primal-dual), min_reduced_cost=float(reduced.min()),
                certified_all_columns=True, dense_fallback=(len(active)==k))
        if len(active)==k:
            raise RuntimeError('full dual certificate failed')
        active = np.union1d(active, violated)
        if len(active) >= .85*k:
            active = np.arange(k)
    raise RuntimeError('no certified assignment')


def solve(cost, k):
    begin=time.perf_counter()
    a=np.asarray(cost)
    if a.ndim != 2 or a.shape[1] % k or a.shape[1] < a.shape[0]:
        raise ValueError('invalid repeated-center cost')
    cap=a.shape[1]//k
    base=a[:, :k]
    if not np.isfinite(base).all():raise ValueError('finite costs required')
    initial=_candidates(base,cap)
    if len(initial) > .5*k and k>256:
        from assignment_legacy_reference import solve as legacy_solve
        rows,cols,record=legacy_solve(cost,k)
        record.update(solver='legacy_dense_broad_candidates',candidate_sizes=[len(initial),k],
                      total_seconds=time.perf_counter()-begin,certified_all_columns=True,dense_fallback=True)
        return rows,cols,record
    for start in range(k, a.shape[1], k):
        if not np.array_equal(a[:, start:start+k], base):
            rows, cols=linear_sum_assignment(a)
            return rows, cols, {'solver':'dense_nonidentical_columns','seconds':time.perf_counter()-begin}
    rows, ids, record=solve_base(base,cap,initial=initial)
    used=np.zeros(k,dtype=np.int64); cols=np.empty(len(rows),dtype=np.int64)
    for row,center in enumerate(ids):
        cols[row]=center+int(used[center])*k
        used[center]+=1
    record['total_seconds']=time.perf_counter()-begin
    return rows,cols,record


def install(k):
    import methods.dm as native
    calls=0
    def adapter(cost,*args,**kwargs):
        nonlocal calls
        if args or kwargs:return linear_sum_assignment(cost,*args,**kwargs)
        rows,cols,record=solve(cost,k)
        calls+=1
        if calls in (1,16):
            from assignment_legacy_reference import solve as legacy_solve
            before=time.perf_counter();old_rows,old_cols,old_record=legacy_solve(cost,k)
            elapsed=time.perf_counter()-before
            old_value=float(np.asarray(cost[old_rows,old_cols],dtype=np.float64).sum())
            new_value=float(np.asarray(cost[rows,cols],dtype=np.float64).sum())
            gap=abs(new_value-old_value)
            if gap>1e-7*max(1.,abs(old_value)):
                raise RuntimeError('live legacy objective equivalence failed')
            record.update(reference_seconds=elapsed,reference_objective_gap=gap,
                          reference_same_center_fraction=float(np.mean(cols%k==old_cols%k)))
            folder=os.environ.get('FBDM_ASSIGNMENT_AUDIT_DIR')
            if folder and calls==1:
                path=Path(folder);path.mkdir(parents=True,exist_ok=True)
                np.save(path/'first_native_cost.npy',np.asarray(cost[:, :k]))
        if calls in (1,16,32) or calls%250==0:
            print(json.dumps(dict(event='exact_assignment_runtime',calls=calls,**record)),flush=True)
        return rows,cols
    native.linear_sum_assignment=adapter


def self_test():
    rng=np.random.default_rng(20260918)
    count=0
    for n,k,cap in [(1,7,2),(16,8,2),(17,11,3),(64,17,5),(65,128,2),(512,512,3)]:
        for kind in range(4):
            base=rng.normal(size=(n,k)).astype(np.float32)
            if kind==1:base=rng.integers(-2,3,size=(n,k)).astype(np.float32)
            if kind==2:base=base*.002+rng.normal(size=(1,k)).astype(np.float32)
            if kind==3:base.fill(0)
            cost=np.tile(base,(1,cap))
            r,c,_=solve(cost,k);rr,cc=linear_sum_assignment(cost)
            assert abs(float(cost[r,c].astype(np.float64).sum()-cost[rr,cc].astype(np.float64).sum()))<1e-7
            assert len(np.unique(c))==n
            count+=1
    return count
