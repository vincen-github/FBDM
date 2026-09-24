"""Same capacitated assignment objective, without duplicating centre nodes.

Each real row has integer mass one.  Centre j has integer demand capacity c.
A zero-cost dummy row supplies K*c-B unused slots. This gives exactly the
rectangular Hungarian objective on centres.repeat(c, 1). No entropy penalty,
queue, learned prices, or change to sample/centre costs is introduced.
POT's network simplex returns a vertex solution; verify integral real rows,
capacity bounds and the primal/dual certificate on every call.
"""
import json
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
import ot


def solve(cost, k):
    a = np.asarray(cost)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('finite 2D costs required')
    n, m = a.shape
    if m < n or m % k:
        raise ValueError('invalid repeated centre capacity')
    cap = m // k
    base = np.ascontiguousarray(a[:, :k], dtype=np.float64)
    for block in range(1, cap):
        if not np.array_equal(a[:, block*k:(block+1)*k], a[:, :k]):
            r,c=linear_sum_assignment(a)
            return r,c,{'solver':'dense_nonidentical_columns'}
    C = np.empty((n + 1, k), dtype=np.float64)
    C[:n] = base
    C[n] = 0
    src = np.ones(n+1, dtype=np.float64)
    src[n] = m - n
    dst = np.full(k, cap, dtype=np.float64)
    start = time.perf_counter()
    plan, info = ot.emd(src,dst,C,log=True,numItermax=1000000,numThreads=1)
    if info.get('warning'):
        raise RuntimeError(f"network simplex failed: {info['warning']}")
    rows = np.arange(n)
    ids = plan[:n].argmax(axis=1)
    if not np.allclose(plan[:n].sum(1),1,rtol=0,atol=1e-9):
        raise RuntimeError('transport row mass violated')
    if not np.allclose(plan[rows,ids],1,rtol=0,atol=1e-9):
        raise RuntimeError('nonintegral real-image assignment')
    counts = np.bincount(ids,minlength=k)
    if counts.max(initial=0)>cap:
        raise RuntimeError('centre capacity violated')
    if not np.allclose(plan.sum(0),dst,rtol=0,atol=1e-9):
        raise RuntimeError('transport column mass violated')
    u,v=info['u'],info['v']
    residual = C-u[:,None]-v[None,:]
    primal=float(base[rows,ids].sum())
    dual=float(src@u+dst@v)
    tolerance=1e-7*max(1.0,abs(primal))
    if float(residual.min()) < -1e-8 or abs(primal-dual)>tolerance:
        raise RuntimeError(f'OT certificate failed: primal={primal}, dual={dual}')
    used=np.zeros(k,dtype=np.int64)
    cols=np.empty(n,dtype=np.int64)
    for row,center in enumerate(ids):
        cols[row]=center+int(used[center])*k
        used[center]+=1
    return rows,cols,dict(solver='capacitated_network_simplex',seconds=time.perf_counter()-start,
        rows=n,centers=k,capacity=cap,active_centers=int((counts>0).sum()),
        primal=primal,dual_gap=abs(primal-dual),min_reduced_cost=float(residual.min()))


def install(k):
    import methods.dm as native
    calls=0
    def adapter(cost,*args,**kwargs):
        nonlocal calls
        if args or kwargs:return linear_sum_assignment(cost,*args,**kwargs)
        rows,cols,record=solve(cost,k)
        calls+=1
        if calls in (1,16,32) or calls%250==0:
            print(json.dumps(dict(event='exact_assignment_runtime',calls=calls,**record)),flush=True)
        return rows,cols
    native.linear_sum_assignment=adapter


def self_test():
    rng=np.random.default_rng(20260914)
    count=0
    for n,k,cap in [(1,7,2),(16,8,2),(17,11,3),(64,17,5),(65,128,2),(128,64,5)]:
        for kind in range(4):
            base=rng.normal(size=(n,k)).astype(np.float32)
            if kind==1:base=rng.integers(-2,3,size=(n,k)).astype(np.float32)
            if kind==2:base=base*.001+rng.normal(size=(1,k)).astype(np.float32)
            if kind==3:base.fill(0)
            cost=np.tile(base,(1,cap))
            r,c=linear_sum_assignment(cost)
            s,t,_=solve(cost,k)
            gap=abs(float(cost[r,c].astype(np.float64).sum()-cost[s,t].astype(np.float64).sum()))
            assert gap<1e-7,(n,k,kind,gap)
            assert len(set(t))==n
            count+=1
    return count


if __name__=='__main__':
    print(json.dumps({'self_tests':self_test()}),flush=True)
    rng=np.random.default_rng(20260914)
    for mode in ('spread','concentrated'):
        base=rng.normal(size=(2048,1024)).astype(np.float32)
        if mode=='concentrated':base=base*.002+rng.normal(size=(1,1024)).astype(np.float32)
        cost=np.tile(base,(1,5))
        begin=time.perf_counter();r,c,record=solve(cost,1024);elapsed=time.perf_counter()-begin
        begin=time.perf_counter();s,t=linear_sum_assignment(cost);dense=time.perf_counter()-begin
        gap=abs(float(cost[r,c].astype(np.float64).sum()-cost[s,t].astype(np.float64).sum()))
        assert gap<1e-7,gap
        print(json.dumps(dict(mode=mode,total_seconds=elapsed,dense_seconds=dense,objective_gap=gap,**record)),flush=True)
