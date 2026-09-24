"""Exact rectangular assignment with union-of-row-top-n column pruning.

An n-row matching occupies at most n-1 other columns for each row. If that
row is assigned outside its n cheapest columns, an unused cheaper/equal one
exists. Repeating this exchange retains an optimum. No approximation or
capacity change; equal-cost optima need not have identical tie-breaking.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment as dense_assignment

def solve(cost, *, force_dense=False):
 a=np.asarray(cost)
 if a.ndim!=2 or not np.isfinite(a).all():
  raise ValueError('finite 2D assignment costs required')
 n,m=a.shape;t=time.perf_counter()
 if force_dense or n==0 or m<=n:
  r,c=dense_assignment(a)
  return r,c,dict(columns=m,full_columns=m,selection_seconds=0.,solver_seconds=time.perf_counter()-t,pruned=False)
 keep=np.unique(np.argpartition(a,kth=n-1,axis=1)[:,:n])
 selection=time.perf_counter()-t;t=time.perf_counter()
 if len(keep)>=.85*m:
  r,c=dense_assignment(a);pruned=False;used=m
 else:
  r,j=dense_assignment(a[:,keep]);c=keep[j];pruned=True;used=len(keep)
 return r,c,dict(columns=used,full_columns=m,selection_seconds=selection,solver_seconds=time.perf_counter()-t,pruned=pruned)

def install():
 import json
 import methods.dm as native
 calls=0
 def adapter(a,*args,**kwargs):
  nonlocal calls
  if args or kwargs:return dense_assignment(a,*args,**kwargs)
  r,c,record=solve(a);calls+=1
  if calls==1 or calls%250==0:print(json.dumps(dict(event='exact_assignment_runtime',calls=calls,**record)),flush=True)
  return r,c
 native.linear_sum_assignment=adapter

def self_test():
 rng=np.random.RandomState(20260910);cases=0
 for n,m in [(1,7),(3,11),(8,31),(17,86),(32,32),(65,320)]:
  for kind in range(4):
   a=rng.normal(size=(n,m)) if kind!=1 else rng.randint(-3,4,size=(n,m)).astype(float)
   if kind==2:a+=rng.normal(size=(n,1))*1000
   if kind==3:a=np.tile(rng.normal(size=(n,max(1,m//5))),(1,6))[:,:m]
   rr,cc=dense_assignment(a);r,c,_=solve(a)
   assert len(set(c.tolist()))==len(r) and np.array_equal(rr,r)
   assert abs(float(a[r,c].sum()-a[rr,cc].sum()))<1e-8
   cases+=1
 return cases
