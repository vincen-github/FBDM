"""Bounded e5 classification checks. Original sources and old checkpoints are immutable."""
import argparse,copy,hashlib,json,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
import numpy as np
import torch
from amp_overflow_guard import distributed_amp_step
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen_source'))
from imagenet_ddp_smoke_20260909 import GlobalAssignmentDM,TwoViews,seed_worker,prepare_ddp
from imagenet_hdf5.dataset import ImageNetHDF5
from training_monitor import evaluate
from optimizer_recipe import make_optimizer,learning_rate
import exact_assignment
from lars_wd_audit import audit_optimizer
from velocity_constraints import validate as validate_constraints, transport_probe
from diagnostic_adapters import make_diagnostic_optimizer,assert_smoke_updates,assert_loss_semantics,checked_forward
import certified_assignment as capacitated_exact_assignment
from logical_assignment_batch import install_preassigned_assignment, logical_batches

def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def digest(items):
    h=hashlib.sha256()
    for n,t in items:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def finite(x):
    if isinstance(x,torch.Tensor):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(tuple,list)):return all(finite(v) for v in x)
    return True

def build(cfg,centers):
    torch.manual_seed(0);random.seed(0);np.random.seed(0)
    vstate=None
    if cfg.head_layers==2:
        control=SimpleNamespace(**{**vars(cfg),'head_layers':4})
        m=GlobalAssignmentDM(control,centers)
        vstate=copy.deepcopy(m.v_net.state_dict());del m
        torch.manual_seed(0);random.seed(0);np.random.seed(0)
    m=GlobalAssignmentDM(cfg,centers);assert m.model.init_weights_argument is None
    m.model=m.model.module
    if vstate is not None:m.v_net.load_state_dict(vstate,strict=True)
    if getattr(cfg,"head_final_bn_freeze_beta",False):
        assert isinstance(m.head[-1],torch.nn.BatchNorm1d) and m.head[-1].affine
        m.head[-1].bias.requires_grad_(False)
    if cfg.fm_weight==0:
        for parameter in m.v_net.parameters():parameter.requires_grad_(False)
    validate_constraints(m,cfg)
    return m

def input_policy(name):
    p=json.loads((ROOT/'policies.json').read_text())[name]
    cfg=SimpleNamespace(**p['config'])
    centers=torch.load(ROOT/p['centers'],map_location='cpu')
    assert hashlib.sha256(centers.contiguous().numpy().tobytes()).hexdigest()==p['centers_sha256']
    assert centers.shape==(cfg.Kprime,cfg.emb) and finite(centers)
    return p,cfg,centers

def optimizer(m,cfg):
    opt=make_diagnostic_optimizer(m,SimpleNamespace(**{**vars(cfg),'optimizer_name':'lars'}))
    if cfg.optimizer_name=='adam':
        opt=torch.optim.Adam([dict(params=g['params'],weight_decay=g['weight_decay']) for g in opt.param_groups],lr=cfg.lr,betas=(.9,.99),eps=1e-8)
    audit_optimizer(opt,m,cfg)
    return opt

def geom(m,views,rank,world):
    buffers={n:b.clone() for n,b in m.named_buffers()}
    old_modes={n:v.training for n,v in m.named_modules()};m.train()
    records=[]
    with torch.no_grad():
        for view,x in enumerate(views):
            with torch.autocast('cuda'):h=m.model(x.cuda());z=m.head(h)
            rec=dict(view=view)
            for name,t in [('backbone',h),('z0',torch.nn.functional.normalize(z.float(),dim=1))]:
                parts=[torch.empty_like(t) for _ in range(world)];dist.all_gather(parts,t.contiguous())
                if rank==0:
                    a=torch.cat(parts).float();u=torch.nn.functional.normalize(a,dim=1);n=len(a)
                    c=a-a.mean(0);c_cpu=c.double().cpu();s=torch.linalg.eigvalsh(c_cpu@c_cpu.T).clamp_min(0);prob=s/s.sum().clamp_min(1e-30)
                    rec[name]=dict(cos=float((u.sum(0).square().sum()-n)/(n*(n-1))),std=float(c.square().mean(0).sqrt().mean()),rank=float(torch.exp(-(prob*prob.clamp_min(1e-30).log()).sum())))
            records.append(rec)
    for n,b in m.named_buffers():b.copy_(buffers[n])
    for n,v in m.named_modules():v.training=old_modes[n]
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--audit',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--validate',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    pol,cfg,centers=input_policy(a.policy)
    assert pol.get('from_scratch') is True and not pol.get('resume')
    assert cfg.init_mode=='random'
    if pol['bridge_kind']=='paper_rho':
        assert cfg.fm_weight==1.3 and cfg.endpoint_weight==0. and 0 < cfg.fm_encoder_grad_scale <= 1.
    else:
        assert pol['bridge_kind']=='pure_regression' and cfg.fm_weight==0. and cfg.endpoint_weight==1.3
    assert cfg.dispatch_ramp_epochs==0 and cfg.fm_time_samples==1
    if a.validate:
        state=torch.load(a.validate/'latest.pt',map_location='cpu');ev=json.loads((a.validate/'evidence.json').read_text());done=json.loads((a.validate/'completion.json').read_text())
        assert done['status']=='ok' and done['smoke'] and state['smoke'] and finite(state) and ev['weights'] is None and ev['pretrained'] is False and state['optimizer'] and state['scaler']
        assert ev['centers_sha256']==pol['centers_sha256'] and ev['evaluator_sha256']==hashlib.sha256((ROOT/'training_monitor.py').read_bytes()).hexdigest()
        assert state['config']==vars(cfg) and ev['initial_parameters_sha256']==pol['initial_parameters_sha256']
        if cfg.fm_weight==0:
            assert done.get('endpoint_semantic_checks',0)==32
            assert digest((n[len('v_net.'):],t) for n,t in state['model'].items() if n.startswith('v_net.'))==pol['diagnostic_velocity_initial_sha256']
        print('SMOKE_FINITE_CONFIG_INIT_OK',flush=True);return
    paths=json.loads((ROOT/'paths.json').read_text())
    m=build(cfg,centers);initial=digest(m.named_parameters())
    assert initial==pol['initial_parameters_sha256'],(a.policy,initial,pol['initial_parameters_sha256'])
    if a.audit:
        assert exact_assignment.self_test()==24
        assert len(ImageNetHDF5(paths['train']))==1281167 and len(ImageNetHDF5(paths['val']))==50000
        o=optimizer(m,cfg);o.load_state_dict(o.state_dict())
        if cfg.optimizer_name=='lars':assert o.param_groups is o.optim.param_groups
        print(json.dumps(dict(status='ok',policy=a.policy,initial=initial,torch=torch.__version__)),flush=True);return
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);assert world==4 and cfg.bs==512
    torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=timedelta(hours=1));torch.backends.cudnn.benchmark=True
    m=torch.nn.SyncBatchNorm.convert_sync_batchnorm(m).cuda(local).train()
    opt=optimizer(m,cfg);start_epoch=0;resume_note=None;resume_audit=None
    if pol.get('resume'):
        assert file_sha(Path(pol['resume']))==pol['resume_sha256']
        ck=torch.load(pol['resume'],map_location='cpu');assert finite(ck) and ck['completed_epochs']==pol['resume_epoch'] and ck['config']==pol['resume_base_config'] and not ck['smoke'] and ck['scaler']
        base_cfg=SimpleNamespace(**pol['resume_base_config']);old=build(base_cfg,centers);old_opt=optimizer(old,base_cfg)
        old.load_state_dict(ck['model'],strict=True);old_opt.load_state_dict(ck['optimizer'])
        incoming={k:v for k,v in ck['model'].items() if not k.startswith('v_net.')}
        current=m.state_dict();assert set(incoming)=={k for k in current if not k.startswith('v_net.')}
        for name,tensor in incoming.items():current[name].copy_(tensor.to(current[name].device))
        m.load_state_dict(current,strict=True)
        old_named=dict(old.named_parameters());new_named=dict(m.named_parameters());old_actual=getattr(old_opt,'optim',old_opt);new_actual=getattr(opt,'optim',opt)
        copied=[];missing=[]
        for name,new_parameter in new_named.items():
            if name.startswith('v_net.'):continue
            old_parameter=old_named[name];state=old_actual.state.get(old_parameter,{})
            if not state:missing.append(name);continue
            new_actual.state[new_parameter]={key:(value.detach().clone().to(new_parameter.device) if isinstance(value,torch.Tensor) else copy.deepcopy(value)) for key,value in state.items()}
            copied.append(name)
        assert not any(new_actual.state.get(p,{}) for n,p in new_named.items() if n.startswith('v_net.'))
        start_epoch=int(ck['completed_epochs']);scaler_state=ck['scaler']
        missing_fields=[k for k in ('rng','rng_state','sampler_state','loader_rng_state','assignment_state') if k not in ck]
        assert len(missing_fields)==5 and cfg.assignment_queue_size==0 and not cfg.assignment_epoch_global
        resume_audit=dict(copied_encoder_head_optimizer_states=len(copied),encoder_head_parameters_without_state=len(missing),fresh_velocity_optimizer_state=True,missing_checkpoint_fields=missing_fields)
        del ck,old,old_opt,old_named,old_actual;torch.cuda.empty_cache()
        assert opt.param_groups is opt.optim.param_groups
        resume_note='Encoder/head model, BN, available LARS states, GradScaler and epoch restored; velocity 2048x4 and its optimizer state are fresh. Per-rank RNG was not stored, so bitwise replay is not claimed.'
    install_preassigned_assignment(m); ddp=DDP(prepare_ddp(m),device_ids=[local],broadcast_buffers=False); capacitated_exact_assignment.install(cfg.Kprime)
    scaler=torch.cuda.amp.GradScaler(init_scale=256. if start_epoch else 128.)
    if start_epoch:scaler.load_state_dict(scaler_state)
    if rank==0:
        a.output.mkdir(parents=True,exist_ok=False)
        (a.output/'lars_weight_decay_actual.json').write_text(json.dumps(audit_optimizer(opt,m,cfg),indent=2))
        (a.output/'evidence.json').write_text(json.dumps(dict(policy=a.policy,config=vars(cfg),weights=None,pretrained=False,initial_parameters_sha256=initial,centers_sha256=pol['centers_sha256'],resume=resume_note,resume_audit=resume_audit,velocity_parameter_count=sum(p.numel() for p in m.v_net.parameters()),torch=torch.__version__,evaluator_sha256=hashlib.sha256((ROOT/'training_monitor.py').read_bytes()).hexdigest()),indent=2))
    dist.barrier()
    if not a.smoke:evaluate(m.model,paths['train'],paths['val'],a.output/('eval_e'+str(start_epoch)),start_epoch,workers=4,input_normalization=getattr(cfg,'input_normalization',True))
    ds=ImageNetHDF5(paths['train'],transform=TwoViews(cfg))
    overflow_total=0;overflow_consecutive=0
    watched={n:p for n,p in m.named_parameters() if n in ['model.conv1.weight','head.0.weight','v_net.v.0.weight']}
    formal_epochs=int(pol.get('max_formal_epochs',5))
    for epoch in range(start_epoch,start_epoch+2 if a.smoke else start_epoch+formal_epochs):
        seed=1000+rank+10000*epoch;torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,seed=0);sampler.set_epoch(epoch)
        loader=DataLoader(ds,sampler=sampler,batch_size=128,num_workers=4,pin_memory=True,drop_last=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed))
        assert len(loader)==2502;m.reset_epoch_stats(len(loader));m.train()
        steps=16 if a.smoke else 2502;totals=torch.zeros(3,device='cuda');begin=time.perf_counter();cached=None
        for step,(views,_) in enumerate(logical_batches(loader,m,getattr(cfg,'logical_assignment_batches',1),steps)):
            if cached is None:cached=[v.clone() for v in views]
            lr=learning_rate(cfg,epoch*2502+step,250200)
            for g in opt.param_groups:g['lr']=lr*g.get('module_lr_multiplier',1.)
            actual=getattr(opt,'optim',opt);assert all(g['lr']==lr*g.get('module_lr_multiplier',1.) for g in actual.param_groups)
            diagnostic=step==0 or (step+1)%250==0 or (a.smoke and step==15)
            before={n:p.detach().clone() for n,p in watched.items()} if diagnostic else {}
            if getattr(cfg,'dispatch_ramp_epochs',0):
                alpha=min(1.,((epoch-start_epoch)*2502+step+1)/(cfg.dispatch_ramp_epochs*2502))
                cfg.fm_weight=1.3*alpha;cfg.endpoint_weight=1.3*(1-alpha)
            opt.zero_grad(set_to_none=True)
            velocity_norm=[]
            hook=m.v_net.register_forward_hook(lambda module,inputs,output: velocity_norm.append(float(output.detach().float().norm(dim=1).mean())))
            try:
                with torch.autocast('cuda'):loss=checked_forward(m,ddp,views,cfg,a.smoke)
            finally:
                hook.remove()
            assert_loss_semantics(m,cfg,loss)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            grads={n:float(p.grad.norm()) if p.grad is not None and bool(torch.isfinite(p.grad).all()) else None for n,p in watched.items()} if diagnostic else {}
            amp_result=distributed_amp_step(scaler=scaler,optimizer=opt,loss=loss)
            if amp_result.overflow:
                overflow_total+=1;overflow_consecutive+=1
                offenders=[n for n,p in m.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
                print(json.dumps(dict(event='amp_overflow_recovered',rank=rank,epoch=epoch+1,step=step+1,scale_before=amp_result.scale_before,scale_after=amp_result.scale_after,local_offenders=offenders[:8],overflow_total=overflow_total,overflow_consecutive=overflow_consecutive)),flush=True)
                if overflow_consecutive>=8:raise RuntimeError('persistent AMP gradient overflow on eight consecutive batches')
            else:overflow_consecutive=0
            totals+=torch.stack([loss.detach(),m.last_losses['fm'],m.last_losses['align']])
            if diagnostic:
                ratios={n:float((p-before[n]).norm()/before[n].norm().clamp_min(1e-12)) for n,p in watched.items()}
                if a.smoke and step==15:assert_smoke_updates(m,cfg,ratios)
                if rank==0:
                    row=dict(epoch=epoch+1,step=step+1,loss=float(loss),fm=float(m.last_losses['fm']),alignment=float(m.last_losses['align']),lr_actual=actual.param_groups[0]['lr'],lr_requested=lr,endpoint=float(m.last_losses['endpoint']),velocity_output_norm_mean=(sum(velocity_norm)/len(velocity_norm) if velocity_norm else None),module_lr_actual={g.get('module_name','native'):g['lr'] for g in actual.param_groups},grads=grads,update_weight_ratio=ratios,scaler=scaler.get_scale(),elapsed=time.perf_counter()-begin)
                    row.update(velocity_energy=float(m.last_losses['velocity_energy']),
                               velocity_effective_norm=float(m.last_losses['velocity_effective_norm']),
                               target_velocity_norm=float(m.last_losses['target_velocity_norm']))
                    print(json.dumps(row),flush=True)
                    with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step+1==steps:break
        torch.cuda.synchronize();seconds=time.perf_counter()-begin;dist.all_reduce(totals);totals/=world*steps
        opt.zero_grad(set_to_none=True);gmetrics=[]
        if rank==0 and cfg.fm_weight:
            probe = transport_probe(m,cached)
            (a.output/('transport_e'+str(epoch+1)+'.json')).write_text(json.dumps(probe,indent=2))
        assert finite(m.state_dict()) and finite(opt.state_dict())
        if rank==0:
            state=dict(model=m.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),centers=centers,config=vars(cfg),completed_epochs=epoch+1,smoke=a.smoke)
            tmp=a.output/'latest.tmp.pt';torch.save(state,tmp);os.replace(tmp,a.output/'latest.pt')
            if epoch+1 in [5,10,20,50,100] and not a.smoke:os.link(a.output/'latest.pt',a.output/('e'+str(epoch+1)+'.pt'))
            (a.output/('epoch'+str(epoch+1)+'.json')).write_text(json.dumps(dict(epoch=epoch+1,steps=steps,seconds=seconds,loss=float(totals[0]),fm=float(totals[1]),alignment=float(totals[2]),geometry=gmetrics,assignment=m.assignment_stats()),indent=2))
        dist.barrier()
        if not a.smoke:evaluate(m.model,paths['train'],paths['val'],a.output/('eval_e'+str(epoch+1)),epoch+1,workers=4,input_normalization=getattr(cfg,'input_normalization',True))
        del loader,cached
        if not a.smoke and cfg.fm_weight and epoch+1==5:
            stop_flag=torch.zeros((),device='cuda',dtype=torch.int32)
            if rank==0:
                first=json.loads((a.output/'eval_e1/result.json').read_text())
                fifth=json.loads((a.output/'eval_e5/result.json').read_text())
                severe=fifth['top1']<5. and fifth['top1']<first['top1'] and fifth['knn5']<1.
                if severe:
                    stop_flag.fill_(1)
                    (a.output/'early_exit.json').write_text(json.dumps(dict(reason='Prospective e5 severe-regression gate; complete checkpoint and evaluation retained',e1=first,e5=fifth),indent=2))
            dist.broadcast(stop_flag,src=0)
            if int(stop_flag):break
    if rank==0:(a.output/'completion.json').write_text(json.dumps(dict(status='ok',smoke=a.smoke,completed_epochs=epoch+1,early_exit=(a.output/'early_exit.json').exists(),endpoint_semantic_checks=getattr(m,'_endpoint_semantic_checks',0))))
    dist.destroy_process_group()
if __name__=='__main__':main()
