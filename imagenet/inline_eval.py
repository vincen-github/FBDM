"""Collective interim evaluation; frozen backbone, isolated RNG, separate clock."""
import hashlib,json,random,time
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from torchvision import transforms as T
from imagenet_hdf5.dataset import ImageNetHDF5

def evaluation_transform(input_normalization=True):
    transforms=[T.Resize(256),T.CenterCrop(224),T.ToTensor()]
    if input_normalization:transforms.append(T.Normalize((.485,.456,.406),(.229,.224,.225)))
    return T.Compose(transforms)

def evaluate(backbone,train_manifest,val_manifest,output,epoch,workers=4,per_class=100,probe_epochs=50,input_normalization=True):
    rank,world=dist.get_rank(),dist.get_world_size()
    local=torch.cuda.current_device()
    output=Path(output)
    modes=[(m,m.training) for m in backbone.modules()]
    buffers={n:t.detach().clone() for n,t in backbone.named_buffers()}
    py_state,np_state=random.getstate(),np.random.get_state()
    start=time.perf_counter()
    result=None
    try:
        with torch.random.fork_rng(devices=[local]):
            torch.manual_seed(0);random.seed(0);np.random.seed(0)
            backbone.eval()
            if rank==0:output.mkdir(parents=True,exist_ok=False)
            dist.barrier()
            transform=evaluation_transform(input_normalization)
            arrays={};index_hashes={};mappings=[]
            for split,manifest,nclass in [('train',train_manifest,per_class),('val',val_manifest,10 if per_class==100 else 1)]:
                ds=ImageNetHDF5(manifest,transform=transform)
                mappings.append(ds.class_to_idx)
                labels=[]
                for shard in ds.shards:
                    with h5py.File(shard,'r') as f:labels.append(np.asarray(f['labels'][:]))
                labels=np.concatenate(labels)
                rng=np.random.RandomState(20260910)
                indices=np.concatenate([rng.choice(np.flatnonzero(labels==c),nclass,replace=False) for c in range(1000)])
                assert len(indices)%world==0
                index_hashes[split]=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
                if rank==0:np.save(output/(split+'_indices.npy'),indices)
                chunk=indices.reshape(world,-1)[rank]
                loader=DataLoader(Subset(ds,chunk.tolist()),batch_size=128,num_workers=workers,
                    pin_memory=False,shuffle=False,generator=torch.Generator().manual_seed(1234+rank))
                xs,ys=[],[]
                with torch.no_grad():
                    for images,target in loader:
                        features=backbone(images.cuda(non_blocking=True))
                        assert bool(torch.isfinite(features).all())
                        xs.append(features.cpu());ys.append(target)
                x=torch.cat(xs).cuda();y=torch.cat(ys).cuda()
                gx=[torch.empty_like(x) for _ in range(world)] if rank==0 else None
                gy=[torch.empty_like(y) for _ in range(world)] if rank==0 else None
                dist.gather(x,gather_list=gx,dst=0);dist.gather(y,gather_list=gy,dst=0)
                if rank==0:arrays[split]=(torch.cat(gx).cpu(),torch.cat(gy).cpu())
                del x,y,gx,gy,xs,ys,loader,ds
                torch.cuda.empty_cache()
            assert mappings[0]==mappings[1]
            if rank==0:
                torch.manual_seed(0)
                train_x,train_y=arrays['train'];val_x,val_y=arrays['val']
                clf=torch.nn.Linear(2048,1000).cuda()
                opt=torch.optim.Adam(clf.parameters(),lr=.01,weight_decay=5e-6)
                sched=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=(1e-6/.01)**(1/probe_epochs))
                for _ in range(probe_epochs):
                    for ids in torch.randperm(len(train_y)).split(1000):
                        opt.zero_grad(set_to_none=True)
                        loss=torch.nn.functional.cross_entropy(clf(train_x[ids].cuda()),train_y[ids].cuda())
                        assert bool(torch.isfinite(loss));loss.backward();opt.step()
                    sched.step()
                correct1=correct5=knn_correct=0
                with torch.no_grad():
                    bank=torch.nn.functional.normalize(train_x.cuda(),dim=1);bank_y=train_y.cuda()
                    for ids in torch.arange(len(val_y)).split(128):
                        q=val_x[ids].cuda();target=val_y[ids].cuda()
                        pred=clf(q).topk(5,dim=1).indices
                        correct1+=int((pred[:,0]==target).sum());correct5+=int((pred==target[:,None]).any(dim=1).sum())
                        neighbors=bank_y[(torch.nn.functional.normalize(q,dim=1)@bank.T).topk(5,dim=1).indices]
                        votes=torch.zeros(len(q),1000,device=q.device)
                        votes.scatter_add_(1,neighbors,torch.ones_like(neighbors,dtype=torch.float32))
                        knn_correct+=int((votes.argmax(1)==target).sum())
                result=dict(status='ok',completed_epochs=epoch,top1=100*correct1/len(val_y),
                    top5=100*correct5/len(val_y),knn5=100*knn_correct/len(val_y),
                    protocol='INTERIM frozen2048 backbone subset100k/10k linear50 + cosine5NN; NOT full ImageNet',
                    train_per_class=per_class,probe_epochs=probe_epochs,subset_index_raw_sha256=index_hashes,
                    bn='eval; buffers unchanged',input_normalization=input_normalization,evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    seconds=time.perf_counter()-start)
                del clf,opt,sched,bank,bank_y,q,pred,votes,neighbors,arrays
            dist.barrier()
            assert all(torch.equal(buffers[n],t) for n,t in backbone.named_buffers()),'Evaluation changed BN buffers'
    finally:
        for m,flag in modes:m.training=flag
        random.setstate(py_state);np.random.set_state(np_state)
        torch.cuda.empty_cache()
    if rank==0:
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(event='inline_eval_complete',**result)),flush=True)
    return result
