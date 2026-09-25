"""Frozen R50 backbone, deterministic single crop, 500-epoch Adam linear probe.

Extract once into disk-backed FP32 features. Train batches are staged to GPU;
no representation gradients, BN updates, ODE, or projection-head evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
from imagenet_hdf5.dataset import ImageNetHDF5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--val-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe-seed', type=int, default=0)
    p.add_argument('--expected-encoder-epochs', type=int, default=100)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.probe_seed)
    state = torch.load(args.checkpoint, map_location='cpu')
    assert state['config']['arch'] == 'resnet50'
    assert state['completed_epochs'] == args.expected_encoder_epochs
    assert not state['smoke']
    input_normalization = state['config'].get('input_normalization', True)
    assert isinstance(input_normalization, bool)
    backbone = models.resnet50(weights=None)
    backbone.fc = torch.nn.Identity()
    encoder = state.get('encoder')
    if encoder is None:
        encoder = {k[len('model.'):]:v for k,v in state['model'].items() if k.startswith('model.')}
    backbone.load_state_dict(encoder, strict=True)
    backbone.requires_grad_(False).cuda().eval()
    completed_epochs = state['completed_epochs']
    del state, encoder
    transforms = [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
    if input_normalization:
        transforms.append(T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)))
    transform = T.Compose(transforms)
    mappings, arrays = [], {}
    for name, manifest, count in [('train',args.train_manifest,1281167),
                                  ('val',args.val_manifest,50000)]:
        dataset = ImageNetHDF5(manifest, transform=transform)
        assert len(dataset) == count and len(dataset.classes) == 1000
        mappings.append(dataset.class_to_idx)
        features = np.lib.format.open_memmap(args.output/(name+'_x.npy'), mode='w+',
                                             dtype=np.float32, shape=(count,2048))
        labels = np.lib.format.open_memmap(args.output/(name+'_y.npy'), mode='w+',
                                           dtype=np.int64, shape=(count,))
        loader = DataLoader(dataset, batch_size=128, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        offset = 0
        with torch.no_grad():
            for images, targets in loader:
                values = backbone(images.cuda(non_blocking=True))
                assert bool(torch.isfinite(values).all())
                n = len(targets)
                features[offset:offset+n] = values.cpu().numpy()
                labels[offset:offset+n] = targets.numpy()
                offset += n
        assert offset == count
        features.flush()
        labels.flush()
        arrays[name] = (features,labels)
        print(json.dumps(dict(event='features_complete', split=name, samples=count)), flush=True)
    assert mappings[0] == mappings[1]
    del backbone
    torch.cuda.empty_cache()
    clf = torch.nn.Linear(2048,1000).cuda()
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1e-6/1e-2)**(1/500))
    train_x, train_y = arrays['train']
    # Keep the ~10 GiB feature matrix in CPU RAM if available. This avoids
    # re-reading the shared filesystem for each of 500 probe epochs.
    train_x = torch.from_numpy(np.array(train_x))
    train_y = torch.from_numpy(np.array(train_y))
    for epoch in range(500):
        clf.train()
        order = torch.randperm(len(train_y))
        for indices in order.split(1000):
            # Unlike the original view(-1,1000), retain ImageNet's final 167 examples.
            x, y = train_x[indices].cuda(), train_y[indices].cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(clf(x),y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('nonfinite probe loss')
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch+1) % 25 == 0:
            print(json.dumps(dict(event='probe_progress', epoch=epoch+1)), flush=True)
    clf.eval()
    val_x, val_y = arrays['val']
    correct1 = correct5 = 0
    with torch.no_grad():
        for start in range(0,len(val_y),1000):
            x = torch.from_numpy(np.array(val_x[start:start+1000])).cuda()
            y = torch.from_numpy(np.array(val_y[start:start+1000])).cuda()
            predictions = clf(x).topk(5,dim=1).indices
            correct1 += int((predictions[:,0] == y).sum())
            correct5 += int((predictions == y[:,None]).any(dim=1).sum())
    result = dict(status='ok', top1=100*correct1/len(val_y), top5=100*correct5/len(val_y),
                  encoder_epochs=completed_epochs, probe_epochs=500, probe_seed=args.probe_seed,
                  feature='2048-d backbone; raw features; no projection or ODE',
                  batchnorm='eval/frozen', input_normalization=input_normalization,
                  checkpoint_sha256=sha(args.checkpoint),
                  train_manifest_sha256=sha(args.train_manifest),
                  val_manifest_sha256=sha(args.val_manifest), transform=repr(transform))
    torch.save(clf.cpu().state_dict(), args.output/'linear_head.pt')
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
