# FBDM

Self-supervised visual representation learning with reference-target assignment and flow matching.

Install the Python dependencies with `pip install -r requirements.txt`.

## Repository structure

```text
FBDM/
├── run.py
├── cifar10/
├── cifar100/
├── stl10/
├── tinyimagenet/
└── imagenet/
```

The four small-dataset directories contain their own configurations, reference tensors, model code, data loaders, and evaluation utilities. ImageNet has a separate distributed training pipeline.

## CIFAR-10, CIFAR-100, STL-10, and Tiny ImageNet

Select a dataset through the repository-level [run.py](run.py). It reads that dataset's `config.json` and launches training from the corresponding directory:

```bash
python run.py cifar10 --check  # Print the training command.
python run.py cifar10          # Launch training.
```

Replace `cifar10` with `cifar100`, `stl10`, or `tinyimagenet`. Prepare the dataset at the location specified in that directory's `datasets/` loader.

Each dataset directory uses the following layout:

| File or directory | Function |
| --- | --- |
| `config.json`, `cfg.py` | Training configuration and command-line options. |
| `centers.pt` | Fixed reference-target tensor. |
| `train_from_scratch_init.py`, `from_scratch_init.py` | Training entry point and initialization helpers. |
| `train.py` | Training loop, evaluation, and checkpoint saving. |
| `model.py` | Backbone, projection head, and velocity network. |
| `methods/dm.py` | Target assignment, flow-matching objective, and alignment loss. |
| `methods/capacitated_assignment.py` | Exact capacity-constrained matching without duplicated center nodes. |
| `methods/utils.py`, `methods/utils_etf_reference.py` | Reference-target construction utilities. |
| `datasets/` | Dataset loading and image augmentations. |
| `eval/get_data.py` | Feature extraction for evaluation. |
| `eval/sgd.py`, `eval/knn.py` | Linear classification and nearest-neighbor evaluation. |
| `instrumentation.py` | Training diagnostics and logging. |

Each dataset keeps its own model and training recipe. Data loading defaults to eight workers; adjust `num_workers` in `config.json` for the available CPU cores. Matching retains the original costs and capacity constraints; tied optima may select different equally optimal targets.

## ImageNet

The ImageNet implementation trains a ResNet-50 encoder, a projection head, and a 20-layer velocity network (hidden width 2048). Reference targets are assigned within each global mini-batch. The objective combines flow matching and view alignment.

The configuration in `imagenet/policies.json` uses a global batch size of 512 on four GPUs, FP32 velocity prediction, and automatic mixed precision for the encoder and projector. The reference contains 4096 centers. `pool_extra_repeats=4` is the extra reference-pool repetition count (M); the number of slots per center is `ceil(batch_size / Kprime) + M`, which is five for this configuration.

The ImageNet experiment uses PyTorch 2.2.1 and torchvision 0.17.1. Install its dependencies with `pip install -r imagenet/requirements.txt` from the repository root.

Set the training and validation manifest paths in `imagenet/paths.json`, then run:

```bash
cd imagenet
python run.py --audit
python -m torch.distributed.run --standalone --nproc_per_node=4 run.py --smoke --output runs/smoke
python run.py --validate runs/smoke
python -m torch.distributed.run --standalone --nproc_per_node=4 run.py --output runs/imagenet
```

Training runs for 100 epochs with a 10-epoch learning-rate warmup and cosine decay. Each GPU process uses four data-loader workers. Atomic `latest.pt` checkpoints are saved after each epoch; snapshots are retained at epochs 3, 5, 10, 20, 50, 75, and 100.

Run full linear evaluation separately on a saved encoder checkpoint:

```bash
python full_linear_eval.py --checkpoint runs/imagenet/e100.pt --train-manifest /path/to/train.manifest.json --val-manifest /path/to/val.manifest.json --output runs/linear --expected-encoder-epochs 100
```

Evaluation freezes the backbone and its batch-normalization statistics, uses raw 2048-dimensional backbone features, and trains a linear classifier for 500 epochs (seed 0). Input preprocessing is `Resize(256)`, `CenterCrop(224)`, and `ToTensor()`, without ImageNet mean/std normalization.

The training entry point is `imagenet/run.py`; `imagenet/frozen_source/` contains the model and objective. Matching solvers live in `imagenet/certified_assignment.py` and its helper modules. `imagenet/imagenet_hdf5/` provides dataset building, loading, and validation utilities. Reference tensors in `imagenet/artifacts/` are required inputs, not pretrained checkpoints.
