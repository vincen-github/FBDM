# FBDM

Self-supervised visual representation learning with reference-target assignment and flow matching.

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
| `methods/utils.py`, `methods/utils_etf_reference.py` | Reference-target construction utilities. |
| `datasets/` | Dataset loading and image augmentations. |
| `eval/get_data.py` | Feature extraction for evaluation. |
| `eval/sgd.py`, `eval/knn.py` | Linear classification and nearest-neighbor evaluation. |
| `instrumentation.py` | Training diagnostics and logging. |

## ImageNet

Use the files inside [imagenet/](imagenet/). Set the dataset manifest paths in `paths.json` and select the training configuration from `policies.json`. The included training runner uses four CUDA GPUs; the full linear evaluator runs separately on a frozen backbone checkpoint.

| File or directory | Function |
| --- | --- |
| [run.py](imagenet/run.py) | Distributed training, configuration audits, short validation runs, and checkpoint saving. |
| [policies.json](imagenet/policies.json), [paths.json](imagenet/paths.json) | Training configuration, reference-tensor checks, and train/validation manifest paths. |
| [artifacts/](imagenet/artifacts/) | Fixed reference-target tensor. |
| [frozen_source/](imagenet/frozen_source/) | Backbone, projection head, velocity network, and FBDM objective used by the runner. |
| [imagenet_ddp_smoke_20260909.py](imagenet/imagenet_ddp_smoke_20260909.py) | Distributed model setup and training-view construction. |
| [logical_assignment_batch.py](imagenet/logical_assignment_batch.py) | Target planning over a matching window spanning multiple physical batches. |
| [certified_assignment.py](imagenet/certified_assignment.py), [capacitated_exact_assignment.py](imagenet/capacitated_exact_assignment.py), [exact_assignment.py](imagenet/exact_assignment.py) | Capacity-constrained matching and solver validation. |
| [optimizer_recipe.py](imagenet/optimizer_recipe.py), [upstream_lars.py](imagenet/upstream_lars.py) | Optimizer construction and learning-rate scheduling. |
| [velocity_constraints.py](imagenet/velocity_constraints.py) | Velocity constraints, energy regularization, and transport diagnostics. |
| [amp_overflow_guard.py](imagenet/amp_overflow_guard.py), [lars_wd_audit.py](imagenet/lars_wd_audit.py), [diagnostic_adapters.py](imagenet/diagnostic_adapters.py) | Numerical safety and training checks. |
| [training_monitor.py](imagenet/training_monitor.py) | Intermediate subset evaluation during training. |
| [full_linear_eval.py](imagenet/full_linear_eval.py) | Full ImageNet linear evaluation with a frozen backbone. |
| [imagenet_hdf5/builder.py](imagenet/imagenet_hdf5/builder.py), [dataset.py](imagenet/imagenet_hdf5/dataset.py), [validator.py](imagenet/imagenet_hdf5/validator.py) | Build, read, and validate sharded HDF5 datasets and their JSON manifests. |
