import argparse
from functools import partial

from datasets import DS_LIST
from methods import METHOD_LIST
from torchvision import models


def get_cfg():
    parser = argparse.ArgumentParser(
        description="CIFAR representation learning with a trainable spherical velocity field"
    )
    parser.add_argument("--method", choices=METHOD_LIST, default="dm")
    parser.add_argument("--dataset", choices=DS_LIST, default="cifar10")
    parser.add_argument(
        "--stl-train-crop-size",
        type=int,
        default=64,
        choices=[64, 80, 96],
        help="STL-10 self-supervised crop resolution.",
    )
    parser.add_argument(
        "--stl-eval-crop-size",
        type=int,
        default=64,
        choices=[64, 96],
        help="STL-10 downstream train/test center-crop resolution.",
    )
    parser.add_argument(
        "--stl-maxpool",
        choices=["enabled", "disabled"],
        default="enabled",
        help="Keep or remove the ResNet stem max-pool for STL-10.",
    )
    parser.add_argument("--arch", choices=[x for x in dir(models) if "resn" in x], default="resnet18")
    parser.add_argument(
        "--init-mode",
        choices=["pretrained", "random"],
        default="pretrained",
        help="Backbone initialization: torchvision ImageNet-1K weights or weights=None.",
    )
    parser.add_argument("--emb", type=int, default=128)
    parser.add_argument("--Kprime", type=int, default=64)
    parser.add_argument("--prior", choices=["etf", "basis"], default="etf")
    parser.add_argument(
        "--centers-file",
        type=str,
        default=None,
        help=(
            "Optional torch file containing a fixed [Kprime, emb] centers tensor "
            "or a dict with key 'centers'. This removes node-dependent eigenspace "
            "choices in controlled ablations."
        ),
    )
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--target-redundancy", type=int, default=1)
    parser.add_argument("--pool-extra-repeats", type=int, default=0)
    parser.add_argument(
        "--capacity-schedule",
        nargs="*",
        default=[],
        metavar="EPOCH:SLOTS",
        help=(
            "Optional zero-based schedule for the number of repeated target slots "
            "per center, for example '0:4 100:6 250:9'. Empty preserves the "
            "historical static target-redundancy/pool-extra-repeats path."
        ),
    )
    parser.add_argument("--target-mode", choices=["shared_point", "shared_center"], default="shared_point")
    parser.add_argument(
        "--assignment-queue-size",
        type=int,
        default=0,
        help=(
            "Number of detached consensus features retained for sliding-window "
            "balanced assignment. Zero preserves the original per-batch Hungarian."
        ),
    )
    parser.add_argument(
        "--assignment-epoch-global",
        action="store_true",
        help=(
            "Use online epoch-wide remaining quotas: each minibatch may be "
            "non-uniform, while the final epoch marginal is exactly uniform."
        ),
    )
    parser.add_argument("--assignment-global-alpha", type=float, default=0.5)
    parser.add_argument("--assignment-global-cap", type=int, default=4)
    parser.add_argument("--assignment-global-temperature", type=float, default=0.05)
    parser.add_argument("--assignment-global-dual-lr", type=float, default=0.0)
    parser.add_argument("--assignment-global-dual-clip", type=float, default=0.2)
    parser.add_argument(
        "--assignment-feature-source",
        choices=["online", "ema"],
        default="online",
    )
    parser.add_argument("--assignment-teacher-momentum", type=float, default=0.99)
    parser.add_argument(
        "--path-geometry",
        choices=["spherical", "euclidean"],
        default="spherical",
        help="Flow-matching interpolation/ODE geometry.",
    )
    parser.add_argument(
        "--time-sampling",
        choices=["path_uniform", "time_uniform"],
        default="path_uniform",
    )
    parser.add_argument(
        "--time-map",
        choices=[
            "linear",
            "frontload_quadratic",
            "frontload_power",
            "frontload_exponential",
            "frontload_rational",
            "frontload_sine",
        ],
        default="frontload_quadratic",
        help=(
            "Map ODE time t to path coordinate s. frontload_quadratic uses "
            "s(t)=1-(1-t)^2 and ds/dt=2(1-t); linear uses s(t)=t. "
            "The power/rate of parameterized maps is set by --time-map-param."
        ),
    )
    parser.add_argument("--time-map-param", type=float, default=2.0)
    parser.add_argument(
        "--flow-path-mode",
        choices=["single", "coarse_fine"],
        default="single",
    )
    parser.add_argument("--flow-stage-rho", type=float, default=0.5)
    parser.add_argument("--flow-stage-split", type=float, default=0.5)
    parser.add_argument(
        "--velocity-mode",
        choices=["standard", "residual"],
        default="standard",
    )
    parser.add_argument("--velocity-hidden", type=int, default=2048)
    parser.add_argument("--velocity-layers", type=int, default=8)
    parser.add_argument(
        "--velocity-hidden-bias",
        choices=["enabled", "disabled"],
        default="enabled",
    )
    parser.add_argument(
        "--velocity-output-bias",
        choices=["enabled", "disabled"],
        default="enabled",
    )
    parser.add_argument(
        "--velocity-tangent-projection",
        action="store_true",
        help="Project flow-matching predictions onto the sphere tangent plane.",
    )
    parser.add_argument(
        "--velocity-radial-weight",
        type=float,
        default=0.0,
        help="Weight for radial velocity-vector MSE on spherical paths.",
    )
    parser.add_argument(
        "--time-feature-mode",
        choices=["raw", "centered"],
        default="raw",
    )
    parser.add_argument(
        "--residual-alpha",
        choices=["constant", "bell"],
        default="constant",
    )
    parser.add_argument("--residual-regularization", type=float, default=0.1)
    parser.add_argument("--normalize-assignment-consensus", action="store_true")
    parser.add_argument("--lambda-param", dest="lambda_param", type=float, default=50.0)
    parser.add_argument("--fm-weight", type=float, default=1.0)
    parser.add_argument(
        "--fm-loss-mode",
        choices=["mse", "huber_matched"],
        default="mse",
        help="Flow-matching residual loss; huber_matched preserves local MSE curvature.",
    )
    parser.add_argument(
        "--fm-time-weight-gamma",
        type=float,
        choices=[-1.0, -0.5, 0.0, 0.5, 1.0],
        default=0.0,
        help=(
            "Exponent gamma for the normalized per-sample flow-matching weight "
            "(s_dot^2)^gamma. Gamma zero preserves the legacy loss exactly. "
            "Nonzero values are supported only for path-uniform rational time."
        ),
    )
    parser.add_argument("--endpoint-weight", type=float, default=0.0)
    parser.add_argument("--soft-occupancy-weight", type=float, default=0.0)
    parser.add_argument("--soft-occupancy-temperature", type=float, default=0.1)
    parser.add_argument("--z0-variance-weight", type=float, default=0.0)
    parser.add_argument("--z0-covariance-weight", type=float, default=0.0)
    parser.add_argument("--z0-uniformity-weight", type=float, default=0.0)
    parser.add_argument("--z0-crosscorr-weight", type=float, default=0.0)
    parser.add_argument("--backbone-alignment-weight", type=float, default=0.0)
    parser.add_argument("--backbone-variance-weight", type=float, default=0.0)
    parser.add_argument("--backbone-covariance-weight", type=float, default=0.0)
    parser.add_argument("--detach-fm-encoder", action="store_true")
    parser.add_argument(
        "--fm-encoder-grad-scale",
        type=float,
        default=1.0,
        help=(
            "Scale only the FM-loss gradient entering encoder/head features; "
            "1 preserves the legacy graph."
        ),
    )
    parser.add_argument("--alignment-space", choices=["z0", "z1"], default="z0")
    parser.add_argument("--train-ode-steps", type=int, default=8)
    parser.add_argument("--eval-ode-steps", type=int, default=50)
    parser.add_argument(
        "--eval-representations",
        nargs="+",
        choices=["backbone", "z0", "z1"],
        default=["backbone", "z0", "z1"],
    )
    parser.add_argument("--linear-probe-epochs", type=int, default=500)
    parser.add_argument(
        "--eval-at-start",
        action="store_true",
        help="Evaluate the untrained backbone once before epoch 0.",
    )
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--encoder-lr-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to --lr for encoder parameters.",
    )
    parser.add_argument(
        "--head-velocity-lr-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to --lr for all non-encoder trainable parameters.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of optimizer steps used for linear learning-rate warmup.",
    )
    parser.add_argument("--adam-l2", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer-name",
        choices=["adam", "radam", "nadam"],
        default="adam",
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="Adam first-moment coefficient.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.99,
        help="Adam second-moment coefficient; beta1 remains fixed at 0.9.",
    )
    parser.add_argument(
        "--adam-amsgrad",
        action="store_true",
        help="Use the AMSGrad maximum second-moment variant of Adam.",
    )
    parser.add_argument(
        "--gradient-centralization",
        action="store_true",
        help="Center gradients of matrix/convolution parameters before each optimizer step.",
    )
    parser.add_argument(
        "--adam-decay-mode",
        choices=["all", "selective"],
        default="all",
        help=(
            "Apply coupled L2 to every trainable parameter, or only to "
            "convolution/linear-style weights while exempting bias and BN affine."
        ),
    )
    parser.add_argument(
        "--adam-decay-scope",
        choices=["all_modules", "encoder_head", "encoder_only"],
        default="all_modules",
        help="Module scope used by selective coupled L2.",
    )
    parser.add_argument(
        "--bn-momentum",
        type=float,
        default=0.1,
        help="Momentum assigned to every BatchNorm module before training.",
    )
    parser.add_argument("--lr-step", choices=["cos", "step", "none"], default="cos")
    parser.add_argument("--step-milestones", type=int, nargs="*", default=[])
    parser.add_argument(
        "--cos-step-milestones",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional zero-based epoch milestones whose multiplicative drops are "
            "applied on top of the cosine learning rate. This reproduces the "
            "historical cosine-plus-late-step schedule."
        ),
    )
    parser.add_argument("--drop-gamma", type=float, default=0.2)
    parser.add_argument("--T0", type=int, default=None)
    parser.add_argument("--Tmult", type=int, default=1)
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--extra-eval-epochs", type=int, nargs="*", default=[])
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Checkpoint cadence in epochs, independent of evaluation cadence.",
    )
    parser.add_argument("--extra-save-epochs", type=int, nargs="*", default=[])
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=1000)
    parser.add_argument(
        "--head-hidden-bias",
        choices=["enabled", "disabled"],
        default="enabled",
    )
    parser.add_argument(
        "--head-output-norm",
        choices=["none", "bn_affine", "bn_noaffine", "ln_noaffine"],
        default="none",
    )
    parser.add_argument(
        "--head-output-bias",
        choices=["enabled", "disabled"],
        default="enabled",
    )
    parser.add_argument("--no-add-bn", dest="add_bn", action="store_false")
    parser.set_defaults(add_bn=True)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--eval-head", action="store_true")
    parser.add_argument("--fname", type=str)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--diag-grad-every-steps", type=int, default=0)
    parser.add_argument("--diag-grad-every-epochs", type=int, default=10)
    parser.add_argument("--diag-natural-every", type=int, default=10)
    parser.add_argument("--diag-natural-batches", type=int, default=1)
    parser.add_argument(
        "--diag-natural-ode-steps",
        type=int,
        default=8,
    )
    addf = partial(parser.add_argument, type=float)
    addf("--cj-bright", default=0.4)
    addf("--cj-contrast", default=0.4)
    addf("--cj-sat", default=0.4)
    addf("--cj-hue", default=0.1)
    addf("--cj-prob", default=0.8)
    addf("--gs-prob", default=0.2)
    addf("--crop-s0", default=0.2)
    addf("--crop-s1", default=1.0)
    addf("--crop-r0", default=0.75)
    addf("--crop-r1", default=4 / 3)
    addf("--hf-prob", default=0.5)
    addf("--blur-prob", default=0.5)
    parser.add_argument(
        "--blur-kernel-size",
        type=int,
        choices=[3, 5, 7, 9],
        default=3,
        help="Odd Gaussian-blur kernel width; three preserves the incumbent.",
    )
    addf("--solarize-prob", default=0.0)
    parser.add_argument("--solarize-threshold", type=int, default=128)
    addf("--autocontrast-prob", default=0.0)
    addf("--equalize-prob", default=0.0)
    addf("--posterize-prob", default=0.0)
    parser.add_argument("--posterize-bits", type=int, default=4)
    addf("--sharpness-prob", default=0.0)
    addf("--sharpness-factor", default=1.5)
    addf("--cutout-prob", default=0.0)
    addf("--kernel-size", default=0.1)
    return parser.parse_args()
