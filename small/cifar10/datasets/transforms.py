import torch
import torchvision.transforms as T


class RandomSolarize(object):
    def __init__(self, threshold=128, p=0.2):
        self.threshold = threshold
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return T.functional.solarize(img, self.threshold)
        return img


class RandomAutocontrast(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return T.functional.autocontrast(img)
        return img


class RandomEqualize(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return T.functional.equalize(img)
        return img


class RandomPosterize(object):
    def __init__(self, bits=4, p=0.1):
        self.bits = bits
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return T.functional.posterize(img, self.bits)
        return img


class RandomSharpness(object):
    def __init__(self, factor=1.5, p=0.1):
        self.factor = factor
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return T.functional.adjust_sharpness(img, self.factor)
        return img


def aug_transform(crop, base_transform, cfg, extra_t=[]):
    """Build the two-view augmentation from the sealed experiment config."""
    if not 0.0 <= cfg.cutout_prob <= 1.0:
        raise ValueError("cutout probability must be in [0, 1]")
    if not 0.0 <= cfg.solarize_prob <= 1.0:
        raise ValueError("solarize probability must be in [0, 1]")
    for name in ("autocontrast_prob", "equalize_prob", "posterize_prob", "sharpness_prob"):
        if not 0.0 <= getattr(cfg, name) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if not 0 <= cfg.solarize_threshold <= 255:
        raise ValueError("solarize threshold must be in [0, 255]")
    if not 1 <= cfg.posterize_bits <= 8:
        raise ValueError("posterize bits must be in [1, 8]")
    if cfg.sharpness_factor < 0:
        raise ValueError("sharpness factor must be nonnegative")
    transforms = [
            T.RandomApply(
                [
                    T.ColorJitter(
                        cfg.cj_bright,
                        cfg.cj_contrast,
                        cfg.cj_sat,
                        cfg.cj_hue,
                    )
                ],
                p=cfg.cj_prob,
            ),
            T.RandomGrayscale(p=cfg.gs_prob),
            T.RandomResizedCrop(
                crop,
                scale=(cfg.crop_s0, cfg.crop_s1),
                ratio=(cfg.crop_r0, cfg.crop_r1),
                interpolation=3,
            ),
            T.RandomHorizontalFlip(p=cfg.hf_prob),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=cfg.blur_prob,
            ),
        ]
    if cfg.solarize_prob:
        transforms.append(
            RandomSolarize(threshold=cfg.solarize_threshold, p=cfg.solarize_prob)
        )
    if cfg.autocontrast_prob:
        transforms.append(RandomAutocontrast(p=cfg.autocontrast_prob))
    if cfg.equalize_prob:
        transforms.append(RandomEqualize(p=cfg.equalize_prob))
    if cfg.posterize_prob:
        transforms.append(
            RandomPosterize(bits=cfg.posterize_bits, p=cfg.posterize_prob)
        )
    if cfg.sharpness_prob:
        transforms.append(
            RandomSharpness(factor=cfg.sharpness_factor, p=cfg.sharpness_prob)
        )
    transforms.extend([*extra_t, base_transform()])
    if cfg.cutout_prob:
        transforms.append(
            T.RandomErasing(
                p=cfg.cutout_prob,
                scale=(0.02, 0.15),
                ratio=(0.5, 2.0),
                value=0.0,
            )
        )
    return T.Compose(transforms)


class MultiSample:
    """Generate ``n`` independently augmented samples."""

    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))
