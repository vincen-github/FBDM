from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from .base import BaseDataset
from .transforms import MultiSample, aug_transform


TINY_IMAGENET_MEAN = (0.480, 0.448, 0.398)
TINY_IMAGENET_STD = (0.277, 0.269, 0.282)


def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)]
    )


class TinyImageNet(BaseDataset):
    """Processed Tiny ImageNet with class-folder train and test splits."""

    def ds_train(self):
        transform = MultiSample(
            aug_transform(64, base_transform, self.aug_cfg),
            n=self.aug_cfg.num_samples,
        )
        return ImageFolder(root="data/tiny-imagenet-200/train", transform=transform)

    def ds_clf(self):
        return ImageFolder(
            root="data/tiny-imagenet-200/train", transform=base_transform()
        )

    def ds_test(self):
        return ImageFolder(
            root="data/tiny-imagenet-200/test", transform=base_transform()
        )
