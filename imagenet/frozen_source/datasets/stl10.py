from torchvision.datasets import STL10 as S10
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))]
    )


def test_transform(size):
    resize = 70 if size == 64 else 96
    return T.Compose(
        [T.Resize(resize, interpolation=3), T.CenterCrop(size), base_transform()]
    )


class STL10(BaseDataset):
    def ds_train(self):
        t = MultiSample(
            aug_transform(
                self.aug_cfg.stl_train_crop_size, base_transform, self.aug_cfg
            ),
            n=self.aug_cfg.num_samples,
        )
        return S10(root="./data", split="train+unlabeled", download=False, transform=t)

    def ds_clf(self):
        t = test_transform(self.aug_cfg.stl_eval_crop_size)
        return S10(root="./data", split="train", download=False, transform=t)

    def ds_test(self):
        t = test_transform(self.aug_cfg.stl_eval_crop_size)
        return S10(root="./data", split="test", download=False, transform=t)
