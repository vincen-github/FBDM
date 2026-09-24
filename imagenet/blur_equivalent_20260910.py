"""CPU separable blur: preserve torchvision .14 PIL conversion and RNG draws."""
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class MatchedBlur(T.GaussianBlur):
    def __init__(self, original, mode):
        super().__init__(original.kernel_size, original.sigma)
        self.mode = mode

    def forward(self, img):
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        if self.mode == 'disabled':
            return img                                                                        
        return separable_blur(img, self.kernel_size, sigma)

def separable_blur(img, kernel_size, sigma):
    was_pil = isinstance(img, Image.Image)
    tensor = F.pil_to_tensor(img) if was_pil else img
    dtype = tensor.dtype
    x = tensor.unsqueeze(0).to(torch.float32)
    kx, ky = kernel_size
    def kernel(k):
        grid = torch.linspace(-(k-1)*.5, (k-1)*.5, steps=k, device=x.device)
        vals = torch.exp(-.5 * (grid / sigma).pow(2))
        return vals / vals.sum()
    x = torch.nn.functional.pad(x, [kx//2, kx//2, ky//2, ky//2], mode='reflect')
    channels = x.shape[1]
    x = torch.nn.functional.conv2d(x, kernel(kx).view(1,1,1,kx).expand(channels,1,1,kx), groups=channels)
    x = torch.nn.functional.conv2d(x, kernel(ky).view(1,1,ky,1).expand(channels,1,ky,1), groups=channels)
    if not dtype.is_floating_point:
        x = x.round()
    x = x.squeeze(0).to(dtype)
    return F.to_pil_image(x, mode=img.mode) if was_pil else x

def replace_blur(compose, mode):
    assert mode in ('original', 'disabled', 'separable')
    count = 0
    for op in compose.transforms:
        if isinstance(op, T.RandomApply):
            for i, child in enumerate(op.transforms):
                if isinstance(child, T.GaussianBlur):
                    count += 1
                    if mode != 'original':
                        op.transforms[i] = MatchedBlur(child, mode)
    assert count == 1, count
