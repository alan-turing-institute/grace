import torch

import torchvision.transforms as transforms


class RandomEdgeCrop:
    """Trim and pad an image at one of its edges to simulate the edge
    of a field-of-view.
    
    Accepts an image stack of size (C,W,H) or (B,C,W,H)
    
    Crops all images in a stack uniformly.
    
    Args:
        max_fraction (float): Max fraction of image dimension that can be
        cropped.
    """
    
    def __init__(self, max_fraction: float = 0.5):
        self.max_fraction = max_fraction
        
    def __call__(self, x: torch.Tensor):
        fraction = torch.rand((1,)) * self.max_fraction * 100
        fraction = fraction.type(torch.IntTensor)
        dims = x.size()
        vignettes = torch.cat(
            (
                torch.zeros(tuple(dims[:-1]) + (fraction,)),
                torch.ones(tuple(dims[:-1]) + (100-fraction,)),
            ), dim=-1
        )
        num_rot = torch.randint(4,(1,))[0]
        vignettes = torch.rot90(vignettes, k=num_rot, dims=[-2,-1])
        vignettes = transforms.functional.resize(vignettes, dims[-2:])
        vignettes = vignettes.reshape(x.size())
        y = torch.mul(x, vignettes)
        
        return y