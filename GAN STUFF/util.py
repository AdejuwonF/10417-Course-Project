from PIL import Image
import torch

def my_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')

class alphaToWhite(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        tranx, transy = torch.where(tensor[3] <= 0)
        tensor[:3,tranx,transy] = 0
        return tensor[:3]


class fillIn():
    def __init__(self):
        pass
    def __call__(self, tensor):
        C, H, W= tensor.shape
        ones = torch.ones((1,1,W))
        tensor[:,0,:] = ones
        tensor[:,H-1,:] = ones

        return tensor